"""Runtime RecommendationService over the frozen AgentRec serving v2 bundle.

The service resolves raw user identity, applies deterministic hard constraints,
and then selects either personalized Uniform Hybrid scoring or the frozen
popularity fallback. It never trains models, embeds text, or reads validation
or test interactions.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import SCHEMA_VERSION_V2, ServingArtifacts, load_serving_artifacts
from .catalog import ProductCatalogIndex, RuntimeProduct
from .contracts import RecommendationRequest, RecommendationResult, RecommendedProduct


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "artifacts/recommendation/serving_v2"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data/processed/recommendation/product_catalog.jsonl"
DEFAULT_DEVICE = "cuda:0"
HYBRID_ALPHA = 0.7
ZSCORE_EPSILON = 1e-6
SEMANTIC_PROFILE_CACHE_SIZE = 256
HYBRID_VARIANT = "uniform"


def _population_zscore_numpy(scores: np.ndarray) -> np.ndarray:
    """Normalize one complete item score vector with population statistics."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Full-catalog scores must be a finite non-empty vector.")
    mean = np.mean(values, dtype=np.float64)
    std = np.std(values, dtype=np.float64)
    normalized = (values - mean) / (std + ZSCORE_EPSILON)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Full-catalog Z-score produced non-finite values.")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _copy_numpy_to_device(torch_module: Any, array: np.ndarray, device: Any):
    """Create device-owned storage without sharing the read-only artifact mmap."""

    # torch.tensor has explicit copy semantics, unlike as_tensor/from_numpy.
    return torch_module.tensor(np.asarray(array), device=device)


class _NumpyScoringBackend:
    """Deterministic CPU backend used by local tests and optional CPU serving."""

    def __init__(self, artifacts: ServingArtifacts, cache_size: int) -> None:
        self._user = artifacts.lightgcn_user_embeddings
        self._items = artifacts.lightgcn_item_embeddings
        self._semantic = artifacts.semantic_item_embeddings
        self._offsets = artifacts.train_seen_offsets
        self._seen = artifacts.train_seen_items
        self._cache_size = cache_size
        self._profiles: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _semantic_profile(self, model_user_row: int) -> np.ndarray:
        with self._lock:
            cached = self._profiles.get(model_user_row)
            if cached is not None:
                self._profiles.move_to_end(model_user_row)
                return cached
        start = int(self._offsets[model_user_row])
        stop = int(self._offsets[model_user_row + 1])
        history = self._seen[start:stop]
        if history.size == 0:
            raise ValueError("Model-known user has no frozen train history.")
        profile = np.mean(self._semantic[history], axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(profile))
        if not math.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError("Semantic train-history profile has invalid norm.")
        profile = np.ascontiguousarray(profile / norm, dtype=np.float32)
        with self._lock:
            self._profiles[model_user_row] = profile
            self._profiles.move_to_end(model_user_row)
            while len(self._profiles) > self._cache_size:
                self._profiles.popitem(last=False)
        return profile

    def score_known_user(
        self, model_user_row: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lightgcn_raw = self._items @ self._user[model_user_row]
        profile = self._semantic_profile(model_user_row)
        semantic_raw = self._semantic @ profile
        lightgcn_z = _population_zscore_numpy(lightgcn_raw)
        semantic_z = _population_zscore_numpy(semantic_raw)
        hybrid = (1.0 - HYBRID_ALPHA) * lightgcn_z + HYBRID_ALPHA * semantic_z
        return lightgcn_z, semantic_z, np.ascontiguousarray(hybrid, dtype=np.float32)


class _TorchScoringBackend:
    """CUDA-resident production backend; PyTorch is imported only at startup."""

    def __init__(
        self,
        artifacts: ServingArtifacts,
        device: str,
        cache_size: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for CUDA recommendation serving.") from exc
        parsed = torch.device(device)
        if parsed.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("RecommendationService production device must be available CUDA.")
        index = 0 if parsed.index is None else parsed.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable.")
        torch.cuda.set_device(index)
        self._torch = torch
        self._device = torch.device("cuda", index)
        self._user = _copy_numpy_to_device(
            torch, artifacts.lightgcn_user_embeddings, self._device
        )
        self._items = _copy_numpy_to_device(
            torch, artifacts.lightgcn_item_embeddings, self._device
        )
        self._semantic = _copy_numpy_to_device(
            torch, artifacts.semantic_item_embeddings, self._device
        )
        self._offsets = artifacts.train_seen_offsets
        self._seen = artifacts.train_seen_items
        self._cache_size = cache_size
        self._profiles: OrderedDict[int, Any] = OrderedDict()
        self._lock = threading.Lock()

    def _semantic_profile(self, model_user_row: int):
        with self._lock:
            cached = self._profiles.get(model_user_row)
            if cached is not None:
                self._profiles.move_to_end(model_user_row)
                return cached
        start = int(self._offsets[model_user_row])
        stop = int(self._offsets[model_user_row + 1])
        history = self._seen[start:stop]
        if history.size == 0:
            raise ValueError("Model-known user has no frozen train history.")
        rows = self._torch.as_tensor(history.astype(np.int64), device=self._device)
        profile = self._semantic[rows].mean(dim=0)
        norm = self._torch.linalg.vector_norm(profile)
        if not bool(self._torch.isfinite(norm)) or float(norm) <= 0.0:
            raise FloatingPointError("Semantic train-history profile has invalid norm.")
        profile = profile / norm
        with self._lock:
            self._profiles[model_user_row] = profile
            self._profiles.move_to_end(model_user_row)
            while len(self._profiles) > self._cache_size:
                self._profiles.popitem(last=False)
        return profile

    def _zscore(self, scores):
        mean = scores.mean()
        std = scores.std(unbiased=False)
        normalized = (scores - mean) / (std + ZSCORE_EPSILON)
        if not bool(self._torch.isfinite(normalized).all()):
            raise FloatingPointError("Full-catalog Z-score produced non-finite values.")
        return normalized

    def score_known_user(
        self, model_user_row: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._torch.inference_mode():
            lightgcn_raw = self._items @ self._user[model_user_row]
            semantic_raw = self._semantic @ self._semantic_profile(model_user_row)
            lightgcn_z = self._zscore(lightgcn_raw)
            semantic_z = self._zscore(semantic_raw)
            hybrid = (1.0 - HYBRID_ALPHA) * lightgcn_z + HYBRID_ALPHA * semantic_z
        return tuple(
            value.detach().to("cpu", dtype=self._torch.float32).numpy()
            for value in (lightgcn_z, semantic_z, hybrid)
        )


class RecommendationService:
    """State-free online recommendation over immutable serving artifacts."""

    def __init__(
        self,
        bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        *,
        device: str = DEFAULT_DEVICE,
        artifacts: ServingArtifacts | None = None,
        catalog: ProductCatalogIndex | None = None,
        scoring_backend: Any | None = None,
        semantic_profile_cache_size: int = SEMANTIC_PROFILE_CACHE_SIZE,
    ) -> None:
        if (
            isinstance(semantic_profile_cache_size, bool)
            or not isinstance(semantic_profile_cache_size, int)
            or semantic_profile_cache_size <= 0
        ):
            raise ValueError("semantic_profile_cache_size must be a positive integer.")
        self.artifacts = artifacts or load_serving_artifacts(
            bundle_dir, project_root=PROJECT_ROOT, validate_catalog=True
        )
        if self.artifacts.manifest["schema_version"] != SCHEMA_VERSION_V2 or (
            self.artifacts.canonical_user_resolver is None
        ):
            raise ValueError("RecommendationService requires a validated serving v2 bundle.")
        self.catalog = catalog or ProductCatalogIndex.from_serving_artifacts(
            catalog_path, self.artifacts
        )
        self._validate_startup_alignment()
        self._backend = scoring_backend or _TorchScoringBackend(
            self.artifacts, device, semantic_profile_cache_size
        )

    @classmethod
    def for_cpu_testing(
        cls,
        *,
        artifacts: ServingArtifacts,
        catalog: ProductCatalogIndex,
    ) -> "RecommendationService":
        """Construct the exact service flow with NumPy math for deterministic tests."""

        return cls(
            artifacts=artifacts,
            catalog=catalog,
            scoring_backend=_NumpyScoringBackend(
                artifacts, SEMANTIC_PROFILE_CACHE_SIZE
            ),
        )

    def _validate_startup_alignment(self) -> None:
        users = self.artifacts.model_user_indices.size
        items = self.artifacts.canonical_item_indices.size
        if self.catalog.size != items:
            raise ValueError("Product Catalog and serving item counts differ.")
        expected_shapes = {
            "lightgcn_user_embeddings": (users, self.artifacts.manifest["embedding_dim"]),
            "lightgcn_item_embeddings": (items, self.artifacts.manifest["embedding_dim"]),
            "semantic_item_embeddings": (items, 1024),
            "train_seen_offsets": (users + 1,),
            "train_observed_mask": (items,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self.artifacts, name).shape != shape:
                raise ValueError(f"Serving artifact {name} shape mismatch at service startup.")

    @staticmethod
    def _applied_constraints(request: RecommendationRequest) -> dict[str, object]:
        return {
            "category": request.category,
            "max_price": request.max_price,
            "required_features": request.required_features,
            "excluded_parent_asins": request.excluded_parent_asins,
        }

    def _identity(self, raw_user_id: str) -> tuple[int | None, int | None, str, str | None]:
        canonical = self.artifacts.resolve_canonical_user(raw_user_id)
        if canonical is None:
            return None, None, "unknown_user", "raw_user_not_in_canonical_mapping"
        model_row = self.artifacts.resolve_model_user_row(canonical)
        if model_row is None:
            return canonical, None, "canonical_known_model_unseen", "canonical_user_not_in_model"
        return canonical, model_row, "personalized", None

    @staticmethod
    def _recommended_product(
        rank: int,
        product: RuntimeProduct,
        score: float,
        components: dict[str, float],
        source: str,
    ) -> RecommendedProduct:
        return RecommendedProduct(
            rank=rank,
            item_index=product.item_index,
            parent_asin=product.parent_asin,
            title=product.title,
            price=product.price,
            categories=product.categories,
            features=product.features,
            score=float(score),
            score_components={key: float(value) for key, value in components.items()},
            score_source=source,
        )

    def _personalized_items(
        self,
        model_user_row: int,
        eligible: np.ndarray,
        top_k: int,
    ) -> tuple[RecommendedProduct, ...]:
        lightgcn_z, semantic_z, hybrid = self._backend.score_known_user(model_user_row)
        if any(values.shape != (self.catalog.size,) for values in (lightgcn_z, semantic_z, hybrid)):
            raise ValueError("Known-user scoring did not return full-catalog vectors.")
        if not all(np.isfinite(values).all() for values in (lightgcn_z, semantic_z, hybrid)):
            raise FloatingPointError("Known-user scoring returned non-finite values.")

        allowed = eligible.copy()
        start = int(self.artifacts.train_seen_offsets[model_user_row])
        stop = int(self.artifacts.train_seen_offsets[model_user_row + 1])
        allowed[self.artifacts.train_seen_items[start:stop]] = False
        candidates = np.flatnonzero(allowed)
        if candidates.size == 0:
            return ()
        # Full-catalog normalization has already happened; only now select candidates.
        order = np.lexsort(
            (self.artifacts.canonical_item_indices[candidates], -hybrid[candidates])
        )
        rows = candidates[order[:top_k]]
        return tuple(
            self._recommended_product(
                rank,
                self.catalog.project_product(int(row)),
                hybrid[row],
                {
                    "lightgcn_z": lightgcn_z[row],
                    "semantic_z": semantic_z[row],
                    "hybrid_score": hybrid[row],
                },
                "hybrid",
            )
            for rank, row in enumerate(rows, 1)
        )

    def _fallback_items(
        self,
        eligible: np.ndarray,
        top_k: int,
    ) -> tuple[RecommendedProduct, ...]:
        canonical = self.artifacts.canonical_item_indices
        results: list[RecommendedProduct] = []
        for popularity_rank, item_index in enumerate(
            self.artifacts.popular_item_indices, 1
        ):
            row = int(np.searchsorted(canonical, item_index))
            if row >= canonical.size or int(canonical[row]) != int(item_index):
                raise ValueError("Popularity item cannot be resolved to a serving row.")
            if not eligible[row]:
                continue
            reciprocal_rank = 1.0 / popularity_rank
            results.append(
                self._recommended_product(
                    len(results) + 1,
                    self.catalog.project_product(row),
                    reciprocal_rank,
                    {"popularity_rank": float(popularity_rank)},
                    "popularity_fallback",
                )
            )
            if len(results) == top_k:
                break
        return tuple(results)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        """Resolve identity and return constrained personalized or fallback items."""

        if not isinstance(request, RecommendationRequest):
            raise TypeError("request must be a RecommendationRequest.")
        canonical, model_row, status, fallback_reason = self._identity(request.user_id)
        eligible = self.catalog.build_eligible_mask(request)
        constraints = self._applied_constraints(request)
        if not eligible.any():
            return RecommendationResult(
                canonical_user_index=canonical,
                model_user_row=model_row,
                personalization_status=status,
                applied_constraints=constraints,
                artifact_version=self.artifacts.manifest["artifact_version"],
                fallback_reason=fallback_reason,
            )
        items = (
            self._personalized_items(model_row, eligible, request.top_k)
            if model_row is not None
            else self._fallback_items(eligible, request.top_k)
        )
        return RecommendationResult(
            canonical_user_index=canonical,
            model_user_row=model_row,
            personalization_status=status,
            applied_constraints=constraints,
            artifact_version=self.artifacts.manifest["artifact_version"],
            items=items,
            fallback_reason=fallback_reason,
        )
