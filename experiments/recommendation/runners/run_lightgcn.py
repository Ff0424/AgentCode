"""
Train and evaluate the AgentRec V2 standard LightGCN baseline.

Pipeline:
    Frozen Dataset Loader -> compact identity mapping -> normalized bipartite
    graph -> epoch-wise uniform unseen-negative sampling -> BPR training ->
    full-item validation -> best checkpoint -> full-item test evaluation ->
    shared ``evaluate_ranking()`` -> Markdown report.

Evaluation creates exactly one Top-max(K) ranking per user. Scores are computed
in user batches against every canonical item; train-seen items are masked before
``torch.topk``. The runner never creates the full users-by-items score matrix.

Example:
    python -m experiments.recommendation.runners.run_lightgcn \
        --config experiments/recommendation/configs/lightgcn.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.recommendation.datasets.load_dataset import (
    RecommendationDataset,
    load_dataset,
)
from experiments.recommendation.evaluation.ranking_metrics import evaluate_ranking
from experiments.recommendation.models.lightgcn import LightGCN


# ============================================================
# 1. Frozen data contract and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "experiments" / "recommendation" / "configs" / "lightgcn.yaml"
)
EXPECTED_TRAIN_INTERACTIONS = 4_257_087
EXPECTED_TRAIN_USERS = 357_974
EXPECTED_TRAIN_ITEMS = 125_684
EXPECTED_TEST_INTERACTIONS = 354_291
EXPECTED_TOTAL_ITEMS = 125_762

REQUIRED_CONFIG_KEYS = {
    "embedding_dim",
    "num_layers",
    "learning_rate",
    "batch_size",
    "epochs",
    "reg_weight",
    "seed",
    "topk",
    "eval_user_batch_size",
    "early_stopping_patience",
    "device",
    "early_stopping_metric",
    "dataset_path",
    "item_mapping_path",
    "parent_asins_path",
    "checkpoint_path",
    "report_path",
    "popularity_report_path",
}


@dataclass(frozen=True)
class ExperimentConfig:
    embedding_dim: int
    num_layers: int
    learning_rate: float
    batch_size: int
    epochs: int
    reg_weight: float
    seed: int
    topk: tuple[int, ...]
    eval_user_batch_size: int
    early_stopping_patience: int
    device: str
    early_stopping_metric: str
    dataset_path: Path
    item_mapping_path: Path
    parent_asins_path: Path
    checkpoint_path: Path
    report_path: Path
    popularity_report_path: Path


@dataclass(frozen=True)
class IdentityMapping:
    canonical_users: np.ndarray
    canonical_items: np.ndarray
    train_user: np.ndarray
    train_item: np.ndarray
    valid_user: np.ndarray
    valid_item: np.ndarray
    test_user: np.ndarray
    test_item: np.ndarray


@dataclass(frozen=True)
class ItemUniverseStats:
    canonical_items: int
    positive_split_observed_items: int
    train_observed_items: int
    train_unseen_items: int
    canonical_without_positive_items: int


@dataclass(frozen=True)
class SeenHistory:
    offsets: np.ndarray
    items: np.ndarray
    positive_pair_keys: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    overall: dict[str, float]
    warm: dict[str, float]
    cold: dict[str, float] | None
    warm_targets: int
    cold_targets: int
    elapsed_seconds: float


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    total_loss: float
    ranking_loss: float
    regularization_loss: float
    train_seconds: float
    validation: EvaluationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the AgentRec V2 LightGCN baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="LightGCN YAML configuration path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional override for the frozen split directory.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Optional override for the best-checkpoint path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional override for the Markdown report path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing checkpoint/report outputs.",
    )
    return parser.parse_args()


def _parse_scalar(raw_value: str) -> Any:
    """Parse the deliberately simple YAML subset without adding PyYAML."""

    value = raw_value.strip()
    if not value:
        raise ValueError("Configuration values must not be empty.")
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LightGCN configuration not found: {path}")
    result: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number}: {raw_line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key or key in result:
            raise ValueError(f"Invalid or duplicate config key on line {line_number}.")
        result[key] = _parse_scalar(raw_value)
    return result


def _require_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _require_float(value: Any, name: str, minimum: float, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (positive and result == 0.0):
        comparator = ">" if positive and minimum == 0.0 else ">="
        raise ValueError(f"{name} must be finite and {comparator} {minimum}.")
    return result


def _resolve_project_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path, args: argparse.Namespace) -> ExperimentConfig:
    raw = _load_simple_yaml(path)
    missing = REQUIRED_CONFIG_KEYS - raw.keys()
    extra = raw.keys() - REQUIRED_CONFIG_KEYS
    if missing or extra:
        raise ValueError(
            f"Configuration keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
        )

    topk_raw = raw["topk"]
    if not isinstance(topk_raw, list) or not topk_raw:
        raise ValueError("topk must be a non-empty YAML list.")
    topk = tuple(_require_int(value, "topk entry") for value in topk_raw)
    if len(set(topk)) != len(topk):
        raise ValueError(f"topk contains duplicate cutoffs: {topk}.")
    if topk != (10, 20, 50):
        raise ValueError(f"V2-06.3 requires topk [10, 20, 50], got {topk}.")

    device = raw["device"]
    metric = raw["early_stopping_metric"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")
    if not isinstance(metric, str) or metric not in {
        f"{prefix}@{k}" for prefix in ("Recall", "NDCG") for k in topk
    }:
        raise ValueError(f"Unsupported early_stopping_metric: {metric!r}.")

    dataset_path = (
        args.dataset_path
        if args.dataset_path is not None
        else _resolve_project_path(raw["dataset_path"], "dataset_path")
    )
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else _resolve_project_path(raw["checkpoint_path"], "checkpoint_path")
    )
    report_path = (
        args.report_path
        if args.report_path is not None
        else _resolve_project_path(raw["report_path"], "report_path")
    )
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    return ExperimentConfig(
        embedding_dim=_require_int(raw["embedding_dim"], "embedding_dim"),
        num_layers=_require_int(raw["num_layers"], "num_layers", minimum=0),
        learning_rate=_require_float(
            raw["learning_rate"], "learning_rate", minimum=0.0, positive=True
        ),
        batch_size=_require_int(raw["batch_size"], "batch_size"),
        epochs=_require_int(raw["epochs"], "epochs"),
        reg_weight=_require_float(
            raw["reg_weight"], "reg_weight", minimum=0.0, positive=False
        ),
        seed=_require_int(raw["seed"], "seed", minimum=0),
        topk=topk,
        eval_user_batch_size=_require_int(
            raw["eval_user_batch_size"], "eval_user_batch_size"
        ),
        early_stopping_patience=_require_int(
            raw["early_stopping_patience"], "early_stopping_patience"
        ),
        device=device.strip(),
        early_stopping_metric=metric,
        dataset_path=dataset_path.resolve(),
        item_mapping_path=_resolve_project_path(
            raw["item_mapping_path"], "item_mapping_path"
        ).resolve(),
        parent_asins_path=_resolve_project_path(
            raw["parent_asins_path"], "parent_asins_path"
        ).resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        report_path=report_path.resolve(),
        popularity_report_path=_resolve_project_path(
            raw["popularity_report_path"], "popularity_report_path"
        ).resolve(),
    )


# ============================================================
# 2. Reproducibility, data validation, and compact identity
# ============================================================

def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_cuda_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(
            f"The full V2-06.3 experiment requires a CUDA device, got {device_name!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run V2-06.3 on the Ubuntu GPU server.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_name!r} does not exist; count={torch.cuda.device_count()}."
        )
    return device


def _validate_frozen_dataset(dataset: RecommendationDataset) -> None:
    train_users = int(np.unique(dataset.train_user).size)
    train_items = int(np.unique(dataset.train_item).size)
    if dataset.train_user.size != EXPECTED_TRAIN_INTERACTIONS:
        raise ValueError(
            f"Train interactions={dataset.train_user.size:,}; expected "
            f"{EXPECTED_TRAIN_INTERACTIONS:,}."
        )
    if train_users != EXPECTED_TRAIN_USERS or train_items != EXPECTED_TRAIN_ITEMS:
        raise ValueError(
            f"Train users/items={train_users:,}/{train_items:,}; expected "
            f"{EXPECTED_TRAIN_USERS:,}/{EXPECTED_TRAIN_ITEMS:,}."
        )
    if dataset.test_user.size != EXPECTED_TEST_INTERACTIONS:
        raise ValueError(
            f"Test interactions={dataset.test_user.size:,}; expected "
            f"{EXPECTED_TEST_INTERACTIONS:,}."
        )
    if np.unique(dataset.test_user).size != dataset.test_user.size:
        raise ValueError("Test split must contain exactly one target per user.")
    if np.unique(dataset.valid_user).size != dataset.valid_user.size:
        raise ValueError("Validation split must contain exactly one target per user.")


def load_canonical_item_universe(
    item_mapping_path: Path,
    parent_asins_path: Path,
) -> np.ndarray:
    """Load the frozen V2-04 canonical item-index universe.

    ``parent_asins_10core.json`` defines catalog membership, while
    ``item_mapping.json`` supplies each member's stable canonical item_index.
    Positive experiment splits must never define this universe because rating
    filtering can legitimately remove catalog items.
    """

    if not parent_asins_path.is_file():
        raise FileNotFoundError(
            f"Canonical parent-ASIN file not found: {parent_asins_path}"
        )
    if not item_mapping_path.is_file():
        raise FileNotFoundError(f"Canonical item mapping not found: {item_mapping_path}")

    with parent_asins_path.open("r", encoding="utf-8") as handle:
        parent_asins = json.load(handle)
    if not isinstance(parent_asins, list):
        raise ValueError("parent_asins_10core.json must contain a JSON list.")
    if len(parent_asins) != EXPECTED_TOTAL_ITEMS:
        raise ValueError(
            f"Canonical parent-ASIN count={len(parent_asins):,}; expected "
            f"{EXPECTED_TOTAL_ITEMS:,}."
        )
    if any(not isinstance(value, str) or not value.strip() for value in parent_asins):
        raise ValueError("Every canonical parent_asin must be a non-empty string.")
    target_parent_asins = set(parent_asins)
    if len(target_parent_asins) != len(parent_asins):
        raise ValueError("Canonical parent-ASIN list contains duplicates.")

    with item_mapping_path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)
    if not isinstance(raw_mapping, dict):
        raise ValueError("item_mapping.json must contain a JSON object.")

    parent_to_item: dict[str, int] = {}
    item_to_parent: dict[int, str] = {}
    for raw_item_index, parent_asin in raw_mapping.items():
        if not isinstance(parent_asin, str):
            raise ValueError(
                f"item_mapping value for key {raw_item_index!r} must be a string."
            )
        if parent_asin not in target_parent_asins:
            continue
        try:
            item_index = int(raw_item_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid canonical item_index key: {raw_item_index!r}."
            ) from exc
        if item_index < 0 or str(item_index) != str(raw_item_index):
            raise ValueError(f"Invalid canonical item_index key: {raw_item_index!r}.")
        if parent_asin in parent_to_item:
            raise ValueError(f"Duplicate mapping for parent_asin {parent_asin!r}.")
        if item_index in item_to_parent:
            raise ValueError(
                f"Canonical item_index {item_index} maps to multiple parent ASINs."
            )
        parent_to_item[parent_asin] = item_index
        item_to_parent[item_index] = parent_asin

    missing_parent_asins = target_parent_asins - parent_to_item.keys()
    if missing_parent_asins:
        example = min(missing_parent_asins)
        raise ValueError(
            f"Canonical item mapping is missing {len(missing_parent_asins):,} target "
            f"parent ASINs; example={example!r}."
        )
    canonical_items = np.asarray(sorted(item_to_parent), dtype=np.int64)
    if canonical_items.size != EXPECTED_TOTAL_ITEMS:
        raise ValueError(
            f"Canonical item universe={canonical_items.size:,}; expected "
            f"{EXPECTED_TOTAL_ITEMS:,}."
        )
    if canonical_items[-1] > np.iinfo(np.int32).max:
        raise ValueError("Canonical item_index exceeds the supported int32 range.")
    return np.ascontiguousarray(canonical_items, dtype=np.int32)


def build_identity_mapping(
    dataset: RecommendationDataset,
    canonical_items: np.ndarray,
) -> IdentityMapping:
    canonical_users = np.unique(
        np.concatenate((dataset.train_user, dataset.valid_user, dataset.test_user))
    ).astype(np.int32, copy=False)
    if (
        canonical_items.ndim != 1
        or canonical_items.size != EXPECTED_TOTAL_ITEMS
        or np.any(canonical_items[1:] <= canonical_items[:-1])
    ):
        raise ValueError(
            "Canonical items must be the complete, strictly increasing V2-04 universe "
            f"of {EXPECTED_TOTAL_ITEMS:,} item indices."
        )

    def map_values(values: np.ndarray, universe: np.ndarray, name: str) -> np.ndarray:
        mapped = np.searchsorted(universe, values).astype(np.int32, copy=False)
        if np.any(mapped >= universe.size) or not np.array_equal(universe[mapped], values):
            raise ValueError(f"Unable to map every {name} to a compact index.")
        return np.ascontiguousarray(mapped)

    return IdentityMapping(
        canonical_users=np.ascontiguousarray(canonical_users),
        canonical_items=np.ascontiguousarray(canonical_items),
        train_user=map_values(dataset.train_user, canonical_users, "train user"),
        train_item=map_values(dataset.train_item, canonical_items, "train item"),
        valid_user=map_values(dataset.valid_user, canonical_users, "valid user"),
        valid_item=map_values(dataset.valid_item, canonical_items, "valid item"),
        test_user=map_values(dataset.test_user, canonical_users, "test user"),
        test_item=map_values(dataset.test_item, canonical_items, "test item"),
    )


def compute_item_universe_stats(mapping: IdentityMapping) -> ItemUniverseStats:
    positive_observed = np.unique(
        np.concatenate((mapping.train_item, mapping.valid_item, mapping.test_item))
    ).size
    train_observed = np.unique(mapping.train_item).size
    return ItemUniverseStats(
        canonical_items=int(mapping.canonical_items.size),
        positive_split_observed_items=int(positive_observed),
        train_observed_items=int(train_observed),
        train_unseen_items=int(mapping.canonical_items.size - train_observed),
        canonical_without_positive_items=int(
            mapping.canonical_items.size - positive_observed
        ),
    )


def build_seen_history(
    train_user: np.ndarray,
    train_item: np.ndarray,
    num_users: int,
    num_items: int,
) -> SeenHistory:
    order = np.lexsort((train_item, train_user))
    sorted_users = train_user[order]
    sorted_items = train_item[order]
    pair_keys = sorted_users.astype(np.int64) * num_items + sorted_items.astype(np.int64)
    if np.unique(pair_keys).size != pair_keys.size:
        raise ValueError("Training split contains duplicate user-item pairs.")

    counts = np.bincount(sorted_users, minlength=num_users)
    offsets = np.empty(num_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    positive_pair_keys = np.sort(pair_keys)
    return SeenHistory(
        offsets=np.ascontiguousarray(offsets),
        items=np.ascontiguousarray(sorted_items, dtype=np.int32),
        positive_pair_keys=np.ascontiguousarray(positive_pair_keys, dtype=np.int64),
    )


# ============================================================
# 3. Sparse graph and negative sampling
# ============================================================

def build_normalized_adjacency(
    train_user: np.ndarray,
    train_item: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """Build D^-1/2 A D^-1/2 for the symmetric bipartite graph."""

    user_degree = np.bincount(train_user, minlength=num_users).astype(np.float32)
    item_degree = np.bincount(train_item, minlength=num_items).astype(np.float32)
    edge_weights = 1.0 / np.sqrt(
        user_degree[train_user] * item_degree[train_item]
    )
    item_nodes = train_item.astype(np.int64) + num_users
    rows = np.concatenate((train_user.astype(np.int64), item_nodes))
    columns = np.concatenate((item_nodes, train_user.astype(np.int64)))
    values = np.concatenate((edge_weights, edge_weights)).astype(np.float32)

    indices = torch.from_numpy(np.stack((rows, columns), axis=0)).to(device)
    weights = torch.from_numpy(values).to(device)
    node_count = num_users + num_items
    return torch.sparse_coo_tensor(
        indices,
        weights,
        size=(node_count, node_count),
        device=device,
    ).coalesce()


def sample_unseen_negatives(
    user_indices: np.ndarray,
    num_items: int,
    positive_pair_keys: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample one item per user, rejecting every train-seen pair."""

    negatives = rng.integers(0, num_items, size=user_indices.size, dtype=np.int32)
    for _ in range(100):
        keys = user_indices.astype(np.int64) * num_items + negatives.astype(np.int64)
        positions = np.searchsorted(positive_pair_keys, keys)
        safe_positions = np.minimum(positions, positive_pair_keys.size - 1)
        collisions = (
            (positions < positive_pair_keys.size)
            & (positive_pair_keys[safe_positions] == keys)
        )
        if not np.any(collisions):
            return negatives
        negatives[collisions] = rng.integers(
            0,
            num_items,
            size=int(collisions.sum()),
            dtype=np.int32,
        )
    raise RuntimeError("Negative sampling did not resolve all train-seen collisions.")


# ============================================================
# 4. Full-item batched evaluation
# ============================================================

def _batch_seen_coordinates(
    users: np.ndarray,
    seen_history: SeenHistory,
) -> tuple[np.ndarray, np.ndarray]:
    counts = seen_history.offsets[users + 1] - seen_history.offsets[users]
    row_indices = np.repeat(np.arange(users.size, dtype=np.int64), counts)
    item_parts = [
        seen_history.items[
            int(seen_history.offsets[user]) : int(seen_history.offsets[user + 1])
        ]
        for user in users
        if seen_history.offsets[user + 1] > seen_history.offsets[user]
    ]
    if not item_parts:
        return row_indices, np.empty(0, dtype=np.int64)
    return row_indices, np.concatenate(item_parts).astype(np.int64, copy=False)


def evaluate_model(
    model: LightGCN,
    evaluation_users: np.ndarray,
    target_local_items: np.ndarray,
    target_canonical_items: np.ndarray,
    canonical_items: np.ndarray,
    train_observed_items: np.ndarray,
    seen_history: SeenHistory,
    topk: tuple[int, ...],
    user_batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    if not (
        evaluation_users.shape
        == target_local_items.shape
        == target_canonical_items.shape
    ):
        raise ValueError("Evaluation users and target arrays must be aligned.")
    max_k = max(topk)
    rankings = np.empty((evaluation_users.size, max_k), dtype=np.int32)

    model.eval()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        final_users, final_items = model.propagate()
        item_matrix = final_items.transpose(0, 1).contiguous()
        for start in range(0, evaluation_users.size, user_batch_size):
            stop = min(start + user_batch_size, evaluation_users.size)
            users = evaluation_users[start:stop]
            user_tensor = torch.from_numpy(users.astype(np.int64, copy=False)).to(device)
            scores = final_users[user_tensor] @ item_matrix

            seen_rows, seen_items = _batch_seen_coordinates(users, seen_history)
            if seen_items.size:
                row_tensor = torch.from_numpy(seen_rows).to(device)
                item_tensor = torch.from_numpy(seen_items).to(device)
                scores[row_tensor, item_tensor] = -torch.inf

            local_topk = torch.topk(
                scores,
                k=max_k,
                dim=1,
                largest=True,
                sorted=True,
            ).indices.cpu().numpy()
            rankings[start:stop] = canonical_items[local_topk]
            del scores, user_tensor, local_topk
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    warm_mask = train_observed_items[target_local_items]
    warm_targets = int(warm_mask.sum())
    cold_targets = int((~warm_mask).sum())
    if warm_targets == 0:
        raise ValueError("Warm-start evaluation contains no targets.")
    overall = evaluate_ranking(rankings, target_canonical_items, topk=topk)
    cold = (
        evaluate_ranking(
            rankings[~warm_mask],
            target_canonical_items[~warm_mask],
            topk=topk,
        )
        if cold_targets
        else None
    )
    # Metrics are macro means. Deriving the warm partition from the overall
    # and small cold partition avoids a second full pass through ~354k users.
    warm = {
        name: float(
            (
                overall[name] * evaluation_users.size
                - (cold[name] * cold_targets if cold is not None else 0.0)
            )
            / warm_targets
        )
        for name in overall
    }
    return EvaluationResult(
        overall=overall,
        warm=warm,
        cold=cold,
        warm_targets=warm_targets,
        cold_targets=cold_targets,
        elapsed_seconds=elapsed_seconds,
    )


# ============================================================
# 5. Training and checkpoint selection
# ============================================================

def train_one_epoch(
    model: LightGCN,
    optimizer: torch.optim.Optimizer,
    train_user: np.ndarray,
    train_item: np.ndarray,
    seen_history: SeenHistory,
    config: ExperimentConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.train()
    order = rng.permutation(train_user.size)
    total_weight = 0
    total_loss_sum = 0.0
    ranking_loss_sum = 0.0
    regularization_loss_sum = 0.0

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, order.size, config.batch_size):
        batch_offsets = order[start : start + config.batch_size]
        users = train_user[batch_offsets]
        positives = train_item[batch_offsets]
        negatives = sample_unseen_negatives(
            users,
            model.num_items,
            seen_history.positive_pair_keys,
            rng,
        )

        user_tensor = torch.from_numpy(users.astype(np.int64, copy=False)).to(device)
        positive_tensor = torch.from_numpy(
            positives.astype(np.int64, copy=False)
        ).to(device)
        negative_tensor = torch.from_numpy(
            negatives.astype(np.int64, copy=False)
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        total_loss, ranking_loss, regularization_loss = model.bpr_loss(
            user_tensor,
            positive_tensor,
            negative_tensor,
            config.reg_weight,
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Training produced non-finite loss: {total_loss}.")
        total_loss.backward()
        optimizer.step()

        weight = int(users.size)
        total_weight += weight
        total_loss_sum += float(total_loss.detach().item()) * weight
        ranking_loss_sum += float(ranking_loss.item()) * weight
        regularization_loss_sum += float(regularization_loss.item()) * weight
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        total_loss_sum / total_weight,
        ranking_loss_sum / total_weight,
        regularization_loss_sum / total_weight,
        elapsed,
    )


def _save_checkpoint_atomic(
    path: Path,
    model: LightGCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    payload = {
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "embedding_dim": config.embedding_dim,
            "num_layers": config.num_layers,
        },
    }
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb+") as stream:
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_best_checkpoint(
    path: Path,
    model: LightGCN,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# ============================================================
# 6. Report generation
# ============================================================

def _read_popularity_metrics(path: Path, topk: tuple[int, ...]) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Popularity baseline report not found: {path}")
    text = path.read_text(encoding="utf-8")
    metrics: dict[str, float] = {}
    for prefix in ("Recall", "NDCG"):
        for k in topk:
            name = f"{prefix}@{k}"
            match = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([0-9.]+)\s*\|", text)
            if match is None:
                raise ValueError(f"Popularity report is missing {name}: {path}")
            metrics[name] = float(match.group(1))
    return metrics


def _metrics_table(metrics: dict[str, float], topk: tuple[int, ...]) -> list[str]:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for prefix in ("Recall", "NDCG"):
        for k in topk:
            name = f"{prefix}@{k}"
            lines.append(f"| {name} | {metrics[name]:.8f} |")
    return lines


def _comparison_table(
    popularity: dict[str, float],
    lightgcn: dict[str, float],
    topk: tuple[int, ...],
) -> list[str]:
    names = [f"Recall@{k}" for k in topk] + [f"NDCG@{k}" for k in topk]
    lines = [
        "| Method | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
        "| Popularity | "
        + " | ".join(f"{popularity[name]:.8f}" for name in names)
        + " |",
        "| LightGCN | "
        + " | ".join(f"{lightgcn[name]:.8f}" for name in names)
        + " |",
        "| Absolute delta | "
        + " | ".join(
            f"{lightgcn[name] - popularity[name]:+.8f}" for name in names
        )
        + " |",
        "| Relative delta | "
        + " | ".join(
            f"{(lightgcn[name] / popularity[name] - 1.0) * 100.0:+.2f}%"
            if popularity[name] != 0.0
            else "n/a"
            for name in names
        )
        + " |",
    ]
    return lines


def _format_report(
    config: ExperimentConfig,
    dataset: RecommendationDataset,
    mapping: IdentityMapping,
    item_universe_stats: ItemUniverseStats,
    graph_edges: int,
    epoch_records: list[EpochRecord],
    best_epoch: int,
    best_validation: EvaluationResult,
    test_result: EvaluationResult,
    training_seconds: float,
    validation_seconds: float,
    peak_gpu_bytes: int,
    popularity_metrics: dict[str, float],
) -> str:
    best_record = next(record for record in epoch_records if record.epoch == best_epoch)
    gpu_name = torch.cuda.get_device_name(torch.device(config.device))
    lines = [
        "# AgentRec V2 LightGCN Baseline",
        "",
        "## Experiment Configuration",
        "",
        f"- Dataset path: `{config.dataset_path}`",
        f"- Checkpoint path: `{config.checkpoint_path}`",
        f"- Report path: `{config.report_path}`",
        "- Fit data: training interactions only",
        "- Validation selects the checkpoint; test is evaluated once afterward",
        "",
        "## Dataset Statistics",
        "",
        "| Split | Interactions | Unique Users | Unique Items |",
        "| --- | ---: | ---: | ---: |",
        f"| Train | {dataset.train_user.size:,} | {np.unique(dataset.train_user).size:,} | {np.unique(dataset.train_item).size:,} |",
        f"| Valid | {dataset.valid_user.size:,} | {np.unique(dataset.valid_user).size:,} | {np.unique(dataset.valid_item).size:,} |",
        f"| Test | {dataset.test_user.size:,} | {np.unique(dataset.test_user).size:,} | {np.unique(dataset.test_item).size:,} |",
        "",
        f"- Canonical users represented in the positive split: {mapping.canonical_users.size:,}",
        f"- Canonical item universe (V2-04 product layer): {item_universe_stats.canonical_items:,}",
        f"- Positive-split observed item universe: {item_universe_stats.positive_split_observed_items:,}",
        f"- Train-observed item universe: {item_universe_stats.train_observed_items:,}",
        f"- Train-unseen/cold items: {item_universe_stats.train_unseen_items:,}",
        f"- Canonical items without a positive split interaction: {item_universe_stats.canonical_without_positive_items:,}",
        "",
        "## Model Architecture",
        "",
        f"LightGCN uses {config.embedding_dim}-dimensional user/item ego embeddings, "
        f"{config.num_layers} rounds of normalized sparse graph propagation, and mean "
        "aggregation over the ego plus propagated layers. It contains no feature "
        "transformation, nonlinear activation, attention, semantic feature, or reranker.",
        "",
        "## Hyperparameters",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
        f"| embedding_dim | {config.embedding_dim} |",
        f"| num_layers | {config.num_layers} |",
        f"| learning_rate | {config.learning_rate} |",
        f"| batch_size | {config.batch_size:,} |",
        f"| maximum_epochs | {config.epochs} |",
        f"| reg_weight | {config.reg_weight} |",
        f"| seed | {config.seed} |",
        f"| topk | {list(config.topk)} |",
        f"| eval_user_batch_size | {config.eval_user_batch_size} |",
        f"| early_stopping_patience | {config.early_stopping_patience} |",
        f"| early_stopping_metric | {config.early_stopping_metric} |",
        "",
        "## Graph Construction",
        "",
        f"The graph has {mapping.canonical_users.size + mapping.canonical_items.size:,} "
        f"nodes and {graph_edges:,} directed edges after adding both directions for every "
        "train interaction. Each edge uses D^-1/2 A D^-1/2 normalization. Canonical IDs "
        "are mapped to compact model-local indices and mapped back before evaluation.",
        "",
        "## BPR Loss and Negative Sampling",
        "",
        "Each train interaction supplies one user-positive pair. One negative item is "
        "sampled uniformly from the full canonical item universe for every pair and every "
        "epoch. Vectorized rejection sampling checks encoded user-item keys against all "
        "train-seen pairs, so a positive or other seen item cannot be sampled. The objective "
        "is mean -log sigmoid(s(u,i+) - s(u,i-)) plus L2 regularization on the sampled ego "
        "embeddings.",
        "",
        "## Training Configuration",
        "",
        f"- Device: `{config.device}` ({gpu_name})",
        "- Optimizer: Adam",
        "- Validation: full-item Top-50 after every epoch",
        "- Seen-item filtering: all train interactions",
        f"- Best validation epoch: **{best_epoch}**",
        f"- Epochs executed: {len(epoch_records)}",
        f"- Best epoch total/ranking/regularization loss: {best_record.total_loss:.8f} / "
        f"{best_record.ranking_loss:.8f} / {best_record.regularization_loss:.8f}",
        "",
        "## Best Validation Metrics Overall",
        "",
        *_metrics_table(best_validation.overall, config.topk),
        "",
        "## Best Validation Metrics Warm Start",
        "",
        *_metrics_table(best_validation.warm, config.topk),
        "",
        "## Test Metrics Overall",
        "",
        *_metrics_table(test_result.overall, config.topk),
        "",
        "## Test Metrics Warm Start",
        "",
        *_metrics_table(test_result.warm, config.topk),
        "",
        "## Warm and Cold Targets",
        "",
        "| Split | Warm Targets | Cold Targets | Cold Rate |",
        "| --- | ---: | ---: | ---: |",
        f"| Valid | {best_validation.warm_targets:,} | {best_validation.cold_targets:,} | "
        f"{best_validation.cold_targets / (best_validation.warm_targets + best_validation.cold_targets):.6%} |",
        f"| Test | {test_result.warm_targets:,} | {test_result.cold_targets:,} | "
        f"{test_result.cold_targets / (test_result.warm_targets + test_result.cold_targets):.6%} |",
        "",
    ]
    if best_validation.cold is not None:
        lines.extend(["## Best Validation Metrics Cold Start", "", *_metrics_table(best_validation.cold, config.topk), ""])
    if test_result.cold is not None:
        lines.extend(["## Test Metrics Cold Start", "", *_metrics_table(test_result.cold, config.topk), ""])
    lines.extend(
        [
            "## Runtime",
            "",
            f"- Generated at UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- Training compute time: {training_seconds:.2f} seconds",
            f"- Total validation evaluation time: {validation_seconds:.2f} seconds",
            f"- Final test evaluation time: {test_result.elapsed_seconds:.2f} seconds",
            f"- Peak allocated GPU memory: {peak_gpu_bytes / (1024 ** 3):.3f} GiB",
            "",
            "## Comparison with Popularity",
            "",
            *_comparison_table(popularity_metrics, test_result.overall, config.topk),
            "",
            "## Conclusion",
            "",
            "The comparison above uses the same frozen temporal split, full canonical item "
            "universe, train-seen filtering, one Top-50 ranking per user, and shared "
            "Recall/NDCG implementation. Validation selected the checkpoint; test metrics "
            "were computed only after selection.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


# ============================================================
# 7. End-to-end experiment
# ============================================================

def run_experiment(config: ExperimentConfig, overwrite: bool) -> Path:
    for output_path in (config.checkpoint_path, config.report_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )

    device = _validate_cuda_device(config.device)
    seed_everything(config.seed)
    dataset = load_dataset(config.dataset_path)
    _validate_frozen_dataset(dataset)
    canonical_items = load_canonical_item_universe(
        config.item_mapping_path,
        config.parent_asins_path,
    )
    mapping = build_identity_mapping(dataset, canonical_items)
    item_universe_stats = compute_item_universe_stats(mapping)
    print(
        "item_universe "
        f"canonical={item_universe_stats.canonical_items:,} "
        f"positive_split_observed={item_universe_stats.positive_split_observed_items:,} "
        f"train_observed={item_universe_stats.train_observed_items:,} "
        f"train_unseen={item_universe_stats.train_unseen_items:,} "
        f"without_positive={item_universe_stats.canonical_without_positive_items:,}",
        flush=True,
    )
    seen_history = build_seen_history(
        mapping.train_user,
        mapping.train_item,
        mapping.canonical_users.size,
        mapping.canonical_items.size,
    )
    adjacency = build_normalized_adjacency(
        mapping.train_user,
        mapping.train_item,
        mapping.canonical_users.size,
        mapping.canonical_items.size,
        device,
    )
    model = LightGCN(
        num_users=mapping.canonical_users.size,
        num_items=mapping.canonical_items.size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        normalized_adjacency=adjacency,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)

    train_observed_items = np.zeros(mapping.canonical_items.size, dtype=bool)
    train_observed_items[np.unique(mapping.train_item)] = True
    epoch_records: list[EpochRecord] = []
    best_metric = -math.inf
    best_epoch = 0
    best_validation: EvaluationResult | None = None
    patience = 0
    total_training_seconds = 0.0
    total_validation_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        total_loss, ranking_loss, reg_loss, train_seconds = train_one_epoch(
            model,
            optimizer,
            mapping.train_user,
            mapping.train_item,
            seen_history,
            config,
            rng,
            device,
        )
        validation = evaluate_model(
            model=model,
            evaluation_users=mapping.valid_user,
            target_local_items=mapping.valid_item,
            target_canonical_items=dataset.valid_item,
            canonical_items=mapping.canonical_items,
            train_observed_items=train_observed_items,
            seen_history=seen_history,
            topk=config.topk,
            user_batch_size=config.eval_user_batch_size,
            device=device,
        )
        record = EpochRecord(
            epoch=epoch,
            total_loss=total_loss,
            ranking_loss=ranking_loss,
            regularization_loss=reg_loss,
            train_seconds=train_seconds,
            validation=validation,
        )
        epoch_records.append(record)
        total_training_seconds += train_seconds
        total_validation_seconds += validation.elapsed_seconds
        current_metric = validation.overall[config.early_stopping_metric]
        print(
            f"epoch={epoch:03d} loss={total_loss:.8f} "
            f"{config.early_stopping_metric}={current_metric:.8f} "
            f"train_s={train_seconds:.2f} valid_s={validation.elapsed_seconds:.2f}",
            flush=True,
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_validation = validation
            patience = 0
            _save_checkpoint_atomic(
                config.checkpoint_path,
                model,
                optimizer,
                epoch,
                config.early_stopping_metric,
                current_metric,
                config,
            )
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break

    if best_validation is None or best_epoch == 0:
        raise RuntimeError("Training completed without selecting a checkpoint.")
    checkpoint = _load_best_checkpoint(config.checkpoint_path, model, device)
    if int(checkpoint["epoch"]) != best_epoch:
        raise ValueError("Reloaded checkpoint epoch does not match selected best epoch.")

    test_result = evaluate_model(
        model=model,
        evaluation_users=mapping.test_user,
        target_local_items=mapping.test_item,
        target_canonical_items=dataset.test_item,
        canonical_items=mapping.canonical_items,
        train_observed_items=train_observed_items,
        seen_history=seen_history,
        topk=config.topk,
        user_batch_size=config.eval_user_batch_size,
        device=device,
    )
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device))
    popularity_metrics = _read_popularity_metrics(
        config.popularity_report_path,
        config.topk,
    )
    report = _format_report(
        config=config,
        dataset=dataset,
        mapping=mapping,
        item_universe_stats=item_universe_stats,
        graph_edges=2 * dataset.train_user.size,
        epoch_records=epoch_records,
        best_epoch=best_epoch,
        best_validation=best_validation,
        test_result=test_result,
        training_seconds=total_training_seconds,
        validation_seconds=total_validation_seconds,
        peak_gpu_bytes=peak_gpu_bytes,
        popularity_metrics=popularity_metrics,
    )
    _write_text_atomic(config.report_path, report)
    return config.report_path


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config.resolve(), args)
        report_path = run_experiment(config, overwrite=args.overwrite)
    except (
        FileNotFoundError,
        FloatingPointError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"LightGCN experiment completed. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
