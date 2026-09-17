"""Deterministic CPU tests for V2-07.4 RecommendationService."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from src.agentrec.recommendation.artifacts import SCHEMA_VERSION_V2, sha256_file
from src.agentrec.recommendation.catalog import ProductCatalogIndex
from src.agentrec.recommendation.contracts import RecommendationRequest
from src.agentrec.recommendation.service import (
    HYBRID_ALPHA,
    RecommendationService,
    _copy_numpy_to_device,
    _population_zscore_numpy,
)


class FakeArtifacts:
    def __init__(self) -> None:
        self.manifest = {
            "schema_version": SCHEMA_VERSION_V2,
            "artifact_version": "fixture-v2",
            "embedding_dim": 2,
        }
        self.canonical_user_resolver = object()
        self.model_user_indices = np.array([2], dtype=np.int32)
        self.model_user_ids = ("known",)
        self.canonical_item_indices = np.array(
            [10, 20, 30, 40, 50, 60], dtype=np.int32
        )
        self.lightgcn_user_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        self.lightgcn_item_embeddings = np.array(
            [[1.0, 0.0], [0.9, 0.0], [0.1, 1.0], [0.0, 0.8], [0.2, 0.2], [0.9, 0.0]],
            dtype=np.float32,
        )
        self.semantic_item_embeddings = np.zeros((6, 1024), dtype=np.float32)
        self.semantic_item_embeddings[0, 0] = 1.0
        self.semantic_item_embeddings[1, 0] = 1.0
        self.semantic_item_embeddings[2, :2] = (0.9, 0.1)
        self.semantic_item_embeddings[3, 1] = 1.0
        self.semantic_item_embeddings[4, 2] = 1.0
        self.semantic_item_embeddings[5, 0] = 1.0
        self.train_seen_offsets = np.array([0, 1], dtype=np.int64)
        self.train_seen_items = np.array([0], dtype=np.int32)
        self.train_observed_mask = np.array([True] * 6, dtype=bool)
        self.popular_item_indices = np.array([40, 20, 60, 30, 10, 50], dtype=np.int32)

    def resolve_canonical_user(self, raw_user_id: str) -> int | None:
        return {"known": 2, "canonical-unseen": 3}.get(raw_user_id)

    def resolve_model_user_row(self, canonical_user_index: int) -> int | None:
        return 0 if canonical_user_index == 2 else None


class ServiceFixture:
    def __init__(self, root: Path) -> None:
        self.artifacts = FakeArtifacts()
        self.catalog_path = root / "catalog.jsonl"
        records = [
            (10, "P10", "Seen Dock", 50.0, ["Docking Stations"], ["HDMI", "USB-C"]),
            (20, "P20", "Dock A", 80.0, ["Docking Stations"], ["HDMI", "USB-C"]),
            (30, "P30", "Dock B", 150.0, ["Docking Stations"], ["HDMI"]),
            (40, "P40", "Mouse", 20.0, ["Computer Mice"], ["USB"]),
            (50, "P50", "Missing Price", None, ["Docking Stations"], ["HDMI", "USB-C"]),
            (60, "P60", "Dock Tie", 80.0, ["Docking Stations"], ["HDMI", "USB-C"]),
        ]
        with self.catalog_path.open("w", encoding="utf-8") as handle:
            for item, parent, title, price, categories, features in reversed(records):
                handle.write(
                    json.dumps(
                        {
                            "item_index": item,
                            "parent_asin": parent,
                            "title": title,
                            "price": price,
                            "categories": categories,
                            "features": features,
                        }
                    )
                    + "\n"
                )
        identity = hashlib.sha256()
        for item, parent, *_ in records:
            identity.update(f"{item}\t{parent}\n".encode("utf-8"))
        self.artifacts.manifest.update(
            canonical_identity_sha256=identity.hexdigest(),
            sources={
                "product_catalog": {
                    "sha256": sha256_file(self.catalog_path),
                }
            },
        )
        self.catalog = ProductCatalogIndex.from_serving_artifacts(
            self.catalog_path, self.artifacts
        )
        self.service = RecommendationService.for_cpu_testing(
            artifacts=self.artifacts, catalog=self.catalog
        )


class RecommendationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ServiceFixture(Path(self.temporary.name))
        self.service = self.fixture.service

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_identity_resolution_covers_all_three_classes(self) -> None:
        known = self.service.recommend(RecommendationRequest(user_id="known", top_k=1))
        unseen = self.service.recommend(
            RecommendationRequest(user_id="canonical-unseen", top_k=1)
        )
        unknown = self.service.recommend(RecommendationRequest(user_id="unknown", top_k=1))
        self.assertEqual((known.canonical_user_index, known.model_user_row), (2, 0))
        self.assertEqual(known.personalization_status, "personalized")
        self.assertEqual(
            (unseen.canonical_user_index, unseen.model_user_row), (3, None)
        )
        self.assertEqual(unseen.personalization_status, "canonical_known_model_unseen")
        self.assertEqual((unknown.canonical_user_index, unknown.model_user_row), (None, None))
        self.assertEqual(unknown.personalization_status, "unknown_user")

    def test_known_user_full_score_shapes_alpha_and_components(self) -> None:
        lightgcn_z, semantic_z, hybrid = self.service._backend.score_known_user(0)
        self.assertEqual(lightgcn_z.shape, (6,))
        self.assertEqual(semantic_z.shape, (6,))
        np.testing.assert_allclose(
            hybrid, (1.0 - HYBRID_ALPHA) * lightgcn_z + HYBRID_ALPHA * semantic_z
        )
        result = self.service.recommend(RecommendationRequest(user_id="known", top_k=2))
        self.assertTrue(all(item.score_source == "hybrid" for item in result.items))
        self.assertEqual(
            set(result.items[0].score_components),
            {"lightgcn_z", "semantic_z", "hybrid_score"},
        )
        self.assertEqual(HYBRID_ALPHA, 0.7)

    def test_semantic_profile_uses_only_train_history_and_is_normalized(self) -> None:
        backend = self.service._backend
        profile = backend._semantic_profile(0)
        np.testing.assert_allclose(profile, self.fixture.artifacts.semantic_item_embeddings[0])
        self.assertAlmostEqual(float(np.linalg.norm(profile)), 1.0, places=6)
        self.assertFalse(hasattr(self.fixture.artifacts, "valid_items"))
        self.assertFalse(hasattr(self.fixture.artifacts, "test_items"))

    def test_full_catalog_zscore_precedes_constraint_filtering(self) -> None:
        _, _, full_hybrid = self.service._backend.score_known_user(0)
        result = self.service.recommend(
            RecommendationRequest(
                user_id="known", top_k=2, category="Docking Stations", max_price=80
            )
        )
        for item in result.items:
            row = self.fixture.catalog.parent_asin_to_serving_row[item.parent_asin]
            self.assertAlmostEqual(item.score, float(full_hybrid[row]), places=6)
        subset = full_hybrid[[1, 5]]
        self.assertFalse(np.allclose(subset, _population_zscore_numpy(subset)))

    def test_train_seen_item_and_excluded_asin_are_never_returned(self) -> None:
        result = self.service.recommend(
            RecommendationRequest(user_id="known", top_k=6, excluded_parent_asins=("P20",))
        )
        ids = {item.parent_asin for item in result.items}
        self.assertNotIn("P10", ids)
        self.assertNotIn("P20", ids)

    def test_category_price_and_feature_hard_constraints(self) -> None:
        result = self.service.recommend(
            RecommendationRequest(
                user_id="known",
                top_k=6,
                category=" docking   stations ",
                max_price=100,
                required_features=("HDMI", "USB-C"),
            )
        )
        self.assertEqual({item.parent_asin for item in result.items}, {"P20", "P60"})
        self.assertTrue(all(item.price is not None and item.price <= 100 for item in result.items))

    def test_zero_eligible_returns_empty_without_relaxation(self) -> None:
        result = self.service.recommend(
            RecommendationRequest(user_id="unknown", category="Impossible Category")
        )
        self.assertEqual(result.returned_count, 0)
        self.assertEqual(result.items, ())

    def test_top_k_above_eligible_count_returns_only_eligible(self) -> None:
        result = self.service.recommend(
            RecommendationRequest(
                user_id="unknown",
                top_k=6,
                category="Docking Stations",
                max_price=80,
                required_features=("USB-C",),
                excluded_parent_asins=("P60",),
            )
        )
        self.assertEqual([item.parent_asin for item in result.items], ["P20", "P10"])

    def test_fallback_preserves_popularity_order_and_constraints(self) -> None:
        for user in ("canonical-unseen", "unknown"):
            with self.subTest(user=user):
                result = self.service.recommend(
                    RecommendationRequest(
                        user_id=user, top_k=3, category="Docking Stations", max_price=100
                    )
                )
                self.assertEqual(
                    [item.parent_asin for item in result.items], ["P20", "P60", "P10"]
                )
                self.assertTrue(
                    all(item.score_source == "popularity_fallback" for item in result.items)
                )
                self.assertTrue(
                    all(set(item.score_components) == {"popularity_rank"} for item in result.items)
                )

    def test_deterministic_tie_break_uses_canonical_item_index(self) -> None:
        first = self.service.recommend(RecommendationRequest(user_id="known", top_k=6))
        second = self.service.recommend(RecommendationRequest(user_id="known", top_k=6))
        first_ids = [item.item_index for item in first.items]
        self.assertEqual(first_ids, [item.item_index for item in second.items])
        self.assertLess(first_ids.index(20), first_ids.index(60))

    def test_response_identity_round_trip_and_immutable_contract(self) -> None:
        result = self.service.recommend(RecommendationRequest(user_id="known", top_k=2))
        for item in result.items:
            row = self.fixture.catalog.parent_asin_to_serving_row[item.parent_asin]
            product = self.fixture.catalog.project_product(row)
            self.assertEqual((item.item_index, item.parent_asin), (product.item_index, product.parent_asin))
        with self.assertRaises(FrozenInstanceError):
            result.items[0].rank = 99
        with self.assertRaises(TypeError):
            result.items[0].score_components["hybrid_score"] = 0.0

    def test_startup_artifact_alignment_mismatch_fails_fast(self) -> None:
        artifacts = self.fixture.artifacts
        artifacts.lightgcn_item_embeddings = artifacts.lightgcn_item_embeddings[:-1]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            RecommendationService(
                artifacts=artifacts,
                catalog=self.fixture.catalog,
                scoring_backend=object(),
            )

    def test_readonly_artifact_is_explicitly_copied_to_owned_device_storage(self) -> None:
        class FakeTorch:
            def __init__(self) -> None:
                self.calls = []

            def tensor(self, array, *, device):
                self.calls.append((array, device))
                return np.array(array, copy=True)

            def as_tensor(self, *_args, **_kwargs):
                raise AssertionError("CUDA artifact initialization must not use as_tensor.")

        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        source.setflags(write=False)
        fake_torch = FakeTorch()
        copied = _copy_numpy_to_device(fake_torch, source, "cuda:0")

        self.assertFalse(source.flags.writeable)
        self.assertTrue(copied.flags.writeable)
        self.assertEqual(copied.dtype, source.dtype)
        self.assertEqual(fake_torch.calls[0][1], "cuda:0")
        copied[0, 0] = 999.0
        self.assertEqual(float(source[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
