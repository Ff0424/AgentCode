"""Deterministic tests for V2-07.2 contracts and serving artifacts."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from experiments.recommendation.models.popularity import PopularityRecommender
from scripts.recommendation.export_serving_artifacts import build_train_seen_csr
from src.agentrec.recommendation.artifacts import (
    EXPECTED_CANONICAL_ITEMS,
    EXPECTED_TRAIN_OBSERVED_ITEMS,
    SCHEMA_VERSION,
    load_serving_artifacts,
    sha256_file,
    validate_manifest_schema,
)
from src.agentrec.recommendation.contracts import (
    RecommendationRequest,
    RecommendationResult,
    RecommendedProduct,
)


ROW_MAPPINGS = {
    "lightgcn_user_embeddings": "row i -> model_user_indices[i] -> canonical user_index",
    "model_user_ids": "row i -> raw/API user_id for model_user_indices[i]",
    "lightgcn_item_embeddings": "row i -> canonical_item_indices[i] -> canonical item_index",
    "train_seen_csr": "model user row u -> serving item rows in offsets[u]:offsets[u+1]",
    "semantic_item_embeddings": "row i -> canonical_item_indices[i] -> canonical item_index",
    "popular_item_indices": "ranked canonical item_index values",
}


class ServingFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bundle = root / "serving"
        self.bundle.mkdir()
        self.arrays = {
            "lightgcn_user_embeddings": np.arange(4, dtype=np.float32).reshape(2, 2),
            "lightgcn_item_embeddings": np.arange(6, dtype=np.float32).reshape(3, 2),
            "model_user_indices": np.array([4, 9], dtype=np.int32),
            "canonical_item_indices": np.array([10, 20, 40], dtype=np.int32),
            "train_seen_offsets": np.array([0, 2, 3], dtype=np.int64),
            "train_seen_items": np.array([0, 1, 1], dtype=np.int32),
            "train_observed_mask": np.array([True, True, False], dtype=bool),
            "popular_item_indices": np.array([20, 10], dtype=np.int32),
        }
        file_specs = {}
        for name, array in self.arrays.items():
            path = self.bundle / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            file_specs[name] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "shape": list(array.shape),
                "dtype": array.dtype.name,
            }
        user_ids_path = self.bundle / "model_user_ids.json"
        user_ids_path.write_text('["user-four","user-nine"]\n', encoding="utf-8")
        file_specs["model_user_ids"] = {
            "path": user_ids_path.name,
            "sha256": sha256_file(user_ids_path),
            "count": 2,
        }
        semantic_path = root / "semantic.npy"
        np.save(semantic_path, np.ones((3, 1024), dtype=np.float32), allow_pickle=False)
        catalog_path = root / "catalog.jsonl"
        with catalog_path.open("w", encoding="utf-8") as handle:
            for item, parent in ((40, "P40"), (10, "P10"), (20, "P20")):
                handle.write(json.dumps({"item_index": item, "parent_asin": parent}) + "\n")
        identity = hashlib.sha256()
        for item, parent in ((10, "P10"), (20, "P20"), (40, "P40")):
            identity.update(f"{item}\t{parent}\n".encode("utf-8"))
        identity_sha256 = identity.hexdigest()
        metadata_path = root / "semantic.json"
        metadata_path.write_text(
            json.dumps(
                {"provenance": {"canonical_identity_sha256": identity_sha256}}
            ) + "\n",
            encoding="utf-8",
        )
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_version": "fixture-v1",
            "created_at": "2026-09-12T00:00:00+00:00",
            "num_model_users": 2,
            "num_canonical_items": 3,
            "num_train_observed_items": 2,
            "num_train_unseen_items": 1,
            "embedding_dim": 2,
            "lightgcn_layers": 1,
            "canonical_identity_sha256": identity_sha256,
            "files": file_specs,
            "external_artifacts": {
                "semantic_item_embeddings": {
                    "path": semantic_path.name,
                    "sha256": sha256_file(semantic_path),
                },
                "semantic_metadata": {
                    "path": metadata_path.name,
                    "sha256": sha256_file(metadata_path),
                },
            },
            "sources": {
                name: {
                    "path": catalog_path.name,
                    "sha256": sha256_file(catalog_path),
                }
                for name in (
                    "lightgcn_checkpoint",
                    "train_split",
                    "user_mapping",
                    "item_mapping",
                    "parent_asins",
                    "product_catalog",
                )
            },
            "row_mappings": ROW_MAPPINGS,
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.bundle / "serving_manifest.json").write_text(
            json.dumps(self.manifest, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


class ContractTests(unittest.TestCase):
    def test_request_validation_and_normalization(self) -> None:
        request = RecommendationRequest(
            user_id="  user-1  ",
            required_features=(" HDMI ", "", " Ethernet "),
            excluded_parent_asins=("B2", "B1", "B2"),
        )
        self.assertEqual(request.user_id, "user-1")
        self.assertEqual(request.required_features, ("HDMI", "Ethernet"))
        self.assertEqual(request.excluded_parent_asins, ("B2", "B1"))
        with self.assertRaises(ValueError):
            RecommendationRequest(user_id=" ")
        with self.assertRaises(ValueError):
            RecommendationRequest(user_id="u", top_k=101)
        with self.assertRaises(TypeError):
            RecommendationRequest(user_id="u", top_k=True)
        with self.assertRaises(ValueError):
            RecommendationRequest(user_id="u", max_price=0)

    def test_response_is_frozen_and_compact(self) -> None:
        item = RecommendedProduct(
            rank=1, item_index=10, parent_asin="P10", title="Title", price=1.0,
            categories=("Electronics",), features=("HDMI",), score=0.5,
            score_components={"hybrid": 0.5}, score_source="cold_aware_hybrid",
        )
        result = RecommendationResult(
            canonical_user_index=4, model_user_row=0,
            personalization_status="personalized", applied_constraints={},
            artifact_version="fixture-v1", items=(item,),
        )
        self.assertEqual(result.returned_count, 1)
        with self.assertRaises(FrozenInstanceError):
            result.model_user_row = 1
        with self.assertRaises(TypeError):
            item.score_components["hybrid"] = 1.0


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = ServingFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_schema_and_all_round_trips(self) -> None:
        validate_manifest_schema(self.fixture.manifest)
        artifacts = load_serving_artifacts(
            self.fixture.bundle, project_root=self.root, validate_catalog=True
        )
        self.assertEqual(
            list(zip(artifacts.model_user_ids, artifacts.model_user_indices.tolist())),
            [("user-four", 4), ("user-nine", 9)],
        )
        self.assertEqual(artifacts.canonical_item_indices.tolist(), [10, 20, 40])
        histories = [
            artifacts.train_seen_items[
                artifacts.train_seen_offsets[row] : artifacts.train_seen_offsets[row + 1]
            ].tolist()
            for row in range(2)
        ]
        self.assertEqual(histories, [[0, 1], [1]])
        for array in (
            artifacts.lightgcn_user_embeddings,
            artifacts.lightgcn_item_embeddings,
            artifacts.model_user_indices,
            artifacts.train_seen_offsets,
            artifacts.train_seen_items,
            artifacts.train_observed_mask,
            artifacts.semantic_item_embeddings,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = array.flat[0]

    def test_checksum_mismatch_and_corruption_fail_fast(self) -> None:
        self.fixture.manifest["files"]["model_user_indices"]["sha256"] = "0" * 64
        self.fixture.write_manifest()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_serving_artifacts(
                self.fixture.bundle, project_root=self.root, validate_catalog=False
            )

    def test_corrupted_npy_with_matching_checksum_fails_fast(self) -> None:
        path = self.fixture.bundle / "model_user_indices.npy"
        path.write_bytes(b"not-a-valid-npy")
        self.fixture.manifest["files"]["model_user_indices"]["sha256"] = sha256_file(path)
        self.fixture.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unable to open"):
            load_serving_artifacts(
                self.fixture.bundle, project_root=self.root, validate_catalog=False
            )

    def test_invalid_csr_is_rejected(self) -> None:
        path = self.fixture.bundle / "train_seen_offsets.npy"
        np.save(path, np.array([0, 3, 2], dtype=np.int64), allow_pickle=False)
        self.fixture.manifest["files"]["train_seen_offsets"]["sha256"] = sha256_file(path)
        self.fixture.write_manifest()
        with self.assertRaisesRegex(ValueError, "CSR offsets"):
            load_serving_artifacts(
                self.fixture.bundle, project_root=self.root, validate_catalog=False
            )

    def test_manifest_schema_rejects_unknown_version(self) -> None:
        self.fixture.manifest["schema_version"] = "999"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_manifest_schema(self.fixture.manifest)

    def test_production_observed_counts_are_frozen(self) -> None:
        mask = np.zeros(EXPECTED_CANONICAL_ITEMS, dtype=bool)
        mask[:EXPECTED_TRAIN_OBSERVED_ITEMS] = True
        self.assertEqual(int(mask.sum()), 125_684)
        self.assertEqual(int((~mask).sum()), 78)

    def test_popularity_tie_break_is_deterministic(self) -> None:
        users = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        items = np.array([20, 10, 20, 10, 40, 30], dtype=np.int32)
        model = PopularityRecommender().fit(users, items)
        self.assertEqual(model.popular_items.tolist(), [10, 20, 30, 40])

    def test_exported_csr_preserves_sorted_user_item_pairs(self) -> None:
        users = np.array([1, 0, 1, 0], dtype=np.int32)
        items = np.array([2, 1, 0, 0], dtype=np.int32)
        offsets, seen = build_train_seen_csr(users, items, num_users=2)
        np.testing.assert_array_equal(offsets, np.array([0, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(seen, np.array([0, 1, 0, 2], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
