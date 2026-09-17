"""Deterministic tests for the V2-07.3 runtime Product Catalog index."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from src.agentrec.recommendation.artifacts import sha256_file
from src.agentrec.recommendation.catalog import (
    ProductCatalogIndex,
    RuntimeProduct,
    normalize_category_member,
    normalize_feature_tokens,
)
from src.agentrec.recommendation.contracts import RecommendationRequest


class ProductCatalogRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog_path = self.root / "product_catalog.jsonl"
        self.records = [
            {
                "item_index": 10,
                "parent_asin": "P10",
                "title": "Dock One",
                "categories": [" Electronics ", "Docking   Stations"],
                "price": 99.0,
                "features": ["Supports HDMI output and USB-C power delivery"],
                "description": "must not enter the runtime product",
            },
            {
                "item_index": 20,
                "parent_asin": "P20",
                "title": "Dock Two",
                "categories": ["Electronics", "Docking Stations"],
                "price": None,
                "features": ["HDMI video output", "Gigabit Ethernet"],
                "description": "ignored",
            },
            {
                "item_index": 40,
                "parent_asin": "P40",
                "title": "Mouse",
                "categories": ["Electronics", "Computer Mice"],
                "price": -1,
                "features": [],
                "description": "ignored",
            },
            {
                "item_index": 70,
                "parent_asin": "P70",
                "title": "Headphones",
                "categories": ["Electronics", "Headphones"],
                "price": 50,
                "features": ["Active noise cancelling with long battery life"],
                "description": "ignored",
            },
            {
                "item_index": 90,
                "parent_asin": "P90",
                "title": "Expensive Dock",
                "categories": ["Electronics", "Docking Stations"],
                "price": 150,
                "features": ["HDMI output only"],
                "description": "ignored",
            },
        ]
        with self.catalog_path.open("w", encoding="utf-8") as handle:
            for record in reversed(self.records):
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.canonical = np.array([10, 20, 40, 70, 90], dtype=np.int32)
        identity = hashlib.sha256()
        by_item = {record["item_index"]: record["parent_asin"] for record in self.records}
        for item_index in self.canonical:
            identity.update(f"{int(item_index)}\t{by_item[int(item_index)]}\n".encode())
        self.identity = identity.hexdigest()
        self.index = ProductCatalogIndex(
            self.catalog_path,
            canonical_item_indices=self.canonical,
            canonical_identity_sha256=self.identity,
            expected_catalog_sha256=sha256_file(self.catalog_path),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(self, **kwargs) -> RecommendationRequest:
        return RecommendationRequest(user_id="user", **kwargs)

    def rows(self, request: RecommendationRequest) -> list[int]:
        return np.flatnonzero(self.index.build_eligible_mask(request)).tolist()

    def test_serving_row_canonical_item_parent_round_trip(self) -> None:
        self.assertEqual(
            [
                (self.index.project_product(row).item_index,
                 self.index.project_product(row).parent_asin)
                for row in range(self.index.size)
            ],
            [(10, "P10"), (20, "P20"), (40, "P40"), (70, "P70"), (90, "P90")],
        )
        self.assertEqual(self.index.parent_asin_to_serving_row["P70"], 3)

    def test_category_exact_normalized_member_match(self) -> None:
        self.assertEqual(normalize_category_member(" Docking   Stations "), "docking stations")
        self.assertEqual(self.rows(self.request(category=" docking stations ")), [0, 1, 4])

    def test_category_mismatch_and_no_substring_match(self) -> None:
        self.assertEqual(self.rows(self.request(category="Docking Station")), [])
        self.assertEqual(self.rows(self.request(category="mouse")), [])

    def test_max_price_and_missing_or_invalid_price_fail_closed(self) -> None:
        self.assertEqual(self.rows(self.request(max_price=100)), [0, 3])
        self.assertNotIn(1, self.rows(self.request(max_price=1_000)))
        self.assertNotIn(2, self.rows(self.request(max_price=1_000)))

    def test_required_features_use_and_semantics(self) -> None:
        self.assertEqual(
            self.rows(self.request(required_features=("HDMI", "USB-C"))),
            [0],
        )
        self.assertEqual(
            self.rows(self.request(required_features=("HDMI", "Ethernet"))),
            [1],
        )

    def test_feature_normalization_and_compound_boundary(self) -> None:
        self.assertEqual(normalize_feature_tokens(" HDMI "), ("hdmi",))
        self.assertEqual(normalize_feature_tokens(" USB-C "), ("usb-c",))
        self.assertEqual(self.rows(self.request(required_features=("usb-c",))), [0])
        self.assertEqual(self.rows(self.request(required_features=("USB",))), [])
        self.assertEqual(
            self.rows(self.request(required_features=("noise   cancelling",))),
            [3],
        )

    def test_missing_features_fail_closed(self) -> None:
        self.assertNotIn(2, self.rows(self.request(required_features=("HDMI",))))

    def test_excluded_asin_and_unknown_asin(self) -> None:
        self.assertEqual(self.rows(self.request(excluded_parent_asins=("P20",))), [0, 2, 3, 4])
        self.assertEqual(self.rows(self.request(excluded_parent_asins=("UNKNOWN",))), [0, 1, 2, 3, 4])

    def test_combined_constraints(self) -> None:
        request = self.request(
            category="Docking Stations",
            max_price=120,
            required_features=("HDMI", "USB-C"),
            excluded_parent_asins=("P20",),
        )
        self.assertEqual(self.rows(request), [0])

    def test_zero_eligible_and_mask_contract(self) -> None:
        mask = self.index.build_eligible_mask(self.request(category="Not A Category"))
        self.assertEqual(mask.shape, (5,))
        self.assertEqual(mask.dtype, np.dtype("bool"))
        self.assertEqual(int(mask.sum()), 0)

    def test_compact_projection_is_immutable_and_excludes_description(self) -> None:
        product = self.index.project_product(0)
        self.assertIsInstance(product, RuntimeProduct)
        self.assertFalse(hasattr(product, "description"))
        self.assertEqual(product.categories, ("Electronics", "Docking   Stations"))
        with self.assertRaises(FrozenInstanceError):
            product.price = 1.0
        with self.assertRaises(IndexError):
            self.index.project_product(5)

    def test_catalog_fingerprint_and_identity_mismatch_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            ProductCatalogIndex(
                self.catalog_path,
                canonical_item_indices=self.canonical,
                canonical_identity_sha256=self.identity,
                expected_catalog_sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ProductCatalogIndex(
                self.catalog_path,
                canonical_item_indices=self.canonical,
                canonical_identity_sha256="0" * 64,
                expected_catalog_sha256=sha256_file(self.catalog_path),
            )


if __name__ == "__main__":
    unittest.main()
