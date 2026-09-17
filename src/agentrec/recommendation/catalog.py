"""Read-only Product Catalog index and deterministic hard-constraint masks.

The catalog is scanned once during process startup and aligned through the
explicit serving-row-to-canonical-item mapping. It retains only compact product
fields needed by recommendation output and hard filtering; descriptions and
model vectors are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .artifacts import ServingArtifacts, sha256_file
from .contracts import RecommendationRequest


FEATURE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-+./][a-z0-9]+)*")


def normalize_category_member(value: str) -> str:
    """Normalize one category member without substring or synonym expansion."""

    if not isinstance(value, str):
        raise TypeError("Category value must be a string.")
    return " ".join(value.split()).casefold()


def normalize_feature_tokens(value: str) -> tuple[str, ...]:
    """Tokenize feature text while preserving compounds such as ``usb-c``."""

    if not isinstance(value, str):
        raise TypeError("Feature value must be a string.")
    normalized = " ".join(value.split()).casefold()
    return tuple(FEATURE_TOKEN_PATTERN.findall(normalized))


@dataclass(frozen=True, slots=True)
class RuntimeProduct:
    """Compact immutable product projection retained by the runtime catalog."""

    serving_row: int
    item_index: int
    parent_asin: str
    title: str | None
    price: float | None
    categories: tuple[str, ...]
    features: tuple[str, ...]


def _optional_title(value: object, line_number: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Catalog title at line {line_number} must be string or null.")
    return value.strip() or None


def _text_tuple(value: object, name: str, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Catalog {name} at line {line_number} must be list[str].")
    return tuple(item.strip() for item in value if item.strip())


def _valid_price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    price = float(value)
    return price if math.isfinite(price) and price > 0.0 else None


class ProductCatalogIndex:
    """Canonical-row-aligned product records and in-memory filtering indices."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        canonical_item_indices: np.ndarray,
        canonical_identity_sha256: str,
        expected_catalog_sha256: str,
    ) -> None:
        self.catalog_path = Path(catalog_path).resolve()
        canonical = np.asarray(canonical_item_indices)
        if canonical.ndim != 1 or canonical.dtype != np.dtype("int32"):
            raise TypeError("canonical_item_indices must be a one-dimensional int32 array.")
        if canonical.size == 0 or np.any(canonical[1:] <= canonical[:-1]):
            raise ValueError("canonical_item_indices must be non-empty and strictly increasing.")
        if not isinstance(canonical_identity_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", canonical_identity_sha256
        ) is None:
            raise ValueError("canonical_identity_sha256 must be a SHA-256 digest.")
        if not isinstance(expected_catalog_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_catalog_sha256
        ) is None:
            raise ValueError("expected_catalog_sha256 must be a SHA-256 digest.")
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Product Catalog not found: {self.catalog_path}")
        if sha256_file(self.catalog_path) != expected_catalog_sha256:
            raise ValueError("Product Catalog fingerprint does not match serving manifest.")

        self._canonical_item_indices = canonical
        self._canonical_item_indices.setflags(write=False)
        self._products = self._load_products(canonical, canonical_identity_sha256)
        self._parent_to_row = MappingProxyType(
            {product.parent_asin: product.serving_row for product in self._products}
        )
        self._prices = np.asarray(
            [np.nan if product.price is None else product.price for product in self._products],
            dtype=np.float64,
        )
        self._prices.setflags(write=False)
        self._category_rows = self._build_category_index()
        self._feature_rows = self._build_feature_index()

    @classmethod
    def from_serving_artifacts(
        cls,
        catalog_path: str | Path,
        artifacts: ServingArtifacts,
    ) -> "ProductCatalogIndex":
        """Bind the Catalog to the already validated serving manifest."""

        source = artifacts.manifest["sources"]["product_catalog"]
        return cls(
            catalog_path,
            canonical_item_indices=artifacts.canonical_item_indices,
            canonical_identity_sha256=artifacts.manifest["canonical_identity_sha256"],
            expected_catalog_sha256=source["sha256"],
        )

    @property
    def size(self) -> int:
        return len(self._products)

    @property
    def parent_asin_to_serving_row(self) -> Mapping[str, int]:
        return self._parent_to_row

    def _load_products(
        self,
        canonical_items: np.ndarray,
        expected_identity: str,
    ) -> tuple[RuntimeProduct, ...]:
        by_item: dict[int, tuple[str, str | None, float | None, tuple[str, ...], tuple[str, ...]]] = {}
        seen_parents: set[str] = set()
        try:
            with self.catalog_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid Product Catalog JSON at line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise TypeError(f"Catalog line {line_number} must be an object.")
                    item_index = record.get("item_index")
                    parent_asin = record.get("parent_asin")
                    if (
                        isinstance(item_index, bool)
                        or not isinstance(item_index, int)
                        or item_index < 0
                    ):
                        raise ValueError(f"Invalid item_index at Catalog line {line_number}.")
                    if not isinstance(parent_asin, str) or not parent_asin.strip():
                        raise ValueError(f"Invalid parent_asin at Catalog line {line_number}.")
                    parent_asin = parent_asin.strip()
                    if item_index in by_item or parent_asin in seen_parents:
                        raise ValueError(f"Duplicate Catalog identity at line {line_number}.")
                    by_item[item_index] = (
                        parent_asin,
                        _optional_title(record.get("title"), line_number),
                        _valid_price(record.get("price")),
                        _text_tuple(record.get("categories"), "categories", line_number),
                        _text_tuple(record.get("features"), "features", line_number),
                    )
                    seen_parents.add(parent_asin)
        except UnicodeError as exc:
            raise ValueError(f"Product Catalog is not valid UTF-8: {self.catalog_path}") from exc

        if len(by_item) != canonical_items.size:
            raise ValueError(
                f"Catalog rows={len(by_item):,}; expected {canonical_items.size:,}."
            )
        identity = hashlib.sha256()
        products: list[RuntimeProduct] = []
        for row, raw_item_index in enumerate(canonical_items):
            item_index = int(raw_item_index)
            if item_index not in by_item:
                raise ValueError(f"Catalog is missing canonical item_index={item_index}.")
            parent_asin, title, price, categories, features = by_item[item_index]
            identity.update(f"{item_index}\t{parent_asin}\n".encode("utf-8"))
            products.append(
                RuntimeProduct(
                    serving_row=row,
                    item_index=item_index,
                    parent_asin=parent_asin,
                    title=title,
                    price=price,
                    categories=categories,
                    features=features,
                )
            )
        if identity.hexdigest() != expected_identity:
            raise ValueError("Catalog canonical item/parent-ASIN identity mismatch.")
        return tuple(products)

    def _build_category_index(self) -> Mapping[str, np.ndarray]:
        rows: dict[str, list[int]] = {}
        for product in self._products:
            for member in {normalize_category_member(value) for value in product.categories}:
                if member:
                    rows.setdefault(member, []).append(product.serving_row)
        result = {}
        for member, values in rows.items():
            array = np.asarray(values, dtype=np.int32)
            array.setflags(write=False)
            result[member] = array
        return MappingProxyType(result)

    def _build_feature_index(self) -> Mapping[str, np.ndarray]:
        # Each posting list contains a row at most once, even if a token repeats.
        rows: dict[str, list[int]] = {}
        for product in self._products:
            tokens = {
                token
                for feature in product.features
                for token in normalize_feature_tokens(feature)
            }
            for token in tokens:
                rows.setdefault(token, []).append(product.serving_row)
        result = {}
        for token, values in rows.items():
            array = np.asarray(values, dtype=np.int32)
            array.setflags(write=False)
            result[token] = array
        return MappingProxyType(result)

    @staticmethod
    def _contains_phrase(product: RuntimeProduct, tokens: tuple[str, ...]) -> bool:
        width = len(tokens)
        for feature in product.features:
            member = normalize_feature_tokens(feature)
            if any(member[start : start + width] == tokens for start in range(len(member) - width + 1)):
                return True
        return False

    def _feature_mask(self, requirement: str) -> np.ndarray:
        tokens = normalize_feature_tokens(requirement)
        mask = np.zeros(self.size, dtype=bool)
        if not tokens:
            return mask
        postings = [self._feature_rows.get(token) for token in tokens]
        if any(rows is None for rows in postings):
            return mask
        candidates = postings[0]
        assert candidates is not None
        for rows in postings[1:]:
            assert rows is not None
            candidates = np.intersect1d(candidates, rows, assume_unique=True)
            if candidates.size == 0:
                return mask
        if len(tokens) == 1:
            mask[candidates] = True
        else:
            matched = [
                int(row)
                for row in candidates
                if self._contains_phrase(self._products[int(row)], tokens)
            ]
            mask[matched] = True
        return mask

    def build_eligible_mask(self, request: RecommendationRequest) -> np.ndarray:
        """Return a new bool mask after deterministic hard-constraint AND filtering."""

        if not isinstance(request, RecommendationRequest):
            raise TypeError("request must be a RecommendationRequest.")
        eligible = np.ones(self.size, dtype=bool)
        if request.category is not None:
            category_rows = self._category_rows.get(
                normalize_category_member(request.category)
            )
            category_mask = np.zeros(self.size, dtype=bool)
            if category_rows is not None:
                category_mask[category_rows] = True
            eligible &= category_mask
        if request.max_price is not None:
            # NaN represents missing/invalid/non-positive price and fails closed.
            eligible &= np.isfinite(self._prices) & (self._prices <= request.max_price)
        for requirement in request.required_features:
            eligible &= self._feature_mask(requirement)
            if not eligible.any():
                break
        for parent_asin in request.excluded_parent_asins:
            row = self._parent_to_row.get(parent_asin)
            if row is not None:
                eligible[row] = False
        return eligible

    def project_product(self, serving_row: int) -> RuntimeProduct:
        """Return the immutable compact record for one exact serving row."""

        if isinstance(serving_row, bool) or not isinstance(serving_row, numbers.Integral):
            raise TypeError("serving_row must be an integer.")
        row = int(serving_row)
        if not 0 <= row < self.size:
            raise IndexError(f"serving_row={row} is outside [0, {self.size}).")
        return self._products[row]
