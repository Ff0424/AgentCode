"""Load the frozen V2-04 canonical product identity universe.

The 10-core parent-ASIN list defines catalog membership. ``item_mapping.json``
provides the stable Recommendation ``item_index``. Returned rows are sorted by
canonical item index and therefore form an explicit row-to-item mapping; callers
must not assume that an item index equals its array row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_CANONICAL_ITEMS = 125_762


@dataclass(frozen=True)
class CanonicalItemUniverse:
    item_indices: np.ndarray
    parent_asins: tuple[str, ...]
    item_indices_sha256: str
    identity_sha256: str


def _load_json(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON in {label} {path}: {exc}") from exc


def load_canonical_item_universe(
    item_mapping_path: str | Path,
    parent_asins_path: str | Path,
    expected_items: int = EXPECTED_CANONICAL_ITEMS,
) -> CanonicalItemUniverse:
    """Return the validated canonical products in ascending item-index order."""

    item_mapping_path = Path(item_mapping_path)
    parent_asins_path = Path(parent_asins_path)
    parent_values = _load_json(parent_asins_path, "canonical parent-ASIN file")
    if not isinstance(parent_values, list) or len(parent_values) != expected_items:
        actual = len(parent_values) if isinstance(parent_values, list) else "non-list"
        raise ValueError(
            f"Canonical parent-ASIN count={actual}; expected {expected_items:,}."
        )
    if any(not isinstance(value, str) or not value.strip() for value in parent_values):
        raise ValueError("Every canonical parent_asin must be a non-empty string.")
    target = set(parent_values)
    if len(target) != expected_items:
        raise ValueError("Canonical parent-ASIN list contains duplicates.")

    raw_mapping = _load_json(item_mapping_path, "canonical item mapping")
    if not isinstance(raw_mapping, dict):
        raise TypeError("item_mapping.json must contain a JSON object.")
    by_item: dict[int, str] = {}
    seen_parents: set[str] = set()
    for raw_index, parent_asin in raw_mapping.items():
        if not isinstance(parent_asin, str):
            raise TypeError(f"item_mapping[{raw_index!r}] must be a string.")
        if parent_asin not in target:
            continue
        if not isinstance(raw_index, str) or not raw_index.isdecimal():
            raise ValueError(f"Invalid canonical item_index key: {raw_index!r}.")
        item_index = int(raw_index)
        if item_index in by_item or parent_asin in seen_parents:
            raise ValueError("Canonical item mapping is not one-to-one.")
        by_item[item_index] = parent_asin
        seen_parents.add(parent_asin)
    missing = target - seen_parents
    if missing:
        raise ValueError(
            f"Canonical mapping misses {len(missing):,} parent ASINs; "
            f"example={min(missing)!r}."
        )

    ordered = sorted(by_item.items())
    indices = np.asarray([item for item, _ in ordered], dtype=np.int64)
    if indices.size != expected_items or np.any(indices[1:] <= indices[:-1]):
        raise ValueError("Canonical item indices must be unique and strictly increasing.")
    if indices[-1] > np.iinfo(np.int32).max:
        raise ValueError("Canonical item_index exceeds int32 range.")
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    parents = tuple(parent for _, parent in ordered)
    indices_hash = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
    identity = hashlib.sha256()
    for item_index, parent_asin in ordered:
        identity.update(f"{item_index}\t{parent_asin}\n".encode("utf-8"))
    indices.setflags(write=False)
    return CanonicalItemUniverse(
        item_indices=indices,
        parent_asins=parents,
        item_indices_sha256=indices_hash,
        identity_sha256=identity.hexdigest(),
    )


def map_canonical_items(values: np.ndarray, universe: np.ndarray, label: str) -> np.ndarray:
    """Map canonical item indices to compact rows with strict membership checks."""

    values = np.asarray(values)
    rows = np.searchsorted(universe, values)
    if np.any(rows >= universe.size) or not np.array_equal(universe[rows], values):
        raise ValueError(f"Every {label} must belong to the canonical item universe.")
    return np.ascontiguousarray(rows, dtype=np.int32)
