"""Build deterministic pure-text product inputs for semantic recommendation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from experiments.recommendation.datasets.canonical_items import CanonicalItemUniverse


SUPPORTED_TEXT_FIELDS = ("title", "categories", "features", "description")
FIELD_LABELS = {
    "title": "Title",
    "categories": "Categories",
    "features": "Features",
    "description": "Description",
}


def _clean_scalar(value, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Product field {field!r} must be a string or null.")
    return " ".join(value.split())


def _clean_list(value, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Product field {field!r} must be a list or null.")
    cleaned: list[str] = []
    for position, element in enumerate(value):
        if not isinstance(element, str):
            raise TypeError(f"{field}[{position}] must be a string.")
        normalized = " ".join(element.split())
        if normalized:
            cleaned.append(normalized)
    return cleaned


def build_product_text(record: dict, text_fields: Iterable[str]) -> str:
    """Render configured semantic fields in a stable labeled template."""

    if not isinstance(record, dict):
        raise TypeError("Product catalog record must be a JSON object.")
    fields = tuple(text_fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("text_fields must be non-empty and duplicate-free.")
    unsupported = set(fields) - set(SUPPORTED_TEXT_FIELDS)
    if unsupported:
        raise ValueError(f"Unsupported semantic text fields: {sorted(unsupported)}.")

    lines: list[str] = []
    for field in fields:
        if field in {"categories", "features"}:
            value = " | ".join(_clean_list(record.get(field), field))
        else:
            value = _clean_scalar(record.get(field), field)
        if value:
            lines.append(f"{FIELD_LABELS[field]}: {value}")
    return "\n".join(lines)


def load_canonical_product_texts(
    product_catalog_path: str | Path,
    universe: CanonicalItemUniverse,
    text_fields: Iterable[str],
) -> list[str]:
    """Load, identity-check, and reorder catalog text to canonical embedding rows."""

    path = Path(product_catalog_path)
    if not path.is_file():
        raise FileNotFoundError(f"Product catalog not found: {path}")
    expected = {
        int(item_index): parent_asin
        for item_index, parent_asin in zip(universe.item_indices, universe.parent_asins)
    }
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid catalog JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Catalog line {line_number} must be a JSON object.")
            item_index = record.get("item_index")
            parent_asin = record.get("parent_asin")
            if isinstance(item_index, bool) or not isinstance(item_index, int):
                raise TypeError(f"Catalog line {line_number} has invalid item_index.")
            if item_index not in expected:
                raise ValueError(f"Catalog line {line_number} is outside canonical scope.")
            if parent_asin != expected[item_index]:
                raise ValueError(
                    f"Catalog identity mismatch for item_index={item_index}: "
                    f"{parent_asin!r} != {expected[item_index]!r}."
                )
            if item_index in records:
                raise ValueError(f"Duplicate catalog item_index={item_index}.")
            records[item_index] = record
    if len(records) != universe.item_indices.size:
        missing = set(expected) - records.keys()
        raise ValueError(
            f"Catalog records={len(records):,}; expected {universe.item_indices.size:,}; "
            f"missing example={min(missing) if missing else None}."
        )
    texts = [build_product_text(records[int(item)], text_fields) for item in universe.item_indices]
    # Preserve identity rows even when all upstream semantic metadata is absent.
    # The empty string is deterministic and avoids inventing ASIN/text content.
    return texts
