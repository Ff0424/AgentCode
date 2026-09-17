"""
Build the canonical AgentRec V2 product catalog for the 10-core item set.

Inputs:
    --item-mapping
        Recommendation integer item index -> Amazon parent_asin JSON mapping.
    --parent-asins
        Ordered 10-core parent_asin JSON list that fixes the catalog scope.
    --product-documents
        Product Documents V2 JSONL used as the authoritative product facts.

Output:
    --output
        Row-oriented product_catalog.jsonl joining Recommendation item indices
        to the normalized product facts consumed by RAG and Agent modules.

The script does not modify upstream Recommendation or RAG artifacts. Output is
written to a temporary file and atomically published only after every 10-core
item has both a valid recommendation mapping and a Product Document V2 record.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


# ============================================================
# 1. Product Document V2 field definitions
# ============================================================

TEXT_LABELS = {
    "Title": "title",
    "Category": "category",
    "Features": "features",
    "Product Details": "product_details",
    "Description": "description",
}


def _require_file(path: Path, label: str) -> None:
    """Fail early when a required input is absent or is not a regular file."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def _load_json(path: Path, label: str):
    """Load one UTF-8 JSON file with a path-specific error message."""

    _require_file(path, label)
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} file {path}: {exc}") from exc
    except UnicodeError as exc:
        raise ValueError(f"{label} file is not valid UTF-8: {path}") from exc


def _validate_parent_asin(value, context: str) -> str:
    """Return a validated parent ASIN without silently coercing its type."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value


# ============================================================
# 2. Canonical 10-core identity mapping
# ============================================================

def load_target_parent_asins(path: Path) -> list[str]:
    """Load the ordered, unique 10-core product identity list."""

    values = _load_json(path, "10-core parent ASIN")
    if not isinstance(values, list):
        raise TypeError("10-core parent ASIN file must contain a JSON list.")
    if not values:
        raise ValueError("10-core parent ASIN list must not be empty.")

    parent_asins = []
    seen = set()
    for position, value in enumerate(values):
        parent_asin = _validate_parent_asin(
            value, f"parent_asins[{position}]"
        )
        if parent_asin in seen:
            raise ValueError(
                f"Duplicate parent_asin {parent_asin!r} at position {position}."
            )
        seen.add(parent_asin)
        parent_asins.append(parent_asin)

    return parent_asins


def load_target_item_indices(
    path: Path,
    target_parent_asins: set[str],
) -> dict[str, int]:
    """Resolve each target parent ASIN to its original recommendation index."""

    mapping = _load_json(path, "item mapping")
    if not isinstance(mapping, dict):
        raise TypeError("item mapping file must contain a JSON object.")

    target_indices = {}
    seen_indices = set()
    for raw_index, raw_parent_asin in mapping.items():
        if not isinstance(raw_index, str) or not raw_index.isdecimal():
            raise ValueError(
                f"item mapping key must be a non-negative integer string: "
                f"{raw_index!r}."
            )
        item_index = int(raw_index)
        if item_index in seen_indices:
            raise ValueError(f"Duplicate item_index in item mapping: {item_index}.")
        seen_indices.add(item_index)

        parent_asin = _validate_parent_asin(
            raw_parent_asin, f"item_mapping[{raw_index!r}]"
        )
        if parent_asin not in target_parent_asins:
            continue
        if parent_asin in target_indices:
            raise ValueError(
                f"Target parent_asin maps to multiple item indices: "
                f"{parent_asin!r}."
            )
        target_indices[parent_asin] = item_index

    missing = target_parent_asins.difference(target_indices)
    if missing:
        examples = sorted(missing)[:10]
        raise ValueError(
            f"Item mapping is missing {len(missing):,} 10-core parent ASINs; "
            f"examples: {examples}."
        )

    return target_indices


# ============================================================
# 3. Product Document V2 parsing and validation
# ============================================================

def _split_nonempty(value: str, separator: str) -> list[str]:
    """Split a structured text field while preserving source order."""

    return [part.strip() for part in value.split(separator) if part.strip()]


def parse_document_text(text: str) -> dict:
    """Parse optional labeled sections from a Product Document V2 string."""

    if not isinstance(text, str):
        raise TypeError("Product document text must be a string.")

    sections = {field_name: [] for field_name in TEXT_LABELS.values()}
    active_field = None

    for raw_line in text.splitlines():
        matched_field = None
        matched_value = None
        for label, field_name in TEXT_LABELS.items():
            prefix = f"{label}:"
            if raw_line.startswith(prefix):
                matched_field = field_name
                matched_value = raw_line[len(prefix) :].strip()
                break

        if matched_field is not None:
            active_field = matched_field
            if matched_value:
                sections[active_field].append(matched_value)
        elif active_field is not None and raw_line.strip():
            # Preserve wrapped section lines until the next recognized label.
            sections[active_field].append(raw_line.strip())

    joined = {
        field_name: "\n".join(parts).strip()
        for field_name, parts in sections.items()
    }

    category_text = joined["category"]
    if "|" in category_text:
        categories = _split_nonempty(category_text, "|")
    elif ">" in category_text:
        categories = _split_nonempty(category_text, ">")
    elif category_text:
        categories = [category_text]
    else:
        categories = []

    feature_text = joined["features"]
    features = _split_nonempty(feature_text, "|") if feature_text else []

    return {
        "title": joined["title"] or None,
        "categories": categories,
        "features": features,
        "description": joined["description"] or None,
    }


def _optional_finite_number(value, context: str):
    """Validate optional JSON numeric metadata and reject bool/NaN/Inf."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number or null.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite when present.")
    return value


def _normalize_optional_price(value):
    """Keep valid finite prices and map unusable source values to null."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Product Documents V2 contains a few corrupted non-numeric prices.
        # Treat them as unavailable instead of inventing or coercing a value.
        return None
    return float(value) if math.isfinite(float(value)) else None


def load_target_documents(
    path: Path,
    target_parent_asins: set[str],
) -> dict[str, dict]:
    """Stream Product Documents V2 and retain facts for target products only."""

    _require_file(path, "Product Documents V2")
    products = {}
    seen_document_ids = set()

    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty Product Document at source line {line_number}."
                    )
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid Product Document JSON at line {line_number}: "
                        f"{exc}."
                    ) from exc
                if not isinstance(document, dict):
                    raise TypeError(
                        f"Product Document line {line_number} must be an object."
                    )

                document_id = _validate_parent_asin(
                    document.get("id"), f"document id at line {line_number}"
                )
                if document_id in seen_document_ids:
                    raise ValueError(
                        f"Duplicate Product Document id {document_id!r} at "
                        f"line {line_number}."
                    )
                seen_document_ids.add(document_id)

                metadata = document.get("metadata")
                if not isinstance(metadata, dict):
                    raise TypeError(
                        f"metadata at Product Document line {line_number} "
                        "must be an object."
                    )
                metadata_id = _validate_parent_asin(
                    metadata.get("parent_asin"),
                    f"metadata.parent_asin at line {line_number}",
                )
                if document_id != metadata_id:
                    raise ValueError(
                        f"Document id and metadata.parent_asin differ at line "
                        f"{line_number}."
                    )
                if document_id not in target_parent_asins:
                    continue

                parsed = parse_document_text(document.get("text"))
                products[document_id] = {
                    "title": parsed["title"],
                    "categories": parsed["categories"],
                    "price": _normalize_optional_price(metadata.get("price")),
                    "average_rating": _optional_finite_number(
                        metadata.get("average_rating"),
                        f"metadata.average_rating at line {line_number}",
                    ),
                    "rating_number": _optional_finite_number(
                        metadata.get("rating_number"),
                        f"metadata.rating_number at line {line_number}",
                    ),
                    "features": parsed["features"],
                    "description": parsed["description"],
                }
    except UnicodeError as exc:
        raise ValueError(f"Product Documents V2 is not valid UTF-8: {path}") from exc

    return products


# ============================================================
# 4. Catalog construction and safe publication
# ============================================================

def build_product_catalog(
    item_mapping_path: Path,
    parent_asins_path: Path,
    product_documents_path: Path,
    output_path: Path,
) -> dict:
    """Build and atomically publish the canonical 10-core product catalog."""

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    parent_asins = load_target_parent_asins(parent_asins_path)
    target_set = set(parent_asins)
    item_indices = load_target_item_indices(item_mapping_path, target_set)
    products = load_target_documents(product_documents_path, target_set)

    missing_products = [asin for asin in parent_asins if asin not in products]
    total_items = len(parent_asins)
    matched_products = total_items - len(missing_products)
    match_rate = matched_products / total_items

    statistics = {
        "total_items": total_items,
        "matched_products": matched_products,
        "missing_products": len(missing_products),
        "match_rate": match_rate,
    }

    print(f"Total items: {total_items:,}")
    print(f"Matched products: {matched_products:,}")
    print(f"Missing products: {len(missing_products):,}")
    print(f"Match rate: {match_rate:.4%}")

    if missing_products:
        raise ValueError(
            f"Product Documents V2 is missing {len(missing_products):,} "
            f"10-core products; examples: {missing_products[:10]}. "
            "No catalog was published."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary output already exists; inspect or remove it first: "
            f"{temporary_path}"
        )

    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as output_file:
            for parent_asin in parent_asins:
                product = products[parent_asin]
                record = {
                    "item_index": int(item_indices[parent_asin]),
                    "parent_asin": parent_asin,
                    "title": product["title"],
                    "categories": product["categories"],
                    "price": product["price"],
                    "average_rating": product["average_rating"],
                    "rating_number": product["rating_number"],
                    "features": product["features"],
                    "description": product["description"],
                }
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )

            # Ensure the complete catalog reaches the filesystem before publish.
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return statistics


# ============================================================
# 5. Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AgentRec V2 canonical 10-core product catalog."
    )
    parser.add_argument(
        "--item-mapping",
        required=True,
        type=Path,
        help="Path to item_mapping.json.",
    )
    parser.add_argument(
        "--parent-asins",
        required=True,
        type=Path,
        help="Path to the ordered parent_asins_10core.json scope.",
    )
    parser.add_argument(
        "--product-documents",
        required=True,
        type=Path,
        help="Path to product_documents_v2.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination product_catalog.jsonl path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_product_catalog(
            item_mapping_path=args.item_mapping,
            parent_asins_path=args.parent_asins,
            product_documents_path=args.product_documents,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Product catalog written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
