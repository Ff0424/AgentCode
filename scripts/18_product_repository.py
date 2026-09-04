"""
Read-only runtime repository for Product Documents V2.

The repository performs one complete JSONL scan during initialization and
keeps normalized products in memory for subsequent constant-time ID lookup.
It does not load embedding or FAISS artifacts, call a model, access the
network, rank products, or write any files.

Example:
    python scripts/18_product_repository.py --product-id B08SW2KQFV
"""

import argparse
import copy
import json
import sys
from pathlib import Path


# ============================================================
# 1. Formal data configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "product_documents_v2.jsonl"
)
EXPECTED_PRODUCT_COUNT = 125_762

TEXT_LABELS = {
    "Title": "title",
    "Category": "category",
    "Features": "features",
    "Product Details": "product_details",
    "Description": "description",
}


# ============================================================
# 2. Read-only Product Documents repository
# ============================================================

class ProductRepository:
    """Load, validate, normalize, and serve Product Documents by product ID."""

    def __init__(self, document_path: Path = DOCUMENT_PATH):
        self.document_path = Path(document_path)
        self.products_by_id: dict[str, dict] = {}
        self._load_and_validate_documents()

    @staticmethod
    def _split_nonempty(value: str, separator: str) -> list[str]:
        """Split a field while preserving source order and repeated values."""

        return [part.strip() for part in value.split(separator) if part.strip()]

    @classmethod
    def _parse_document_text(cls, text: str) -> dict:
        """Parse optional labeled sections without assuming all are present."""

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
                # Preserve a section if future document versions wrap it over
                # multiple lines before the next recognized label.
                sections[active_field].append(raw_line.strip())

        joined = {
            field_name: "\n".join(parts).strip()
            for field_name, parts in sections.items()
        }

        category_text = joined["category"]
        if "|" in category_text:
            categories = cls._split_nonempty(category_text, "|")
        elif ">" in category_text:
            # Product Documents V2 currently represents category hierarchy
            # with ``>``; supporting it keeps the normalized value list-shaped.
            categories = cls._split_nonempty(category_text, ">")
        elif category_text:
            categories = [category_text]
        else:
            categories = []

        feature_text = joined["features"]
        features = (
            cls._split_nonempty(feature_text, "|")
            if feature_text
            else []
        )

        return {
            "title": joined["title"] or None,
            "categories": categories,
            "features": features,
            "product_details": joined["product_details"] or None,
            "description": joined["description"] or None,
        }

    def _load_and_validate_documents(self) -> None:
        """Scan the formal JSONL once and build the authoritative ID mapping."""

        if not self.document_path.is_file():
            raise FileNotFoundError(
                f"Product Documents V2 file not found: {self.document_path}"
            )

        try:
            with self.document_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        raise ValueError(
                            f"Empty JSONL record at source line {line_number}."
                        )
                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON at source line {line_number}: {exc}"
                        ) from exc

                    if not isinstance(document, dict):
                        raise ValueError(
                            f"Source line {line_number} must contain a JSON object."
                        )

                    product_id = document.get("id")
                    if not isinstance(product_id, str) or not product_id.strip():
                        raise ValueError(
                            f"Invalid or empty id at source line {line_number}."
                        )

                    metadata = document.get("metadata")
                    if not isinstance(metadata, dict):
                        raise ValueError(
                            f"metadata must be an object at source line {line_number}."
                        )
                    parent_asin = metadata.get("parent_asin")
                    if not isinstance(parent_asin, str) or not parent_asin.strip():
                        raise ValueError(
                            "metadata.parent_asin must be a non-empty string at "
                            f"source line {line_number}."
                        )
                    if product_id != parent_asin:
                        raise ValueError(
                            f"id does not match metadata.parent_asin at source "
                            f"line {line_number}: {product_id!r} != {parent_asin!r}."
                        )
                    if product_id in self.products_by_id:
                        raise ValueError(
                            f"Duplicate product ID {product_id!r} at source "
                            f"line {line_number}."
                        )

                    text = document.get("text")
                    if not isinstance(text, str):
                        raise ValueError(
                            f"text must be a string at source line {line_number}."
                        )
                    parsed = self._parse_document_text(text)

                    # Keep only the stable runtime-facing structure; the raw
                    # metadata object itself is not exposed to callers.
                    self.products_by_id[product_id] = {
                        "product_id": product_id,
                        **parsed,
                        "store": metadata.get("store"),
                        "price": metadata.get("price"),
                        "average_rating": metadata.get("average_rating"),
                        "rating_number": metadata.get("rating_number"),
                        "text": text,
                    }
        except UnicodeError as exc:
            raise ValueError(
                f"Product Documents V2 is not valid UTF-8: {exc}"
            ) from exc

        actual_count = len(self.products_by_id)
        if actual_count != EXPECTED_PRODUCT_COUNT:
            raise ValueError(
                f"Product document count is {actual_count:,}; "
                f"expected exactly {EXPECTED_PRODUCT_COUNT:,}."
            )

    @staticmethod
    def _validate_product_id(product_id, label: str = "product_id") -> None:
        if not isinstance(product_id, str):
            raise TypeError(
                f"{label} must be a string, found {type(product_id).__name__}."
            )
        if not product_id.strip():
            raise ValueError(f"{label} must be a non-empty string.")

    def get(self, product_id: str) -> dict:
        """Return a defensive copy of one normalized product."""

        self._validate_product_id(product_id)
        try:
            product = self.products_by_id[product_id]
        except KeyError:
            raise KeyError(f"Product ID not found: {product_id}") from None

        # Deep copy protects nested category/feature lists in the internal cache.
        return copy.deepcopy(product)

    def get_many(self, product_ids: list[str]) -> list[dict]:
        """Return products in input order, including repeated requested IDs."""

        if not isinstance(product_ids, list):
            raise TypeError("product_ids must be a list of strings.")
        if not product_ids:
            return []

        for position, product_id in enumerate(product_ids):
            self._validate_product_id(product_id, f"product_ids[{position}]")
            if product_id not in self.products_by_id:
                raise KeyError(f"Product ID not found: {product_id}")

        return [copy.deepcopy(self.products_by_id[item]) for item in product_ids]


# ============================================================
# 3. Minimal read-only CLI for one manual lookup
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Read one normalized product from Product Documents V2."
    )
    parser.add_argument(
        "--product-id",
        required=True,
        help="Exact product ID / parent ASIN to retrieve.",
    )
    return parser.parse_args()


def display_value(value) -> str:
    return "(not available)" if value is None or value == "" else str(value)


def main() -> int:
    args = parse_args()
    repository = ProductRepository()
    product = repository.get(args.product_id)

    print(f"Product ID: {product['product_id']}")
    print(f"Title: {display_value(product['title'])}")
    print(
        "Categories: "
        + (" > ".join(product["categories"]) or "(not available)")
    )
    print(f"Price: {display_value(product['price'])}")
    print(f"Average Rating: {display_value(product['average_rating'])}")
    print(f"Rating Number: {display_value(product['rating_number'])}")
    print(f"Store: {display_value(product['store'])}")

    print("Features:")
    if product["features"]:
        for feature in product["features"]:
            print(f"  - {feature}")
    else:
        print("  (not available)")

    print("Product Details:")
    print(f"  {display_value(product['product_details'])}")
    print("Description:")
    print(f"  {display_value(product['description'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
