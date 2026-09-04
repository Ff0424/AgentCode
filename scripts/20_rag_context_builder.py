"""
RAGContextBuilder converts already-ranked recommendation results into a
compact, deterministic, LLM-ready product context.

It does not retrieve products, rerank candidates, parse user intent,
or call an LLM.
"""

import argparse
import importlib.util
import math
import numbers
import sys
from pathlib import Path


# ============================================================
# 1. Defaults and optional CLI dependency path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOMMENDATION_SERVICE_V2_PATH = (
    PROJECT_ROOT / "scripts" / "19_recommendation_service_v2.py"
)

DEFAULT_MAX_CONTEXT_CHARS = 12_000
DEFAULT_MAX_DESCRIPTION_CHARS = 1_200
DEFAULT_MAX_PRODUCT_DETAILS_CHARS = 1_000
DEFAULT_MAX_FEATURES = 8


# ============================================================
# 2. Deterministic RAG context builder
# ============================================================

class RAGContextBuilder:
    """Format validated recommendation results under a character budget."""

    def __init__(
        self,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        max_product_details_chars: int = DEFAULT_MAX_PRODUCT_DETAILS_CHARS,
        max_features: int = DEFAULT_MAX_FEATURES,
    ):
        self.max_context_chars = self._validate_positive_integer(
            max_context_chars, "max_context_chars"
        )
        self.max_description_chars = self._validate_positive_integer(
            max_description_chars, "max_description_chars"
        )
        self.max_product_details_chars = self._validate_positive_integer(
            max_product_details_chars, "max_product_details_chars"
        )
        self.max_features = self._validate_positive_integer(
            max_features, "max_features"
        )

    @staticmethod
    def _validate_positive_integer(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer; bool is not accepted.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return int(value)

    @staticmethod
    def _normalize_text(value, field_name: str):
        """Normalize optional text without coercing inconsistent data types."""

        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"product.{field_name} must be a string, None, or empty string."
            )
        normalized = " ".join(value.split())
        return normalized or None

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Truncate normalized text on a word boundary within the full budget."""

        if len(text) <= max_chars:
            return text
        ellipsis = "..."
        if max_chars <= len(ellipsis):
            return ellipsis[:max_chars]

        content_budget = max_chars - len(ellipsis)
        prefix = text[:content_budget].rstrip()
        last_space = prefix.rfind(" ")
        if last_space > 0:
            prefix = prefix[:last_space].rstrip()
        return prefix + ellipsis

    @staticmethod
    def _finite_numeric(value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        converted = float(value)
        return converted if math.isfinite(converted) else None

    @classmethod
    def _format_price(cls, value):
        numeric = cls._finite_numeric(value)
        return None if numeric is None else f"${numeric:.2f}"

    @classmethod
    def _format_rating(cls, value):
        numeric = cls._finite_numeric(value)
        if numeric is None:
            return None
        return f"{numeric:g} / 5"

    @classmethod
    def _format_rating_count(cls, value):
        numeric = cls._finite_numeric(value)
        if numeric is None or numeric < 0.0 or not numeric.is_integer():
            return None
        return str(int(numeric))

    @classmethod
    def _normalize_string_list(cls, value, field_name: str) -> list[str]:
        if value is None or value == "":
            return []
        if not isinstance(value, list):
            raise TypeError(f"product.{field_name} must be a list of strings.")
        normalized = []
        for position, item in enumerate(value):
            if not isinstance(item, str):
                raise TypeError(
                    f"product.{field_name}[{position}] must be a string."
                )
            cleaned = " ".join(item.split())
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, found {type(query).__name__}.")
        stripped = query.strip()
        if not stripped:
            raise ValueError("query must be a non-empty string.")
        return stripped

    @staticmethod
    def _validate_recommendations(recommendations) -> None:
        """Validate the rank-ordered, unique minimum recommendation contract."""

        if not isinstance(recommendations, list):
            raise TypeError("recommendations must be a list of dictionaries.")

        seen_ranks = set()
        seen_product_ids = set()
        for position, recommendation in enumerate(recommendations, start=1):
            context = f"recommendations[{position - 1}]"
            if not isinstance(recommendation, dict):
                raise TypeError(f"{context} must be a dictionary.")
            for required in ("rank", "product_id", "product"):
                if required not in recommendation:
                    raise ValueError(f"{context} is missing required field {required!r}.")

            rank = recommendation["rank"]
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError(f"{context}.rank must be a positive integer.")
            if rank <= 0:
                raise ValueError(f"{context}.rank must be a positive integer.")
            if rank in seen_ranks:
                raise ValueError(f"Duplicate recommendation rank: {rank}.")
            if rank != position:
                raise ValueError(
                    f"Recommendations must be ordered rank 1, 2, 3, ...; "
                    f"position {position} has rank {rank}."
                )

            product_id = recommendation["product_id"]
            if not isinstance(product_id, str) or not product_id.strip():
                raise ValueError(f"{context}.product_id must be a non-empty string.")
            if product_id in seen_product_ids:
                raise ValueError(f"Duplicate recommendation product_id: {product_id}.")

            product = recommendation["product"]
            if not isinstance(product, dict):
                raise TypeError(f"{context}.product must be a dictionary.")
            if product.get("product_id") != product_id:
                raise ValueError(
                    f"{context}.product.product_id does not match recommendation "
                    f"product_id {product_id!r}."
                )

            seen_ranks.add(rank)
            seen_product_ids.add(product_id)

    def _build_product_block(self, recommendation: dict) -> tuple[str, bool]:
        """Build one complete fact-only block and report field truncation."""

        rank = recommendation["rank"]
        product_id = recommendation["product_id"]
        product = recommendation["product"]
        lines = [f"[Product {rank}]", f"Product ID: {product_id}"]
        truncated = False

        title = self._normalize_text(product.get("title"), "title")
        store = self._normalize_text(product.get("store"), "store")
        details = self._normalize_text(
            product.get("product_details"), "product_details"
        )
        description = self._normalize_text(
            product.get("description"), "description"
        )
        categories = self._normalize_string_list(
            product.get("categories"), "categories"
        )
        features = self._normalize_string_list(product.get("features"), "features")

        if title:
            lines.append(f"Title: {title}")
        price = self._format_price(product.get("price"))
        if price:
            lines.append(f"Price: {price}")
        rating = self._format_rating(product.get("average_rating"))
        if rating:
            lines.append(f"Average Rating: {rating}")
        rating_count = self._format_rating_count(product.get("rating_number"))
        if rating_count:
            lines.append(f"Rating Count: {rating_count}")
        if store:
            lines.append(f"Store: {store}")
        if categories:
            lines.append("Categories: " + " > ".join(categories))

        if len(features) > self.max_features:
            truncated = True
        selected_features = features[: self.max_features]
        if selected_features:
            lines.append("Features:")
            lines.extend(f"- {feature}" for feature in selected_features)

        if details:
            shortened = self._truncate_text(details, self.max_product_details_chars)
            truncated = truncated or shortened != details
            lines.append(f"Product Details: {shortened}")
        if description:
            shortened = self._truncate_text(description, self.max_description_chars)
            truncated = truncated or shortened != description
            lines.append(f"Description: {shortened}")

        return "\n".join(lines), truncated

    def _build_minimal_first_block(self, recommendation: dict) -> tuple[str, bool]:
        """Fit the first product's identity and title into an extreme budget."""

        rank = recommendation["rank"]
        product_id = recommendation["product_id"]
        title = self._normalize_text(
            recommendation["product"].get("title"), "title"
        )
        base = f"[Product {rank}]\nProduct ID: {product_id}"
        if len(base) > self.max_context_chars:
            return "", True
        if not title:
            return base, True

        title_prefix = "\nTitle: "
        remaining = self.max_context_chars - len(base) - len(title_prefix)
        if remaining <= 0:
            return base, True
        shortened_title = self._truncate_text(title, remaining)
        return base + title_prefix + shortened_title, True

    def build(self, query: str, recommendations: list[dict]) -> dict:
        """Return structured metadata plus a deterministic fact-only context."""

        stripped_query = self._validate_query(query)
        self._validate_recommendations(recommendations)
        input_count = len(recommendations)
        if not recommendations:
            return {
                "query": stripped_query,
                "context": "",
                "included_product_ids": [],
                "included_count": 0,
                "input_count": 0,
                "truncated": False,
                "context_chars": 0,
            }

        blocks = []
        included_product_ids = []
        truncated = False

        for recommendation in recommendations:
            block, block_truncated = self._build_product_block(recommendation)
            separator = "\n\n" if blocks else ""
            proposed_length = sum(len(item) for item in blocks)
            proposed_length += 2 * max(0, len(blocks) - 1)
            proposed_length += len(separator) + len(block)

            if proposed_length > self.max_context_chars:
                truncated = True
                if not blocks:
                    minimal_block, minimal_truncated = (
                        self._build_minimal_first_block(recommendation)
                    )
                    if minimal_block:
                        blocks.append(minimal_block)
                        included_product_ids.append(recommendation["product_id"])
                    truncated = truncated or minimal_truncated
                # Strict prefix policy: never skip a rank to seek a shorter block.
                break

            blocks.append(block)
            included_product_ids.append(recommendation["product_id"])
            truncated = truncated or block_truncated

        if len(included_product_ids) < input_count:
            truncated = True
        context = "\n\n".join(blocks)
        if len(context) > self.max_context_chars:
            raise RuntimeError("Internal error: context exceeded max_context_chars.")

        return {
            "query": stripped_query,
            "context": context,
            "included_product_ids": included_product_ids,
            "included_count": int(len(included_product_ids)),
            "input_count": int(input_count),
            "truncated": bool(truncated),
            "context_chars": int(len(context)),
        }


# ============================================================
# 3. Optional V2-backed smoke-test CLI
# ============================================================

def _load_recommendation_service_v2():
    """Dynamically load V2 for CLI use without making the builder depend on it."""

    if not RECOMMENDATION_SERVICE_V2_PATH.is_file():
        raise FileNotFoundError(
            f"Recommendation service V2 not found: {RECOMMENDATION_SERVICE_V2_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "agentcode_recommendation_service_v2_for_context",
        RECOMMENDATION_SERVICE_V2_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create an import spec for {RECOMMENDATION_SERVICE_V2_PATH}."
        )

    # Dynamic CLI imports must not create bytecode cache files.
    sys.dont_write_bytecode = True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Could not load {RECOMMENDATION_SERVICE_V2_PATH}: {exc}"
        ) from exc
    service_class = getattr(module, "RecommendationServiceV2", None)
    if not isinstance(service_class, type):
        raise ImportError(
            "RecommendationServiceV2 was not found in "
            f"{RECOMMENDATION_SERVICE_V2_PATH}."
        )
    return service_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic RAG context from V2 recommendations."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-pool-k", type=int, default=20)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-rating", type=float, default=None)
    parser.add_argument("--min-rating-count", type=int, default=None)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service_class = _load_recommendation_service_v2()
    recommendations = service_class().recommend(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank_pool_k=args.rerank_pool_k,
        max_price=args.max_price,
        min_rating=args.min_rating,
        min_rating_count=args.min_rating_count,
    )
    result = RAGContextBuilder(
        max_context_chars=args.max_context_chars
    ).build(args.query, recommendations)

    print(f"Query: {result['query']}")
    print(f"Input recommendations: {result['input_count']}")
    print(f"Included products: {result['included_count']}")
    print(f"Context chars: {result['context_chars']}")
    print(f"Truncated: {result['truncated']}")
    print("\n===== RAG CONTEXT =====\n")
    print(result["context"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
