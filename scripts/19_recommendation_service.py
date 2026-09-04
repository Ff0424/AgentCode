"""
Transparent runtime recommendation and heuristic reranking service.

This module composes the existing ProductRetriever and ProductRepository:

    semantic retrieval -> product lookup -> hard filters -> reranking -> Top-K

It does not duplicate model/index/document loading, call an LLM, infer filters
from natural language, train a model, access the network, or write artifacts.

Example:
    python scripts/19_recommendation_service.py \
        --query "USB-C hub with HDMI and ethernet" \
        --top-k 5 \
        --candidate-k 50
"""

import argparse
import importlib.util
import math
import numbers
import sys
from pathlib import Path


# ============================================================
# 1. Local dependencies and fixed reranking configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RETRIEVAL_SERVICE_PATH = SCRIPTS_DIR / "17_retrieval_service.py"
PRODUCT_REPOSITORY_PATH = SCRIPTS_DIR / "18_product_repository.py"

SEMANTIC_WEIGHT = 0.70
RATING_WEIGHT = 0.20
POPULARITY_WEIGHT = 0.10
WEIGHT_SUM_TOLERANCE = 1e-12


def _load_class(module_path: Path, module_name: str, class_name: str):
    """Load a class from a numeric-prefixed local script without sys.path hacks."""

    if not module_path.is_file():
        raise FileNotFoundError(f"Required local module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_path}.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not load local module {module_path}: {exc}") from exc

    loaded_class = getattr(module, class_name, None)
    if not isinstance(loaded_class, type):
        raise ImportError(f"{class_name} was not found in {module_path}.")
    return loaded_class


def _load_default_dependencies():
    """Resolve the existing runtime components only when defaults are needed."""

    retriever_class = _load_class(
        RETRIEVAL_SERVICE_PATH,
        "agentcode_retrieval_service_17",
        "ProductRetriever",
    )
    repository_class = _load_class(
        PRODUCT_REPOSITORY_PATH,
        "agentcode_product_repository_18",
        "ProductRepository",
    )
    return retriever_class, repository_class


# ============================================================
# 2. Recommendation and heuristic reranking service
# ============================================================

class RecommendationService:
    """Combine semantic candidates with explicit filters and fixed heuristics."""

    def __init__(self, retriever=None, repository=None):
        self._validate_weights()

        if retriever is None or repository is None:
            retriever_class, repository_class = _load_default_dependencies()
            if retriever is None:
                retriever = retriever_class()
            if repository is None:
                repository = repository_class()

        if not callable(getattr(retriever, "search", None)):
            raise TypeError("retriever must provide a callable search() method.")
        if not callable(getattr(repository, "get_many", None)):
            raise TypeError("repository must provide a callable get_many() method.")

        self.retriever = retriever
        self.repository = repository

    @staticmethod
    def _validate_weights() -> None:
        weights = (SEMANTIC_WEIGHT, RATING_WEIGHT, POPULARITY_WEIGHT)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise RuntimeError("Recommendation weights must be finite and non-negative.")
        if not math.isclose(
            sum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise RuntimeError("Recommendation weights must sum to exactly 1.0.")

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, found {type(query).__name__}.")
        if not query.strip():
            raise ValueError("query must be a non-empty string.")

    @staticmethod
    def _validate_positive_integer(value, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validate_optional_number(value, name: str, minimum: float, maximum=None):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be None or a finite number.")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite.")
        if converted < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        if maximum is not None and converted > maximum:
            raise ValueError(f"{name} must be at most {maximum}.")
        return converted

    @staticmethod
    def _validate_optional_count(value, name: str):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be None or a non-negative integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return value

    def _index_size(self) -> int:
        index = getattr(self.retriever, "index", None)
        index_size = getattr(index, "ntotal", None)
        if isinstance(index_size, bool) or not isinstance(index_size, numbers.Integral):
            raise RuntimeError("retriever.index.ntotal is unavailable or invalid.")
        if int(index_size) <= 0:
            raise RuntimeError("retriever.index.ntotal must be positive.")
        return int(index_size)

    @staticmethod
    def _finite_number_or_none(value):
        """Normalize product numeric fields without accepting bool or NaN/Inf."""

        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        converted = float(value)
        return converted if math.isfinite(converted) else None

    @classmethod
    def _passes_hard_constraints(
        cls,
        product: dict,
        max_price,
        min_rating,
        min_rating_count,
    ) -> bool:
        price = cls._finite_number_or_none(product.get("price"))
        rating = cls._finite_number_or_none(product.get("average_rating"))
        rating_count = cls._finite_number_or_none(product.get("rating_number"))

        if max_price is not None and (price is None or price > max_price):
            return False
        if min_rating is not None and (rating is None or rating < min_rating):
            return False
        if min_rating_count is not None and (
            rating_count is None or rating_count < min_rating_count
        ):
            return False
        return True

    @classmethod
    def _rating_score(cls, product: dict) -> float:
        rating = cls._finite_number_or_none(product.get("average_rating"))
        if rating is None:
            return 0.0
        return float(min(1.0, max(0.0, rating / 5.0)))

    @classmethod
    def _log_popularity(cls, product: dict) -> float:
        rating_count = cls._finite_number_or_none(product.get("rating_number"))
        if rating_count is None or rating_count < 0.0:
            rating_count = 0.0
        return float(math.log1p(rating_count))

    @staticmethod
    def _validate_candidate_alignment(candidates: list, products: list) -> None:
        if len(products) != len(candidates):
            raise RuntimeError(
                "Repository result count does not match retrieval candidate count."
            )
        for position, (candidate, product) in enumerate(zip(candidates, products)):
            if not isinstance(candidate, dict) or not isinstance(product, dict):
                raise RuntimeError(
                    f"Candidate/product at position {position} must be a dictionary."
                )
            candidate_id = candidate.get("product_id")
            if product.get("product_id") != candidate_id:
                raise RuntimeError(
                    "Repository results are not aligned with retrieval candidates "
                    f"at position {position}: {candidate_id!r} != "
                    f"{product.get('product_id')!r}."
                )

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        max_price: float | None = None,
        min_rating: float | None = None,
        min_rating_count: int | None = None,
    ) -> list[dict]:
        """Retrieve, hard-filter, and deterministically rerank product candidates."""

        self._validate_query(query)
        self._validate_positive_integer(top_k, "top_k")
        self._validate_positive_integer(candidate_k, "candidate_k")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")
        index_size = self._index_size()
        if candidate_k > index_size:
            raise ValueError(
                f"candidate_k={candidate_k:,} exceeds FAISS index size "
                f"{index_size:,}."
            )

        max_price = self._validate_optional_number(
            max_price, "max_price", minimum=0.0
        )
        min_rating = self._validate_optional_number(
            min_rating, "min_rating", minimum=0.0, maximum=5.0
        )
        min_rating_count = self._validate_optional_count(
            min_rating_count, "min_rating_count"
        )

        candidates = self.retriever.search(query, top_k=candidate_k)
        if not isinstance(candidates, list):
            raise RuntimeError("retriever.search() must return a list.")
        if len(candidates) > candidate_k:
            raise RuntimeError("Retriever returned more candidates than requested.")

        candidate_ids = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise RuntimeError(f"Retrieval candidate {position} is not a dictionary.")
            product_id = candidate.get("product_id")
            if not isinstance(product_id, str) or not product_id:
                raise RuntimeError(
                    f"Retrieval candidate {position} has an invalid product_id."
                )
            candidate_ids.append(product_id)

        products = self.repository.get_many(candidate_ids)
        self._validate_candidate_alignment(candidates, products)

        filtered = []
        for candidate, product in zip(candidates, products):
            if not self._passes_hard_constraints(
                product,
                max_price,
                min_rating,
                min_rating_count,
            ):
                continue

            semantic_score = self._finite_number_or_none(candidate.get("score"))
            if semantic_score is None:
                raise RuntimeError(
                    f"Candidate {candidate['product_id']!r} has an invalid score."
                )
            retrieval_rank = candidate.get("rank")
            if (
                isinstance(retrieval_rank, bool)
                or not isinstance(retrieval_rank, numbers.Integral)
                or int(retrieval_rank) <= 0
            ):
                raise RuntimeError(
                    f"Candidate {candidate['product_id']!r} has an invalid rank."
                )

            filtered.append(
                {
                    "product_id": candidate["product_id"],
                    "retrieval_rank": int(retrieval_rank),
                    "retrieval_score": float(semantic_score),
                    "rating_score": self._rating_score(product),
                    "log_popularity": self._log_popularity(product),
                    "product": product,
                }
            )

        if not filtered:
            return []

        popularity_values = [item["log_popularity"] for item in filtered]
        popularity_min = min(popularity_values)
        popularity_max = max(popularity_values)
        popularity_range = popularity_max - popularity_min

        for item in filtered:
            if popularity_range == 0.0:
                popularity_score = 0.0
            else:
                popularity_score = (
                    item["log_popularity"] - popularity_min
                ) / popularity_range

            item["popularity_score"] = float(popularity_score)
            item["final_score"] = float(
                SEMANTIC_WEIGHT * item["retrieval_score"]
                + RATING_WEIGHT * item["rating_score"]
                + POPULARITY_WEIGHT * item["popularity_score"]
            )
            del item["log_popularity"]

        # Explicit secondary keys make equal-score ordering reproducible.
        filtered.sort(
            key=lambda item: (
                -item["final_score"],
                -item["retrieval_score"],
                item["retrieval_rank"],
                item["product_id"],
            )
        )

        recommendations = filtered[:top_k]
        for rank, recommendation in enumerate(recommendations, start=1):
            recommendation["rank"] = int(rank)
        return recommendations


# ============================================================
# 3. Minimal CLI for one manual recommendation request
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve, filter, and heuristically rerank local products."
    )
    parser.add_argument("--query", required=True, help="Non-empty shopping query.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-rating", type=float, default=None)
    parser.add_argument("--min-rating-count", type=int, default=None)
    return parser.parse_args()


def _display(value) -> str:
    return "(not available)" if value is None or value == "" else str(value)


def main() -> int:
    args = parse_args()
    service = RecommendationService()
    recommendations = service.recommend(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_price=args.max_price,
        min_rating=args.min_rating,
        min_rating_count=args.min_rating_count,
    )

    print(f"Query: {args.query}\n")
    if not recommendations:
        print("No products satisfied the active constraints.")
        return 0

    for item in recommendations:
        product = item["product"]
        print(f"{item['rank']}. product_id={item['product_id']}")
        print(f"   title={_display(product.get('title'))}")
        print(f"   price={_display(product.get('price'))}")
        print(f"   average_rating={_display(product.get('average_rating'))}")
        print(f"   rating_number={_display(product.get('rating_number'))}")
        print(f"   retrieval_rank={item['retrieval_rank']}")
        print(f"   retrieval_score={item['retrieval_score']:.8f}")
        print(f"   rating_score={item['rating_score']:.8f}")
        print(f"   popularity_score={item['popularity_score']:.8f}")
        print(f"   final_score={item['final_score']:.8f}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
