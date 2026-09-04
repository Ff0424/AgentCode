"""
Relevance-first runtime recommendation and heuristic reranking service.

V2 preserves the global linear V1 service as an unchanged baseline and reuses
its ProductRetriever/ProductRepository integration and validation helpers. Its
only behavioral change is to hard-filter first, limit reranking to a semantic
relevance pool, normalize scores inside that pool, and then select final Top-K.

Example:
    python scripts/19_recommendation_service_v2.py \
        --query "USB-C hub with HDMI and ethernet" \
        --top-k 10 \
        --candidate-k 50 \
        --rerank-pool-k 20
"""

import argparse
import importlib.util
import math
import numbers
import sys
from pathlib import Path


# ============================================================
# 1. V1 loading and fixed V2 ranking configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_SERVICE_PATH = PROJECT_ROOT / "scripts" / "19_recommendation_service.py"

SEMANTIC_WEIGHT = 0.70
RATING_WEIGHT = 0.20
POPULARITY_WEIGHT = 0.10
WEIGHT_SUM_TOLERANCE = 1e-12


def _load_v1_service_class():
    """Load the numeric-prefixed V1 module without requiring a file rename."""

    if not V1_SERVICE_PATH.is_file():
        raise FileNotFoundError(f"V1 recommendation service not found: {V1_SERVICE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "agentcode_recommendation_service_v1_19",
        V1_SERVICE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {V1_SERVICE_PATH}.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Could not load V1 service {V1_SERVICE_PATH}: {exc}") from exc

    service_class = getattr(module, "RecommendationService", None)
    if not isinstance(service_class, type):
        raise ImportError(f"RecommendationService was not found in {V1_SERVICE_PATH}.")
    return service_class


_RecommendationServiceV1 = _load_v1_service_class()


# ============================================================
# 2. Relevance-first recommendation service
# ============================================================

class RecommendationServiceV2(_RecommendationServiceV1):
    """Rerank only the strongest post-filter semantic candidates."""

    def __init__(self, retriever=None, repository=None):
        self._validate_v2_weights()
        super().__init__(retriever=retriever, repository=repository)

    @staticmethod
    def _validate_v2_weights() -> None:
        weights = (SEMANTIC_WEIGHT, RATING_WEIGHT, POPULARITY_WEIGHT)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise RuntimeError("V2 recommendation weights must be finite and non-negative.")
        if not math.isclose(
            sum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise RuntimeError("V2 recommendation weights must sum to exactly 1.0.")

    @staticmethod
    def _normalize_min_max(values: list[float], equal_value: float) -> list[float]:
        """Return stable [0, 1] values and explicitly handle a zero range."""

        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        value_range = maximum - minimum
        if value_range == 0.0:
            return [float(equal_value)] * len(values)
        return [
            float(min(1.0, max(0.0, (value - minimum) / value_range)))
            for value in values
        ]

    @staticmethod
    def _validate_retrieval_candidates(candidates: list, candidate_k: int) -> None:
        if not isinstance(candidates, list):
            raise RuntimeError("retriever.search() must return a list.")
        if len(candidates) > candidate_k:
            raise RuntimeError("Retriever returned more candidates than requested.")

        seen_ids = set()
        seen_ranks = set()
        previous_rank = 0
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise RuntimeError(f"Retrieval candidate {position} is not a dictionary.")

            product_id = candidate.get("product_id")
            if not isinstance(product_id, str) or not product_id.strip():
                raise RuntimeError(
                    f"Retrieval candidate {position} has an invalid product_id."
                )
            if product_id in seen_ids:
                raise RuntimeError(f"Duplicate retrieval product_id: {product_id}.")

            retrieval_rank = candidate.get("rank")
            if (
                isinstance(retrieval_rank, bool)
                or not isinstance(retrieval_rank, numbers.Integral)
                or int(retrieval_rank) <= 0
            ):
                raise RuntimeError(
                    f"Retrieval candidate {product_id!r} has an invalid rank."
                )
            retrieval_rank = int(retrieval_rank)
            if retrieval_rank in seen_ranks:
                raise RuntimeError(f"Duplicate retrieval rank: {retrieval_rank}.")
            if retrieval_rank <= previous_rank:
                raise RuntimeError(
                    "Retrieval candidates are not in ascending retrieval-rank order."
                )

            seen_ids.add(product_id)
            seen_ranks.add(retrieval_rank)
            previous_rank = retrieval_rank

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        rerank_pool_k: int = 20,
        max_price: float | None = None,
        min_rating: float | None = None,
        min_rating_count: int | None = None,
    ) -> list[dict]:
        """Filter candidates, form a relevance pool, normalize, and rerank."""

        self._validate_query(query)
        self._validate_positive_integer(top_k, "top_k")
        self._validate_positive_integer(candidate_k, "candidate_k")
        self._validate_positive_integer(rerank_pool_k, "rerank_pool_k")
        if not top_k <= rerank_pool_k <= candidate_k:
            raise ValueError(
                "Expected top_k <= rerank_pool_k <= candidate_k, found "
                f"{top_k} <= {rerank_pool_k} <= {candidate_k}."
            )

        index_size = self._index_size()
        if candidate_k > index_size:
            raise ValueError(
                f"candidate_k={candidate_k:,} exceeds FAISS index size "
                f"{index_size:,}."
            )

        # Reuse V1's exact public-parameter semantics for all hard constraints.
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
        self._validate_retrieval_candidates(candidates, candidate_k)
        candidate_ids = [candidate["product_id"] for candidate in candidates]
        products = self.repository.get_many(candidate_ids)
        self._validate_candidate_alignment(candidates, products)

        # Filtering retains the list order returned by semantic retrieval.
        filtered = []
        for candidate, product in zip(candidates, products):
            if not self._passes_hard_constraints(
                product,
                max_price,
                min_rating,
                min_rating_count,
            ):
                continue

            filtered.append(
                {
                    "product_id": candidate["product_id"],
                    "retrieval_rank": int(candidate["rank"]),
                    "candidate": candidate,
                    "product": product,
                }
            )

        if not filtered:
            return []

        actual_pool_size = min(rerank_pool_k, len(filtered))
        relevance_pool = filtered[:actual_pool_size]

        # Compute every ranking signal only after relevance-pool selection.
        for item in relevance_pool:
            raw_score = self._finite_number_or_none(item["candidate"].get("score"))
            if raw_score is None:
                raise RuntimeError(
                    f"Candidate {item['product_id']!r} has an invalid score."
                )
            item["retrieval_score"] = float(raw_score)
            item["semantic_score_raw"] = float(raw_score)
            item["rating_score"] = float(self._rating_score(item["product"]))
            item["log_popularity"] = float(self._log_popularity(item["product"]))
            del item["candidate"]

        semantic_scores = self._normalize_min_max(
            [item["semantic_score_raw"] for item in relevance_pool],
            equal_value=1.0,
        )
        popularity_scores = self._normalize_min_max(
            [item["log_popularity"] for item in relevance_pool],
            equal_value=0.0,
        )

        for item, semantic_score, popularity_score in zip(
            relevance_pool,
            semantic_scores,
            popularity_scores,
        ):
            item["semantic_score"] = float(semantic_score)
            item["popularity_score"] = float(popularity_score)
            item["final_score"] = float(
                SEMANTIC_WEIGHT * item["semantic_score"]
                + RATING_WEIGHT * item["rating_score"]
                + POPULARITY_WEIGHT * item["popularity_score"]
            )
            # The public retrieval_score already preserves this raw value.
            del item["semantic_score_raw"]
            del item["log_popularity"]

        relevance_pool.sort(
            key=lambda item: (
                -item["final_score"],
                -item["semantic_score"],
                -item["retrieval_score"],
                item["retrieval_rank"],
                item["product_id"],
            )
        )

        recommendations = relevance_pool[:top_k]
        for final_rank, recommendation in enumerate(recommendations, start=1):
            recommendation["rank"] = int(final_rank)
        return recommendations


# ============================================================
# 3. Minimal CLI for one manual V2 recommendation request
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run relevance-first heuristic product recommendation V2."
    )
    parser.add_argument("--query", required=True, help="Non-empty shopping query.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-pool-k", type=int, default=20)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-rating", type=float, default=None)
    parser.add_argument("--min-rating-count", type=int, default=None)
    return parser.parse_args()


def _display(value) -> str:
    return "(not available)" if value is None or value == "" else str(value)


def main() -> int:
    args = parse_args()
    service = RecommendationServiceV2()
    recommendations = service.recommend(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank_pool_k=args.rerank_pool_k,
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
        print(f"   retrieval_rank={item['retrieval_rank']}")
        print(f"   title={_display(product.get('title'))}")
        print(f"   price={_display(product.get('price'))}")
        print(f"   average_rating={_display(product.get('average_rating'))}")
        print(f"   rating_number={_display(product.get('rating_number'))}")
        print(f"   retrieval_score_raw={item['retrieval_score']:.8f}")
        print(f"   semantic_score={item['semantic_score']:.8f}")
        print(f"   rating_score={item['rating_score']:.8f}")
        print(f"   popularity_score={item['popularity_score']:.8f}")
        print(f"   final_score={item['final_score']:.8f}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
