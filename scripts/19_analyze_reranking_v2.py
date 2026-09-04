"""Read-only sanity analysis for relevance-first RecommendationServiceV2."""

import importlib.util
import math
import numbers
import statistics
import sys
from pathlib import Path


# ============================================================
# 1. Fixed analysis configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOMMENDATION_SERVICE_PATH = (
    PROJECT_ROOT / "scripts" / "19_recommendation_service_v2.py"
)

QUERY = "USB-C hub with HDMI and ethernet"
CANDIDATE_K = 50
RERANK_POOL_K = 20
ANALYSIS_K = 10
MOVER_COUNT = 5


# ============================================================
# 2. Local RecommendationServiceV2 loading
# ============================================================

def load_recommendation_service_class():
    """Load the numeric-prefixed V2 module without changing sys.path."""

    if not RECOMMENDATION_SERVICE_PATH.is_file():
        raise FileNotFoundError(
            f"Recommendation service V2 not found: {RECOMMENDATION_SERVICE_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "agentcode_recommendation_service_v2_19",
        RECOMMENDATION_SERVICE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create an import spec for {RECOMMENDATION_SERVICE_PATH}."
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Could not load {RECOMMENDATION_SERVICE_PATH}: {exc}"
        ) from exc
    service_class = getattr(module, "RecommendationServiceV2", None)
    if not isinstance(service_class, type):
        raise ImportError(
            "RecommendationServiceV2 was not found in "
            f"{RECOMMENDATION_SERVICE_PATH}."
        )
    return service_class


# ============================================================
# 3. Retrieval and V2 result validation
# ============================================================

def require_product_id(value, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} has a missing or invalid product_id.")
    return value


def require_rank(value, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{context} has a missing or invalid rank: {value!r}.")
    rank = int(value)
    if rank <= 0:
        raise ValueError(f"{context} rank must be positive, found {rank}.")
    return rank


def require_score(value, field: str, context: str, unit_interval=False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{context} has an invalid {field}: {value!r}.")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{context} has a non-finite {field}: {score!r}.")
    if unit_interval and not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{context} {field}={score!r} is outside the [0, 1] range."
        )
    return score


def validate_retrieval_results(results: list) -> dict:
    if not isinstance(results, list):
        raise TypeError("ProductRetriever.search() must return a list.")
    if len(results) != CANDIDATE_K:
        raise ValueError(
            f"Retrieval returned {len(results)} results; expected {CANDIDATE_K}."
        )

    by_id = {}
    seen_ranks = set()
    for position, result in enumerate(results, start=1):
        context = f"retrieval result {position}"
        if not isinstance(result, dict):
            raise TypeError(f"{context} must be a dictionary.")
        product_id = require_product_id(result.get("product_id"), context)
        rank = require_rank(result.get("rank"), context)
        score = require_score(result.get("score"), "score", context)
        if product_id in by_id:
            raise ValueError(f"Duplicate retrieval product_id: {product_id}.")
        if rank in seen_ranks:
            raise ValueError(f"Duplicate retrieval rank: {rank}.")
        if rank != position:
            raise ValueError(
                f"Retrieval rank/order mismatch at position {position}: rank={rank}."
            )
        by_id[product_id] = {"rank": rank, "score": score}
        seen_ranks.add(rank)
    return by_id


def validate_and_merge_results(recommendations: list, retrieval_by_id: dict) -> list:
    if not isinstance(recommendations, list):
        raise TypeError("RecommendationServiceV2.recommend() must return a list.")
    if len(recommendations) != RERANK_POOL_K:
        raise ValueError(
            f"V2 returned {len(recommendations)} results; "
            f"expected exactly {RERANK_POOL_K}."
        )

    merged = []
    seen_ids = set()
    seen_final_ranks = set()
    for position, item in enumerate(recommendations, start=1):
        context = f"V2 recommendation {position}"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be a dictionary.")
        product_id = require_product_id(item.get("product_id"), context)
        final_rank = require_rank(item.get("rank"), context)
        retrieval_rank = require_rank(item.get("retrieval_rank"), context)

        if product_id in seen_ids:
            raise ValueError(f"Duplicate V2 product_id: {product_id}.")
        if final_rank in seen_final_ranks:
            raise ValueError(f"Duplicate final rank: {final_rank}.")
        if final_rank != position:
            raise ValueError(
                f"Final rank/order mismatch at position {position}: rank={final_rank}."
            )
        if product_id not in retrieval_by_id:
            raise ValueError(
                f"V2 product_id {product_id} is absent from Retrieval Top-50."
            )
        if retrieval_rank > RERANK_POOL_K:
            raise ValueError(
                f"V2 product {product_id} has retrieval_rank={retrieval_rank}; "
                f"expected <= {RERANK_POOL_K} without hard constraints."
            )

        retrieval_reference = retrieval_by_id[product_id]
        if retrieval_rank != retrieval_reference["rank"]:
            raise ValueError(
                f"Retrieval rank mismatch for {product_id}: V2={retrieval_rank}, "
                f"retrieval={retrieval_reference['rank']}."
            )
        retrieval_score = require_score(
            item.get("retrieval_score"), "retrieval_score", context
        )
        if not math.isclose(
            retrieval_score,
            retrieval_reference["score"],
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"Retrieval score mismatch for {product_id}: V2={retrieval_score}, "
                f"retrieval={retrieval_reference['score']}."
            )

        product = item.get("product")
        if not isinstance(product, dict) or product.get("product_id") != product_id:
            raise ValueError(f"Product object ID mismatch for {product_id}.")

        merged.append(
            {
                "final_rank": final_rank,
                "retrieval_rank": retrieval_rank,
                "rank_change": int(retrieval_rank - final_rank),
                "product_id": product_id,
                "title": product.get("title"),
                "retrieval_score": float(retrieval_score),
                "semantic_score": require_score(
                    item.get("semantic_score"), "semantic_score", context, True
                ),
                "rating_score": require_score(
                    item.get("rating_score"), "rating_score", context, True
                ),
                "popularity_score": require_score(
                    item.get("popularity_score"), "popularity_score", context, True
                ),
                "final_score": require_score(
                    item.get("final_score"), "final_score", context
                ),
            }
        )
        seen_ids.add(product_id)
        seen_final_ranks.add(final_rank)

    expected_ranks = set(range(1, RERANK_POOL_K + 1))
    if seen_final_ranks != expected_ranks:
        raise ValueError("V2 final ranks are not exactly 1 through 20.")
    expected_pool_ids = {
        product_id
        for product_id, value in retrieval_by_id.items()
        if value["rank"] <= RERANK_POOL_K
    }
    if seen_ids != expected_pool_ids:
        raise ValueError(
            "V2 result IDs do not exactly match the unfiltered Retrieval Top-20."
        )
    return merged


# ============================================================
# 4. Summary calculations and stdout tables
# ============================================================

def printable_title(value) -> str:
    if value is None or value == "":
        return "(not available)"
    return " ".join(str(value).split())


def print_table(title: str, rows: list) -> None:
    print(f"\n{title}")
    print(
        "final_rank\tretrieval_rank\trank_change\tproduct_id\t"
        "retrieval_score\tsemantic_score\trating_score\t"
        "popularity_score\tfinal_score\ttitle"
    )
    for item in rows:
        print(
            f"{item['final_rank']}\t{item['retrieval_rank']}\t"
            f"{item['rank_change']:+d}\t{item['product_id']}\t"
            f"{item['retrieval_score']:.8f}\t{item['semantic_score']:.8f}\t"
            f"{item['rating_score']:.8f}\t{item['popularity_score']:.8f}\t"
            f"{item['final_score']:.8f}\t{printable_title(item['title'])}"
        )


def print_analysis(rows: list, candidate_count: int) -> None:
    final_top = rows[:ANALYSIS_K]
    retrieval_top_ids = {
        item["product_id"] for item in rows if item["retrieval_rank"] <= ANALYSIS_K
    }
    final_top_ids = {item["product_id"] for item in final_top}
    retained = len(retrieval_top_ids & final_top_ids)

    print(f"Query: {QUERY}")
    print(f"Candidate count: {candidate_count}")
    print(f"Rerank pool size: {len(rows)}")
    print(
        f"Retrieval Top-{ANALYSIS_K} retained in Final Top-{ANALYSIS_K}: "
        f"{retained}/{ANALYSIS_K}"
    )
    print(
        f"Final Top-{ANALYSIS_K} mean retrieval rank: "
        f"{statistics.fmean(item['retrieval_rank'] for item in final_top):.4f}"
    )
    print(
        f"Final Top-{ANALYSIS_K} mean retrieval score: "
        f"{statistics.fmean(item['retrieval_score'] for item in final_top):.8f}"
    )
    print(
        f"Final Top-{ANALYSIS_K} count with retrieval_rank > 20: "
        f"{sum(item['retrieval_rank'] > 20 for item in final_top)}"
    )
    print(
        f"Final Top-{ANALYSIS_K} count with retrieval_rank > 30: "
        f"{sum(item['retrieval_rank'] > 30 for item in final_top)}"
    )

    print_table(f"Final Top-{ANALYSIS_K}", final_top)
    upward = sorted(
        rows,
        key=lambda item: (-item["rank_change"], item["final_rank"], item["product_id"]),
    )[:MOVER_COUNT]
    downward = sorted(
        rows,
        key=lambda item: (item["rank_change"], item["final_rank"], item["product_id"]),
    )[:MOVER_COUNT]
    print_table(f"Top {MOVER_COUNT} upward movers", upward)
    print_table(f"Top {MOVER_COUNT} downward movers", downward)

    by_retrieval_rank = sorted(rows, key=lambda item: item["retrieval_rank"])
    print("\nRetrieval Top-10 products and their final ranks")
    print("retrieval_rank\tfinal_rank\trank_change\tproduct_id\ttitle")
    for item in by_retrieval_rank[:ANALYSIS_K]:
        print(
            f"{item['retrieval_rank']}\t{item['final_rank']}\t"
            f"{item['rank_change']:+d}\t{item['product_id']}\t"
            f"{printable_title(item['title'])}"
        )


# ============================================================
# 5. Read-only analysis orchestration
# ============================================================

def analyze() -> None:
    service_class = load_recommendation_service_class()
    service = service_class()
    retrieval_results = service.retriever.search(QUERY, top_k=CANDIDATE_K)
    recommendation_results = service.recommend(
        QUERY,
        top_k=RERANK_POOL_K,
        candidate_k=CANDIDATE_K,
        rerank_pool_k=RERANK_POOL_K,
    )
    retrieval_by_id = validate_retrieval_results(retrieval_results)
    merged = validate_and_merge_results(recommendation_results, retrieval_by_id)
    print_analysis(merged, len(retrieval_results))


def main() -> int:
    analyze()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
