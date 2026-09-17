"""
Model-independent top-K ranking metrics for AgentRec recommendation experiments.

Inputs:
    - One ranked item sequence per evaluation user.
    - One non-empty ground-truth item collection (or one scalar item) per user.
    - Positive integer cutoffs, conventionally K = [10, 20, 50].

Outputs:
    ``evaluate_ranking`` returns macro-averaged metrics using conventional keys
    such as ``Recall@10`` and ``NDCG@10``. Items may be canonical integer IDs or
    any other hashable identity type. The module does not depend on a specific
    model, tensor framework, or candidate-generation implementation.
"""

import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any


# ============================================================
# 1. Input normalization and validation
# ============================================================

DEFAULT_TOPK = (10, 20, 50)


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"K must be a positive integer, got {k!r}.")
    return k


def _validate_topk(topk: Iterable[int]) -> tuple[int, ...]:
    if isinstance(topk, (str, bytes)):
        raise TypeError("topk must be an iterable of positive integers.")
    try:
        values = tuple(_validate_k(k) for k in topk)
    except TypeError as exc:
        raise TypeError("topk must be an iterable of positive integers.") from exc
    if not values:
        raise ValueError("topk must contain at least one cutoff.")
    if len(set(values)) != len(values):
        raise ValueError(f"topk contains duplicate cutoffs: {values}.")
    return values


def _as_ranked_list(ranked_items: Iterable[Hashable], context: str) -> list[Hashable]:
    if isinstance(ranked_items, (str, bytes)):
        raise TypeError(f"{context} must be a ranked iterable, not one string.")
    try:
        ranking = list(ranked_items)
    except TypeError as exc:
        raise TypeError(f"{context} must be an iterable of item IDs.") from exc

    try:
        unique_count = len(set(ranking))
    except TypeError as exc:
        raise TypeError(f"Every item in {context} must be hashable.") from exc
    if unique_count != len(ranking):
        raise ValueError(f"{context} contains duplicate item IDs.")
    return ranking


def _as_relevant_set(relevant_items: Any, context: str) -> set[Hashable]:
    # A scalar item is convenient for the current leave-one-out split. Strings
    # are scalar product IDs rather than character collections.
    if isinstance(relevant_items, (str, bytes)):
        values = [relevant_items]
    else:
        try:
            values = list(relevant_items)
        except TypeError:
            values = [relevant_items]

    try:
        relevant = set(values)
    except TypeError as exc:
        raise TypeError(f"Every item in {context} must be hashable.") from exc
    if not relevant:
        raise ValueError(f"{context} must contain at least one relevant item.")
    return relevant


# ============================================================
# 2. Per-user Recall@K and NDCG@K
# ============================================================

def recall_at_k(
    ranked_items: Sequence[Hashable],
    relevant_items: Any,
    k: int,
) -> float:
    """Return binary-relevance Recall@K for one user."""

    cutoff = _validate_k(k)
    ranking = _as_ranked_list(ranked_items, "ranked_items")
    relevant = _as_relevant_set(relevant_items, "relevant_items")
    hits = sum(item in relevant for item in ranking[:cutoff])
    return float(hits / len(relevant))


def ndcg_at_k(
    ranked_items: Sequence[Hashable],
    relevant_items: Any,
    k: int,
) -> float:
    """Return binary-relevance normalized discounted cumulative gain at K."""

    cutoff = _validate_k(k)
    ranking = _as_ranked_list(ranked_items, "ranked_items")
    relevant = _as_relevant_set(relevant_items, "relevant_items")

    dcg = sum(
        1.0 / math.log2(rank + 2.0)
        for rank, item in enumerate(ranking[:cutoff])
        if item in relevant
    )
    ideal_hits = min(cutoff, len(relevant))
    ideal_dcg = sum(
        1.0 / math.log2(rank + 2.0) for rank in range(ideal_hits)
    )
    return float(dcg / ideal_dcg)


# ============================================================
# 3. Macro-averaged experiment evaluation
# ============================================================

def evaluate_ranking(
    ranked_items_by_user: Iterable[Iterable[Hashable]],
    relevant_items_by_user: Iterable[Any],
    topk: Iterable[int] = DEFAULT_TOPK,
) -> dict[str, float]:
    """Compute macro-averaged Recall@K and NDCG@K across evaluation users.

    Args:
        ranked_items_by_user: Ranked, duplicate-free item IDs for every user.
        relevant_items_by_user: Matching non-empty ground-truth collections, or
            scalar ground-truth item IDs for leave-one-out evaluation.
        topk: Positive unique ranking cutoffs. Defaults to (10, 20, 50).

    Returns:
        A flat metric mapping, for example ``{"Recall@10": 0.4,
        "NDCG@10": 0.25, ...}``.

    Raises:
        TypeError: For non-iterable or unhashable inputs.
        ValueError: For invalid K, empty ground truth, duplicate rankings, no
            users, or different prediction/ground-truth user counts.
    """

    cutoffs = _validate_topk(topk)
    try:
        rankings = list(ranked_items_by_user)
        ground_truth = list(relevant_items_by_user)
    except TypeError as exc:
        raise TypeError(
            "ranked_items_by_user and relevant_items_by_user must be iterable."
        ) from exc

    if len(rankings) != len(ground_truth):
        raise ValueError(
            "Prediction and ground-truth user counts differ: "
            f"{len(rankings)} != {len(ground_truth)}."
        )
    if not rankings:
        raise ValueError("At least one evaluation user is required.")

    recall_totals = {k: 0.0 for k in cutoffs}
    ndcg_totals = {k: 0.0 for k in cutoffs}
    for user_offset, (raw_ranking, raw_relevant) in enumerate(
        zip(rankings, ground_truth)
    ):
        ranking = _as_ranked_list(raw_ranking, f"ranking for user {user_offset}")
        relevant = _as_relevant_set(
            raw_relevant, f"ground truth for user {user_offset}"
        )
        for k in cutoffs:
            hits = sum(item in relevant for item in ranking[:k])
            recall_totals[k] += hits / len(relevant)

            dcg = sum(
                1.0 / math.log2(rank + 2.0)
                for rank, item in enumerate(ranking[:k])
                if item in relevant
            )
            ideal_dcg = sum(
                1.0 / math.log2(rank + 2.0)
                for rank in range(min(k, len(relevant)))
            )
            ndcg_totals[k] += dcg / ideal_dcg

    user_count = len(rankings)
    metrics: dict[str, float] = {}
    for k in cutoffs:
        metrics[f"Recall@{k}"] = float(recall_totals[k] / user_count)
        metrics[f"NDCG@{k}"] = float(ndcg_totals[k] / user_count)
    return metrics


if __name__ == "__main__":
    example_metrics = evaluate_ranking(
        ranked_items_by_user=[[10, 20, 30], [40, 50, 60]],
        relevant_items_by_user=[20, 99],
    )
    for metric_name, value in example_metrics.items():
        print(f"{metric_name}: {value:.6f}")
