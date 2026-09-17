"""Exact leave-one-out Recall/NDCG metrics from held-out target ranks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PartitionMetrics:
    overall: dict[str, float]
    warm: dict[str, float]
    cold: dict[str, float] | None
    warm_targets: int
    cold_targets: int


def metrics_from_target_ranks(ranks: np.ndarray, topk: tuple[int, ...]) -> dict[str, float]:
    """Compute LOO metrics where rank 0 denotes a target missing from Top-max(K)."""

    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("Target ranks must be a non-empty one-dimensional array.")
    if np.any(ranks < 0):
        raise ValueError("Target ranks cannot be negative.")
    metrics = {}
    for k in topk:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("Every K must be a positive integer.")
        hits = (ranks > 0) & (ranks <= k)
        metrics[f"Recall@{k}"] = float(hits.mean())
        gains = np.zeros(ranks.size, dtype=np.float64)
        gains[hits] = 1.0 / np.log2(ranks[hits].astype(np.float64) + 1.0)
        metrics[f"NDCG@{k}"] = float(gains.mean())
    return metrics


def evaluate_rank_partitions(
    ranks: np.ndarray,
    warm_mask: np.ndarray,
    topk: tuple[int, ...],
) -> PartitionMetrics:
    warm_mask = np.asarray(warm_mask, dtype=bool)
    if warm_mask.shape != np.asarray(ranks).shape:
        raise ValueError("warm_mask must align with target ranks.")
    warm_count = int(warm_mask.sum())
    cold_count = int((~warm_mask).sum())
    if warm_count == 0:
        raise ValueError("Evaluation requires at least one warm target.")
    return PartitionMetrics(
        overall=metrics_from_target_ranks(ranks, topk),
        warm=metrics_from_target_ranks(np.asarray(ranks)[warm_mask], topk),
        cold=(
            metrics_from_target_ranks(np.asarray(ranks)[~warm_mask], topk)
            if cold_count
            else None
        ),
        warm_targets=warm_count,
        cold_targets=cold_count,
    )
