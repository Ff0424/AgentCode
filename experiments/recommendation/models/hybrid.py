"""Score normalization and fusion for AgentRec V2 hybrid recommendation."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def per_user_zscore(scores: Tensor, epsilon: float) -> Tensor:
    """Normalize each user's full-catalog scores with stable population Z-score."""

    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating two-dimensional tensor.")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a finite positive number.")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive number.")
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False)
    normalized = (scores - mean) / (std + epsilon)
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("Z-score normalization produced non-finite values.")
    return normalized


def fuse_uniform(lightgcn_scores: Tensor, semantic_scores: Tensor, alpha: float) -> Tensor:
    """Fuse normalized collaborative and semantic scores with semantic weight alpha."""

    if lightgcn_scores.shape != semantic_scores.shape:
        raise ValueError("LightGCN and semantic score shapes must match.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a finite number in [0, 1].")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1].")
    return (1.0 - alpha) * lightgcn_scores + alpha * semantic_scores


def fuse_cold_aware(
    lightgcn_scores: Tensor,
    semantic_scores: Tensor,
    alpha: float,
    train_observed_items: Tensor,
) -> Tensor:
    """Use uniform fusion for warm items and semantic-only scores for cold items."""

    fused = fuse_uniform(lightgcn_scores, semantic_scores, alpha)
    if train_observed_items.ndim != 1 or train_observed_items.dtype != torch.bool:
        raise ValueError("train_observed_items must be a one-dimensional bool tensor.")
    if train_observed_items.numel() != fused.shape[1]:
        raise ValueError("Cold-item mask length must equal the item score dimension.")
    return torch.where(train_observed_items.unsqueeze(0), fused, semantic_scores)
