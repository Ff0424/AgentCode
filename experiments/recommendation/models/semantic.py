"""Core tensor operations for the pure-text semantic recommendation baseline."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def build_mean_user_profiles(
    train_users: Tensor,
    train_item_rows: Tensor,
    item_embeddings: Tensor,
    num_users: int,
    normalize: bool,
    interaction_batch_size: int = 65_536,
) -> tuple[Tensor, Tensor]:
    """Mean-pool train-history item vectors into one profile per compact user."""

    if train_users.ndim != 1 or train_item_rows.ndim != 1:
        raise ValueError("Training user/item tensors must be one-dimensional.")
    if train_users.shape != train_item_rows.shape or train_users.numel() == 0:
        raise ValueError("Training user/item tensors must be aligned and non-empty.")
    if item_embeddings.ndim != 2 or not item_embeddings.is_floating_point():
        raise ValueError("item_embeddings must be a floating two-dimensional tensor.")
    profiles = torch.zeros(
        (num_users, item_embeddings.shape[1]),
        dtype=item_embeddings.dtype,
        device=item_embeddings.device,
    )
    if interaction_batch_size <= 0:
        raise ValueError("interaction_batch_size must be positive.")
    # Chunk gathering so millions of history vectors never form one huge tensor.
    for start in range(0, train_users.numel(), interaction_batch_size):
        stop = min(start + interaction_batch_size, train_users.numel())
        profiles.index_add_(
            0,
            train_users[start:stop],
            item_embeddings[train_item_rows[start:stop]],
        )
    counts = torch.bincount(train_users, minlength=num_users)
    if torch.any(counts == 0):
        raise ValueError("Every compact user must have at least one train interaction.")
    profiles.div_(counts.to(profiles.dtype).unsqueeze(1))
    if normalize:
        profiles = F.normalize(profiles, p=2, dim=1)
    return profiles.contiguous(), counts


def validate_normalized_embeddings(embeddings: Tensor, tolerance: float = 2e-4) -> None:
    """Require finite, nonzero, approximately unit-length embedding rows."""

    if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite two-dimensional tensor.")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if torch.any(norms == 0) or torch.max(torch.abs(norms - 1.0)).item() > tolerance:
        raise ValueError("Configured normalized embeddings are not unit length.")
