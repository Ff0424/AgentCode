"""
Standard LightGCN model for AgentRec V2 collaborative recommendation.

Inputs:
    - A symmetric, degree-normalized sparse user-item adjacency matrix.
    - Compact model-local user/item indices for BPR training triples.

Outputs:
    - Layer-averaged final user and item embeddings.
    - BPR ranking loss with L2 regularization on ego embeddings.

The implementation intentionally contains no feature transformation,
non-linear activation, attention, semantic feature, or reranking component.
Canonical identities remain the runner's responsibility: this model only uses
compact indices and never changes the Product Catalog mapping.
"""

from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# 1. Construction validation
# ============================================================

def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(value)


def _validate_index_tensor(values: Tensor, name: str) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {tuple(values.shape)}.")
    if values.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long; got {values.dtype}.")


# ============================================================
# 2. Standard LightGCN
# ============================================================

class LightGCN(nn.Module):
    """Pure collaborative LightGCN with mean layer aggregation."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        normalized_adjacency: Tensor,
    ) -> None:
        super().__init__()
        self.num_users = _positive_int(num_users, "num_users")
        self.num_items = _positive_int(num_items, "num_items")
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.num_layers = _non_negative_int(num_layers, "num_layers")

        expected_shape = (
            self.num_users + self.num_items,
            self.num_users + self.num_items,
        )
        if not isinstance(normalized_adjacency, Tensor):
            raise TypeError("normalized_adjacency must be a torch.Tensor.")
        if normalized_adjacency.layout != torch.sparse_coo:
            raise TypeError(
                "normalized_adjacency must use torch.sparse_coo layout."
            )
        if tuple(normalized_adjacency.shape) != expected_shape:
            raise ValueError(
                "normalized_adjacency shape is "
                f"{tuple(normalized_adjacency.shape)}; expected {expected_shape}."
            )
        if not normalized_adjacency.is_coalesced():
            normalized_adjacency = normalized_adjacency.coalesce()

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        # The graph is rebuilt deterministically by the runner and is omitted
        # from checkpoints to avoid duplicating a large sparse artifact.
        self.register_buffer(
            "normalized_adjacency",
            normalized_adjacency,
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the standard Xavier initialization for ego embeddings."""

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self) -> tuple[Tensor, Tensor]:
        """Propagate over the full graph and mean-aggregate all layers."""

        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        layer_embeddings = [ego_embeddings]
        current_embeddings = ego_embeddings
        for _ in range(self.num_layers):
            current_embeddings = torch.sparse.mm(
                self.normalized_adjacency,
                current_embeddings,
            )
            layer_embeddings.append(current_embeddings)

        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return torch.split(
            final_embeddings,
            (self.num_users, self.num_items),
            dim=0,
        )

    def bpr_loss(
        self,
        user_indices: Tensor,
        positive_item_indices: Tensor,
        negative_item_indices: Tensor,
        reg_weight: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mean implicit-feedback BPR loss for one triple batch.

        L2 regularization is applied to the batch's ego embeddings, while the
        ranking scores use graph-propagated, layer-averaged embeddings.
        """

        _validate_index_tensor(user_indices, "user_indices")
        _validate_index_tensor(positive_item_indices, "positive_item_indices")
        _validate_index_tensor(negative_item_indices, "negative_item_indices")
        if not (
            user_indices.shape
            == positive_item_indices.shape
            == negative_item_indices.shape
        ):
            raise ValueError("BPR user/positive/negative tensors must have equal shapes.")
        if user_indices.numel() == 0:
            raise ValueError("A BPR batch must contain at least one triple.")
        if isinstance(reg_weight, bool) or not isinstance(reg_weight, (int, float)):
            raise TypeError("reg_weight must be a finite non-negative number.")
        reg_weight = float(reg_weight)
        if not math.isfinite(reg_weight) or reg_weight < 0.0:
            raise ValueError("reg_weight must be a finite non-negative number.")

        final_users, final_items = self.propagate()
        batch_users = final_users[user_indices]
        batch_positive = final_items[positive_item_indices]
        batch_negative = final_items[negative_item_indices]

        positive_scores = torch.sum(batch_users * batch_positive, dim=1)
        negative_scores = torch.sum(batch_users * batch_negative, dim=1)
        ranking_loss = -F.logsigmoid(positive_scores - negative_scores).mean()

        ego_users = self.user_embedding(user_indices)
        ego_positive = self.item_embedding(positive_item_indices)
        ego_negative = self.item_embedding(negative_item_indices)
        regularization_loss = 0.5 * (
            ego_users.square().sum()
            + ego_positive.square().sum()
            + ego_negative.square().sum()
        ) / user_indices.numel()
        total_loss = ranking_loss + reg_weight * regularization_loss
        return total_loss, ranking_loss.detach(), regularization_loss.detach()
