"""
Deterministic popularity baseline for AgentRec V2 recommendation experiments.

Inputs:
    ``fit(train_user, train_item)`` consumes aligned one-dimensional canonical
    user/item index arrays from the frozen training split.

Outputs:
    ``recommend(user_ids, k)`` returns a two-dimensional ``int32`` array with
    one duplicate-free Top-K canonical item ranking per requested user. Items
    seen by that user in the training split are excluded.

The model has no learned parameters beyond training-set item interaction
frequencies. Frequency ties are resolved by ascending canonical item index so
that repeated runs are deterministic.
"""

from __future__ import annotations

import numbers

import numpy as np


# ============================================================
# 1. Input validation
# ============================================================

def _as_index_array(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a one-dimensional non-negative integer array."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices; got {array.dtype}.")
    if array.size and np.any(array < 0):
        raise ValueError(f"{name} contains negative indices.")
    return array


def _validate_k(k: int) -> int:
    """Reject booleans and non-positive/non-integral cutoff values."""

    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    return int(k)


# ============================================================
# 2. Popularity recommender
# ============================================================

class PopularityRecommender:
    """Rank train-observed items by interaction frequency.

    User histories are stored as user-sorted NumPy arrays rather than a large
    Python dictionary of sets. A small temporary set is created only for the
    user currently being ranked.
    """

    def __init__(self) -> None:
        self._popular_items: np.ndarray | None = None
        self._popular_counts: np.ndarray | None = None
        self._seen_users: np.ndarray | None = None
        self._seen_offsets: np.ndarray | None = None
        self._seen_items: np.ndarray | None = None

    @property
    def popular_items(self) -> np.ndarray:
        """Return the fitted global popularity order as a read-only array."""

        self._require_fitted()
        assert self._popular_items is not None
        return self._popular_items

    @property
    def popular_counts(self) -> np.ndarray:
        """Return interaction counts aligned with ``popular_items``."""

        self._require_fitted()
        assert self._popular_counts is not None
        return self._popular_counts

    def fit(
        self,
        train_user: np.ndarray,
        train_item: np.ndarray,
    ) -> "PopularityRecommender":
        """Fit item frequencies and train-seen histories.

        Args:
            train_user: Canonical user index for every training interaction.
            train_item: Canonical item index aligned row-for-row with users.

        Returns:
            This fitted recommender.

        Raises:
            TypeError: If either input is not an integer array.
            ValueError: If inputs are empty, misaligned, multidimensional, or
                contain negative indices.
        """

        users = _as_index_array(train_user, "train_user")
        items = _as_index_array(train_item, "train_item")
        if users.shape != items.shape:
            raise ValueError(
                f"train_user/train_item shapes differ: {users.shape} != {items.shape}."
            )
        if users.size == 0:
            raise ValueError("The training split must contain at least one interaction.")

        unique_items, counts = np.unique(items, return_counts=True)
        # np.lexsort uses the last key as primary: count descending, then ID ascending.
        popularity_order = np.lexsort((unique_items, -counts))
        self._popular_items = np.ascontiguousarray(
            unique_items[popularity_order], dtype=np.int32
        )
        self._popular_counts = np.ascontiguousarray(
            counts[popularity_order], dtype=np.int64
        )

        # Preserve an already user-sorted training split without allocating a
        # full argsort index; otherwise build a deterministic user/item order.
        if users.size < 2 or bool(np.all(users[:-1] <= users[1:])):
            sorted_users = users
            sorted_items = items
        else:
            order = np.lexsort((items, users))
            sorted_users = users[order]
            sorted_items = items[order]

        seen_users, first_offsets = np.unique(sorted_users, return_index=True)
        self._seen_users = np.ascontiguousarray(seen_users, dtype=np.int32)
        self._seen_offsets = np.ascontiguousarray(
            np.append(first_offsets, sorted_users.size), dtype=np.int64
        )
        self._seen_items = np.ascontiguousarray(sorted_items, dtype=np.int32)

        for array in (
            self._popular_items,
            self._popular_counts,
            self._seen_users,
            self._seen_offsets,
            self._seen_items,
        ):
            array.setflags(write=False)
        return self

    def recommend(self, user_ids: np.ndarray, k: int) -> np.ndarray:
        """Return train-seen-filtered global Top-K rankings for users.

        Unknown users receive the unfiltered global popularity ranking. The
        returned row order strictly matches the input ``user_ids`` order.
        """

        self._require_fitted()
        users = _as_index_array(user_ids, "user_ids")
        cutoff = _validate_k(k)

        assert self._popular_items is not None
        if cutoff > self._popular_items.size:
            raise ValueError(
                f"k={cutoff} exceeds {self._popular_items.size} train-observed items."
            )
        rankings = np.empty((users.size, cutoff), dtype=np.int32)
        if users.size == 0:
            return rankings

        for row, raw_user_id in enumerate(users):
            user_id = int(raw_user_id)
            seen_items = self._seen_items_for_user(user_id)
            if seen_items.size == 0:
                rankings[row] = self._popular_items[:cutoff]
                continue

            seen = {int(item_id) for item_id in seen_items}
            output_offset = 0
            for raw_item_id in self._popular_items:
                item_id = int(raw_item_id)
                if item_id in seen:
                    continue
                rankings[row, output_offset] = item_id
                output_offset += 1
                if output_offset == cutoff:
                    break
            if output_offset != cutoff:
                raise ValueError(
                    f"User {user_id} has fewer than {cutoff} unseen train items."
                )
        return rankings

    def _seen_items_for_user(self, user_id: int) -> np.ndarray:
        """Return the fitted training-item slice for one canonical user."""

        assert self._seen_users is not None
        assert self._seen_offsets is not None
        assert self._seen_items is not None

        position = int(np.searchsorted(self._seen_users, user_id))
        if position >= self._seen_users.size or int(self._seen_users[position]) != user_id:
            return self._seen_items[:0]
        start = int(self._seen_offsets[position])
        stop = int(self._seen_offsets[position + 1])
        return self._seen_items[start:stop]

    def _require_fitted(self) -> None:
        if self._popular_items is None:
            raise RuntimeError("PopularityRecommender.fit() must be called first.")
