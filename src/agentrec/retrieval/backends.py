"""Replaceable vector-search backends for normalized chunk embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendSearchResult:
    rows: np.ndarray
    scores: np.ndarray


class ChunkRetrievalBackend(Protocol):
    @property
    def row_count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def search(
        self, query_embedding: np.ndarray, *, top_k: int,
        candidate_rows: np.ndarray | None = None,
    ) -> BackendSearchResult: ...


class NumPyExactBackend:
    """Exact inner-product baseline with stable deterministic full sorting."""

    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.dtype != np.dtype(np.float32):
            raise ValueError("embeddings must be a two-dimensional float32 array.")
        self._embeddings = embeddings

    @property
    def row_count(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def dimension(self) -> int:
        return int(self._embeddings.shape[1])

    def search(
        self, query_embedding: np.ndarray, *, top_k: int,
        candidate_rows: np.ndarray | None = None,
    ) -> BackendSearchResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer; bool is not accepted.")
        if top_k > self.row_count:
            raise ValueError(f"top_k={top_k} exceeds row_count={self.row_count}.")
        query = np.asarray(query_embedding)
        if query.shape != (self.dimension,) or query.dtype != np.dtype(np.float32):
            raise ValueError(
                f"query_embedding must have shape ({self.dimension},) and dtype float32."
            )
        if not query.flags.c_contiguous or not np.isfinite(query).all():
            raise ValueError("query_embedding must be contiguous and finite.")
        if not np.isclose(np.linalg.norm(query), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("query_embedding must be L2 normalized.")

        if candidate_rows is None:
            rows = np.arange(self.row_count, dtype=np.int64)
            scores = np.asarray(self._embeddings @ query, dtype=np.float32)
        else:
            raw_rows = np.asarray(candidate_rows)
            if raw_rows.ndim != 1 or raw_rows.dtype.kind not in "iu":
                raise ValueError("candidate_rows must be a one-dimensional integer array.")
            rows = np.unique(raw_rows.astype(np.int64, copy=False))
            if rows.size and (rows[0] < 0 or rows[-1] >= self.row_count):
                raise ValueError("candidate_rows contains an out-of-range global row.")
            if not rows.size:
                return BackendSearchResult(
                    rows=np.empty(0, dtype=np.int64),
                    scores=np.empty(0, dtype=np.float32),
                )
            scores = np.asarray(self._embeddings[rows] @ query, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("Vector search produced NaN or Inf scores.")

        # rows are ascending before stable sort, so ties resolve to lower global row.
        order = np.argsort(-scores, kind="stable")[:min(top_k, rows.size)]
        return BackendSearchResult(
            rows=np.ascontiguousarray(rows[order], dtype=np.int64),
            scores=np.ascontiguousarray(scores[order], dtype=np.float32),
        )
