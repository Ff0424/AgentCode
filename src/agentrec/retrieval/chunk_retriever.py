"""Application-neutral orchestration for query encoding and chunk retrieval."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from .artifact_loader import ChunkRetrievalArtifacts
from .backends import ChunkRetrievalBackend
from .contracts import RetrievedChunk, RetrievalResult
from .query_encoder import QueryEncoder


class ChunkRetriever:
    def __init__(
        self, *, encoder: QueryEncoder, backend: ChunkRetrievalBackend,
        artifacts: ChunkRetrievalArtifacts,
    ) -> None:
        if encoder.embedding_dimension != backend.dimension:
            raise ValueError("Query encoder and retrieval backend dimensions differ.")
        if backend.dimension != artifacts.dimension or backend.row_count != artifacts.row_count:
            raise ValueError("Retrieval backend and artifact dimensions/rows differ.")
        self.encoder = encoder
        self.backend = backend
        self.artifacts = artifacts

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return query.strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer; bool is not accepted.")
        return top_k

    @staticmethod
    def _validate_candidate_items(
        values: Collection[int] | None,
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
            raise TypeError("candidate_item_indices must be a collection of integers.")
        result: set[int] = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("candidate_item_indices must contain non-negative integers.")
            result.add(value)
        return tuple(sorted(result))

    def search(
        self, query: str, *, top_k: int = 10,
        candidate_item_indices: Collection[int] | None = None,
    ) -> RetrievalResult:
        normalized_query = self._validate_query(query)
        requested = self._validate_top_k(top_k)
        if requested > self.artifacts.row_count:
            raise ValueError(
                f"top_k={requested} exceeds chunk_count={self.artifacts.row_count}."
            )
        candidate_items = self._validate_candidate_items(candidate_item_indices)
        candidate_rows = (
            None if candidate_items is None
            else self.artifacts.rows_for_items(candidate_items)
        )
        if candidate_rows is not None and not candidate_rows.size:
            return RetrievalResult(
                query=normalized_query, requested_top_k=requested,
                returned_count=0, candidate_restricted=True, chunks=(),
            )
        encoded = self.encoder.encode([normalized_query])
        if encoded.shape != (1, self.backend.dimension):
            raise ValueError("QueryEncoder violated its output shape contract.")
        result = self.backend.search(
            encoded[0], top_k=requested, candidate_rows=candidate_rows
        )
        chunks: list[RetrievedChunk] = []
        for rank, (row_value, score_value) in enumerate(
            zip(result.rows, result.scores), start=1
        ):
            row = int(row_value)
            chunk = self.artifacts.get_chunk(row)
            chunks.append(RetrievedChunk(
                rank=rank, embedding_row=row, chunk_index=chunk.chunk_index,
                chunk_id=chunk.chunk_id, parent_asin=chunk.parent_asin,
                item_index=chunk.item_index, chunk_type=chunk.chunk_type,
                part_index=chunk.part_index, text=chunk.text,
                similarity_score=float(score_value),
            ))
        return RetrievalResult(
            query=normalized_query, requested_top_k=requested,
            returned_count=len(chunks),
            candidate_restricted=candidate_items is not None,
            chunks=tuple(chunks),
        )
