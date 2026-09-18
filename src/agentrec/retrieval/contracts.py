"""Immutable public contracts for chunk vector retrieval."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.agentrec.knowledge import KnowledgeChunkType


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RetrievedChunk(BaseModel):
    """One grounded knowledge chunk returned by vector retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Annotated[int, Field(strict=True, ge=1)]
    embedding_row: Annotated[int, Field(strict=True, ge=0)]
    chunk_index: Annotated[int, Field(strict=True, ge=0)]
    chunk_id: NonEmptyText
    parent_asin: NonEmptyText
    item_index: Annotated[int, Field(strict=True, ge=0)]
    chunk_type: KnowledgeChunkType
    part_index: Annotated[int, Field(strict=True, ge=0)]
    text: NonEmptyText
    similarity_score: float

    @field_validator("similarity_score", mode="before")
    @classmethod
    def validate_similarity_score(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("similarity_score must be a finite number.")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("similarity_score must be finite.")
        if score < -1.00001 or score > 1.00001:
            raise ValueError("normalized-vector similarity_score must be in [-1, 1].")
        return min(1.0, max(-1.0, score))


class RetrievalResult(BaseModel):
    """Stable Agent-neutral result for one natural-language query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: NonEmptyText
    requested_top_k: Annotated[int, Field(strict=True, ge=1)]
    returned_count: Annotated[int, Field(strict=True, ge=0)]
    candidate_restricted: bool
    chunks: tuple[RetrievedChunk, ...]

    @field_validator("chunks")
    @classmethod
    def validate_ranks(cls, value: tuple[RetrievedChunk, ...]) -> tuple[RetrievedChunk, ...]:
        if [chunk.rank for chunk in value] != list(range(1, len(value) + 1)):
            raise ValueError("Retrieved chunk ranks must be exactly 1..N.")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.returned_count != len(self.chunks):
            raise ValueError("returned_count must equal len(chunks).")
