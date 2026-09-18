"""Immutable runtime contracts for grounded candidate evidence."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from ..knowledge import KnowledgeChunkType


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Text = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class EvidenceStatus(str, Enum):
    """Evidence is retrieved context, never an automatic fact verification."""

    RETRIEVED_UNVERIFIED = "retrieved_unverified"


class EvidenceCandidate(BaseModel):
    """Minimal canonical candidate identity accepted by the evidence service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText


class EvidenceSnippet(BaseModel):
    """One identity-safe KnowledgeChunk projection without embedding internals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Annotated[int, Field(strict=True, ge=1)]
    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    chunk_id: NonEmptyText
    chunk_type: KnowledgeChunkType
    part_index: Annotated[int, Field(strict=True, ge=0)]
    text: NonEmptyText
    similarity_score: float

    @field_validator("similarity_score", mode="before")
    @classmethod
    def validate_score(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("similarity_score must be a finite number.")
        score = float(value)
        if not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
            raise ValueError("similarity_score must be finite and in [-1, 1].")
        return min(1.0, max(-1.0, score))


class EvidenceProvenance(BaseModel):
    """Artifact and implementation provenance shared by one retrieval snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_artifact_version: NonEmptyText
    knowledge_chunks_sha256: Sha256Text
    retrieval_artifact_version: NonEmptyText
    embedding_model: NonEmptyText
    retrieval_backend: NonEmptyText


class ProductEvidence(BaseModel):
    """Retrieved, unverified snippets for exactly one candidate product."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    snippets: Annotated[tuple[EvidenceSnippet, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_identity_and_ranks(self) -> "ProductEvidence":
        if any(
            snippet.item_index != self.item_index
            or snippet.parent_asin != self.parent_asin
            for snippet in self.snippets
        ):
            raise ValueError("Every evidence snippet must match its product identity.")
        if tuple(value.rank for value in self.snippets) != tuple(
            range(1, len(self.snippets) + 1)
        ):
            raise ValueError("Product evidence ranks must be exactly 1..N.")
        if len({value.chunk_id for value in self.snippets}) != len(self.snippets):
            raise ValueError("Product evidence chunk_id values must be unique.")
        return self


class RequirementEvidence(BaseModel):
    """Pre-mutation evidence snapshot for all candidates of one requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    retrieved_at_plan_version: Annotated[int, Field(strict=True, ge=0)]
    requirement_id: NonEmptyText
    query: NonEmptyText
    candidates: Annotated[tuple[EvidenceCandidate, ...], Field(min_length=1)]
    products: Annotated[tuple[ProductEvidence, ...], Field(min_length=1)]
    status: EvidenceStatus = EvidenceStatus.RETRIEVED_UNVERIFIED
    provenance: EvidenceProvenance

    @model_validator(mode="after")
    def validate_candidate_coverage(self) -> "RequirementEvidence":
        candidate_pairs = tuple(
            (value.item_index, value.parent_asin) for value in self.candidates
        )
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise ValueError("Evidence candidates must have unique canonical identities.")
        product_pairs = tuple(
            (value.item_index, value.parent_asin) for value in self.products
        )
        if product_pairs != candidate_pairs:
            raise ValueError("Product evidence must exactly cover candidates in order.")
        return self


class SelectedProductEvidence(BaseModel):
    """Post-mutation evidence retained only for the product added to the plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    selected_at_plan_version: Annotated[int, Field(strict=True, ge=1)]
    requirement_id: NonEmptyText
    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    query: NonEmptyText
    snippets: Annotated[tuple[EvidenceSnippet, ...], Field(min_length=1)]
    status: EvidenceStatus = EvidenceStatus.RETRIEVED_UNVERIFIED
    provenance: EvidenceProvenance

    @model_validator(mode="after")
    def validate_selected_identity(self) -> "SelectedProductEvidence":
        if any(
            value.item_index != self.item_index
            or value.parent_asin != self.parent_asin
            for value in self.snippets
        ):
            raise ValueError("Selected evidence snippets must match the selected product.")
        return self
