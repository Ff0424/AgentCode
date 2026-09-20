"""Immutable contracts for grounded deterministic final responses.

These types contain only response-safe product facts, verified constraint
claims, conflict summaries, and their rendered result envelope. They perform
no projection, rendering, service calls, or workflow mutation.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from ..domain import ItemSource
from ..knowledge import KnowledgeChunkType
from ..replanning import CandidateStatusSummary
from ..verification import ConstraintVerificationStatus


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CurrencyCode = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=3,
        to_upper=True,
    ),
]
FiniteMoney = Annotated[float, Field(allow_inf_nan=False)]


class ResponseKind(str, Enum):
    READY = "ready"
    CONFLICT = "conflict"


class ConflictReason(str, Enum):
    NO_RECOMMENDATION_CANDIDATES = "no_recommendation_candidates"
    REPLAN_ATTEMPTS_EXHAUSTED = "replan_attempts_exhausted"


class ResponseErrorCode(str, Enum):
    PROJECTION_FAILED = "projection_failed"
    RENDER_FAILED = "render_failed"


class EvidenceReference(BaseModel):
    """Verbatim evidence retained for internal claim provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyText
    chunk_type: KnowledgeChunkType
    excerpt: NonEmptyText


class VerifiedConstraintClaim(BaseModel):
    """One hard constraint with deterministic supporting evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    parent_asin: NonEmptyText
    original_constraint: NonEmptyText
    canonical_constraint: NonEmptyText
    status: Literal[ConstraintVerificationStatus.SUPPORTED]
    supporting_evidence: Annotated[
        tuple[EvidenceReference, ...], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> "VerifiedConstraintClaim":
        chunk_ids = tuple(value.chunk_id for value in self.supporting_evidence)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("supporting_evidence chunk_id values must be unique.")
        return self


class ProductDecisionSummary(BaseModel):
    """Response-safe projection of one product already selected in the plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    category: NonEmptyText
    parent_asin: NonEmptyText
    title: NonEmptyText
    price: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    quantity: Annotated[int, Field(strict=True, ge=1, le=100)]
    source: ItemSource
    selection_rationale: NonEmptyText
    verified_claims: tuple[VerifiedConstraintClaim, ...] = ()


class ConflictDecisionSummary(BaseModel):
    """Sanitized description of one supported terminal conflict shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    category: NonEmptyText
    required_features: tuple[NonEmptyText, ...] = ()
    reason: ConflictReason
    replan_attempts_performed: Annotated[int, Field(strict=True, ge=0, le=1)]
    candidate_pool_sizes: Annotated[
        tuple[Annotated[int, Field(strict=True, ge=1, le=50)], ...],
        Field(min_length=1, max_length=2),
    ]
    verification_status_counts: CandidateStatusSummary | None = None
    failed_constraints: tuple[NonEmptyText, ...] = ()
    unknown_constraints: tuple[NonEmptyText, ...] = ()
    contradicted_constraints: tuple[NonEmptyText, ...] = ()
    user_action_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_conflict_shape(self) -> "ConflictDecisionSummary":
        if self.reason is ConflictReason.NO_RECOMMENDATION_CANDIDATES:
            if self.verification_status_counts is not None:
                raise ValueError(
                    "No-candidate conflict cannot contain verification counts."
                )
            if (
                self.failed_constraints
                or self.unknown_constraints
                or self.contradicted_constraints
            ):
                raise ValueError(
                    "No-candidate conflict cannot claim verification failures."
                )
            expected_attempts = 1 if self.candidate_pool_sizes == (5, 10) else 0
            if self.candidate_pool_sizes not in {(5,), (5, 10)}:
                raise ValueError(
                    "No-candidate pool sizes must be (5,) or (5, 10)."
                )
            if self.replan_attempts_performed != expected_attempts:
                raise ValueError(
                    "No-candidate replan count must match candidate_pool_sizes."
                )
            return self

        counts = self.verification_status_counts
        if self.replan_attempts_performed != 1:
            raise ValueError("Exhaustion requires exactly one replan attempt.")
        if self.candidate_pool_sizes != (5, 10):
            raise ValueError("Exhaustion requires candidate_pool_sizes=(5, 10).")
        if counts is None:
            raise ValueError("Exhaustion requires verification status counts.")
        if counts.eligible_count != 0:
            raise ValueError("Exhaustion requires zero eligible candidates.")
        if not self.failed_constraints:
            raise ValueError("Exhaustion requires at least one failed constraint.")
        return self


class ReadyResponseContext(BaseModel):
    """Fully projected facts permitted in a READY final response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[ResponseKind.READY] = ResponseKind.READY
    plan_id: NonEmptyText
    plan_version: Annotated[int, Field(strict=True, ge=0)]
    currency: CurrencyCode
    products: Annotated[tuple[ProductDecisionSummary, ...], Field(min_length=1)]
    total_spent: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    remaining_budget: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_unique_products(self) -> "ReadyResponseContext":
        requirement_ids = tuple(value.requirement_id for value in self.products)
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("Response products must have unique requirement_id values.")
        parent_asins = tuple(value.parent_asin for value in self.products)
        if len(set(parent_asins)) != len(parent_asins):
            raise ValueError("Response products must have unique parent_asin values.")
        return self


class ConflictResponseContext(BaseModel):
    """Fully projected facts permitted in a CONFLICT final response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[ResponseKind.CONFLICT] = ResponseKind.CONFLICT
    plan_id: NonEmptyText
    plan_version: Annotated[int, Field(strict=True, ge=0)]
    currency: CurrencyCode
    decision: ConflictDecisionSummary
    total_spent: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    remaining_budget: FiniteMoney


GroundedResponseContext: TypeAlias = Annotated[
    ReadyResponseContext | ConflictResponseContext,
    Field(discriminator="kind"),
]


class FinalResponseResult(BaseModel):
    """Rendered public text plus its internal structured decision summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ResponseKind
    text: NonEmptyText
    decision_summary: tuple[ProductDecisionSummary, ...] | ConflictDecisionSummary

    @model_validator(mode="after")
    def validate_kind_and_summary(self) -> "FinalResponseResult":
        if self.kind is ResponseKind.READY:
            if not isinstance(self.decision_summary, tuple) or not self.decision_summary:
                raise ValueError(
                    "READY response requires a non-empty product decision tuple."
                )
        elif not isinstance(self.decision_summary, ConflictDecisionSummary):
            raise ValueError(
                "CONFLICT response requires a ConflictDecisionSummary."
            )
        return self
