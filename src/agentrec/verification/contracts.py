"""Immutable contracts for deterministic evidence-based feature verification."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConstraintVerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class ConstraintVerificationReason(str, Enum):
    EXPLICIT_SUPPORT = "explicit_support"
    EXPLICIT_CONTRADICTION = "explicit_contradiction"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CandidateVerificationStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNVERIFIED = "unverified"


class ConstraintVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint: NonEmptyText
    canonical_constraint: NonEmptyText
    status: ConstraintVerificationStatus
    reason: ConstraintVerificationReason
    supporting_chunk_ids: tuple[NonEmptyText, ...] = ()
    contradicting_chunk_ids: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def validate_status_evidence(self) -> "ConstraintVerification":
        support = self.supporting_chunk_ids
        contradict = self.contradicting_chunk_ids
        if len(set(support)) != len(support) or len(set(contradict)) != len(contradict):
            raise ValueError("Verification chunk IDs must be unique.")
        expected = {
            (True, False): (
                ConstraintVerificationStatus.SUPPORTED,
                ConstraintVerificationReason.EXPLICIT_SUPPORT,
            ),
            (False, True): (
                ConstraintVerificationStatus.CONTRADICTED,
                ConstraintVerificationReason.EXPLICIT_CONTRADICTION,
            ),
            (True, True): (
                ConstraintVerificationStatus.UNKNOWN,
                ConstraintVerificationReason.CONFLICTING_EVIDENCE,
            ),
            (False, False): (
                ConstraintVerificationStatus.UNKNOWN,
                ConstraintVerificationReason.INSUFFICIENT_EVIDENCE,
            ),
        }[(bool(support), bool(contradict))]
        if (self.status, self.reason) != expected:
            raise ValueError("Constraint status/reason does not match its evidence IDs.")
        return self


class CandidateVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    status: CandidateVerificationStatus
    constraints: Annotated[tuple[ConstraintVerification, ...], Field(min_length=1)]
    verified_at_plan_version: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def validate_aggregate(self) -> "CandidateVerification":
        statuses = tuple(value.status for value in self.constraints)
        expected = (
            CandidateVerificationStatus.INELIGIBLE
            if ConstraintVerificationStatus.CONTRADICTED in statuses
            else CandidateVerificationStatus.UNVERIFIED
            if ConstraintVerificationStatus.UNKNOWN in statuses
            else CandidateVerificationStatus.ELIGIBLE
        )
        if self.status is not expected:
            raise ValueError("Candidate status does not match constraint aggregation.")
        return self


class RequirementVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    requirement_id: NonEmptyText
    verified_at_plan_version: Annotated[int, Field(strict=True, ge=0)]
    candidates: Annotated[tuple[CandidateVerification, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_candidates(self) -> "RequirementVerification":
        pairs = tuple((value.item_index, value.parent_asin) for value in self.candidates)
        if len(set(pairs)) != len(pairs):
            raise ValueError("Candidate verification identities must be unique.")
        if any(
            value.verified_at_plan_version != self.verified_at_plan_version
            for value in self.candidates
        ):
            raise ValueError("Candidate verification versions must match the requirement.")
        return self

    @property
    def eligible_parent_asins(self) -> tuple[str, ...]:
        return tuple(
            value.parent_asin
            for value in self.candidates
            if value.status is CandidateVerificationStatus.ELIGIBLE
        )


class SelectedCandidateVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    requirement_id: NonEmptyText
    selected_at_plan_version: Annotated[int, Field(strict=True, ge=1)]
    candidate: CandidateVerification

    @model_validator(mode="after")
    def validate_version_and_eligibility(self) -> "SelectedCandidateVerification":
        if self.candidate.status is not CandidateVerificationStatus.ELIGIBLE:
            raise ValueError("Only an eligible candidate can be selected.")
        if self.selected_at_plan_version != self.candidate.verified_at_plan_version + 1:
            raise ValueError("Selected verification must follow verification by one version.")
        return self
