"""Immutable runtime contracts for bounded candidate-pool re-planning."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FailureType(str, Enum):
    NO_RECOMMENDATION_CANDIDATES = "no_recommendation_candidates"
    NO_ELIGIBLE_CANDIDATES = "no_eligible_candidates"
    BUDGET_CONFLICT = "budget_conflict"
    PLANNER_SELECTION_INVALID = "planner_selection_invalid"
    STALE_PLAN_VERSION = "stale_plan_version"
    IDENTITY_INTEGRITY_FAILURE = "identity_integrity_failure"
    PROVENANCE_INTEGRITY_FAILURE = "provenance_integrity_failure"
    EXECUTION_FAILURE = "execution_failure"
    REPLAN_ATTEMPTS_EXHAUSTED = "replan_attempts_exhausted"


class VerificationFailurePattern(str, Enum):
    ALL_CANDIDATES_UNVERIFIED = "all_candidates_unverified"
    ALL_CANDIDATES_INELIGIBLE = "all_candidates_ineligible"
    MIXED_VERIFICATION_FAILURE = "mixed_verification_failure"


class FailureRecoverability(str, Enum):
    RECOVERABLE = "recoverable"
    REQUIRES_USER = "requires_user"
    NON_RECOVERABLE = "non_recoverable"
    SYSTEM_ERROR = "system_error"


class FailureReason(str, Enum):
    VERIFICATION_RETURNED_ZERO_ELIGIBLE = "verification_returned_zero_eligible"
    BOUNDED_REPLAN_ATTEMPT_EXHAUSTED = "bounded_replan_attempt_exhausted"


class ReplanAction(str, Enum):
    EXPAND_CANDIDATE_POOL = "expand_candidate_pool"
    STOP_CONFLICT = "stop_conflict"


class ReplanReason(str, Enum):
    EXPAND_AFTER_ZERO_ELIGIBLE = "expand_after_zero_eligible"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"


class CandidateStatusSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_candidates: Annotated[int, Field(strict=True, ge=1)]
    eligible_count: Annotated[int, Field(strict=True, ge=0)]
    unverified_count: Annotated[int, Field(strict=True, ge=0)]
    ineligible_count: Annotated[int, Field(strict=True, ge=0)]
    supported_constraint_count: Annotated[int, Field(strict=True, ge=0)]
    contradicted_constraint_count: Annotated[int, Field(strict=True, ge=0)]
    unknown_constraint_count: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def validate_candidate_counts(self) -> "CandidateStatusSummary":
        if self.total_candidates != (
            self.eligible_count + self.unverified_count + self.ineligible_count
        ):
            raise ValueError("Candidate status counts must sum to total_candidates.")
        return self


class FailureDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    requirement_id: NonEmptyText
    diagnosed_at_plan_version: Annotated[int, Field(strict=True, ge=0)]
    attempt: Annotated[int, Field(strict=True, ge=0, le=1)]
    failure_type: FailureType
    recoverability: FailureRecoverability
    verification_pattern: VerificationFailurePattern
    failed_constraints: tuple[NonEmptyText, ...]
    candidate_status_summary: CandidateStatusSummary
    source_error_code: NonEmptyText
    reason: FailureReason

    @model_validator(mode="after")
    def validate_bounded_failure(self) -> "FailureDiagnosis":
        if self.candidate_status_summary.eligible_count != 0:
            raise ValueError("Bounded re-plan diagnosis requires zero eligible candidates.")
        if self.attempt == 0:
            expected = (
                FailureType.NO_ELIGIBLE_CANDIDATES,
                FailureRecoverability.RECOVERABLE,
                FailureReason.VERIFICATION_RETURNED_ZERO_ELIGIBLE,
            )
        else:
            expected = (
                FailureType.REPLAN_ATTEMPTS_EXHAUSTED,
                FailureRecoverability.NON_RECOVERABLE,
                FailureReason.BOUNDED_REPLAN_ATTEMPT_EXHAUSTED,
            )
        if (self.failure_type, self.recoverability, self.reason) != expected:
            raise ValueError("Failure type, recoverability, and reason disagree with attempt.")
        return self


class ReplanDirective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    requirement_id: NonEmptyText
    source_plan_version: Annotated[int, Field(strict=True, ge=0)]
    attempt: Annotated[int, Field(strict=True, ge=1, le=1)]
    action: ReplanAction
    previous_top_k: Annotated[int, Field(strict=True, ge=1, le=50)]
    next_top_k: Annotated[int, Field(strict=True, ge=1, le=50)] | None
    preserved_category: NonEmptyText
    preserved_max_budget: float | None
    preserved_required_features: tuple[NonEmptyText, ...]
    reason: ReplanReason

    @model_validator(mode="after")
    def validate_action_contract(self) -> "ReplanDirective":
        if self.action is ReplanAction.EXPAND_CANDIDATE_POOL:
            if (
                self.attempt != 1
                or self.previous_top_k != 5
                or self.next_top_k != 10
                or self.reason is not ReplanReason.EXPAND_AFTER_ZERO_ELIGIBLE
            ):
                raise ValueError("Candidate-pool expansion must be the frozen 5-to-10 attempt.")
        elif (
            self.next_top_k is not None
            or self.reason is not ReplanReason.ATTEMPTS_EXHAUSTED
        ):
            raise ValueError("STOP_CONFLICT cannot define another candidate-pool size.")
        return self
