"""Deterministic diagnosis of the only recoverable V2-09.5 failure."""

from __future__ import annotations

from ..verification import (
    CandidateVerificationStatus,
    ConstraintVerificationStatus,
    RequirementVerification,
)
from .contracts import (
    CandidateStatusSummary,
    FailureDiagnosis,
    FailureReason,
    FailureRecoverability,
    FailureType,
    VerificationFailurePattern,
)


class FailureDiagnosisService:
    """Classify a zero-eligible verification snapshot without an LLM."""

    def diagnose_zero_eligible(
        self,
        *,
        verification: RequirementVerification,
        plan_id: str,
        requirement_id: str,
        plan_version: int,
        attempt: int,
    ) -> FailureDiagnosis:
        if not isinstance(verification, RequirementVerification):
            raise TypeError("verification must be a RequirementVerification.")
        if (
            verification.plan_id != plan_id
            or verification.requirement_id != requirement_id
            or verification.verified_at_plan_version != plan_version
        ):
            raise ValueError("Verification identity or plan version is stale.")
        if verification.eligible_parent_asins:
            raise ValueError("Failure diagnosis requires zero eligible candidates.")
        if isinstance(attempt, bool) or attempt not in {0, 1}:
            raise ValueError("attempt must be 0 or 1.")

        statuses = tuple(value.status for value in verification.candidates)
        unverified = statuses.count(CandidateVerificationStatus.UNVERIFIED)
        ineligible = statuses.count(CandidateVerificationStatus.INELIGIBLE)
        if unverified == len(statuses):
            pattern = VerificationFailurePattern.ALL_CANDIDATES_UNVERIFIED
        elif ineligible == len(statuses):
            pattern = VerificationFailurePattern.ALL_CANDIDATES_INELIGIBLE
        else:
            pattern = VerificationFailurePattern.MIXED_VERIFICATION_FAILURE

        constraints = tuple(
            constraint
            for candidate in verification.candidates
            for constraint in candidate.constraints
        )
        failed = tuple(dict.fromkeys(
            value.constraint
            for value in constraints
            if value.status is not ConstraintVerificationStatus.SUPPORTED
        ))
        summary = CandidateStatusSummary(
            total_candidates=len(statuses),
            eligible_count=statuses.count(CandidateVerificationStatus.ELIGIBLE),
            unverified_count=unverified,
            ineligible_count=ineligible,
            supported_constraint_count=sum(
                value.status is ConstraintVerificationStatus.SUPPORTED
                for value in constraints
            ),
            contradicted_constraint_count=sum(
                value.status is ConstraintVerificationStatus.CONTRADICTED
                for value in constraints
            ),
            unknown_constraint_count=sum(
                value.status is ConstraintVerificationStatus.UNKNOWN
                for value in constraints
            ),
        )
        return FailureDiagnosis(
            plan_id=plan_id,
            requirement_id=requirement_id,
            diagnosed_at_plan_version=plan_version,
            attempt=attempt,
            failure_type=(
                FailureType.NO_ELIGIBLE_CANDIDATES
                if attempt == 0
                else FailureType.REPLAN_ATTEMPTS_EXHAUSTED
            ),
            recoverability=(
                FailureRecoverability.RECOVERABLE
                if attempt == 0
                else FailureRecoverability.NON_RECOVERABLE
            ),
            verification_pattern=pattern,
            failed_constraints=failed,
            candidate_status_summary=summary,
            source_error_code="constraint_verification:no_eligible_candidates",
            reason=(
                FailureReason.VERIFICATION_RETURNED_ZERO_ELIGIBLE
                if attempt == 0
                else FailureReason.BOUNDED_REPLAN_ATTEMPT_EXHAUSTED
            ),
        )
