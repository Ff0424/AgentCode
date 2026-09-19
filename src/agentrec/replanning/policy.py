"""Closed deterministic policy for one candidate-pool expansion."""

from __future__ import annotations

from ..domain import ShoppingRequirement
from .contracts import (
    FailureDiagnosis,
    FailureRecoverability,
    FailureType,
    ReplanAction,
    ReplanDirective,
    ReplanReason,
)


class BoundedReplanPolicy:
    """Map a validated diagnosis to expansion or a terminal conflict."""

    def decide(
        self,
        *,
        diagnosis: FailureDiagnosis,
        requirement: ShoppingRequirement,
        current_top_k: int,
        current_attempt: int,
        max_replan_attempts: int,
    ) -> ReplanDirective:
        if not isinstance(diagnosis, FailureDiagnosis):
            raise TypeError("diagnosis must be a FailureDiagnosis.")
        if not isinstance(requirement, ShoppingRequirement):
            raise TypeError("requirement must be a ShoppingRequirement.")
        if diagnosis.requirement_id != requirement.requirement_id:
            raise ValueError("Diagnosis and requirement identities differ.")
        if diagnosis.attempt != current_attempt:
            raise ValueError("Diagnosis attempt is stale.")

        expand = (
            diagnosis.failure_type is FailureType.NO_ELIGIBLE_CANDIDATES
            and diagnosis.recoverability is FailureRecoverability.RECOVERABLE
            and current_attempt == 0
            and max_replan_attempts == 1
            and current_top_k == 5
        )
        return ReplanDirective(
            plan_id=diagnosis.plan_id,
            requirement_id=requirement.requirement_id,
            source_plan_version=diagnosis.diagnosed_at_plan_version,
            attempt=1,
            action=(
                ReplanAction.EXPAND_CANDIDATE_POOL
                if expand else ReplanAction.STOP_CONFLICT
            ),
            previous_top_k=current_top_k,
            next_top_k=10 if expand else None,
            preserved_category=requirement.category,
            preserved_max_budget=requirement.max_budget,
            preserved_required_features=requirement.required_features,
            reason=(
                ReplanReason.EXPAND_AFTER_ZERO_ELIGIBLE
                if expand else ReplanReason.ATTEMPTS_EXHAUSTED
            ),
        )
