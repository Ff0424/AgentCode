"""Immutable integration contracts for prepared multi-requirement goals.

These contracts bind planning projection, deterministic budget allocation, and
the initial workflow runtime state.  They represent preparation only: no graph,
recommendation, evidence, verification, or response execution has occurred.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..planning import (
    GoalBudgetAllocation,
    GoalRequirementProjection,
    ShoppingGoalExtractionDecision,
)
from ..workflows import ShoppingWorkflowState


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_CENT = Decimal("0.01")


def _money_key(value: float) -> Decimal:
    """Return the V2-10.4 P0 monetary representation used for comparisons."""

    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


class GoalExecutionStatus(str, Enum):
    """Terminal statuses for the preparation-only goal entry point."""

    CLARIFICATION_REQUIRED = "clarification_required"
    PREPARED = "prepared"


class PreparedGoalExecution(BaseModel):
    """Validated runtime envelope ready for a later goal workflow stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    projection: GoalRequirementProjection
    budget_allocation: GoalBudgetAllocation
    initial_workflow_state: ShoppingWorkflowState
    recursion_limit: Annotated[int, Field(strict=True, ge=1)]

    @model_validator(mode="after")
    def validate_runtime_bindings(self) -> "PreparedGoalExecution":
        projection_ids = tuple(
            value.requirement_id for value in self.projection.requirements
        )
        allocation_ids = tuple(
            value.requirement_id for value in self.budget_allocation.allocations
        )
        if projection_ids != allocation_ids:
            raise ValueError(
                "Projection and budget allocation requirement IDs must match in order."
            )
        if _money_key(self.projection.total_budget) != _money_key(
            self.budget_allocation.total_budget
        ):
            raise ValueError(
                "Projection and budget allocation total budgets must match."
            )

        state = self.initial_workflow_state
        plan = state.agent_state.shopping_plan
        if plan.requirements != self.projection.requirements:
            raise ValueError(
                "Initial ShoppingPlan requirements must equal projected requirements."
            )
        if _money_key(plan.total_budget) != _money_key(self.projection.total_budget):
            raise ValueError(
                "Initial ShoppingPlan total budget must equal projection total budget."
            )
        if state.goal_budget_allocation != self.budget_allocation:
            raise ValueError(
                "Initial workflow state must preserve the goal budget allocation."
            )

        downstream_values = (
            state.recommendation_args,
            state.last_tool_result,
            state.selected_parent_asin,
            state.evaluation,
            state.planner_decision,
            state.current_evidence,
            state.current_verification,
            state.current_failure_diagnosis,
            state.current_replan_directive,
            state.route,
        )
        if any(value is not None for value in downstream_values) or any(
            (
                state.selected_evidence,
                state.selected_verifications,
                state.failure_history,
                state.replan_history,
            )
        ):
            raise ValueError(
                "Prepared workflow state must not contain downstream execution output."
            )
        if (
            state.current_evidence_attempt is not None
            or state.current_verification_attempt is not None
            or state.agent_state.current_requirement_id is not None
            or state.agent_state.last_tool_result is not None
            or state.agent_state.pending_action is not None
            or state.agent_state.error_state is not None
        ):
            raise ValueError(
                "Prepared workflow state must not contain active execution state."
            )
        return self


class GoalExecutionResult(BaseModel):
    """Preparation result or a safe clarification response boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GoalExecutionStatus
    goal_decision: ShoppingGoalExtractionDecision
    prepared_execution: PreparedGoalExecution | None = None
    clarification_question: NonEmptyText | None = None
    memory_error: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "GoalExecutionResult":
        if self.memory_error not in {None, "memory_store_failed", "memory_merge_failed"}:
            raise ValueError("memory_error must be a supported sanitized error code.")
        if self.status is GoalExecutionStatus.CLARIFICATION_REQUIRED:
            if not self.goal_decision.clarification_needed:
                raise ValueError(
                    "CLARIFICATION_REQUIRED requires a clarification goal decision."
                )
            if self.clarification_question is None:
                raise ValueError(
                    "CLARIFICATION_REQUIRED requires clarification_question."
                )
            if self.prepared_execution is not None:
                raise ValueError(
                    "CLARIFICATION_REQUIRED cannot contain prepared_execution."
                )
            if self.clarification_question != self.goal_decision.clarification_question:
                raise ValueError(
                    "clarification_question must match the validated goal decision."
                )
        elif self.status is GoalExecutionStatus.PREPARED:
            if self.goal_decision.clarification_needed:
                raise ValueError("PREPARED requires a complete goal decision.")
            if self.prepared_execution is None:
                raise ValueError("PREPARED requires prepared_execution.")
            if self.clarification_question is not None:
                raise ValueError("PREPARED cannot contain clarification_question.")
        return self
