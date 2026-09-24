"""Immutable integration contracts for multi-requirement Goal execution.

The preparation snapshot binds planning projection, deterministic budget
allocation, and the initial workflow state.  The result envelope additionally
binds terminal workflow and response output without duplicating their logic.
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
from ..response import FinalResponseResult, ResponseErrorCode, ResponseKind
from ..workflows import ShoppingWorkflowState, WorkflowRoute


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_CENT = Decimal("0.01")


def _money_key(value: float) -> Decimal:
    """Return the V2-10.4 P0 monetary representation used for comparisons."""

    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


class GoalExecutionStatus(str, Enum):
    """Preparation and terminal statuses for the Goal execution entry points."""

    CLARIFICATION_REQUIRED = "clarification_required"
    PREPARED = "prepared"
    READY = "ready"
    CONFLICT = "conflict"
    ERROR = "error"


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
    """Validated preparation or terminal result for one Goal execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GoalExecutionStatus
    goal_decision: ShoppingGoalExtractionDecision
    prepared_execution: PreparedGoalExecution | None = None
    clarification_question: NonEmptyText | None = None
    workflow_state: ShoppingWorkflowState | None = None
    final_response: FinalResponseResult | None = None
    response_error: ResponseErrorCode | None = None
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
            if any(
                value is not None
                for value in (
                    self.workflow_state,
                    self.final_response,
                    self.response_error,
                    self.memory_error,
                )
            ):
                raise ValueError(
                    "CLARIFICATION_REQUIRED cannot contain runtime output or memory_error."
                )
            if self.clarification_question != self.goal_decision.clarification_question:
                raise ValueError(
                    "clarification_question must match the validated goal decision."
                )
            return self

        if self.goal_decision.clarification_needed:
            raise ValueError(f"{self.status.value.upper()} requires a complete goal decision.")
        if self.prepared_execution is None:
            raise ValueError(f"{self.status.value.upper()} requires prepared_execution.")
        if self.clarification_question is not None:
            raise ValueError(
                f"{self.status.value.upper()} cannot contain clarification_question."
            )

        if self.status is GoalExecutionStatus.PREPARED:
            if any(
                value is not None
                for value in (
                    self.workflow_state,
                    self.final_response,
                    self.response_error,
                )
            ):
                raise ValueError("PREPARED cannot contain terminal runtime output.")
            return self

        if self.workflow_state is None:
            raise ValueError(f"{self.status.value.upper()} requires workflow_state.")
        expected_route = {
            GoalExecutionStatus.READY: WorkflowRoute.READY,
            GoalExecutionStatus.CONFLICT: WorkflowRoute.CONFLICT,
            GoalExecutionStatus.ERROR: WorkflowRoute.ERROR,
        }[self.status]
        if self.workflow_state.route is not expected_route:
            raise ValueError(
                "Goal execution status must match the terminal workflow route."
            )
        self._validate_terminal_binding()

        if self.status is GoalExecutionStatus.ERROR:
            if self.final_response is not None or self.response_error is not None:
                raise ValueError("ERROR cannot contain final response output.")
            return self

        expected_kind = (
            ResponseKind.READY
            if self.status is GoalExecutionStatus.READY
            else ResponseKind.CONFLICT
        )
        if self.final_response is not None:
            if self.response_error is not None:
                raise ValueError(
                    "A terminal response and response_error are mutually exclusive."
                )
            if self.final_response.kind is not expected_kind:
                raise ValueError(
                    "Final response kind must match the Goal execution status."
                )
        elif self.response_error not in {
            ResponseErrorCode.PROJECTION_FAILED,
            ResponseErrorCode.RENDER_FAILED,
        }:
            raise ValueError(
                "READY and CONFLICT require either a final response or a safe response_error."
            )
        return self

    def _validate_terminal_binding(self) -> None:
        """Bind terminal runtime state to the immutable preparation snapshot."""

        prepared = self.prepared_execution
        terminal = self.workflow_state
        if prepared is None or terminal is None:  # guarded by validate_status_payload
            raise ValueError("Terminal binding requires prepared and workflow state.")
        initial = prepared.initial_workflow_state
        initial_agent = initial.agent_state
        terminal_agent = terminal.agent_state
        initial_plan = initial_agent.shopping_plan
        terminal_plan = terminal_agent.shopping_plan

        if (
            terminal_agent.user_id != initial_agent.user_id
            or terminal_agent.session_id != initial_agent.session_id
            or terminal_plan.plan_id != initial_plan.plan_id
            or terminal_plan.user_id != initial_plan.user_id
            or terminal_plan.currency != initial_plan.currency
            or _money_key(terminal_plan.total_budget)
            != _money_key(initial_plan.total_budget)
        ):
            raise ValueError(
                "Terminal workflow identity must match the prepared Goal execution."
            )

        def immutable_requirement_fields(requirement):
            return (
                requirement.requirement_id,
                requirement.category,
                requirement.quantity,
                requirement.max_budget,
                requirement.required_features,
                requirement.soft_preferences,
                requirement.priority,
            )

        initial_requirements = tuple(
            immutable_requirement_fields(value) for value in initial_plan.requirements
        )
        terminal_requirements = tuple(
            immutable_requirement_fields(value) for value in terminal_plan.requirements
        )
        if terminal_requirements != initial_requirements:
            raise ValueError(
                "Terminal requirements must preserve prepared identity, order, and constraints."
            )
        if (
            initial.goal_budget_allocation != prepared.budget_allocation
            or terminal.goal_budget_allocation != prepared.budget_allocation
        ):
            raise ValueError(
                "Terminal workflow state must preserve the prepared Goal allocation."
            )
        if terminal_plan.version < initial_plan.version:
            raise ValueError("Terminal ShoppingPlan version cannot move backwards.")
