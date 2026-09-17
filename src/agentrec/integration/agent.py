"""End-to-end AgentRec task integration without provider or runtime ownership.

This layer sequences validated requirement extraction into a new ShoppingPlan
and invokes the existing LangGraph workflow. All dependencies are injected;
product facts still originate from RecommendationToolResult and plan facts are
still owned by ShoppingPlanService and the immutable domain model.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..domain import AgentState
from ..planning import RequirementExtractionDecision
from ..services import ShoppingPlanService
from ..workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RequirementInput(BaseModel):
    """One ordered natural-language requirement with a system-owned identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    user_request: NonEmptyText


class AgentExecutionStatus(str, Enum):
    READY = "ready"
    CONFLICT = "conflict"
    CLARIFICATION_REQUIRED = "clarification_required"
    ERROR = "error"


class AgentExecutionResult(BaseModel):
    """Serializable result of extraction plus optional workflow execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentExecutionStatus
    extraction_decisions: tuple[RequirementExtractionDecision, ...]
    clarification_requirement_ids: tuple[str, ...] = ()
    workflow_state: ShoppingWorkflowState | None = None


class AgentTaskRunner:
    """Compose existing boundaries into one deterministic user-task execution."""

    def __init__(
        self,
        *,
        requirement_extractor: Any,
        planner: Any,
        recommendation_tool: Any,
        shopping_plan_service: ShoppingPlanService,
    ) -> None:
        dependencies = (
            (requirement_extractor, "extract", "requirement_extractor"),
            (planner, "decide", "planner"),
            (recommendation_tool, "recommend", "recommendation_tool"),
            (shopping_plan_service, "create_plan", "shopping_plan_service"),
        )
        for dependency, method, name in dependencies:
            if dependency is None or not callable(getattr(dependency, method, None)):
                raise TypeError(f"{name} must provide {method}().")
        self._extractor = requirement_extractor
        self._planner = planner
        self._recommendation_tool = recommendation_tool
        self._plan_service = shopping_plan_service

    def run(
        self,
        *,
        user_id: str,
        session_id: str,
        plan_id: str,
        total_budget: float,
        requirements: tuple[RequirementInput, ...],
        currency: str = "USD",
        recursion_limit: int = 50,
    ) -> AgentExecutionResult:
        if not isinstance(requirements, tuple) or not requirements:
            raise ValueError("requirements must be a non-empty tuple.")
        if any(not isinstance(value, RequirementInput) for value in requirements):
            raise TypeError("Every requirements entry must be RequirementInput.")
        if isinstance(recursion_limit, bool) or not isinstance(recursion_limit, int) or recursion_limit < 1:
            raise ValueError("recursion_limit must be a positive integer.")

        decisions = tuple(
            self._extractor.extract(user_request=value.user_request)
            for value in requirements
        )
        clarification_ids = tuple(
            request.requirement_id
            for request, decision in zip(requirements, decisions, strict=True)
            if decision.clarification_needed
        )
        if clarification_ids:
            return AgentExecutionResult(
                status=AgentExecutionStatus.CLARIFICATION_REQUIRED,
                extraction_decisions=decisions,
                clarification_requirement_ids=clarification_ids,
            )

        domain_requirements = tuple(
            decision.to_shopping_requirement(requirement_id=request.requirement_id)
            for request, decision in zip(requirements, decisions, strict=True)
        )
        plan = self._plan_service.create_plan(
            plan_id=plan_id,
            user_id=user_id,
            currency=currency,
            total_budget=total_budget,
            requirements=domain_requirements,
        )
        initial = ShoppingWorkflowState(
            agent_state=AgentState(
                user_id=user_id,
                session_id=session_id,
                shopping_plan=plan,
            )
        )
        graph = build_shopping_workflow(
            self._recommendation_tool,
            self._plan_service,
            planner=self._planner,
        )
        output = ShoppingWorkflowState.model_validate(
            graph.invoke(initial, config={"recursion_limit": recursion_limit})
        )
        status = {
            WorkflowRoute.READY: AgentExecutionStatus.READY,
            WorkflowRoute.CONFLICT: AgentExecutionStatus.CONFLICT,
            WorkflowRoute.ERROR: AgentExecutionStatus.ERROR,
        }.get(output.route)
        if status is None:
            raise RuntimeError(f"Workflow ended with invalid route={output.route!r}.")
        return AgentExecutionResult(
            status=status,
            extraction_decisions=decisions,
            workflow_state=output,
        )
