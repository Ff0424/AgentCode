"""End-to-end AgentRec task integration without provider or runtime ownership.

This layer sequences validated requirement extraction into a new ShoppingPlan
and invokes the existing LangGraph workflow. All dependencies are injected;
product facts still originate from RecommendationToolResult and plan facts are
still owned by ShoppingPlanService and the immutable domain model.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..decision import (
    AgentIntent,
    DecisionContext,
    DecisionExecutionPlan,
)
from ..domain import AgentState
from ..memory import MemoryStore, RequirementMemoryMerger
from ..planning import (
    DeterministicBudgetAllocator,
    GoalRequirementProjection,
    GoalToRequirementProjector,
    RequirementExtractionDecision,
    ShoppingGoalExtractionDecision,
)
from ..response import (
    DeterministicFinalResponseRenderer,
    FinalResponseResult,
    GroundedResponseProjector,
    ResponseErrorCode,
)
from ..services import ShoppingPlanService
from ..workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from .goal import (
    GoalExecutionResult,
    GoalExecutionStatus,
    PreparedGoalExecution,
)


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
    final_response: FinalResponseResult | None = None
    response_error: ResponseErrorCode | None = None
    memory_error: str | None = None
    decision_plan: DecisionExecutionPlan | None = None


class AgentTaskRunner:
    """Compose existing boundaries into one deterministic user-task execution."""

    def __init__(
        self,
        *,
        requirement_extractor: Any,
        planner: Any,
        recommendation_tool: Any,
        evidence_service: Any,
        verification_service: Any,
        shopping_plan_service: ShoppingPlanService,
        response_projector: Any | None = None,
        response_renderer: Any | None = None,
        memory_store: MemoryStore | None = None,
        memory_merger: RequirementMemoryMerger | None = None,
        decision_policy: Any | None = None,
        decision_executor: Any | None = None,
        goal_extractor: Any | None = None,
        goal_projector: Any | None = None,
        budget_allocator: Any | None = None,
    ) -> None:
        dependencies = (
            (requirement_extractor, "extract", "requirement_extractor"),
            (planner, "decide", "planner"),
            (recommendation_tool, "recommend", "recommendation_tool"),
            (evidence_service, "retrieve", "evidence_service"),
            (verification_service, "verify", "verification_service"),
            (shopping_plan_service, "create_plan", "shopping_plan_service"),
        )
        for dependency, method, name in dependencies:
            if dependency is None or not callable(getattr(dependency, method, None)):
                raise TypeError(f"{name} must provide {method}().")
        self._extractor = requirement_extractor
        self._planner = planner
        self._recommendation_tool = recommendation_tool
        self._evidence_service = evidence_service
        self._verification_service = verification_service
        self._plan_service = shopping_plan_service
        self._response_projector = (
            GroundedResponseProjector()
            if response_projector is None
            else response_projector
        )
        self._response_renderer = (
            DeterministicFinalResponseRenderer()
            if response_renderer is None
            else response_renderer
        )
        if not callable(getattr(self._response_projector, "project", None)):
            raise TypeError("response_projector must provide project().")
        if not callable(getattr(self._response_renderer, "render", None)):
            raise TypeError("response_renderer must provide render().")
        if (memory_store is None) != (memory_merger is None):
            raise TypeError(
                "memory_store and memory_merger must be provided together."
            )
        if memory_store is not None and not callable(
            getattr(memory_store, "get_confirmed_preferences", None)
        ):
            raise TypeError(
                "memory_store must provide get_confirmed_preferences()."
            )
        if memory_merger is not None and not callable(
            getattr(memory_merger, "merge", None)
        ):
            raise TypeError("memory_merger must provide merge().")
        self._memory_store = memory_store
        self._memory_merger = memory_merger
        if (decision_policy is None) != (decision_executor is None):
            raise ValueError(
                "decision_policy and decision_executor must be provided together."
            )
        if decision_policy is not None and not callable(
            getattr(decision_policy, "decide", None)
        ):
            raise TypeError("decision_policy must provide decide().")
        if decision_executor is not None and not callable(
            getattr(decision_executor, "execute", None)
        ):
            raise TypeError("decision_executor must provide execute().")
        self._decision_policy = decision_policy
        self._decision_executor = decision_executor
        if goal_extractor is not None and not callable(
            getattr(goal_extractor, "extract", None)
        ):
            raise TypeError("goal_extractor must provide extract().")
        self._goal_extractor = goal_extractor
        self._goal_projector = (
            GoalToRequirementProjector()
            if goal_projector is None
            else goal_projector
        )
        self._budget_allocator = (
            DeterministicBudgetAllocator()
            if budget_allocator is None
            else budget_allocator
        )
        if not callable(getattr(self._goal_projector, "project", None)):
            raise TypeError("goal_projector must provide project().")
        if not callable(getattr(self._budget_allocator, "allocate", None)):
            raise TypeError("budget_allocator must provide allocate().")

    def run_goal(
        self,
        *,
        user_id: str,
        session_id: str,
        plan_id: str,
        user_request: str,
        currency: str = "USD",
        recursion_limit: int = 50,
    ) -> GoalExecutionResult:
        """Prepare one extracted shopping goal without invoking the workflow."""

        if self._goal_extractor is None:
            raise RuntimeError("run_goal requires a configured goal_extractor.")
        if (
            isinstance(recursion_limit, bool)
            or not isinstance(recursion_limit, int)
            or recursion_limit < 1
        ):
            raise ValueError("recursion_limit must be a positive integer.")

        decision = ShoppingGoalExtractionDecision.model_validate(
            self._goal_extractor.extract(user_request=user_request)
        )
        if decision.clarification_needed:
            return GoalExecutionResult(
                status=GoalExecutionStatus.CLARIFICATION_REQUIRED,
                goal_decision=decision,
                clarification_question=decision.clarification_question,
            )

        original_projection = self._goal_projector.project(decision)
        final_projection = original_projection
        memory_error: str | None = None

        # Current memory has no requirement/category scope. Applying it to a
        # multi-goal bundle could leak one preference across unrelated products.
        if (
            len(original_projection.requirements) == 1
            and self._memory_store is not None
            and self._memory_merger is not None
        ):
            requirement = original_projection.requirements[0]
            try:
                preferences = self._memory_store.get_confirmed_preferences(user_id)
            except Exception:
                memory_error = "memory_store_failed"
            else:
                try:
                    merged = self._memory_merger.merge(requirement, preferences)
                except Exception:
                    memory_error = "memory_merge_failed"
                else:
                    final_projection = GoalRequirementProjection(
                        total_budget=original_projection.total_budget,
                        requirements=(merged,),
                        allocation_preferences=(
                            original_projection.allocation_preferences
                        ),
                    )

        # Allocation must succeed before any business plan is created.
        allocation = self._budget_allocator.allocate(final_projection)
        projection_ids = tuple(
            value.requirement_id for value in final_projection.requirements
        )
        allocation_ids = tuple(
            value.requirement_id for value in allocation.allocations
        )
        if projection_ids != allocation_ids:
            raise ValueError(
                "Projection and allocation requirement IDs must match in order."
            )
        cent = Decimal("0.01")
        if Decimal(str(final_projection.total_budget)).quantize(
            cent, rounding=ROUND_HALF_UP
        ) != Decimal(str(allocation.total_budget)).quantize(
            cent, rounding=ROUND_HALF_UP
        ):
            raise ValueError("Projection and allocation total budgets must match.")

        plan = self._plan_service.create_plan(
            plan_id=plan_id,
            user_id=user_id,
            currency=currency,
            total_budget=final_projection.total_budget,
            requirements=final_projection.requirements,
        )
        initial = ShoppingWorkflowState(
            agent_state=AgentState(
                user_id=user_id,
                session_id=session_id,
                shopping_plan=plan,
            ),
            goal_budget_allocation=allocation,
        )
        prepared = PreparedGoalExecution(
            projection=final_projection,
            budget_allocation=allocation,
            initial_workflow_state=initial,
            recursion_limit=recursion_limit,
        )
        return GoalExecutionResult(
            status=GoalExecutionStatus.PREPARED,
            goal_decision=decision,
            prepared_execution=prepared,
            memory_error=memory_error,
        )

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
        memory_error: str | None = None
        confirmed_preferences = ()
        if self._memory_store is not None and self._memory_merger is not None:
            try:
                confirmed_preferences = (
                    self._memory_store.get_confirmed_preferences(user_id)
                )
            except Exception:
                # Memory is optional augmentation; never expose internal failures.
                memory_error = "memory_store_failed"
            else:
                augmented_requirements = []
                for requirement in domain_requirements:
                    try:
                        augmented = self._memory_merger.merge(
                            requirement,
                            confirmed_preferences,
                        )
                    except Exception:
                        # Fail open per requirement and retain the extracted facts.
                        memory_error = "memory_merge_failed"
                        augmented = requirement
                    augmented_requirements.append(augmented)
                domain_requirements = tuple(augmented_requirements)

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
        decision_plan: DecisionExecutionPlan | None = None
        if self._decision_policy is not None and self._decision_executor is not None:
            decision_context = DecisionContext(
                intent=AgentIntent.RECOMMENDATION,
                memory_available=bool(confirmed_preferences),
                requirement_complete=True,
                candidate_available=False,
                verification_required=True,
                response_required=True,
            )
            try:
                directive = self._decision_policy.decide(decision_context)
                decision_plan = self._decision_executor.execute(directive)
            except Exception:
                # Decision metadata is optional and cannot alter workflow execution.
                decision_plan = None

        graph = build_shopping_workflow(
            self._recommendation_tool,
            self._plan_service,
            planner=self._planner,
            evidence_service=self._evidence_service,
            verification_service=self._verification_service,
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

        final_response: FinalResponseResult | None = None
        response_error: ResponseErrorCode | None = None
        if status in (AgentExecutionStatus.READY, AgentExecutionStatus.CONFLICT):
            try:
                response_context = self._response_projector.project(output)
            except Exception:
                # Internal projection details must not cross the integration boundary.
                response_error = ResponseErrorCode.PROJECTION_FAILED
            else:
                try:
                    final_response = self._response_renderer.render(response_context)
                except Exception:
                    # Preserve the successful workflow result while reporting a safe code.
                    response_error = ResponseErrorCode.RENDER_FAILED

        return AgentExecutionResult(
            status=status,
            extraction_decisions=decisions,
            workflow_state=output,
            final_response=final_response,
            response_error=response_error,
            memory_error=memory_error,
            decision_plan=decision_plan,
        )
