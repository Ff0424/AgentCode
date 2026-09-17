"""Minimal deterministic LangGraph workflow for AgentRec shopping plans.

The graph sequences trusted recommendation-tool output into immutable Shopping
Plan mutations. Runtime dependencies are captured by node closures and never
stored in serializable workflow state. P0 intentionally contains no LLM,
natural-language parsing, automatic replanning, persistence, or API behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ..domain import AgentState, PlanStatus, RequirementStatus, ShoppingRequirement
from ..planning import (
    SelectCandidateDecision,
    SelectRequirementDecision,
    validate_planner_decision,
)
from ..services import ShoppingPlanServiceError
from ..tools import RecommendationToolArgs
from .routes import WorkflowAction, WorkflowRoute
from .state import ShoppingWorkflowState


TOP_K = 5
SELECTED_REASON = "selected_by_rank_policy"


def _state(value: ShoppingWorkflowState | dict[str, Any]) -> ShoppingWorkflowState:
    """Normalize LangGraph input so direct node tests and graph runs agree."""

    return ShoppingWorkflowState.model_validate(value)


def _agent_update(state: AgentState, **changes: Any) -> AgentState:
    return AgentState.model_validate(state.model_copy(update=changes).model_dump())


def _current_requirement(state: ShoppingWorkflowState) -> ShoppingRequirement:
    requirement_id = state.agent_state.current_requirement_id
    matches = tuple(
        requirement
        for requirement in state.agent_state.shopping_plan.requirements
        if requirement.requirement_id == requirement_id
    )
    if len(matches) != 1:
        raise ValueError("Workflow state has no valid current requirement.")
    return matches[0]


def select_requirement_node(
    value: ShoppingWorkflowState | dict[str, Any],
) -> dict[str, Any]:
    """Select the first unfinished requirement in stable domain order."""

    state = _state(value)
    plan = state.agent_state.shopping_plan
    requirement = next(
        (
            item
            for item in plan.requirements
            if item.status
            in {RequirementStatus.PENDING, RequirementStatus.CANDIDATE_SELECTED}
        ),
        None,
    )
    if requirement is not None:
        agent_state = _agent_update(
            state.agent_state,
            current_requirement_id=requirement.requirement_id,
            pending_action=WorkflowAction.RECOMMEND.value,
            error_state=None,
        )
        route = WorkflowRoute.CONTINUE
    elif plan.status in {PlanStatus.READY, PlanStatus.COMPLETED}:
        agent_state = _agent_update(
            state.agent_state,
            current_requirement_id=None,
            pending_action=WorkflowAction.READY.value,
            error_state=None,
        )
        route = WorkflowRoute.READY
    elif plan.status == PlanStatus.CONFLICT:
        agent_state = _agent_update(
            state.agent_state,
            current_requirement_id=None,
            pending_action=WorkflowAction.CONFLICT.value,
        )
        route = WorkflowRoute.CONFLICT
    else:
        agent_state = _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.CONFLICT.value,
            error_state="workflow_invariant:no_selectable_requirement",
        )
        route = WorkflowRoute.ERROR
    return {
        "agent_state": agent_state,
        "recommendation_args": None,
        "last_tool_result": None,
        "selected_parent_asin": None,
        "evaluation": None,
        "planner_decision": None,
        "route": route,
    }


def _planner_error(state: ShoppingWorkflowState, code: str) -> dict[str, Any]:
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.CONFLICT.value,
            error_state=f"planner_validation:{code}",
        ),
        "planner_decision": None,
        "route": WorkflowRoute.ERROR,
    }


def _validate_plan_binding(state: ShoppingWorkflowState, decision: Any) -> bool:
    plan = state.agent_state.shopping_plan
    return decision.plan_id == plan.plan_id and decision.plan_version == plan.version


def requirement_planner_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    planner: Any,
) -> dict[str, Any]:
    """Ask the injected planner which unfinished requirement to process."""

    state = _state(value)
    plan = state.agent_state.shopping_plan
    context = {
        "decision_key": f"select_requirement:{plan.version}",
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "requirements": tuple(
            {
                "requirement_id": item.requirement_id,
                "category": item.category,
                "status": item.status.value,
                "priority": item.priority,
            }
            for item in plan.requirements
        ),
    }
    raw_decision = planner.decide(context=context)
    try:
        decision = validate_planner_decision(raw_decision)
    except ValidationError:
        return _planner_error(state, "schema_invalid")
    if not isinstance(decision, SelectRequirementDecision):
        return _planner_error(state, "action_not_permitted")
    if not _validate_plan_binding(state, decision):
        return _planner_error(state, "stale_plan_version")
    matches = tuple(
        requirement
        for requirement in plan.requirements
        if requirement.requirement_id == decision.requirement_id
    )
    if len(matches) != 1 or matches[0].status not in {
        RequirementStatus.PENDING,
        RequirementStatus.CANDIDATE_SELECTED,
    }:
        return _planner_error(state, "invalid_requirement_id")
    return {
        "agent_state": _agent_update(
            state.agent_state,
            current_requirement_id=decision.requirement_id,
            pending_action=WorkflowAction.RECOMMEND.value,
            error_state=None,
        ),
        "recommendation_args": None,
        "last_tool_result": None,
        "selected_parent_asin": None,
        "evaluation": None,
        "planner_decision": decision,
        "route": WorkflowRoute.CONTINUE,
    }


def recommend_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    recommendation_tool: Any,
) -> dict[str, Any]:
    """Execute the injected recommendation adapter with the fixed P0 policy."""

    state = _state(value)
    requirement = _current_requirement(state)
    args = RecommendationToolArgs(
        top_k=TOP_K,
        category=requirement.category,
        max_price=requirement.max_budget,
        required_features=requirement.required_features,
    )
    result = recommendation_tool.recommend(
        user_id=state.agent_state.user_id,
        args=args,
        excluded_parent_asins=tuple(
            item.parent_asin for item in state.agent_state.shopping_plan.selected_items
        ),
    )
    route = (
        WorkflowRoute.CONFLICT
        if result.returned_count == 0
        else WorkflowRoute.CONTINUE
    )
    agent_state = _agent_update(
        state.agent_state,
        last_tool_result=result.model_dump(mode="json"),
        pending_action=(
            WorkflowAction.CONFLICT.value
            if route == WorkflowRoute.CONFLICT
            else WorkflowAction.SELECT_CANDIDATE.value
        ),
        error_state=(
            "no_recommendation_candidates"
            if route == WorkflowRoute.CONFLICT
            else None
        ),
    )
    return {
        "agent_state": agent_state,
        "recommendation_args": args,
        "last_tool_result": result,
        "selected_parent_asin": None,
        "evaluation": None,
        "route": route,
    }


def select_candidate_node(
    value: ShoppingWorkflowState | dict[str, Any],
) -> dict[str, Any]:
    """Select rank one without an LLM or a second ranking policy."""

    state = _state(value)
    result = state.last_tool_result
    candidates = () if result is None else tuple(
        item for item in result.items if item.rank == 1
    )
    if len(candidates) != 1:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="candidate_selection:rank_one_missing_or_ambiguous",
            ),
            "selected_parent_asin": None,
            "route": WorkflowRoute.ERROR,
        }
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.UPDATE_PLAN.value,
            error_state=None,
        ),
        "selected_parent_asin": candidates[0].parent_asin,
        "route": WorkflowRoute.CONTINUE,
    }


def candidate_selector_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    planner: Any,
) -> dict[str, Any]:
    """Ask the planner to choose only an identity from the trusted candidates."""

    state = _state(value)
    result = state.last_tool_result
    requirement = _current_requirement(state)
    if result is None or result.returned_count == 0:
        return _planner_error(state, "candidate_result_missing")
    plan = state.agent_state.shopping_plan
    context = {
        "decision_key": f"select_candidate:{plan.version}:{requirement.requirement_id}",
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "requirement_id": requirement.requirement_id,
        # Compact facts are read-only context; the decision returns identity only.
        "candidates": tuple(item.model_dump(mode="json") for item in result.items),
    }
    raw_decision = planner.decide(context=context)
    try:
        decision = validate_planner_decision(raw_decision)
    except ValidationError:
        return _planner_error(state, "schema_invalid")
    if not isinstance(decision, SelectCandidateDecision):
        return _planner_error(state, "action_not_permitted")
    if not _validate_plan_binding(state, decision):
        return _planner_error(state, "stale_plan_version")
    if decision.requirement_id != requirement.requirement_id:
        return _planner_error(state, "invalid_requirement_id")
    matches = tuple(
        item for item in result.items if item.parent_asin == decision.parent_asin
    )
    if len(matches) != 1:
        return _planner_error(state, "candidate_not_in_tool_result")
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.UPDATE_PLAN.value,
            error_state=None,
        ),
        "selected_parent_asin": decision.parent_asin,
        "planner_decision": decision,
        "route": WorkflowRoute.CONTINUE,
    }


def update_plan_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    shopping_plan_service: Any,
) -> dict[str, Any]:
    """Apply the selected trusted candidate through ShoppingPlanService."""

    state = _state(value)
    if state.recommendation_args is None or state.last_tool_result is None:
        raise ValueError("Plan update requires recommendation args and tool result.")
    if state.selected_parent_asin is None:
        raise ValueError("Plan update requires selected_parent_asin.")
    requirement = _current_requirement(state)
    already_selected = sum(
        item.quantity
        for item in state.agent_state.shopping_plan.selected_items
        if item.requirement_id == requirement.requirement_id
    )
    quantity = requirement.quantity - already_selected
    try:
        plan = shopping_plan_service.select_item(
            state.agent_state.shopping_plan,
            requirement_id=requirement.requirement_id,
            recommendation_result=state.last_tool_result,
            recommendation_args=state.recommendation_args,
            selected_parent_asin=state.selected_parent_asin,
            quantity=quantity,
            selected_reason=(
                state.planner_decision.reason
                if isinstance(state.planner_decision, SelectCandidateDecision)
                else SELECTED_REASON
            ),
        )
    except ShoppingPlanServiceError as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"shopping_plan_service:{type(exc).__name__}",
            ),
            "route": WorkflowRoute.ERROR,
        }
    return {
        "agent_state": _agent_update(
            state.agent_state,
            shopping_plan=plan,
            pending_action=WorkflowAction.EVALUATE.value,
            error_state=None,
        ),
        "evaluation": None,
        "route": WorkflowRoute.CONTINUE,
    }


def evaluate_plan_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    shopping_plan_service: Any,
) -> dict[str, Any]:
    """Project domain facts and choose ready, conflict, or continue."""

    state = _state(value)
    evaluation = shopping_plan_service.evaluate_plan(state.agent_state.shopping_plan)
    if evaluation.is_ready:
        route = WorkflowRoute.READY
        action = WorkflowAction.READY
    elif evaluation.has_hard_constraint_conflict:
        route = WorkflowRoute.CONFLICT
        action = WorkflowAction.CONFLICT
    elif evaluation.pending_requirement_count > 0:
        route = WorkflowRoute.CONTINUE
        action = WorkflowAction.SELECT_REQUIREMENT
    else:
        route = WorkflowRoute.ERROR
        action = WorkflowAction.CONFLICT
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=action.value,
            error_state=(
                "workflow_invariant:unevaluable_plan"
                if route == WorkflowRoute.ERROR
                else state.agent_state.error_state
            ),
        ),
        "evaluation": evaluation,
        "route": route,
    }


def ready_node(value: ShoppingWorkflowState | dict[str, Any]) -> dict[str, Any]:
    state = _state(value)
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.READY.value,
            error_state=None,
        ),
        "route": WorkflowRoute.READY,
    }


def conflict_node(value: ShoppingWorkflowState | dict[str, Any]) -> dict[str, Any]:
    state = _state(value)
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.CONFLICT.value,
        ),
        "route": WorkflowRoute.CONFLICT,
    }


def _route(value: ShoppingWorkflowState | dict[str, Any]) -> str:
    state = _state(value)
    if state.route is None:
        raise ValueError("Workflow route is not set.")
    return state.route.value


def _bound_node(function: Callable[..., dict[str, Any]], **dependencies: Any):
    def node(state: ShoppingWorkflowState) -> dict[str, Any]:
        return function(state, **dependencies)

    return node


def build_shopping_workflow(
    recommendation_tool: Any,
    shopping_plan_service: Any,
    planner: Any | None = None,
):
    """Build and compile the P0 graph using injected runtime dependencies."""

    if recommendation_tool is None or not callable(
        getattr(recommendation_tool, "recommend", None)
    ):
        raise TypeError("recommendation_tool must provide recommend().")
    for method in ("select_item", "evaluate_plan"):
        if not callable(getattr(shopping_plan_service, method, None)):
            raise TypeError(f"shopping_plan_service must provide {method}().")
    if planner is not None and not callable(getattr(planner, "decide", None)):
        raise TypeError("planner must provide decide().")

    graph = StateGraph(ShoppingWorkflowState)
    requirement_node = "requirement_planner" if planner is not None else "select_requirement"
    candidate_node = "candidate_selector" if planner is not None else "select_candidate"
    if planner is None:
        graph.add_node("select_requirement", select_requirement_node)
        graph.add_node("select_candidate", select_candidate_node)
    else:
        graph.add_node(
            "requirement_planner",
            _bound_node(requirement_planner_node, planner=planner),
        )
        graph.add_node(
            "candidate_selector",
            _bound_node(candidate_selector_node, planner=planner),
        )
    graph.add_node(
        "recommend",
        _bound_node(recommend_node, recommendation_tool=recommendation_tool),
    )
    graph.add_node(
        "update_plan",
        _bound_node(update_plan_node, shopping_plan_service=shopping_plan_service),
    )
    graph.add_node(
        "evaluate_plan",
        _bound_node(evaluate_plan_node, shopping_plan_service=shopping_plan_service),
    )
    graph.add_node("ready", ready_node)
    graph.add_node("conflict", conflict_node)

    graph.add_edge(START, requirement_node)
    graph.add_conditional_edges(
        requirement_node,
        _route,
        {
            WorkflowRoute.CONTINUE.value: "recommend",
            WorkflowRoute.READY.value: "ready",
            WorkflowRoute.CONFLICT.value: "conflict",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "recommend",
        _route,
        {
            WorkflowRoute.CONTINUE.value: candidate_node,
            WorkflowRoute.CONFLICT.value: "conflict",
        },
    )
    graph.add_conditional_edges(
        candidate_node,
        _route,
        {
            WorkflowRoute.CONTINUE.value: "update_plan",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "update_plan",
        _route,
        {
            WorkflowRoute.CONTINUE.value: "evaluate_plan",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "evaluate_plan",
        _route,
        {
            WorkflowRoute.CONTINUE.value: requirement_node,
            WorkflowRoute.READY.value: "ready",
            WorkflowRoute.CONFLICT.value: "conflict",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_edge("ready", END)
    graph.add_edge("conflict", END)
    return graph.compile()
