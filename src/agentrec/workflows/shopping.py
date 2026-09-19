"""Minimal deterministic LangGraph workflow for AgentRec shopping plans.

The graph sequences trusted recommendation-tool output into immutable Shopping
Plan mutations. Runtime dependencies are captured by node closures and never
stored in serializable workflow state. Its only automatic re-plan is the
bounded deterministic 5-to-10 candidate-pool expansion after zero eligible
candidates; it does not relax requirements or mutate the plan before success.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ..domain import AgentState, PlanStatus, RequirementStatus, ShoppingRequirement
from ..evidence import (
    EvidenceCandidate,
    RequirementEvidence,
    SelectedProductEvidence,
)
from ..planning import (
    SelectCandidateDecision,
    SelectRequirementDecision,
    validate_planner_decision,
)
from ..replanning import (
    BoundedReplanPolicy,
    FailureDiagnosisService,
    ReplanAction,
)
from ..services import ShoppingPlanServiceError
from ..tools import RecommendationToolArgs
from ..verification import (
    CandidateVerificationStatus,
    RequirementVerification,
    SelectedCandidateVerification,
)
from .routes import WorkflowAction, WorkflowRoute
from .state import ShoppingWorkflowState


TOP_K = 5
REPLAN_TOP_K = 10
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
        "current_evidence": None,
        "current_verification": None,
        "current_evidence_attempt": None,
        "current_verification_attempt": None,
        "recommendation_top_k": TOP_K,
        "replan_attempt": 0,
        "current_failure_diagnosis": None,
        "current_replan_directive": None,
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
        "current_evidence": None,
        "current_verification": None,
        "current_evidence_attempt": None,
        "current_verification_attempt": None,
        "recommendation_top_k": TOP_K,
        "replan_attempt": 0,
        "current_failure_diagnosis": None,
        "current_replan_directive": None,
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
        top_k=state.recommendation_top_k,
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
            else WorkflowAction.RETRIEVE_EVIDENCE.value
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
        "current_evidence": None,
        "current_verification": None,
        "current_evidence_attempt": None,
        "current_verification_attempt": None,
        "route": route,
    }


def _evidence_candidates(result: Any) -> tuple[EvidenceCandidate, ...]:
    if result is None or result.returned_count == 0:
        raise ValueError("Evidence retrieval requires recommendation candidates.")
    return tuple(
        EvidenceCandidate(item_index=item.item_index, parent_asin=item.parent_asin)
        for item in result.items
    )


def _validate_current_evidence(
    state: ShoppingWorkflowState,
) -> RequirementEvidence:
    evidence = state.current_evidence
    requirement = _current_requirement(state)
    plan = state.agent_state.shopping_plan
    if evidence is None:
        raise ValueError("Current requirement evidence is missing.")
    if state.current_evidence_attempt != state.replan_attempt:
        raise ValueError("Current requirement evidence belongs to another attempt.")
    if (
        evidence.plan_id != plan.plan_id
        or evidence.retrieved_at_plan_version != plan.version
        or evidence.requirement_id != requirement.requirement_id
    ):
        raise ValueError("Current requirement evidence is stale.")
    if evidence.candidates != _evidence_candidates(state.last_tool_result):
        raise ValueError("Evidence candidate identities differ from the Tool Result.")
    return evidence


def retrieve_evidence_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    evidence_service: Any,
) -> dict[str, Any]:
    """Retrieve unverified evidence inside the trusted recommendation boundary."""

    state = _state(value)
    requirement = _current_requirement(state)
    plan = state.agent_state.shopping_plan
    try:
        candidates = _evidence_candidates(state.last_tool_result)
        evidence = evidence_service.retrieve(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            requirement=requirement,
            candidates=candidates,
        )
        if not isinstance(evidence, RequirementEvidence):
            raise TypeError("Evidence service returned an invalid result type.")
        expected = (plan.plan_id, plan.version, requirement.requirement_id, candidates)
        actual = (
            evidence.plan_id,
            evidence.retrieved_at_plan_version,
            evidence.requirement_id,
            evidence.candidates,
        )
        if actual != expected:
            raise ValueError("Evidence service returned a stale or mismatched snapshot.")
    except Exception as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"evidence_retrieval:{type(exc).__name__}",
            ),
            "current_evidence": None,
            "current_verification": None,
            "current_evidence_attempt": None,
            "current_verification_attempt": None,
            "route": WorkflowRoute.ERROR,
        }
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.VERIFY_CONSTRAINTS.value,
            error_state=None,
        ),
        "current_evidence": evidence,
        "current_verification": None,
        "current_evidence_attempt": state.replan_attempt,
        "current_verification_attempt": None,
        "selected_parent_asin": None,
        "route": WorkflowRoute.CONTINUE,
    }


def verify_constraints_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    verification_service: Any,
) -> dict[str, Any]:
    """Verify hard features and expose only deterministic eligible candidates."""

    state = _state(value)
    requirement = _current_requirement(state)
    plan = state.agent_state.shopping_plan
    try:
        evidence = _validate_current_evidence(state)
        if not requirement.required_features:
            return {
                "agent_state": _agent_update(
                    state.agent_state,
                    pending_action=WorkflowAction.SELECT_CANDIDATE.value,
                    error_state=None,
                ),
                "current_verification": None,
                "current_verification_attempt": None,
                "route": WorkflowRoute.CONTINUE,
            }
        verification = verification_service.verify(
            requirement=requirement,
            evidence=evidence,
            current_plan_id=plan.plan_id,
            current_plan_version=plan.version,
        )
        if not isinstance(verification, RequirementVerification):
            raise TypeError("Verification service returned an invalid result type.")
        expected_pairs = tuple(
            (item.item_index, item.parent_asin) for item in evidence.products
        )
        actual_pairs = tuple(
            (item.item_index, item.parent_asin) for item in verification.candidates
        )
        if (
            verification.plan_id != plan.plan_id
            or verification.requirement_id != requirement.requirement_id
            or verification.verified_at_plan_version != plan.version
            or actual_pairs != expected_pairs
        ):
            raise ValueError("Verification snapshot is stale or identity-mismatched.")
        if not verification.eligible_parent_asins:
            return {
                "agent_state": _agent_update(
                    state.agent_state,
                    pending_action=WorkflowAction.DIAGNOSE_FAILURE.value,
                    error_state="constraint_verification:no_eligible_candidates",
                ),
                "current_verification": verification,
                "current_verification_attempt": state.replan_attempt,
                "route": WorkflowRoute.DIAGNOSE,
            }
    except Exception as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"constraint_verification:{type(exc).__name__}",
            ),
            "current_verification": None,
            "current_verification_attempt": None,
            "route": WorkflowRoute.ERROR,
        }
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.SELECT_CANDIDATE.value,
            error_state=None,
        ),
        "current_verification": verification,
        "current_verification_attempt": state.replan_attempt,
        "route": WorkflowRoute.CONTINUE,
    }


def diagnose_failure_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    diagnosis_service: Any,
) -> dict[str, Any]:
    """Classify a zero-eligible result without changing ShoppingPlan."""

    state = _state(value)
    requirement = _current_requirement(state)
    plan = state.agent_state.shopping_plan
    verification = state.current_verification
    result = state.last_tool_result
    if (
        result is None
        or result.returned_count == 0
        or verification is None
        or state.current_evidence is None
        or state.current_evidence_attempt != state.replan_attempt
        or state.current_verification_attempt != state.replan_attempt
    ):
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="replan_diagnosis:invalid_or_stale_runtime_state",
            ),
            "route": WorkflowRoute.ERROR,
        }
    try:
        diagnosis = diagnosis_service.diagnose_zero_eligible(
            verification=verification,
            plan_id=plan.plan_id,
            requirement_id=requirement.requirement_id,
            plan_version=plan.version,
            attempt=state.replan_attempt,
        )
    except Exception as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"replan_diagnosis:{type(exc).__name__}",
            ),
            "route": WorkflowRoute.ERROR,
        }
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=WorkflowAction.REPLAN.value,
            error_state=state.agent_state.error_state,
        ),
        "current_failure_diagnosis": diagnosis,
        "failure_history": (*state.failure_history, diagnosis),
        "route": WorkflowRoute.REPLAN,
    }


def replan_policy_node(
    value: ShoppingWorkflowState | dict[str, Any],
    *,
    replan_policy: Any,
) -> dict[str, Any]:
    """Choose the closed 5-to-10 expansion or a terminal conflict."""

    state = _state(value)
    diagnosis = state.current_failure_diagnosis
    if diagnosis is None:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="replan_policy:diagnosis_missing",
            ),
            "route": WorkflowRoute.ERROR,
        }
    try:
        directive = replan_policy.decide(
            diagnosis=diagnosis,
            requirement=_current_requirement(state),
            current_top_k=state.recommendation_top_k,
            current_attempt=state.replan_attempt,
            max_replan_attempts=state.max_replan_attempts,
        )
    except Exception as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"replan_policy:{type(exc).__name__}",
            ),
            "route": WorkflowRoute.ERROR,
        }
    retry = directive.action is ReplanAction.EXPAND_CANDIDATE_POOL
    return {
        "agent_state": _agent_update(
            state.agent_state,
            pending_action=(
                WorkflowAction.RETRY_RECOMMENDATION.value
                if retry else WorkflowAction.CONFLICT.value
            ),
            error_state=(
                state.agent_state.error_state
                if retry else "constraint_verification:replan_attempts_exhausted"
            ),
        ),
        "current_replan_directive": directive,
        "replan_history": (*state.replan_history, directive),
        "route": WorkflowRoute.RETRY if retry else WorkflowRoute.CONFLICT,
    }


def reset_for_retry_node(
    value: ShoppingWorkflowState | dict[str, Any],
) -> dict[str, Any]:
    """Clear attempt-local state before the one permitted recommendation retry."""

    state = _state(value)
    directive = state.current_replan_directive
    plan = state.agent_state.shopping_plan
    if (
        directive is None
        or directive.action is not ReplanAction.EXPAND_CANDIDATE_POOL
        or directive.source_plan_version != plan.version
        or directive.attempt != state.replan_attempt + 1
        or directive.next_top_k != REPLAN_TOP_K
    ):
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="replan_reset:invalid_directive",
            ),
            "route": WorkflowRoute.ERROR,
        }
    return {
        "agent_state": _agent_update(
            state.agent_state,
            last_tool_result=None,
            pending_action=WorkflowAction.RETRY_RECOMMENDATION.value,
            error_state=None,
        ),
        "recommendation_args": None,
        "last_tool_result": None,
        "selected_parent_asin": None,
        "evaluation": None,
        "planner_decision": None,
        "current_evidence": None,
        "current_verification": None,
        "current_evidence_attempt": None,
        "current_verification_attempt": None,
        "recommendation_top_k": directive.next_top_k,
        "replan_attempt": directive.attempt,
        "route": WorkflowRoute.RETRY,
    }


def _eligible_parent_asins(state: ShoppingWorkflowState) -> tuple[str, ...]:
    requirement = _current_requirement(state)
    result = state.last_tool_result
    if result is None:
        raise ValueError("Candidate selection requires a Tool Result.")
    if not requirement.required_features:
        if state.current_verification is not None:
            raise ValueError("Feature-free requirement must skip verification.")
        return tuple(item.parent_asin for item in result.items)
    verification = state.current_verification
    if verification is None:
        raise ValueError("Hard-feature candidate selection requires verification.")
    plan = state.agent_state.shopping_plan
    if (
        verification.plan_id != plan.plan_id
        or verification.requirement_id != requirement.requirement_id
        or verification.verified_at_plan_version != plan.version
        or state.current_verification_attempt != state.replan_attempt
        or state.current_verification_attempt != state.current_evidence_attempt
    ):
        raise ValueError("Candidate verification is stale.")
    return verification.eligible_parent_asins


def select_candidate_node(
    value: ShoppingWorkflowState | dict[str, Any],
) -> dict[str, Any]:
    """Select rank one without an LLM or a second ranking policy."""

    state = _state(value)
    try:
        evidence = _validate_current_evidence(state)
        eligible = _eligible_parent_asins(state)
    except ValueError as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"evidence_validation:{type(exc).__name__}",
            ),
            "selected_parent_asin": None,
            "route": WorkflowRoute.ERROR,
        }
    result = state.last_tool_result
    candidates = () if result is None else tuple(
        item for item in result.items if item.parent_asin in eligible
    )
    candidates = tuple(sorted(candidates, key=lambda item: item.rank)[:1])
    if len(candidates) != 1:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="candidate_selection:eligible_candidate_missing_or_ambiguous",
            ),
            "selected_parent_asin": None,
            "route": WorkflowRoute.ERROR,
        }
    if not any(
        product.parent_asin == candidates[0].parent_asin
        and product.item_index == candidates[0].item_index
        for product in evidence.products
    ):
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="evidence_validation:selected_candidate_missing",
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
    try:
        evidence = _validate_current_evidence(state)
        eligible = _eligible_parent_asins(state)
    except ValueError:
        return _planner_error(state, "stale_or_missing_evidence_or_verification")
    evidence_by_parent = {value.parent_asin: value for value in evidence.products}
    context = {
        "decision_key": f"select_candidate:{plan.version}:{requirement.requirement_id}",
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "requirement_id": requirement.requirement_id,
        # item_index remains system-only. Evidence text is untrusted data, not instructions.
        "candidates": tuple({
            "rank": item.rank,
            "parent_asin": item.parent_asin,
            "title": item.title,
            "price": item.price,
            "score": item.score,
            "score_source": item.score_source,
            "evidence_status": evidence.status.value,
            "evidence": tuple({
                "rank": snippet.rank,
                "chunk_id": snippet.chunk_id,
                "chunk_type": snippet.chunk_type.value,
                "part_index": snippet.part_index,
                "text": snippet.text,
                "similarity_score": snippet.similarity_score,
            } for snippet in evidence_by_parent[item.parent_asin].snippets),
        } for item in result.items if item.parent_asin in eligible),
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
    if decision.parent_asin not in set(eligible):
        return _planner_error(state, "candidate_not_eligible")
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
    try:
        evidence = _validate_current_evidence(state)
        eligible = _eligible_parent_asins(state)
    except ValueError as exc:
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state=f"evidence_validation:{type(exc).__name__}",
            ),
            "route": WorkflowRoute.ERROR,
        }
    if state.recommendation_args is None or state.last_tool_result is None:
        raise ValueError("Plan update requires recommendation args and tool result.")
    if state.selected_parent_asin is None:
        raise ValueError("Plan update requires selected_parent_asin.")
    if state.selected_parent_asin not in set(eligible):
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="constraint_verification:selected_candidate_not_eligible",
            ),
            "route": WorkflowRoute.ERROR,
        }
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
    selected_tool_items = tuple(
        item for item in state.last_tool_result.items
        if item.parent_asin == state.selected_parent_asin
    )
    selected_products = tuple(
        item for item in evidence.products
        if item.parent_asin == state.selected_parent_asin
    )
    if (
        len(selected_tool_items) != 1
        or len(selected_products) != 1
        or selected_tool_items[0].item_index != selected_products[0].item_index
    ):
        return {
            "agent_state": _agent_update(
                state.agent_state,
                pending_action=WorkflowAction.CONFLICT.value,
                error_state="evidence_validation:selected_identity_mismatch",
            ),
            "route": WorkflowRoute.ERROR,
        }
    product_evidence = selected_products[0]
    selected_snapshot = SelectedProductEvidence(
        plan_id=plan.plan_id,
        selected_at_plan_version=plan.version,
        requirement_id=requirement.requirement_id,
        item_index=product_evidence.item_index,
        parent_asin=product_evidence.parent_asin,
        query=evidence.query,
        snippets=product_evidence.snippets,
        status=evidence.status,
        provenance=evidence.provenance,
    )
    selected_history = tuple(
        item for item in state.selected_evidence
        if item.requirement_id != requirement.requirement_id
    ) + (selected_snapshot,)
    selected_verifications = state.selected_verifications
    if state.current_verification is not None:
        matches = tuple(
            item for item in state.current_verification.candidates
            if item.parent_asin == state.selected_parent_asin
        )
        if len(matches) != 1 or matches[0].status is not CandidateVerificationStatus.ELIGIBLE:
            return {
                "agent_state": _agent_update(
                    state.agent_state,
                    pending_action=WorkflowAction.CONFLICT.value,
                    error_state="constraint_verification:selected_candidate_not_eligible",
                ),
                "route": WorkflowRoute.ERROR,
            }
        selected_verifications = tuple(
            item for item in state.selected_verifications
            if item.requirement_id != requirement.requirement_id
        ) + (SelectedCandidateVerification(
            plan_id=plan.plan_id,
            requirement_id=requirement.requirement_id,
            selected_at_plan_version=plan.version,
            candidate=matches[0],
        ),)
    return {
        "agent_state": _agent_update(
            state.agent_state,
            shopping_plan=plan,
            pending_action=WorkflowAction.EVALUATE.value,
            error_state=None,
        ),
        "evaluation": None,
        "current_evidence": None,
        "current_verification": None,
        "current_evidence_attempt": None,
        "current_verification_attempt": None,
        "current_failure_diagnosis": None,
        "current_replan_directive": None,
        "selected_evidence": selected_history,
        "selected_verifications": selected_verifications,
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
    *,
    evidence_service: Any,
    verification_service: Any,
    diagnosis_service: Any | None = None,
    replan_policy: Any | None = None,
):
    """Build the shopping graph with one deterministic bounded re-plan."""

    if recommendation_tool is None or not callable(
        getattr(recommendation_tool, "recommend", None)
    ):
        raise TypeError("recommendation_tool must provide recommend().")
    for method in ("select_item", "evaluate_plan"):
        if not callable(getattr(shopping_plan_service, method, None)):
            raise TypeError(f"shopping_plan_service must provide {method}().")
    if planner is not None and not callable(getattr(planner, "decide", None)):
        raise TypeError("planner must provide decide().")
    if evidence_service is None or not callable(getattr(evidence_service, "retrieve", None)):
        raise TypeError("evidence_service must provide retrieve().")
    if verification_service is None or not callable(getattr(verification_service, "verify", None)):
        raise TypeError("verification_service must provide verify().")
    diagnosis_service = diagnosis_service or FailureDiagnosisService()
    replan_policy = replan_policy or BoundedReplanPolicy()
    if not callable(getattr(diagnosis_service, "diagnose_zero_eligible", None)):
        raise TypeError("diagnosis_service must provide diagnose_zero_eligible().")
    if not callable(getattr(replan_policy, "decide", None)):
        raise TypeError("replan_policy must provide decide().")

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
        "retrieve_evidence",
        _bound_node(retrieve_evidence_node, evidence_service=evidence_service),
    )
    graph.add_node(
        "verify_constraints",
        _bound_node(verify_constraints_node, verification_service=verification_service),
    )
    graph.add_node(
        "diagnose_failure",
        _bound_node(diagnose_failure_node, diagnosis_service=diagnosis_service),
    )
    graph.add_node(
        "replan_policy",
        _bound_node(replan_policy_node, replan_policy=replan_policy),
    )
    graph.add_node("reset_for_retry", reset_for_retry_node)
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
            WorkflowRoute.CONTINUE.value: "retrieve_evidence",
            WorkflowRoute.CONFLICT.value: "conflict",
        },
    )
    graph.add_conditional_edges(
        "retrieve_evidence",
        _route,
        {
            WorkflowRoute.CONTINUE.value: "verify_constraints",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "verify_constraints",
        _route,
        {
            WorkflowRoute.CONTINUE.value: candidate_node,
            WorkflowRoute.DIAGNOSE.value: "diagnose_failure",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "diagnose_failure",
        _route,
        {
            WorkflowRoute.REPLAN.value: "replan_policy",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "replan_policy",
        _route,
        {
            WorkflowRoute.RETRY.value: "reset_for_retry",
            WorkflowRoute.CONFLICT.value: "conflict",
            WorkflowRoute.ERROR.value: END,
        },
    )
    graph.add_conditional_edges(
        "reset_for_retry",
        _route,
        {
            WorkflowRoute.RETRY.value: "recommend",
            WorkflowRoute.ERROR.value: END,
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
