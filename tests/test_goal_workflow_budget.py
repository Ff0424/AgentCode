"""Deterministic tests for Goal budget derivation and workflow enforcement."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from src.agentrec.domain import (
    AgentState,
    ItemSource,
    PlanStatus,
    RequirementStatus,
    ShoppingPlan,
    ShoppingPlanItem,
    ShoppingRequirement,
)
from src.agentrec.planning import GoalBudgetAllocation, RequirementBudgetAllocation
from src.agentrec.recommendation.catalog import ProductCatalogIndex
from src.agentrec.recommendation.contracts import RecommendationRequest
from src.agentrec.services import RequirementConstraintError, ShoppingPlanService
from src.agentrec.tools import RecommendationToolItem, RecommendationToolResult
from src.agentrec.verification import EvidenceConstraintVerifier
from src.agentrec.workflows import (
    GoalAllocationExhaustedError,
    GoalBudgetDerivationError,
    RecommendationBudgetProvenance,
    ShoppingWorkflowState,
    WorkflowRoute,
    build_shopping_workflow,
    derive_goal_recommendation_budget,
)
from src.agentrec.workflows.shopping import recommend_node
from tests.fake_evidence import FakeEvidenceService
from tests.test_bounded_replan_workflow import (
    AttemptEvidenceService,
    PoolAwareTool,
)


def requirement(
    identifier: str = "req-001",
    *,
    category: str = "Dock",
    quantity: int = 1,
    max_budget: float | None = None,
    features: tuple[str, ...] = (),
    status: RequirementStatus = RequirementStatus.PENDING,
) -> ShoppingRequirement:
    return ShoppingRequirement(
        requirement_id=identifier,
        category=category,
        quantity=quantity,
        max_budget=max_budget,
        required_features=features,
        status=status,
    )


def allocation(
    *values: tuple[str, float], total: float | None = None, unallocated: float = 0
) -> GoalBudgetAllocation:
    allocated_total = sum(value for _identifier, value in values)
    return GoalBudgetAllocation(
        total_budget=allocated_total + unallocated if total is None else total,
        allocations=tuple(
            RequirementBudgetAllocation(
                requirement_id=identifier, allocated_budget=value
            )
            for identifier, value in values
        ),
        unallocated_budget=unallocated,
    )


def plan_for(
    target: ShoppingRequirement,
    *,
    selected_items: tuple[ShoppingPlanItem, ...] = (),
    total_budget: float = 500,
    extra_requirements: tuple[ShoppingRequirement, ...] = (),
) -> ShoppingPlan:
    requirements = (target, *extra_requirements)
    if not selected_items:
        status = PlanStatus.DRAFT
    elif all(value.status is RequirementStatus.SATISFIED for value in requirements):
        status = PlanStatus.READY
    elif any(value.status is RequirementStatus.CONFLICT for value in requirements):
        status = PlanStatus.CONFLICT
    else:
        status = PlanStatus.IN_PROGRESS
    return ShoppingPlan(
        plan_id="plan",
        user_id="user",
        total_budget=total_budget,
        requirements=requirements,
        selected_items=selected_items,
        status=status,
        version=1 if selected_items else 0,
    )


def selected_item(
    *,
    requirement_id: str = "req-001",
    category: str = "Dock",
    price: float = 40,
    quantity: int = 1,
    parent_asin: str = "P-OLD",
) -> ShoppingPlanItem:
    return ShoppingPlanItem(
        slot_id=f"{requirement_id}:1",
        requirement_id=requirement_id,
        category=category,
        parent_asin=parent_asin,
        title="Existing",
        price=price,
        quantity=quantity,
        selected_reason="existing",
        constraints_satisfied=True,
        source=ItemSource.HYBRID,
        recommendation_score=0.5,
    )


def state_for(
    target: ShoppingRequirement,
    goal_allocation: GoalBudgetAllocation | None,
    *,
    selected_items: tuple[ShoppingPlanItem, ...] = (),
    total_budget: float = 500,
) -> ShoppingWorkflowState:
    plan = plan_for(
        target,
        selected_items=selected_items,
        total_budget=total_budget,
    )
    return ShoppingWorkflowState(
        agent_state=AgentState(
            user_id="user",
            session_id="session",
            shopping_plan=plan,
            current_requirement_id=target.requirement_id,
        ),
        goal_budget_allocation=goal_allocation,
    )


class _RecordingTool:
    def __init__(self, *, price: float = 30, empty: bool = False) -> None:
        self.price = price
        self.empty = empty
        self.calls = []

    def recommend(self, *, user_id, args, excluded_parent_asins=()):
        self.calls.append((user_id, args, excluded_parent_asins))
        if self.empty:
            return RecommendationToolResult(
                personalization_status="personalized",
                fallback_reason=None,
                returned_count=0,
                items=(),
            )
        return RecommendationToolResult(
            personalization_status="personalized",
            fallback_reason=None,
            returned_count=1,
            items=(RecommendationToolItem(
                rank=1,
                item_index=1,
                parent_asin="P-NEW",
                title="Candidate",
                price=self.price,
                score=0.9,
                score_source="hybrid",
            ),),
        )


class _FailingPlanService(ShoppingPlanService):
    def select_item(self, *args, **kwargs):
        raise RequirementConstraintError("deterministic mutation failure")


class GoalBudgetDerivationTests(unittest.TestCase):
    def test_allocation_lookup_and_effective_budget_rules(self) -> None:
        cases = (
            (200, None, 200),
            (200, 150, 150),
            (100, 150, 100),
        )
        for allocated, cap, expected in cases:
            with self.subTest(allocated=allocated, cap=cap):
                target = requirement(quantity=1, max_budget=cap)
                source = plan_for(target)
                before = (source.model_dump(), target.model_dump())
                result = derive_goal_recommendation_budget(
                    plan=source,
                    requirement=target,
                    allocation=allocation(("req-001", allocated), total=allocated),
                )
                self.assertEqual(result.effective_subtotal_budget, expected)
                self.assertEqual(result.derived_unit_max_price, expected)
                self.assertEqual((source.model_dump(), target.model_dump()), before)

    def test_quantity_floor_and_explicit_cap(self) -> None:
        cases = (
            (100, None, 1, 100),
            (100, None, 2, 50),
            (100, None, 3, 33.33),
            (200, 150, 2, 75),
            (100, 150, 2, 50),
        )
        for allocated, cap, quantity, expected in cases:
            with self.subTest(quantity=quantity, cap=cap):
                target = requirement(quantity=quantity, max_budget=cap)
                result = derive_goal_recommendation_budget(
                    plan=plan_for(target),
                    requirement=target,
                    allocation=allocation(("req-001", allocated), total=allocated),
                )
                self.assertEqual(result.derived_unit_max_price, expected)

    def test_retained_subtotal_and_other_requirement_are_handled(self) -> None:
        target = requirement(quantity=2, status=RequirementStatus.CANDIDATE_SELECTED)
        other = requirement("req-002", category="Mouse", status=RequirementStatus.SATISFIED)
        items = (
            selected_item(price=40),
            selected_item(
                requirement_id="req-002",
                category="Mouse",
                price=300,
                parent_asin="P-MOUSE",
            ),
        )
        source = ShoppingPlan(
            plan_id="plan",
            user_id="user",
            total_budget=500,
            requirements=(target, other),
            selected_items=items,
            status="in_progress",
            version=2,
        )
        result = derive_goal_recommendation_budget(
            plan=source,
            requirement=target,
            allocation=allocation(
                ("req-001", 100), ("req-002", 400), total=500
            ),
        )
        self.assertEqual(result.retained_subtotal_before_call, 40)
        self.assertEqual(result.remaining_quantity_before_call, 1)
        self.assertEqual(result.derived_unit_max_price, 60)
        self.assertEqual(result.plan_version_before_call, 2)

    def test_missing_satisfied_and_exhausted_contexts_fail_closed(self) -> None:
        target = requirement()
        with self.assertRaises(GoalBudgetDerivationError):
            derive_goal_recommendation_budget(
                plan=plan_for(target),
                requirement=target,
                allocation=allocation(("other", 100), total=100),
            )

        satisfied = requirement(status=RequirementStatus.SATISFIED)
        item = selected_item(price=50)
        with self.assertRaises(GoalBudgetDerivationError):
            derive_goal_recommendation_budget(
                plan=plan_for(satisfied, selected_items=(item,)),
                requirement=satisfied,
                allocation=allocation(("req-001", 100), total=100),
            )

        partial = requirement(quantity=2, status=RequirementStatus.CANDIDATE_SELECTED)
        with self.assertRaises(GoalAllocationExhaustedError):
            derive_goal_recommendation_budget(
                plan=plan_for(partial, selected_items=(selected_item(price=100),)),
                requirement=partial,
                allocation=allocation(("req-001", 100), total=100),
            )

        subcent = requirement(quantity=2)
        with self.assertRaises(GoalAllocationExhaustedError):
            derive_goal_recommendation_budget(
                plan=plan_for(subcent, total_budget=0.01),
                requirement=subcent,
                allocation=allocation(("req-001", 0.01), total=0.01),
            )

        invalid_plan = ShoppingPlan.model_construct(
            plan_id="plan",
            user_id="user",
            currency="USD",
            total_budget=100.0,
            requirements=(target,),
            selected_items=(selected_item(quantity=2),),
            status=PlanStatus.IN_PROGRESS,
            version=1,
        )
        with self.assertRaises(GoalBudgetDerivationError):
            derive_goal_recommendation_budget(
                plan=invalid_plan,
                requirement=target,
                allocation=allocation(("req-001", 100), total=100),
            )

    def test_derivation_is_deterministic(self) -> None:
        target = requirement(quantity=3)
        kwargs = dict(
            plan=plan_for(target),
            requirement=target,
            allocation=allocation(("req-001", 100), total=100),
        )
        self.assertEqual(
            derive_goal_recommendation_budget(**kwargs),
            derive_goal_recommendation_budget(**kwargs),
        )

    def test_provenance_contract_round_trip_frozen_and_strict(self) -> None:
        value = RecommendationBudgetProvenance(
            requirement_id=" req-001 ",
            plan_version_before_call=2,
            allocated_budget=100,
            explicit_max_budget=None,
            effective_subtotal_budget=100,
            retained_subtotal_before_call=40,
            remaining_quantity_before_call=1,
            derived_unit_max_price=60,
        )
        self.assertEqual(value.requirement_id, "req-001")
        self.assertEqual(
            RecommendationBudgetProvenance.model_validate_json(value.model_dump_json()),
            value,
        )
        with self.assertRaises(ValidationError):
            value.derived_unit_max_price = 61
        with self.assertRaises(ValidationError):
            RecommendationBudgetProvenance(
                **value.model_dump(), unexpected=True
            )
        with self.assertRaises(ValidationError):
            RecommendationBudgetProvenance(
                **{**value.model_dump(), "derived_unit_max_price": 59.99}
            )


class GoalWorkflowBudgetTests(unittest.TestCase):
    def test_existing_catalog_price_boundary_is_inclusive(self) -> None:
        catalog = ProductCatalogIndex.__new__(ProductCatalogIndex)
        catalog._products = (None, None, None)
        catalog._prices = np.asarray((29.99, 30.00, 30.01), dtype=np.float64)
        catalog._category_rows = {}
        catalog._feature_rows = {}
        catalog._parent_to_row = {}
        eligible = catalog.build_eligible_mask(RecommendationRequest(
            user_id="user", max_price=30.00
        ))
        self.assertEqual(eligible.tolist(), [True, True, False])

    def test_state_provenance_identity_order_and_legacy_invariants(self) -> None:
        first = requirement("req-001")
        second = requirement("req-002", category="Mouse")
        plan = ShoppingPlanService().create_plan(
            plan_id="plan",
            user_id="user",
            currency="USD",
            total_budget=200,
            requirements=(first, second),
        )
        first_provenance = RecommendationBudgetProvenance(
            requirement_id="req-001",
            plan_version_before_call=0,
            allocated_budget=100,
            explicit_max_budget=None,
            effective_subtotal_budget=100,
            retained_subtotal_before_call=0,
            remaining_quantity_before_call=1,
            derived_unit_max_price=100,
        )
        second_provenance = first_provenance.model_copy(update={
            "requirement_id": "req-002"
        })
        goal_allocation = allocation(
            ("req-001", 100), ("req-002", 100), total=200
        )
        base_agent = AgentState(
            user_id="user",
            session_id="session",
            shopping_plan=plan,
            current_requirement_id="req-001",
        )
        accepted = ShoppingWorkflowState(
            agent_state=base_agent,
            goal_budget_allocation=goal_allocation,
            current_recommendation_budget_provenance=first_provenance,
        )
        self.assertEqual(
            accepted.current_recommendation_budget_provenance,
            first_provenance,
        )
        with self.assertRaises(ValidationError):
            ShoppingWorkflowState(
                agent_state=base_agent,
                current_recommendation_budget_provenance=first_provenance,
            )
        with self.assertRaises(ValidationError):
            ShoppingWorkflowState(
                agent_state=base_agent,
                goal_budget_allocation=goal_allocation,
                selected_recommendation_budget_provenance=(
                    second_provenance,
                    first_provenance,
                ),
            )
        with self.assertRaises(ValidationError):
            ShoppingWorkflowState(
                agent_state=base_agent,
                goal_budget_allocation=goal_allocation,
                selected_recommendation_budget_provenance=(
                    first_provenance,
                    first_provenance,
                ),
            )

    def test_goal_and_legacy_recommend_branches(self) -> None:
        target = requirement(quantity=3, features=("HDMI",))
        tool = _RecordingTool()
        goal_state = state_for(
            target,
            allocation(("req-001", 100), total=100),
            total_budget=100,
        )
        update = recommend_node(goal_state, recommendation_tool=tool)
        self.assertEqual(update["recommendation_args"].max_price, 33.33)
        self.assertEqual(update["recommendation_args"].top_k, 5)
        self.assertEqual(update["recommendation_args"].required_features, ("HDMI",))
        self.assertEqual(
            update["current_recommendation_budget_provenance"].requirement_id,
            "req-001",
        )
        self.assertIsNone(target.max_budget)

        legacy_target = requirement(max_budget=120)
        legacy_tool = _RecordingTool()
        legacy = state_for(legacy_target, None)
        legacy_update = recommend_node(legacy, recommendation_tool=legacy_tool)
        self.assertEqual(legacy_update["recommendation_args"].max_price, 120)
        self.assertIsNone(legacy_update["current_recommendation_budget_provenance"])

    def test_exclusions_remain_unchanged(self) -> None:
        target = requirement(quantity=2, status=RequirementStatus.CANDIDATE_SELECTED)
        old = selected_item(price=40)
        tool = _RecordingTool()
        state = state_for(
            target,
            allocation(("req-001", 100), total=100),
            selected_items=(old,),
            total_budget=100,
        )
        update = recommend_node(state, recommendation_tool=tool)
        self.assertEqual(update["recommendation_args"].max_price, 60)
        self.assertEqual(tool.calls[0][2], ("P-OLD",))

    def test_allocation_exhaustion_does_not_call_tool(self) -> None:
        target = requirement(quantity=2, status=RequirementStatus.CANDIDATE_SELECTED)
        old = selected_item(price=100)
        tool = _RecordingTool()
        state = state_for(
            target,
            allocation(("req-001", 100), total=100),
            selected_items=(old,),
            total_budget=100,
        )
        before = state.agent_state.shopping_plan
        update = recommend_node(state, recommendation_tool=tool)
        self.assertEqual(tool.calls, [])
        self.assertEqual(update["route"], WorkflowRoute.CONFLICT)
        self.assertEqual(
            update["agent_state"].error_state,
            "goal_budget:allocation_exhausted",
        )
        self.assertIsNone(update["last_tool_result"])
        self.assertEqual(update["agent_state"].shopping_plan, before)

    def test_goal_graph_retains_selected_provenance_after_success(self) -> None:
        target = requirement(quantity=3)
        tool = _RecordingTool(price=30)
        initial = ShoppingWorkflowState(
            agent_state=AgentState(
                user_id="user", session_id="session",
                shopping_plan=plan_for(target, total_budget=100),
            ),
            goal_budget_allocation=allocation(("req-001", 100), total=100),
        )
        graph = build_shopping_workflow(
            tool,
            ShoppingPlanService(),
            evidence_service=FakeEvidenceService(),
            verification_service=EvidenceConstraintVerifier(),
        )
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial, config={"recursion_limit": 50})
        )
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual(len(final.selected_recommendation_budget_provenance), 1)
        provenance = final.selected_recommendation_budget_provenance[0]
        self.assertEqual(provenance.plan_version_before_call, 0)
        self.assertEqual(provenance.derived_unit_max_price, 33.33)
        self.assertIsNone(final.current_recommendation_budget_provenance)
        item = final.agent_state.shopping_plan.selected_items[0]
        self.assertEqual((item.price, item.quantity), (30, 3))

    def test_failed_plan_mutation_does_not_append_selected_provenance(self) -> None:
        target = requirement()
        initial = ShoppingWorkflowState(
            agent_state=AgentState(
                user_id="user", session_id="session",
                shopping_plan=plan_for(target, total_budget=100),
            ),
            goal_budget_allocation=allocation(("req-001", 100), total=100),
        )
        graph = build_shopping_workflow(
            _RecordingTool(price=30),
            _FailingPlanService(),
            evidence_service=FakeEvidenceService(),
            verification_service=EvidenceConstraintVerifier(),
        )
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial, config={"recursion_limit": 50})
        )
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(final.selected_recommendation_budget_provenance, ())
        self.assertEqual(final.agent_state.shopping_plan, initial.agent_state.shopping_plan)

    def test_retry_preserves_goal_budget_and_changes_only_top_k(self) -> None:
        target = requirement(
            category="Hubs", max_budget=500, features=("HDMI",)
        )
        initial = ShoppingWorkflowState(
            agent_state=AgentState(
                user_id="user", session_id="session",
                shopping_plan=plan_for(target, total_budget=83.33),
            ),
            goal_budget_allocation=allocation(
                ("req-001", 83.33), total=83.33
            ),
        )
        tool = PoolAwareTool()
        graph = build_shopping_workflow(
            tool,
            ShoppingPlanService(),
            evidence_service=AttemptEvidenceService(success_on_expanded_pool=True),
            verification_service=EvidenceConstraintVerifier(),
        )
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial, config={"recursion_limit": 80})
        )
        self.assertEqual([value.top_k for value in tool.calls], [5, 10])
        self.assertEqual([value.max_price for value in tool.calls], [83.33, 83.33])
        self.assertEqual(final.goal_budget_allocation, initial.goal_budget_allocation)
        self.assertEqual(len(final.selected_recommendation_budget_provenance), 1)


if __name__ == "__main__":
    unittest.main()
