"""Fail-closed response tests for Goal recommendation budget provenance."""

from __future__ import annotations

import unittest

from src.agentrec.domain import AgentState, ShoppingRequirement
from src.agentrec.planning import GoalBudgetAllocation, RequirementBudgetAllocation
from src.agentrec.response import (
    ConflictReason,
    GroundedResponseProjectionError,
    GroundedResponseProjector,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.tools import (
    RecommendationToolArgs,
    RecommendationToolItem,
    RecommendationToolResult,
)
from src.agentrec.verification import EvidenceConstraintVerifier
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.fake_evidence import FakeEvidenceService
from tests.test_bounded_replan_workflow import AttemptEvidenceService, PoolAwareTool
from tests.test_shopping_workflow import FakeRecommendationTool


PROJECTOR = GroundedResponseProjector()


class _PartialConflictTool:
    """Return one successful first requirement, then a pool for verification."""

    def __init__(self) -> None:
        self.calls: list[RecommendationToolArgs] = []

    def recommend(self, *, user_id, args, excluded_parent_asins=()):
        self.calls.append(args)
        if args.category == "Dock":
            return RecommendationToolResult(
                personalization_status="personalized",
                fallback_reason=None,
                returned_count=1,
                items=(RecommendationToolItem(
                    rank=1,
                    item_index=100,
                    parent_asin="P-DOCK",
                    title="Dock",
                    price=20,
                    score=1,
                    score_source="hybrid",
                ),),
            )
        return RecommendationToolResult(
            personalization_status="personalized",
            fallback_reason=None,
            returned_count=args.top_k,
            items=tuple(
                RecommendationToolItem(
                    rank=rank,
                    item_index=200 + rank,
                    parent_asin=f"P-HUB-{rank}",
                    title=f"Hub {rank}",
                    price=50,
                    score=1 / rank,
                    score_source="hybrid",
                )
                for rank in range(1, args.top_k + 1)
            ),
        )


def _allocation(
    total: float,
    *values: tuple[str, float],
    unallocated: float = 0,
) -> GoalBudgetAllocation:
    return GoalBudgetAllocation(
        total_budget=total,
        allocations=tuple(
            RequirementBudgetAllocation(
                requirement_id=requirement_id,
                allocated_budget=amount,
            )
            for requirement_id, amount in values
        ),
        unallocated_budget=unallocated,
    )


def _state(
    requirements: tuple[ShoppingRequirement, ...],
    allocation: GoalBudgetAllocation,
) -> ShoppingWorkflowState:
    plan = ShoppingPlanService().create_plan(
        plan_id="goal-plan",
        user_id="user",
        currency="USD",
        total_budget=allocation.total_budget,
        requirements=requirements,
    )
    return ShoppingWorkflowState(
        agent_state=AgentState(
            user_id="user",
            session_id="session",
            shopping_plan=plan,
        ),
        goal_budget_allocation=allocation,
    )


def _run(
    initial: ShoppingWorkflowState,
    tool,
    *,
    evidence_service=None,
) -> ShoppingWorkflowState:
    graph = build_shopping_workflow(
        tool,
        ShoppingPlanService(),
        evidence_service=evidence_service or FakeEvidenceService(),
        verification_service=EvidenceConstraintVerifier(),
    )
    return ShoppingWorkflowState.model_validate(
        graph.invoke(initial, config={"recursion_limit": 80})
    )


def _single_ready(*, price: float = 30, allocation_amount: float = 30):
    requirement = ShoppingRequirement(
        requirement_id="mouse",
        category="Mouse",
        quantity=1,
    )
    initial = _state(
        (requirement,),
        _allocation(
            100,
            ("mouse", allocation_amount),
            unallocated=100 - allocation_amount,
        ),
    )
    return _run(initial, FakeRecommendationTool({"Mouse": price}))


def _three_ready() -> ShoppingWorkflowState:
    requirements = (
        ShoppingRequirement(
            requirement_id="dock",
            category="Dock",
            max_budget=200,
            required_features=("HDMI",),
        ),
        ShoppingRequirement(requirement_id="mouse", category="Mouse"),
        ShoppingRequirement(
            requirement_id="headphones",
            category="Headphones",
            max_budget=250,
        ),
    )
    initial = _state(
        requirements,
        _allocation(
            500,
            ("dock", 150),
            ("mouse", 100),
            ("headphones", 250),
        ),
    )
    return _run(
        initial,
        FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180}),
    )


def _goal_no_candidates() -> ShoppingWorkflowState:
    requirement = ShoppingRequirement(
        requirement_id="mouse",
        category="Mouse",
        quantity=2,
        max_budget=150,
    )
    initial = _state(
        (requirement,),
        _allocation(100, ("mouse", 100)),
    )
    return _run(
        initial,
        FakeRecommendationTool({"Mouse": 30}, empty_category="Mouse"),
    )


def _goal_verification_exhausted() -> ShoppingWorkflowState:
    requirement = ShoppingRequirement(
        requirement_id="dock",
        category="Hubs",
        max_budget=500,
        required_features=("HDMI",),
    )
    initial = _state(
        (requirement,),
        _allocation(100, ("dock", 100)),
    )
    return _run(
        initial,
        PoolAwareTool(),
        evidence_service=AttemptEvidenceService(success_on_expanded_pool=False),
    )


def _goal_allocation_exhausted(*, executable: bool = False) -> ShoppingWorkflowState:
    requirement = ShoppingRequirement(requirement_id="mouse", category="Mouse")
    amount = 10 if executable else 0
    initial = _state(
        (requirement,),
        _allocation(100, ("mouse", amount), unallocated=100 - amount),
    )
    if executable:
        return initial.model_copy(update={
            "agent_state": initial.agent_state.model_copy(update={
                "current_requirement_id": "mouse",
                "pending_action": "conflict",
                "error_state": "goal_budget:allocation_exhausted",
            }),
            "route": WorkflowRoute.CONFLICT,
        })
    return _run(initial, FakeRecommendationTool({"Mouse": 5}))


class GoalReadyResponseProvenanceTests(unittest.TestCase):
    def test_one_and_three_requirement_ready_projection(self) -> None:
        one = _single_ready()
        three = _three_ready()
        self.assertEqual(PROJECTOR.project(one).products[0].requirement_id, "mouse")
        self.assertEqual(
            tuple(value.requirement_id for value in PROJECTOR.project(three).products),
            ("dock", "mouse", "headphones"),
        )
        self.assertEqual(
            tuple(
                value.requirement_id
                for value in three.selected_recommendation_budget_provenance
            ),
            ("dock", "mouse", "headphones"),
        )

    def test_provenance_is_matched_by_requirement_identity(self) -> None:
        final = _three_ready()
        reversed_history = tuple(
            reversed(final.selected_recommendation_budget_provenance)
        )
        context = PROJECTOR.project(final.model_copy(update={
            "selected_recommendation_budget_provenance": reversed_history,
        }))
        self.assertEqual(len(context.products), 3)

    def test_selected_price_below_and_at_cap(self) -> None:
        for price in (29.99, 30.0):
            with self.subTest(price=price):
                self.assertEqual(
                    PROJECTOR.project(_single_ready(price=price)).products[0].price,
                    price,
                )

    def test_raw_price_above_cap_is_rejected(self) -> None:
        final = _single_ready()
        plan = final.agent_state.shopping_plan
        item = plan.selected_items[0].model_copy(update={"price": 30.001})
        changed_plan = plan.model_copy(update={"selected_items": (item,)})
        changed_evaluation = final.evaluation.model_copy(update={
            "total_spent": 30.001,
            "remaining_budget": 69.999,
        })
        changed = final.model_copy(update={
            "agent_state": final.agent_state.model_copy(update={"shopping_plan": changed_plan}),
            "evaluation": changed_evaluation,
        })
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(changed)

    def test_missing_and_tampered_selected_provenance_fail_closed(self) -> None:
        final = _single_ready()
        provenance = final.selected_recommendation_budget_provenance[0]
        corruptions = (
            (),
            (provenance.model_copy(update={"allocated_budget": 29}),),
            (provenance.model_copy(update={"explicit_max_budget": 25}),),
            (provenance.model_copy(update={"effective_subtotal_budget": 29}),),
            (provenance.model_copy(update={"derived_unit_max_price": 29}),),
            (provenance.model_copy(update={"retained_subtotal_before_call": 1}),),
            (provenance.model_copy(update={"remaining_quantity_before_call": 2}),),
            (provenance.model_copy(update={"plan_version_before_call": 4}),),
        )
        for history in corruptions:
            with self.subTest(history=history), self.assertRaises(
                GroundedResponseProjectionError
            ):
                PROJECTOR.project(final.model_copy(update={
                    "selected_recommendation_budget_provenance": history,
                }))

    def test_selected_subtotal_and_quantity_are_defensively_checked(self) -> None:
        final = _single_ready()
        plan = final.agent_state.shopping_plan
        item = plan.selected_items[0]
        for changes in ({"price": 31}, {"quantity": 2}):
            changed_item = item.model_copy(update=changes)
            changed_plan = plan.model_copy(update={"selected_items": (changed_item,)})
            changed = final.model_copy(update={
                "agent_state": final.agent_state.model_copy(
                    update={"shopping_plan": changed_plan}
                )
            })
            with self.subTest(changes=changes), self.assertRaises(
                GroundedResponseProjectionError
            ):
                PROJECTOR.project(changed)


class GoalConflictResponseProvenanceTests(unittest.TestCase):
    def test_valid_goal_no_candidates_uses_derived_cap(self) -> None:
        final = _goal_no_candidates()
        self.assertEqual(final.recommendation_args.max_price, 50)
        self.assertNotEqual(
            final.recommendation_args.max_price,
            final.agent_state.shopping_plan.requirements[0].max_budget,
        )
        context = PROJECTOR.project(final)
        self.assertEqual(
            context.decision.reason,
            ConflictReason.NO_RECOMMENDATION_CANDIDATES,
        )

    def test_goal_no_candidates_rejects_current_provenance_and_args_tampering(self) -> None:
        final = _goal_no_candidates()
        provenance = final.current_recommendation_budget_provenance
        corruptions = (
            {"current_recommendation_budget_provenance": None},
            {"current_recommendation_budget_provenance": provenance.model_copy(
                update={"allocated_budget": 90}
            )},
            {"recommendation_args": final.recommendation_args.model_copy(
                update={"max_price": None}
            )},
            {"recommendation_args": final.recommendation_args.model_copy(
                update={"max_price": 51}
            )},
            {"recommendation_args": final.recommendation_args.model_copy(
                update={"category": "Wrong"}
            )},
            {"recommendation_args": final.recommendation_args.model_copy(
                update={"required_features": ("Other",)}
            )},
        )
        for changes in corruptions:
            with self.subTest(changes=changes), self.assertRaises(
                GroundedResponseProjectionError
            ):
                PROJECTOR.project(final.model_copy(update=changes))

    def test_goal_no_candidates_rejects_changed_allocation(self) -> None:
        final = _goal_no_candidates()
        changed = _allocation(100, ("mouse", 90), unallocated=10)
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"goal_budget_allocation": changed}))

    def test_valid_goal_retry_exhaustion_uses_final_retry_provenance(self) -> None:
        final = _goal_verification_exhausted()
        self.assertEqual(final.recommendation_top_k, 10)
        self.assertEqual(final.recommendation_args.top_k, 10)
        self.assertIsNotNone(final.current_recommendation_budget_provenance)
        self.assertEqual(
            final.recommendation_args.max_price,
            final.current_recommendation_budget_provenance.derived_unit_max_price,
        )
        self.assertEqual(
            PROJECTOR.project(final).decision.reason,
            ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
        )

    def test_goal_retry_exhaustion_rejects_wrong_final_cap(self) -> None:
        final = _goal_verification_exhausted()
        bad_args = final.recommendation_args.model_copy(update={"max_price": 99})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"recommendation_args": bad_args}))

    def test_retry_exhaustion_validates_prior_selected_provenance(self) -> None:
        requirements = (
            ShoppingRequirement(requirement_id="dock", category="Dock"),
            ShoppingRequirement(
                requirement_id="hub",
                category="Hubs",
                required_features=("HDMI",),
            ),
        )
        initial = _state(
            requirements,
            _allocation(200, ("dock", 30), ("hub", 100), unallocated=70),
        )
        final = _run(
            initial,
            _PartialConflictTool(),
            evidence_service=AttemptEvidenceService(success_on_expanded_pool=False),
        )
        self.assertEqual(
            PROJECTOR.project(final).decision.reason,
            ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
        )
        provenance = final.selected_recommendation_budget_provenance[0]
        corrupted = provenance.model_copy(update={"allocated_budget": 29})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={
                "selected_recommendation_budget_provenance": (corrupted,),
            }))


class GoalAllocationExhaustedProjectionTests(unittest.TestCase):
    def test_valid_exhaustion_projects_without_recommendation_claims(self) -> None:
        final = _goal_allocation_exhausted()
        self.assertIsNone(final.recommendation_args)
        self.assertIsNone(final.last_tool_result)
        self.assertIsNone(final.current_recommendation_budget_provenance)
        self.assertIsNone(final.current_evidence)
        self.assertIsNone(final.current_verification)
        context = PROJECTOR.project(final)
        self.assertEqual(
            context.decision.reason,
            ConflictReason.GOAL_ALLOCATION_EXHAUSTED,
        )
        self.assertEqual(context.decision.candidate_pool_sizes, ())
        self.assertEqual(context.decision.replan_attempts_performed, 0)
        self.assertIsNone(context.decision.verification_status_counts)
        self.assertTrue(context.decision.user_action_required)

    def test_executable_budget_cannot_forge_exhaustion(self) -> None:
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(_goal_allocation_exhausted(executable=True))

    def test_legacy_state_cannot_claim_goal_allocation_exhaustion(self) -> None:
        goal = _goal_allocation_exhausted()
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(goal.model_copy(update={"goal_budget_allocation": None}))

    def test_invalid_goal_context_is_not_misclassified_as_exhaustion(self) -> None:
        final = _goal_allocation_exhausted()
        invalid = final.goal_budget_allocation.model_copy(update={"allocations": ()})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"goal_budget_allocation": invalid}))

    def test_prior_selected_provenance_is_validated(self) -> None:
        requirements = (
            ShoppingRequirement(requirement_id="dock", category="Dock"),
            ShoppingRequirement(requirement_id="mouse", category="Mouse"),
        )
        initial = _state(
            requirements,
            _allocation(100, ("dock", 30), ("mouse", 0), unallocated=70),
        )
        final = _run(initial, FakeRecommendationTool({"Dock": 20, "Mouse": 5}))
        self.assertEqual(final.route, WorkflowRoute.CONFLICT)
        self.assertEqual(len(final.agent_state.shopping_plan.selected_items), 1)
        self.assertEqual(
            PROJECTOR.project(final).decision.reason,
            ConflictReason.GOAL_ALLOCATION_EXHAUSTED,
        )
        provenance = final.selected_recommendation_budget_provenance[0]
        corrupted = provenance.model_copy(update={"allocated_budget": 29})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={
                "selected_recommendation_budget_provenance": (corrupted,),
            }))


class LegacyResponseCompatibilityTests(unittest.TestCase):
    def test_legacy_ready_and_conflicts_need_no_budget_provenance(self) -> None:
        requirement = ShoppingRequirement(
            requirement_id="mouse",
            category="Mouse",
            max_budget=50,
        )
        initial = _state(
            (requirement,),
            _allocation(50, ("mouse", 50)),
        ).model_copy(update={"goal_budget_allocation": None})
        ready = _run(initial, FakeRecommendationTool({"Mouse": 30}))
        self.assertEqual(PROJECTOR.project(ready).products[0].price, 30)

        conflict = _run(
            initial,
            FakeRecommendationTool({"Mouse": 30}, empty_category="Mouse"),
        )
        self.assertEqual(
            PROJECTOR.project(conflict).decision.reason,
            ConflictReason.NO_RECOMMENDATION_CANDIDATES,
        )
        bad_args = conflict.recommendation_args.model_copy(update={"max_price": 49})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(conflict.model_copy(update={"recommendation_args": bad_args}))


if __name__ == "__main__":
    unittest.main()
