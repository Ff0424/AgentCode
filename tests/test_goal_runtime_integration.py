"""Deterministic integration tests for V2-10.5c.2 Goal runtime execution."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.domain import AgentState, ShoppingRequirement
from src.agentrec.integration import (
    AgentTaskRunner,
    GoalExecutionResult,
    GoalExecutionStatus,
)
from src.agentrec.planning import (
    AllocationPreferenceType,
    FakePlanner,
    FakeGoalExtractor,
    GoalAllocationPreference,
    GoalBudgetAllocation,
    GoalRequirementProposal,
    RequirementBudgetAllocation,
    SelectCandidateDecision,
    SelectRequirementDecision,
    ShoppingGoalExtractionDecision,
)
from src.agentrec.response import (
    ConflictReason,
    DeterministicFinalResponseRenderer,
    GroundedResponseProjector,
    ResponseErrorCode,
    ResponseKind,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.verification import EvidenceConstraintVerifier
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute
from tests.fake_evidence import FakeEvidenceService
from tests.test_bounded_replan_workflow import AttemptEvidenceService, PoolAwareTool
from tests.test_shopping_workflow import FakeRecommendationTool


def golden_goal() -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        total_budget=500,
        requirement_proposals=(
            GoalRequirementProposal(category="Dock", required_features=("HDMI",)),
            GoalRequirementProposal(category="Mouse"),
            GoalRequirementProposal(category="Headphones"),
        ),
        allocation_preferences=(
            GoalAllocationPreference(
                target_index=1,
                preference=AllocationPreferenceType.SAVE_MORE,
            ),
            GoalAllocationPreference(
                target_index=2,
                preference=AllocationPreferenceType.ALLOCATE_MORE,
            ),
        ),
    )


def single_goal(
    *,
    category: str = "Dock",
    total_budget: float = 100,
    quantity: int = 1,
    required_features: tuple[str, ...] = ("HDMI",),
) -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        total_budget=total_budget,
        requirement_proposals=(GoalRequirementProposal(
            category=category,
            quantity=quantity,
            required_features=required_features,
        ),),
    )


def clarification_goal() -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        requirement_proposals=(GoalRequirementProposal(category="Dock"),),
        clarification_needed=True,
        clarification_question="What is your total budget?",
    )


class _UnusedExtractor:
    def extract(self, **_kwargs):
        raise AssertionError("legacy requirement extractor must not be called")


class _CountingRecommendationTool(FakeRecommendationTool):
    pass


class _RejectingWorkflowRunner:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("workflow must not execute")


class _ReturningWorkflowRunner:
    def __init__(self, route: WorkflowRoute) -> None:
        self.route = route
        self.calls = []

    def execute(self, initial_state, *, recursion_limit=50):
        self.calls.append((initial_state, recursion_limit))
        return initial_state.model_copy(update={"route": self.route})


class _RaisingWorkflowRunner:
    def execute(self, *_args, **_kwargs):
        raise RuntimeError("runtime boundary failed")


class _FailingProjector:
    def project(self, _state):
        raise RuntimeError("private projection failure")


class _FailingRenderer:
    def render(self, _context):
        raise RuntimeError("private rendering failure")


class _ZeroAllocator:
    def allocate(self, projection):
        return GoalBudgetAllocation(
            total_budget=projection.total_budget,
            allocations=(RequirementBudgetAllocation(
                requirement_id=projection.requirements[0].requirement_id,
                allocated_budget=0,
            ),),
            unallocated_budget=projection.total_budget,
        )


def make_runner(
    decision: ShoppingGoalExtractionDecision,
    *,
    recommendation_tool=None,
    evidence_service=None,
    workflow_runner=None,
    budget_allocator=None,
    response_projector=None,
    response_renderer=None,
) -> AgentTaskRunner:
    requirement_ids = tuple(
        f"req-{index:03d}"
        for index in range(1, len(decision.requirement_proposals) + 1)
    )
    decisions = {}
    for version, (requirement_id, proposal) in enumerate(
        zip(requirement_ids, decision.requirement_proposals, strict=True)
    ):
        decisions[f"select_requirement:{version}"] = SelectRequirementDecision(
            plan_id="goal-plan",
            plan_version=version,
            requirement_id=requirement_id,
            reason=f"process {requirement_id}",
        )
        decisions[
            f"select_candidate:{version}:{requirement_id}"
        ] = SelectCandidateDecision(
            plan_id="goal-plan",
            plan_version=version,
            requirement_id=requirement_id,
            parent_asin=f"P-{proposal.category.upper()}",
            reason=f"select {requirement_id}",
        )
    return AgentTaskRunner(
        requirement_extractor=_UnusedExtractor(),
        planner=FakePlanner(decisions_by_key=decisions),
        recommendation_tool=(
            recommendation_tool
            or FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180})
        ),
        evidence_service=evidence_service or FakeEvidenceService(),
        verification_service=EvidenceConstraintVerifier(),
        shopping_plan_service=ShoppingPlanService(),
        response_projector=response_projector,
        response_renderer=response_renderer,
        goal_extractor=FakeGoalExtractor(decision),
        budget_allocator=budget_allocator,
        workflow_runner=workflow_runner,
    )


def run_goal(runner: AgentTaskRunner) -> GoalExecutionResult:
    return runner.run_goal(
        user_id="goal-user",
        session_id="goal-session",
        plan_id="goal-plan",
        user_request="prepare a shopping bundle",
        recursion_limit=80,
    )


class GoalRuntimeIntegrationTests(unittest.TestCase):
    def test_golden_goal_executes_to_ready_with_preserved_allocation(self) -> None:
        result = run_goal(make_runner(golden_goal()))
        prepared = result.prepared_execution
        terminal = result.workflow_state
        plan = terminal.agent_state.shopping_plan

        self.assertEqual(result.status, GoalExecutionStatus.READY)
        self.assertEqual(terminal.route, WorkflowRoute.READY)
        self.assertEqual(len(plan.selected_items), 3)
        self.assertEqual(plan.total_spent, 330)
        self.assertEqual(plan.remaining_budget, 170)
        self.assertEqual(
            tuple(value.allocated_budget for value in prepared.budget_allocation.allocations),
            (166.67, 83.33, 250.0),
        )
        self.assertEqual(terminal.goal_budget_allocation, prepared.budget_allocation)
        self.assertEqual(len(terminal.selected_recommendation_budget_provenance), 3)
        self.assertEqual(result.final_response.kind, ResponseKind.READY)
        self.assertIsNone(result.response_error)

    def test_clarification_short_circuits_runtime(self) -> None:
        workflow_runner = _RejectingWorkflowRunner()
        result = run_goal(make_runner(
            clarification_goal(),
            workflow_runner=workflow_runner,
        ))
        self.assertEqual(result.status, GoalExecutionStatus.CLARIFICATION_REQUIRED)
        self.assertIsNone(result.prepared_execution)
        self.assertIsNone(result.workflow_state)
        self.assertEqual(workflow_runner.calls, 0)

    def test_no_candidate_conflict_is_rendered(self) -> None:
        tool = _CountingRecommendationTool(
            {"Dock": 50},
            empty_category="Dock",
        )
        result = run_goal(make_runner(single_goal(), recommendation_tool=tool))
        self.assertEqual(result.status, GoalExecutionStatus.CONFLICT)
        self.assertEqual(result.workflow_state.route, WorkflowRoute.CONFLICT)
        self.assertEqual(
            result.final_response.decision_summary.reason,
            ConflictReason.NO_RECOMMENDATION_CANDIDATES,
        )
        self.assertEqual(
            result.workflow_state.goal_budget_allocation,
            result.prepared_execution.budget_allocation,
        )

    def test_verification_exhaustion_preserves_budget_and_changes_only_pool(self) -> None:
        tool = PoolAwareTool()
        result = run_goal(make_runner(
            single_goal(category="Hubs", total_budget=100),
            recommendation_tool=tool,
            evidence_service=AttemptEvidenceService(success_on_expanded_pool=False),
        ))
        state = result.workflow_state
        self.assertEqual(result.status, GoalExecutionStatus.CONFLICT)
        self.assertEqual(
            result.final_response.decision_summary.reason,
            ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
        )
        self.assertEqual([value.top_k for value in tool.calls], [5, 10])
        self.assertEqual(
            [value.max_price for value in tool.calls],
            [100, 100],
        )
        self.assertEqual(
            state.goal_budget_allocation,
            result.prepared_execution.budget_allocation,
        )

    def test_quantity_uses_one_sku_with_derived_per_unit_cap(self) -> None:
        tool = _CountingRecommendationTool({"Dock": 30})
        result = run_goal(make_runner(
            single_goal(quantity=3),
            recommendation_tool=tool,
        ))
        self.assertEqual(result.status, GoalExecutionStatus.READY)
        self.assertEqual(tool.calls[0][1].max_price, 33.33)
        item = result.workflow_state.agent_state.shopping_plan.selected_items[0]
        self.assertEqual(item.quantity, 3)
        self.assertEqual(result.workflow_state.agent_state.shopping_plan.total_spent, 90)

    def test_allocation_exhaustion_does_not_call_recommendation(self) -> None:
        tool = _CountingRecommendationTool({"Dock": 50})
        result = run_goal(make_runner(
            single_goal(),
            recommendation_tool=tool,
            budget_allocator=_ZeroAllocator(),
        ))
        self.assertEqual(tool.calls, [])
        self.assertEqual(result.status, GoalExecutionStatus.CONFLICT)
        self.assertEqual(
            result.final_response.decision_summary.reason,
            ConflictReason.GOAL_ALLOCATION_EXHAUSTED,
        )
        self.assertNotIn("没有返回候选商品", result.final_response.text)

    def test_workflow_error_returns_terminal_state_without_response(self) -> None:
        workflow_runner = _ReturningWorkflowRunner(WorkflowRoute.ERROR)
        result = run_goal(make_runner(
            single_goal(),
            workflow_runner=workflow_runner,
        ))
        self.assertEqual(result.status, GoalExecutionStatus.ERROR)
        self.assertIsNotNone(result.prepared_execution)
        self.assertEqual(result.workflow_state.route, WorkflowRoute.ERROR)
        self.assertIsNone(result.final_response)
        self.assertIsNone(result.response_error)

    def test_response_failures_are_isolated(self) -> None:
        cases = (
            (_FailingProjector(), None, ResponseErrorCode.PROJECTION_FAILED),
            (GroundedResponseProjector(), _FailingRenderer(), ResponseErrorCode.RENDER_FAILED),
        )
        for projector, renderer, expected in cases:
            with self.subTest(expected=expected):
                result = run_goal(make_runner(
                    golden_goal(),
                    response_projector=projector,
                    response_renderer=renderer,
                ))
                self.assertEqual(result.status, GoalExecutionStatus.READY)
                self.assertIsNotNone(result.workflow_state)
                self.assertIsNotNone(result.prepared_execution)
                self.assertIsNone(result.final_response)
                self.assertEqual(result.response_error, expected)

        conflict = run_goal(make_runner(
            single_goal(),
            recommendation_tool=_CountingRecommendationTool(
                {"Dock": 50},
                empty_category="Dock",
            ),
            response_projector=_FailingProjector(),
        ))
        self.assertEqual(conflict.status, GoalExecutionStatus.CONFLICT)
        self.assertIsNone(conflict.final_response)
        self.assertEqual(
            conflict.response_error,
            ResponseErrorCode.PROJECTION_FAILED,
        )

    def test_prepared_snapshot_is_unchanged_by_execution(self) -> None:
        runner = make_runner(single_goal())
        prepared_result = runner.prepare_goal(
            user_id="goal-user",
            session_id="goal-session",
            plan_id="goal-plan",
            user_request="prepare a shopping bundle",
            recursion_limit=80,
        )
        prepared = prepared_result.prepared_execution
        before = prepared.model_dump(mode="json")
        runner._workflow_runner.execute(  # intentional boundary-level assertion
            prepared.initial_workflow_state,
            recursion_limit=prepared.recursion_limit,
        )
        self.assertEqual(prepared.model_dump(mode="json"), before)

    def test_goal_result_contract_rejects_terminal_mismatches(self) -> None:
        valid = run_goal(make_runner(golden_goal()))
        prepared = valid.prepared_execution
        ready = valid.workflow_state
        conflict = ready.model_copy(update={"route": WorkflowRoute.CONFLICT})

        invalid_payloads = (
            {"status": GoalExecutionStatus.READY, "workflow_state": conflict},
            {"status": GoalExecutionStatus.CONFLICT, "workflow_state": ready},
            {
                "status": GoalExecutionStatus.ERROR,
                "workflow_state": ready.model_copy(update={"route": WorkflowRoute.ERROR}),
                "final_response": valid.final_response,
            },
            {
                "status": GoalExecutionStatus.READY,
                "workflow_state": ready,
                "final_response": None,
            },
            {
                "status": GoalExecutionStatus.READY,
                "workflow_state": ready,
                "response_error": ResponseErrorCode.RENDER_FAILED,
            },
        )
        base = {
            "goal_decision": valid.goal_decision,
            "prepared_execution": prepared,
            "final_response": valid.final_response,
        }
        for changes in invalid_payloads:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                GoalExecutionResult(**(base | changes))

        changed_allocation = prepared.budget_allocation.model_copy(update={
            "allocations": tuple(reversed(prepared.budget_allocation.allocations)),
        })
        changed_plan = ready.agent_state.shopping_plan.model_copy(update={
            "plan_id": "other-plan",
        })
        changed_requirement = ready.agent_state.shopping_plan.requirements[0].model_copy(
            update={"category": "Other"}
        )
        requirements = (
            changed_requirement,
            *ready.agent_state.shopping_plan.requirements[1:],
        )
        identity_states = (
            ready.model_copy(update={"goal_budget_allocation": changed_allocation}),
            ready.model_copy(update={
                "agent_state": ready.agent_state.model_copy(update={
                    "shopping_plan": changed_plan,
                }),
            }),
            ready.model_copy(update={
                "agent_state": ready.agent_state.model_copy(update={
                    "shopping_plan": ready.agent_state.shopping_plan.model_copy(
                        update={"requirements": requirements}
                    ),
                }),
            }),
            ready.model_copy(update={
                "agent_state": ready.agent_state.model_copy(update={
                    "shopping_plan": ready.agent_state.shopping_plan.model_copy(
                        update={"version": -1}
                    ),
                }),
            }),
        )
        for workflow_state in identity_states:
            with self.subTest(workflow_state=workflow_state), self.assertRaises(
                ValidationError
            ):
                GoalExecutionResult(
                    status=GoalExecutionStatus.READY,
                    goal_decision=valid.goal_decision,
                    prepared_execution=prepared,
                    workflow_state=workflow_state,
                    final_response=valid.final_response,
                )

    def test_invalid_nonterminal_route_is_not_disguised_as_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid route"):
            run_goal(make_runner(
                single_goal(),
                workflow_runner=_ReturningWorkflowRunner(WorkflowRoute.CONTINUE),
            ))

    def test_runtime_exception_is_propagated(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "runtime boundary failed"):
            run_goal(make_runner(
                single_goal(),
                workflow_runner=_RaisingWorkflowRunner(),
            ))


if __name__ == "__main__":
    unittest.main()
