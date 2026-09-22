"""Focused tests for V2-10.5b.1 goal execution preparation."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from src.agentrec.domain import AgentState, ShoppingRequirement
from src.agentrec.integration import (
    AgentTaskRunner,
    GoalExecutionResult,
    GoalExecutionStatus,
    PreparedGoalExecution,
)
from src.agentrec.memory import (
    MemoryScope,
    MemoryStore,
    PreferenceItem,
    PreferenceStatus,
    PreferenceType,
    RequirementMemoryMerger,
)
from src.agentrec.planning import (
    AllocationPreferenceType,
    FakeGoalExtractor,
    GoalAllocationPreference,
    GoalBudgetAllocation,
    GoalRequirementProjection,
    GoalRequirementProposal,
    RequirementBudgetAllocation,
    ShoppingGoalExtractionDecision,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.workflows import ShoppingWorkflowState


class _Unused:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _PlanServiceSpy(ShoppingPlanService):
    def __init__(self) -> None:
        self.create_calls = 0

    def create_plan(self, **kwargs):
        self.create_calls += 1
        return super().create_plan(**kwargs)


class _CallRejector:
    def __init__(self, method: str) -> None:
        self.calls = 0
        setattr(self, method, self._reject)

    def _reject(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("dependency must not be called")


class _FailingAllocator:
    def allocate(self, projection):
        raise RuntimeError("allocation failed")


class _BadAllocator:
    def __init__(self, allocation: GoalBudgetAllocation) -> None:
        self.allocation = allocation

    def allocate(self, projection):
        return self.allocation


class _CountingStore:
    def __init__(self, preferences=(), *, fail=False) -> None:
        self.preferences = preferences
        self.fail = fail
        self.calls = 0

    def get_confirmed_preferences(self, user_id):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private store failure")
        return self.preferences


class _CountingMerger:
    def __init__(self, *, fail=False) -> None:
        self.delegate = RequirementMemoryMerger()
        self.fail = fail
        self.calls = 0

    def merge(self, requirement, preferences):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private merge failure")
        return self.delegate.merge(requirement, preferences)


def golden_decision() -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        total_budget=500,
        requirement_proposals=(
            GoalRequirementProposal(
                category="Docking Stations", required_features=("HDMI",)
            ),
            GoalRequirementProposal(category="Mouse"),
            GoalRequirementProposal(category="Headphones"),
        ),
        allocation_preferences=(
            GoalAllocationPreference(
                target_index=1, preference=AllocationPreferenceType.SAVE_MORE
            ),
            GoalAllocationPreference(
                target_index=2, preference=AllocationPreferenceType.ALLOCATE_MORE
            ),
        ),
    )


def single_decision(*, quantity: int = 1) -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        total_budget=500,
        requirement_proposals=(
            GoalRequirementProposal(
                category="Docking Stations",
                quantity=quantity,
                required_features=("HDMI",),
            ),
        ),
    )


def clarification_decision() -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        requirement_proposals=(GoalRequirementProposal(category="Dock"),),
        clarification_needed=True,
        clarification_question="What is your total budget?",
    )


def preference() -> PreferenceItem:
    now = datetime(2026, 9, 22, tzinfo=timezone.utc)
    return PreferenceItem(
        id="pref-1",
        user_id="user-1",
        scope=MemoryScope.USER_PROFILE,
        type=PreferenceType.FEATURE,
        value="USB-C",
        status=PreferenceStatus.CONFIRMED,
        created_at=now,
        updated_at=now,
    )


def runner(
    decision: ShoppingGoalExtractionDecision | None = None,
    *,
    goal_extractor=True,
    goal_projector=None,
    budget_allocator=None,
    plan_service=None,
    memory_store=None,
    memory_merger=None,
) -> AgentTaskRunner:
    return AgentTaskRunner(
        requirement_extractor=_Unused(),
        planner=_Unused(),
        recommendation_tool=_Unused(),
        evidence_service=_Unused(),
        verification_service=_Unused(),
        shopping_plan_service=plan_service or ShoppingPlanService(),
        goal_extractor=(
            FakeGoalExtractor(decision or golden_decision())
            if goal_extractor
            else None
        ),
        goal_projector=goal_projector,
        budget_allocator=budget_allocator,
        memory_store=memory_store,
        memory_merger=memory_merger,
    )


def prepare(value: AgentTaskRunner) -> GoalExecutionResult:
    return value.run_goal(
        user_id="user-1",
        session_id="session-1",
        plan_id="plan-1",
        user_request="prepare my trip",
    )


class GoalExecutionPreparationTests(unittest.TestCase):
    def test_golden_multi_goal_prepares_expected_allocation(self) -> None:
        result = prepare(runner())
        prepared = result.prepared_execution
        self.assertEqual(result.status, GoalExecutionStatus.PREPARED)
        self.assertEqual(
            tuple(value.allocated_budget for value in prepared.budget_allocation.allocations),
            (166.67, 83.33, 250.0),
        )
        plan = prepared.initial_workflow_state.agent_state.shopping_plan
        self.assertEqual(plan.requirements, prepared.projection.requirements)
        self.assertEqual(plan.total_budget, 500)
        self.assertEqual(plan.version, 0)

    def test_goal_decision_is_only_total_budget_source(self) -> None:
        decision = single_decision().model_copy(update={"total_budget": 321.45})
        result = prepare(runner(decision))
        prepared = result.prepared_execution
        self.assertEqual(prepared.projection.total_budget, 321.45)
        self.assertEqual(
            prepared.initial_workflow_state.agent_state.shopping_plan.total_budget,
            321.45,
        )

    def test_clarification_short_circuits_all_preparation_dependencies(self) -> None:
        projector = _CallRejector("project")
        allocator = _CallRejector("allocate")
        plan_service = _PlanServiceSpy()
        store = _CountingStore()
        merger = _CountingMerger()
        result = prepare(runner(
            clarification_decision(),
            goal_projector=projector,
            budget_allocator=allocator,
            plan_service=plan_service,
            memory_store=store,
            memory_merger=merger,
        ))
        self.assertEqual(result.status, GoalExecutionStatus.CLARIFICATION_REQUIRED)
        self.assertEqual(result.clarification_question, "What is your total budget?")
        self.assertIsNone(result.prepared_execution)
        self.assertEqual(
            (projector.calls, allocator.calls, plan_service.create_calls, store.calls, merger.calls),
            (0, 0, 0, 0, 0),
        )

    def test_multi_requirement_skips_memory_completely(self) -> None:
        store = _CountingStore((preference(),))
        merger = _CountingMerger()
        result = prepare(runner(
            memory_store=store,
            memory_merger=merger,
        ))
        self.assertEqual((store.calls, merger.calls), (0, 0))
        self.assertIsNone(result.memory_error)
        self.assertEqual(
            result.prepared_execution.projection.requirements[0].required_features,
            ("HDMI",),
        )

    def test_single_requirement_confirmed_memory_is_merged(self) -> None:
        store = _CountingStore((preference(),))
        merger = _CountingMerger()
        result = prepare(runner(
            single_decision(), memory_store=store, memory_merger=merger
        ))
        projection = result.prepared_execution.projection
        self.assertEqual((store.calls, merger.calls), (1, 1))
        self.assertEqual(
            projection.requirements[0].required_features,
            ("HDMI", "USB-C"),
        )
        self.assertEqual(projection.total_budget, 500)
        self.assertEqual(projection.requirements[0].requirement_id, "req-001")

    def test_memory_store_and_merger_fail_open_with_safe_codes(self) -> None:
        cases = (
            (_CountingStore(fail=True), _CountingMerger(), "memory_store_failed"),
            (_CountingStore(), _CountingMerger(fail=True), "memory_merge_failed"),
        )
        for store, merger, code in cases:
            with self.subTest(code=code):
                result = prepare(runner(
                    single_decision(), memory_store=store, memory_merger=merger
                ))
                self.assertEqual(result.status, GoalExecutionStatus.PREPARED)
                self.assertEqual(result.memory_error, code)
                self.assertEqual(
                    result.prepared_execution.projection.requirements[0].required_features,
                    ("HDMI",),
                )

    def test_allocator_failure_occurs_before_plan_creation(self) -> None:
        service = _PlanServiceSpy()
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            prepare(runner(
                single_decision(),
                budget_allocator=_FailingAllocator(),
                plan_service=service,
            ))
        self.assertEqual(service.create_calls, 0)

    def test_cross_contract_id_order_and_total_are_rejected_before_plan(self) -> None:
        bad_allocations = (
            GoalBudgetAllocation(
                total_budget=500,
                allocations=(RequirementBudgetAllocation(
                    requirement_id="wrong", allocated_budget=500
                ),),
                unallocated_budget=0,
            ),
            GoalBudgetAllocation(
                total_budget=501,
                allocations=(RequirementBudgetAllocation(
                    requirement_id="req-001", allocated_budget=501
                ),),
                unallocated_budget=0,
            ),
        )
        for allocation in bad_allocations:
            service = _PlanServiceSpy()
            with self.subTest(allocation=allocation), self.assertRaises(ValueError):
                prepare(runner(
                    single_decision(),
                    budget_allocator=_BadAllocator(allocation),
                    plan_service=service,
                ))
            self.assertEqual(service.create_calls, 0)

        reversed_allocation = GoalBudgetAllocation(
            total_budget=500,
            allocations=(
                RequirementBudgetAllocation(
                    requirement_id="req-002", allocated_budget=250
                ),
                RequirementBudgetAllocation(
                    requirement_id="req-001", allocated_budget=250
                ),
            ),
            unallocated_budget=0,
        )
        two_requirements = ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(
                GoalRequirementProposal(category="Dock"),
                GoalRequirementProposal(category="Mouse"),
            ),
        )
        service = _PlanServiceSpy()
        with self.assertRaises(ValueError):
            prepare(runner(
                two_requirements,
                budget_allocator=_BadAllocator(reversed_allocation),
                plan_service=service,
            ))
        self.assertEqual(service.create_calls, 0)

    def test_prepared_state_has_no_fake_downstream_output_and_round_trips(self) -> None:
        result = prepare(runner())
        state = result.prepared_execution.initial_workflow_state
        self.assertIsNone(state.route)
        self.assertIsNone(state.recommendation_args)
        self.assertIsNone(state.last_tool_result)
        self.assertIsNone(state.current_evidence)
        self.assertIsNone(state.current_verification)
        restored = GoalExecutionResult.model_validate_json(result.model_dump_json())
        self.assertEqual(restored, result)

    def test_repeated_preparation_is_deterministic(self) -> None:
        value = runner()
        self.assertEqual(prepare(value), prepare(value))

    def test_quantity_is_preserved_without_per_unit_calculation(self) -> None:
        result = prepare(runner(single_decision(quantity=3)))
        prepared = result.prepared_execution
        self.assertEqual(prepared.projection.requirements[0].quantity, 3)
        self.assertEqual(prepared.budget_allocation.allocations[0].allocated_budget, 500)
        self.assertIsNone(prepared.projection.requirements[0].max_budget)

    def test_missing_goal_extractor_only_blocks_run_goal(self) -> None:
        value = runner(goal_extractor=False)
        with self.assertRaisesRegex(RuntimeError, "goal_extractor"):
            prepare(value)

    def test_goal_result_cross_field_rules(self) -> None:
        decision = clarification_decision()
        with self.assertRaises(ValidationError):
            GoalExecutionResult(
                status=GoalExecutionStatus.CLARIFICATION_REQUIRED,
                goal_decision=decision,
            )
        with self.assertRaises(ValidationError):
            GoalExecutionResult(
                status=GoalExecutionStatus.PREPARED,
                goal_decision=golden_decision(),
            )

        valid = prepare(runner(single_decision()))
        with self.assertRaises(ValidationError):
            valid.status = GoalExecutionStatus.CLARIFICATION_REQUIRED
        with self.assertRaises(ValidationError):
            GoalExecutionResult(
                status=valid.status,
                goal_decision=valid.goal_decision,
                prepared_execution=valid.prepared_execution,
                unsupported=True,
            )
        with self.assertRaises(ValidationError):
            GoalExecutionResult(
                status=valid.status,
                goal_decision=valid.goal_decision,
                prepared_execution=valid.prepared_execution,
                memory_error="private_exception_text",
            )

    def test_workflow_state_allocation_validation_and_legacy_default(self) -> None:
        result = prepare(runner(single_decision()))
        state = result.prepared_execution.initial_workflow_state
        legacy = ShoppingWorkflowState(agent_state=state.agent_state)
        self.assertIsNone(legacy.goal_budget_allocation)

        wrong_id = GoalBudgetAllocation(
            total_budget=500,
            allocations=(RequirementBudgetAllocation(
                requirement_id="wrong", allocated_budget=500
            ),),
            unallocated_budget=0,
        )
        with self.assertRaises(ValidationError):
            ShoppingWorkflowState(
                agent_state=state.agent_state,
                goal_budget_allocation=wrong_id,
            )
        wrong_total = GoalBudgetAllocation(
            total_budget=501,
            allocations=(RequirementBudgetAllocation(
                requirement_id="req-001", allocated_budget=501
            ),),
            unallocated_budget=0,
        )
        with self.assertRaises(ValidationError):
            ShoppingWorkflowState(
                agent_state=state.agent_state,
                goal_budget_allocation=wrong_total,
            )

    def test_prepared_contract_rejects_plan_projection_mismatch(self) -> None:
        result = prepare(runner(single_decision()))
        prepared = result.prepared_execution
        changed = ShoppingRequirement(
            requirement_id="req-001", category="Different"
        )
        plan = ShoppingPlanService().create_plan(
            plan_id="plan-1",
            user_id="user-1",
            currency="USD",
            total_budget=500,
            requirements=(changed,),
        )
        state = ShoppingWorkflowState(
            agent_state=AgentState(
                user_id="user-1", session_id="session-1", shopping_plan=plan
            ),
            goal_budget_allocation=prepared.budget_allocation,
        )
        with self.assertRaises(ValidationError):
            PreparedGoalExecution(
                projection=prepared.projection,
                budget_allocation=prepared.budget_allocation,
                initial_workflow_state=state,
                recursion_limit=50,
            )


if __name__ == "__main__":
    unittest.main()
