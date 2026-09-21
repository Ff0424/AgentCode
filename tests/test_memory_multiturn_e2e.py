"""V2-09.7.5 deterministic multi-turn memory pipeline validation."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.agentrec.integration import AgentExecutionStatus, AgentTaskRunner, RequirementInput
from src.agentrec.memory import (
    MemoryScope,
    MemoryStore,
    PreferenceItem,
    PreferenceStatus,
    PreferenceType,
    RequirementMemoryMerger,
)
from src.agentrec.planning import (
    FakePlanner,
    FakeRequirementExtractor,
    RequirementExtractionDecision,
    SelectCandidateDecision,
    SelectRequirementDecision,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.verification import EvidenceConstraintVerifier
from tests.fake_evidence import FakeEvidenceService
from tests.test_shopping_workflow import FakeRecommendationTool


NOW = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)


def memory_item(
    preference_type: PreferenceType,
    value: str,
    status: PreferenceStatus,
) -> PreferenceItem:
    return PreferenceItem(
        id=f"{preference_type.value}-{status.value}",
        user_id="user-1",
        scope=MemoryScope.USER_PROFILE,
        type=preference_type,
        value=value,
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


class ObservingMerger:
    """Observe calls while delegating all merge semantics to production code."""

    def __init__(self) -> None:
        self.delegate = RequirementMemoryMerger()
        self.calls = []

    def merge(self, requirement, preferences):
        self.calls.append((requirement, preferences))
        return self.delegate.merge(requirement, preferences)


class FailingStore:
    def get_confirmed_preferences(self, user_id):
        raise RuntimeError("private memory infrastructure detail")


def build_runner(
    *,
    max_budget: float | None = None,
    memory_store=None,
    memory_merger=None,
) -> AgentTaskRunner:
    extraction = RequirementExtractionDecision(
        category="Dock",
        quantity=1,
        max_budget=max_budget,
        required_features=("HDMI",),
        soft_preferences=(),
        priority=3,
        clarification_needed=False,
        reason="deterministic turn-two extraction",
    )
    planner = FakePlanner(decisions_by_key={
        "select_requirement:0": SelectRequirementDecision(
            plan_id="memory-e2e",
            plan_version=0,
            requirement_id="dock",
            reason="select dock requirement",
        ),
        "select_candidate:0:dock": SelectCandidateDecision(
            plan_id="memory-e2e",
            plan_version=0,
            requirement_id="dock",
            parent_asin="P-DOCK",
            reason="select first eligible dock",
        ),
    })
    return AgentTaskRunner(
        requirement_extractor=FakeRequirementExtractor(extraction),
        planner=planner,
        recommendation_tool=FakeRecommendationTool({"Dock": 120}),
        evidence_service=FakeEvidenceService(),
        verification_service=EvidenceConstraintVerifier(),
        shopping_plan_service=ShoppingPlanService(),
        memory_store=memory_store,
        memory_merger=memory_merger,
    )


def execute(runner: AgentTaskRunner):
    return runner.run(
        user_id="user-1",
        session_id="session-1",
        plan_id="memory-e2e",
        total_budget=500,
        requirements=(RequirementInput(
            requirement_id="dock",
            user_request="recommend a dock",
        ),),
    )


def final_requirement(result):
    return result.workflow_state.agent_state.shopping_plan.requirements[0]


class MemoryMultiTurnE2ETests(unittest.TestCase):
    def test_confirmed_preference_flows_through_store_merger_and_runner(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            PreferenceType.FEATURE, "USB-C", PreferenceStatus.CONFIRMED
        ))
        merger = ObservingMerger()
        result = execute(build_runner(memory_store=store, memory_merger=merger))
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(final_requirement(result).required_features, ("HDMI", "USB-C"))
        self.assertEqual(len(merger.calls), 1)
        self.assertEqual(
            tuple(value.status for value in merger.calls[0][1]),
            (PreferenceStatus.CONFIRMED,),
        )

    def test_candidate_preference_does_not_affect_next_turn(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            PreferenceType.FEATURE, "USB-C", PreferenceStatus.CANDIDATE
        ))
        merger = ObservingMerger()
        result = execute(build_runner(memory_store=store, memory_merger=merger))
        self.assertEqual(final_requirement(result).required_features, ("HDMI",))
        self.assertEqual(len(merger.calls), 1)
        self.assertEqual(merger.calls[0][1], ())

    def test_rejected_preference_does_not_affect_next_turn(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            PreferenceType.FEATURE, "USB-C", PreferenceStatus.REJECTED
        ))
        merger = ObservingMerger()
        result = execute(build_runner(memory_store=store, memory_merger=merger))
        self.assertEqual(final_requirement(result).required_features, ("HDMI",))
        self.assertEqual(merger.calls[0][1], ())

    def test_current_turn_budget_overrides_confirmed_memory(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            PreferenceType.BUDGET, "500", PreferenceStatus.CONFIRMED
        ))
        result = execute(build_runner(
            max_budget=200,
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(final_requirement(result).max_budget, 200)

    def test_memory_failure_isolated_from_workflow_and_response(self) -> None:
        baseline = execute(build_runner())
        failed = execute(build_runner(
            memory_store=FailingStore(),
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(failed.memory_error, "memory_store_failed")
        self.assertEqual(failed.status, baseline.status)
        self.assertEqual(final_requirement(failed), final_requirement(baseline))
        self.assertEqual(failed.final_response, baseline.final_response)
        self.assertIsNotNone(failed.workflow_state)
        self.assertNotEqual(failed.status, AgentExecutionStatus.ERROR)

    def test_repeated_multi_turn_execution_is_deterministic_and_safe(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            PreferenceType.FEATURE, "USB-C", PreferenceStatus.CONFIRMED
        ))
        runner = build_runner(
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        )
        first = execute(runner)
        second = execute(runner)
        self.assertEqual(final_requirement(first), final_requirement(second))
        self.assertEqual(first.final_response, second.final_response)
        self.assertEqual(first.workflow_state, second.workflow_state)
        self.assertTrue(
            {"memory", "preferences", "memory_error"}.isdisjoint(
                first.workflow_state.model_fields
            )
        )
        public_text = first.final_response.text.casefold()
        for forbidden in ("embedding", "artifact", "similarity", "score"):
            self.assertNotIn(forbidden, public_text)


if __name__ == "__main__":
    unittest.main()
