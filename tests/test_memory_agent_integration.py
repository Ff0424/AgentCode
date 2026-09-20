"""AgentTaskRunner tests for optional fail-open memory augmentation."""

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
    preference_id: str,
    preference_type: PreferenceType,
    value: str,
    *,
    status: PreferenceStatus = PreferenceStatus.CONFIRMED,
) -> PreferenceItem:
    return PreferenceItem(
        id=preference_id,
        user_id="user-1",
        scope=MemoryScope.USER_PROFILE,
        type=preference_type,
        value=value,
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def build_runner(
    *,
    max_budget: float | None = 200,
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
        reason="deterministic extraction",
    )
    planner = FakePlanner(decisions_by_key={
        "select_requirement:0": SelectRequirementDecision(
            plan_id="plan",
            plan_version=0,
            requirement_id="dock",
            reason="select requirement",
        ),
        "select_candidate:0:dock": SelectCandidateDecision(
            plan_id="plan",
            plan_version=0,
            requirement_id="dock",
            parent_asin="P-DOCK",
            reason="select candidate",
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


def run(runner: AgentTaskRunner):
    return runner.run(
        user_id="user-1",
        session_id="session-1",
        plan_id="plan",
        total_budget=500,
        requirements=(RequirementInput(
            requirement_id="dock",
            user_request="dock with HDMI",
        ),),
    )


def final_requirement(result):
    return result.workflow_state.agent_state.shopping_plan.requirements[0]


class _FailingMemoryStore:
    def get_confirmed_preferences(self, user_id):
        raise RuntimeError("private store failure details")


class _FailingMemoryMerger:
    def merge(self, requirement, preferences):
        raise RuntimeError("private merger failure details")


class MemoryAgentIntegrationTests(unittest.TestCase):
    def test_no_memory_and_empty_memory_keep_existing_behavior(self) -> None:
        without_memory = run(build_runner())
        empty_store = MemoryStore()
        with_empty_memory = run(build_runner(
            memory_store=empty_store,
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(without_memory.status, AgentExecutionStatus.READY)
        self.assertEqual(final_requirement(without_memory), final_requirement(with_empty_memory))
        self.assertIsNone(with_empty_memory.memory_error)

    def test_confirmed_feature_is_injected_before_workflow(self) -> None:
        store = MemoryStore()
        store.add(memory_item("feature", PreferenceType.FEATURE, "USB-C"))
        result = run(build_runner(
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(
            final_requirement(result).required_features,
            ("HDMI", "USB-C"),
        )
        self.assertIsNone(result.memory_error)

    def test_candidate_memory_is_not_injected(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            "feature",
            PreferenceType.FEATURE,
            "USB-C",
            status=PreferenceStatus.CANDIDATE,
        ))
        result = run(build_runner(
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(final_requirement(result).required_features, ("HDMI",))

    def test_rejected_memory_is_not_injected(self) -> None:
        store = MemoryStore()
        store.add(memory_item(
            "feature",
            PreferenceType.FEATURE,
            "USB-C",
            status=PreferenceStatus.REJECTED,
        ))
        result = run(build_runner(
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(final_requirement(result).required_features, ("HDMI",))

    def test_current_requirement_budget_has_priority(self) -> None:
        store = MemoryStore()
        store.add(memory_item("budget", PreferenceType.BUDGET, "500"))
        result = run(build_runner(
            max_budget=200,
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(final_requirement(result).max_budget, 200)

    def test_memory_store_failure_is_isolated(self) -> None:
        result = run(build_runner(
            memory_store=_FailingMemoryStore(),
            memory_merger=RequirementMemoryMerger(),
        ))
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(final_requirement(result).required_features, ("HDMI",))
        self.assertEqual(result.memory_error, "memory_store_failed")
        self.assertIsNotNone(result.final_response)

    def test_memory_merger_failure_is_isolated(self) -> None:
        result = run(build_runner(
            memory_store=MemoryStore(),
            memory_merger=_FailingMemoryMerger(),
        ))
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(final_requirement(result).required_features, ("HDMI",))
        self.assertEqual(result.memory_error, "memory_merge_failed")
        self.assertIsNotNone(result.final_response)

    def test_memory_does_not_add_workflow_state_fields(self) -> None:
        store = MemoryStore()
        store.add(memory_item("feature", PreferenceType.FEATURE, "USB-C"))
        result = run(build_runner(
            memory_store=store,
            memory_merger=RequirementMemoryMerger(),
        ))
        fields = set(result.workflow_state.model_fields)
        self.assertTrue({"memory", "preferences", "memory_error"}.isdisjoint(fields))
        self.assertEqual(result.workflow_state.route.value, "ready")


if __name__ == "__main__":
    unittest.main()
