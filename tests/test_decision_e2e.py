"""Deterministic end-to-end validation for AgentTaskRunner decision metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.agentrec.decision import (
    AgentIntent,
    DecisionAction,
    DecisionContext,
    DecisionExecutor,
    DeterministicDecisionPolicy,
)
from src.agentrec.integration import AgentTaskRunner, RequirementInput
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


FORBIDDEN_RESPONSE_TERMS = (
    "embedding",
    "score",
    "similarity",
    "artifact",
    "backend",
    "model",
)


class _FailingPolicy:
    def decide(self, context):
        raise RuntimeError("private decision failure")


def resolve_context(context: DecisionContext):
    policy = DeterministicDecisionPolicy()
    return DecisionExecutor().execute(policy.decide(context))


def confirmed_memory_store() -> MemoryStore:
    store = MemoryStore()
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    store.add(PreferenceItem(
        id="preference-1",
        user_id="user",
        scope=MemoryScope.USER_PROFILE,
        type=PreferenceType.FEATURE,
        value="USB-C",
        status=PreferenceStatus.CONFIRMED,
        created_at=now,
        updated_at=now,
    ))
    return store


def build_runner(
    *,
    decision_policy,
    decision_executor,
    memory_store=None,
    memory_merger=None,
) -> AgentTaskRunner:
    extraction = RequirementExtractionDecision(
        category="Dock",
        quantity=1,
        max_budget=200,
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
        decision_policy=decision_policy,
        decision_executor=decision_executor,
    )


def run(runner: AgentTaskRunner):
    return runner.run(
        user_id="user",
        session_id="session",
        plan_id="plan",
        total_budget=500,
        requirements=(
            RequirementInput(requirement_id="dock", user_request="dock with HDMI"),
        ),
    )


class DecisionE2ETests(unittest.TestCase):
    def test_recommendation_e2e_without_memory(self) -> None:
        result = run(build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        self.assertFalse(result.decision_plan.use_memory)
        self.assertTrue(result.decision_plan.retrieve_products)
        self.assertTrue(result.decision_plan.verify_evidence)
        self.assertTrue(result.decision_plan.generate_response)

    def test_recommendation_e2e_with_memory(self) -> None:
        runner = build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
            memory_store=confirmed_memory_store(),
            memory_merger=RequirementMemoryMerger(),
        )

        result = run(runner)

        self.assertTrue(result.decision_plan.use_memory)
        self.assertTrue(result.decision_plan.retrieve_products)
        self.assertTrue(result.decision_plan.verify_evidence)
        self.assertTrue(result.decision_plan.generate_response)

    def test_information_e2e(self) -> None:
        plan = resolve_context(DecisionContext(
            intent=AgentIntent.INFORMATION,
            memory_available=False,
            requirement_complete=True,
            candidate_available=False,
            verification_required=False,
            response_required=True,
        ))
        self.assertTrue(plan.generate_response)
        self.assertTrue(plan.stop)

    def test_clarification_e2e(self) -> None:
        plan = resolve_context(DecisionContext(
            intent=AgentIntent.RECOMMENDATION,
            memory_available=True,
            requirement_complete=False,
            candidate_available=False,
            verification_required=False,
            response_required=True,
        ))
        self.assertTrue(plan.ask_clarification)
        self.assertTrue(plan.stop)

    def test_repeated_execution_is_deterministic(self) -> None:
        runner = build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
        )
        first = run(runner)
        second = run(runner)
        self.assertEqual(first.decision_plan, second.decision_plan)

    def test_policy_failure_isolation_keeps_workflow_running(self) -> None:
        result = run(build_runner(
            decision_policy=_FailingPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        self.assertIsNone(result.decision_plan)
        self.assertIsNotNone(result.workflow_state)
        self.assertIsNotNone(result.final_response)

    def test_final_response_does_not_leak_internal_terms(self) -> None:
        result = run(build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        text = result.final_response.text.casefold()
        for term in FORBIDDEN_RESPONSE_TERMS:
            self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
