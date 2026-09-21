"""AgentTaskRunner tests for isolated decision-plan metadata generation."""

from __future__ import annotations

import unittest

from src.agentrec.decision import (
    DecisionExecutionPlan,
    DecisionExecutor,
    DeterministicDecisionPolicy,
)
from src.agentrec.integration import AgentExecutionStatus, AgentTaskRunner, RequirementInput
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


class _FailingPolicy:
    def decide(self, context):
        raise RuntimeError("private policy failure")


class _FailingExecutor:
    def execute(self, directive):
        raise RuntimeError("private executor failure")


def build_runner(*, decision_policy=None, decision_executor=None) -> AgentTaskRunner:
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


class DecisionIntegrationTests(unittest.TestCase):
    def test_no_decision_dependency_keeps_old_behavior(self) -> None:
        result = run(build_runner())
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertIsNone(result.decision_plan)
        self.assertIsNotNone(result.workflow_state)

    def test_decision_enabled_produces_plan(self) -> None:
        result = run(build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        self.assertIsInstance(result.decision_plan, DecisionExecutionPlan)
        self.assertEqual(result.status, AgentExecutionStatus.READY)

    def test_recommendation_plan_mapping_is_correct(self) -> None:
        result = run(build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        self.assertEqual(
            result.decision_plan,
            DecisionExecutionPlan(
                retrieve_products=True,
                verify_evidence=True,
                generate_response=True,
            ),
        )

    def test_policy_failure_is_isolated(self) -> None:
        result = run(build_runner(
            decision_policy=_FailingPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        self.assertIsNone(result.decision_plan)
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertIsNotNone(result.final_response)

    def test_executor_failure_is_isolated(self) -> None:
        result = run(build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=_FailingExecutor(),
        ))
        self.assertIsNone(result.decision_plan)
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertIsNotNone(result.final_response)

    def test_workflow_state_has_no_decision_fields(self) -> None:
        result = run(build_runner(
            decision_policy=DeterministicDecisionPolicy(),
            decision_executor=DecisionExecutor(),
        ))
        fields = type(result.workflow_state).model_fields
        self.assertTrue(
            {"decision", "decision_plan", "decision_directive"}.isdisjoint(fields)
        )


if __name__ == "__main__":
    unittest.main()
