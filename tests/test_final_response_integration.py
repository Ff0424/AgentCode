"""Integration tests for deterministic final-response projection and rendering."""

from __future__ import annotations

import unittest

from src.agentrec.integration import AgentExecutionStatus, AgentTaskRunner, RequirementInput
from src.agentrec.planning import (
    FakePlanner,
    FakeRequirementExtractor,
    RequirementExtractionDecision,
    SelectCandidateDecision,
    SelectRequirementDecision,
)
from src.agentrec.response import (
    DeterministicFinalResponseRenderer,
    GroundedResponseProjector,
    ResponseErrorCode,
    ResponseKind,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.verification import EvidenceConstraintVerifier
from tests.fake_evidence import FakeEvidenceService
from tests.test_shopping_workflow import FakeRecommendationTool


def _decision(*, clarification: bool = False) -> RequirementExtractionDecision:
    return RequirementExtractionDecision(
        category="Mouse",
        quantity=1,
        max_budget=None if clarification else 50,
        required_features=(),
        soft_preferences=(),
        priority=3,
        clarification_needed=clarification,
        reason="deterministic extraction",
    )


def _planner(*, candidate: str = "P-MOUSE") -> FakePlanner:
    return FakePlanner(decisions_by_key={
        "select_requirement:0": SelectRequirementDecision(
            plan_id="plan", plan_version=0, requirement_id="mouse", reason="select"
        ),
        "select_candidate:0:mouse": SelectCandidateDecision(
            plan_id="plan", plan_version=0, requirement_id="mouse",
            parent_asin=candidate, reason="select candidate",
        ),
    })


def _runner(
    *,
    empty: bool = False,
    clarification: bool = False,
    candidate: str = "P-MOUSE",
    response_projector=None,
    response_renderer=None,
) -> AgentTaskRunner:
    return AgentTaskRunner(
        requirement_extractor=FakeRequirementExtractor(_decision(clarification=clarification)),
        planner=_planner(candidate=candidate),
        recommendation_tool=FakeRecommendationTool(
            {"Mouse": 30}, empty_category="Mouse" if empty else None
        ),
        evidence_service=FakeEvidenceService(),
        verification_service=EvidenceConstraintVerifier(),
        shopping_plan_service=ShoppingPlanService(),
        response_projector=response_projector,
        response_renderer=response_renderer,
    )


def _run(runner: AgentTaskRunner):
    return runner.run(
        user_id="u",
        session_id="s",
        plan_id="plan",
        total_budget=100,
        requirements=(RequirementInput(
            requirement_id="mouse", user_request="inexpensive mouse"
        ),),
    )


class _FailingRenderer:
    def render(self, context):
        raise RuntimeError("sensitive renderer detail")


class _FailingProjector:
    def project(self, state):
        raise RuntimeError("sensitive projector detail")


class _ObservingProjector:
    def __init__(self) -> None:
        self.unchanged = False

    def project(self, state):
        before = state.model_dump(mode="json")
        result = GroundedResponseProjector().project(state)
        self.unchanged = state.model_dump(mode="json") == before
        return result


class FinalResponseIntegrationTests(unittest.TestCase):
    def test_ready_result_contains_rendered_final_response(self) -> None:
        result = _run(_runner())
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(result.final_response.kind, ResponseKind.READY)
        self.assertIn("P-MOUSE", result.final_response.text)
        self.assertIsNone(result.response_error)

    def test_conflict_result_contains_rendered_final_response(self) -> None:
        result = _run(_runner(empty=True))
        self.assertEqual(result.status, AgentExecutionStatus.CONFLICT)
        self.assertEqual(result.final_response.kind, ResponseKind.CONFLICT)
        self.assertIn("没有返回候选商品", result.final_response.text)
        self.assertIsNone(result.response_error)

    def test_clarification_does_not_enter_response_pipeline(self) -> None:
        result = _run(_runner(clarification=True))
        self.assertEqual(result.status, AgentExecutionStatus.CLARIFICATION_REQUIRED)
        self.assertIsNone(result.final_response)
        self.assertIsNone(result.response_error)

    def test_error_route_does_not_enter_response_pipeline(self) -> None:
        result = _run(_runner(candidate="NOT-IN-RESULT"))
        self.assertEqual(result.status, AgentExecutionStatus.ERROR)
        self.assertIsNone(result.final_response)
        self.assertIsNone(result.response_error)

    def test_renderer_failure_is_isolated_from_workflow_result(self) -> None:
        result = _run(_runner(response_renderer=_FailingRenderer()))
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(result.workflow_state.route.value, "ready")
        self.assertIsNone(result.final_response)
        self.assertEqual(result.response_error, ResponseErrorCode.RENDER_FAILED)

    def test_projector_failure_is_isolated_from_workflow_result(self) -> None:
        result = _run(_runner(response_projector=_FailingProjector()))
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(result.workflow_state.route.value, "ready")
        self.assertIsNone(result.final_response)
        self.assertEqual(result.response_error, ResponseErrorCode.PROJECTION_FAILED)

    def test_response_pipeline_does_not_mutate_workflow_state(self) -> None:
        projector = _ObservingProjector()
        result = _run(_runner(response_projector=projector))
        self.assertTrue(projector.unchanged)
        self.assertEqual(result.workflow_state.agent_state.shopping_plan.total_spent, 30)


if __name__ == "__main__":
    unittest.main()
