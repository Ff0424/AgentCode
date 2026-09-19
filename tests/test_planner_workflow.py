"""Deterministic V2-08.4b planner-node integration tests."""

from __future__ import annotations

import unittest

from src.agentrec.planning import (
    FakePlanner,
    SelectCandidateDecision,
    SelectRequirementDecision,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.test_shopping_workflow import FakeRecommendationTool, initial_state
from tests.fake_evidence import FakeEvidenceService
from src.agentrec.verification import EvidenceConstraintVerifier


PRICES = {"Dock": 120, "Mouse": 30, "Headphones": 180}


def planner_decisions(*, invalid_requirement: bool = False, invalid_candidate: bool = False):
    decisions = {}
    for version, requirement_id in enumerate(("dock", "mouse", "headphones")):
        selected_requirement = "missing" if invalid_requirement and version == 0 else requirement_id
        decisions[f"select_requirement:{version}"] = SelectRequirementDecision(
            plan_id="workflow-demo",
            plan_version=version,
            requirement_id=selected_requirement,
            reason=f"process {selected_requirement}",
        )
        selected_asin = "P-NOT-RETURNED" if invalid_candidate and version == 0 else f"P-{requirement_id.upper()}"
        decisions[f"select_candidate:{version}:{requirement_id}"] = SelectCandidateDecision(
            plan_id="workflow-demo",
            plan_version=version,
            requirement_id=requirement_id,
            parent_asin=selected_asin,
            reason=f"select {selected_asin}",
        )
    return decisions


def invoke_with_planner(planner: FakePlanner):
    tool = FakeRecommendationTool(PRICES)
    graph = build_shopping_workflow(
        tool, ShoppingPlanService(), planner=planner,
        evidence_service=FakeEvidenceService(),
        verification_service=EvidenceConstraintVerifier(),
    )
    output = graph.invoke(initial_state(), config={"recursion_limit": 50})
    return ShoppingWorkflowState.model_validate(output), tool


class PlannerWorkflowTests(unittest.TestCase):
    def test_fake_planner_selects_requirements_and_candidates(self) -> None:
        final, tool = invoke_with_planner(
            FakePlanner(decisions_by_key=planner_decisions())
        )
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual(final.agent_state.shopping_plan.total_spent, 330)
        self.assertEqual(
            [item.parent_asin for item in final.agent_state.shopping_plan.selected_items],
            ["P-DOCK", "P-MOUSE", "P-HEADPHONES"],
        )
        self.assertEqual(len(tool.calls), 3)
        self.assertEqual(
            [item.selected_reason for item in final.agent_state.shopping_plan.selected_items],
            ["select P-DOCK", "select P-MOUSE", "select P-HEADPHONES"],
        )

    def test_invalid_requirement_is_rejected_before_tool_call(self) -> None:
        final, tool = invoke_with_planner(
            FakePlanner(decisions_by_key=planner_decisions(invalid_requirement=True))
        )
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(final.agent_state.error_state, "planner_validation:invalid_requirement_id")
        self.assertEqual(tool.calls, [])
        self.assertEqual(final.agent_state.shopping_plan.version, 0)

    def test_candidate_not_in_tool_result_is_rejected(self) -> None:
        final, tool = invoke_with_planner(
            FakePlanner(decisions_by_key=planner_decisions(invalid_candidate=True))
        )
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(final.agent_state.error_state, "planner_validation:candidate_not_in_tool_result")
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(final.agent_state.shopping_plan.version, 0)

    def test_stale_plan_version_is_rejected(self) -> None:
        decisions = planner_decisions()
        decisions["select_requirement:0"] = SelectRequirementDecision(
            plan_id="workflow-demo",
            plan_version=99,
            requirement_id="dock",
            reason="stale snapshot",
        )
        final, tool = invoke_with_planner(FakePlanner(decisions_by_key=decisions))
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(final.agent_state.error_state, "planner_validation:stale_plan_version")
        self.assertEqual(tool.calls, [])

    def test_existing_deterministic_fallback_still_runs(self) -> None:
        tool = FakeRecommendationTool(PRICES)
        graph = build_shopping_workflow(
            tool, ShoppingPlanService(), evidence_service=FakeEvidenceService(),
            verification_service=EvidenceConstraintVerifier(),
        )
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial_state(), config={"recursion_limit": 50})
        )
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual(final.agent_state.shopping_plan.total_spent, 330)
        self.assertIsNone(final.planner_decision)

    def test_repeated_planner_execution_is_deterministic(self) -> None:
        planner = FakePlanner(decisions_by_key=planner_decisions())
        first, _ = invoke_with_planner(planner)
        second, _ = invoke_with_planner(planner)
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))

    def test_planner_instance_is_not_serialized_into_state(self) -> None:
        planner = FakePlanner(decisions_by_key=planner_decisions())
        final, _ = invoke_with_planner(planner)
        payload = final.model_dump_json()
        self.assertNotIn("FakePlanner", payload)
        self.assertNotIn("_decisions_by_key", payload)


if __name__ == "__main__":
    unittest.main()
