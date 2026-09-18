"""Workflow-level V2-09.3 identity, version and failure tests."""

import unittest

from src.agentrec.services import ShoppingPlanService
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.fake_evidence import FakeEvidenceService
from tests.test_shopping_workflow import FakeRecommendationTool, initial_state


class GroundedWorkflowTests(unittest.TestCase):
    def test_evidence_is_selected_only_and_versions_are_not_off_by_one(self):
        evidence = FakeEvidenceService()
        graph = build_shopping_workflow(
            FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180}),
            ShoppingPlanService(), evidence_service=evidence)
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial_state(), config={"recursion_limit": 50}))
        self.assertEqual([call[1] for call in evidence.calls], [0, 1, 2])
        self.assertEqual([item.selected_at_plan_version for item in final.selected_evidence], [1, 2, 3])
        self.assertIsNone(final.current_evidence)
        self.assertEqual(len(final.selected_evidence), 3)

    def test_retrieval_failure_is_error_and_does_not_mutate_plan(self):
        graph = build_shopping_workflow(
            FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180}),
            ShoppingPlanService(), evidence_service=FakeEvidenceService(fail=True))
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial_state(), config={"recursion_limit": 50}))
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(final.agent_state.shopping_plan.version, 0)
        self.assertTrue(final.agent_state.error_state.startswith("evidence_retrieval:"))


if __name__ == "__main__":
    unittest.main()
