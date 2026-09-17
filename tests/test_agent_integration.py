"""V2-08.6 end-to-end tests using only deterministic fake components."""

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
from src.agentrec.services import ShoppingPlanService
from tests.test_shopping_workflow import FakeRecommendationTool


REQUESTS = {
    "dock request": ("dock", "Dock", 120, ("HDMI",)),
    "mouse request": ("mouse", "Mouse", 30, ()),
    "headphones request": ("headphones", "Headphones", 180, ()),
}


def extraction(category, budget, features=(), *, clarification=False):
    return RequirementExtractionDecision(
        category=category,
        quantity=1,
        max_budget=budget,
        required_features=features,
        soft_preferences=(),
        priority=3,
        clarification_needed=clarification,
        reason="deterministic extraction",
    )


def planner_for(plan_id, requirement_ids):
    decisions = {}
    for version, requirement_id in enumerate(requirement_ids):
        decisions[f"select_requirement:{version}"] = SelectRequirementDecision(
            plan_id=plan_id, plan_version=version, requirement_id=requirement_id,
            reason=f"process {requirement_id}",
        )
        decisions[f"select_candidate:{version}:{requirement_id}"] = SelectCandidateDecision(
            plan_id=plan_id, plan_version=version, requirement_id=requirement_id,
            parent_asin=f"P-{requirement_id.upper()}", reason=f"select {requirement_id}",
        )
    return FakePlanner(decisions_by_key=decisions)


def runner(request_names, prices, *, empty_category=None, plan_id="integration-plan"):
    decisions = {
        name: extraction(category, budget, features)
        for name, (_rid, category, budget, features) in REQUESTS.items()
        if name in request_names
    }
    return AgentTaskRunner(
        requirement_extractor=FakeRequirementExtractor(decisions_by_request=decisions),
        planner=planner_for(plan_id, [REQUESTS[name][0] for name in request_names]),
        recommendation_tool=FakeRecommendationTool(prices, empty_category=empty_category),
        shopping_plan_service=ShoppingPlanService(),
    )


def inputs(names):
    return tuple(
        RequirementInput(requirement_id=REQUESTS[name][0], user_request=name)
        for name in names
    )


class AgentIntegrationTests(unittest.TestCase):
    def test_single_requirement_success(self) -> None:
        service = runner(("headphones request",), {"Headphones": 180})
        result = service.run(
            user_id="u", session_id="s", plan_id="integration-plan",
            total_budget=200, requirements=inputs(("headphones request",)),
        )
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(result.workflow_state.agent_state.shopping_plan.total_spent, 180)

    def test_multiple_requirement_success(self) -> None:
        names = tuple(REQUESTS)
        service = runner(names, {"Dock": 120, "Mouse": 30, "Headphones": 180})
        result = service.run(
            user_id="u", session_id="s", plan_id="integration-plan",
            total_budget=500, requirements=inputs(names),
        )
        plan = result.workflow_state.agent_state.shopping_plan
        self.assertEqual(result.status, AgentExecutionStatus.READY)
        self.assertEqual(plan.total_spent, 330)
        self.assertEqual(plan.version, 3)

    def test_missing_budget_stops_for_clarification_before_plan(self) -> None:
        decision = extraction("Headphones", None, clarification=True)
        service = AgentTaskRunner(
            requirement_extractor=FakeRequirementExtractor(decision),
            planner=planner_for("integration-plan", ("headphones",)),
            recommendation_tool=FakeRecommendationTool({"Headphones": 180}),
            shopping_plan_service=ShoppingPlanService(),
        )
        result = service.run(
            user_id="u", session_id="s", plan_id="integration-plan",
            total_budget=500,
            requirements=(RequirementInput(requirement_id="headphones", user_request="good headphones"),),
        )
        self.assertEqual(result.status, AgentExecutionStatus.CLARIFICATION_REQUIRED)
        self.assertEqual(result.clarification_requirement_ids, ("headphones",))
        self.assertIsNone(result.workflow_state)

    def test_empty_recommendation_is_conflict(self) -> None:
        service = runner(
            ("headphones request",), {"Headphones": 180}, empty_category="Headphones"
        )
        result = service.run(
            user_id="u", session_id="s", plan_id="integration-plan",
            total_budget=200, requirements=inputs(("headphones request",)),
        )
        self.assertEqual(result.status, AgentExecutionStatus.CONFLICT)
        self.assertEqual(result.workflow_state.agent_state.error_state, "no_recommendation_candidates")

    def test_overall_budget_conflict(self) -> None:
        names = tuple(REQUESTS)
        decisions = {
            "dock request": extraction("Dock", 120, ("HDMI",)),
            "mouse request": extraction("Mouse", 30),
            "headphones request": extraction("Headphones", 400),
        }
        service = AgentTaskRunner(
            requirement_extractor=FakeRequirementExtractor(decisions_by_request=decisions),
            planner=planner_for("integration-plan", ("dock", "mouse", "headphones")),
            recommendation_tool=FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 400}),
            shopping_plan_service=ShoppingPlanService(),
        )
        result = service.run(
            user_id="u", session_id="s", plan_id="integration-plan",
            total_budget=500, requirements=inputs(names),
        )
        self.assertEqual(result.status, AgentExecutionStatus.CONFLICT)
        self.assertEqual(result.workflow_state.agent_state.shopping_plan.total_spent, 550)

    def test_repeated_execution_is_deterministic(self) -> None:
        names = tuple(REQUESTS)
        service = runner(names, {"Dock": 120, "Mouse": 30, "Headphones": 180})
        kwargs = dict(
            user_id="u", session_id="s", plan_id="integration-plan",
            total_budget=500, requirements=inputs(names),
        )
        first = service.run(**kwargs)
        second = service.run(**kwargs)
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))


if __name__ == "__main__":
    unittest.main()
