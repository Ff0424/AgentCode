"""Deterministic LangGraph skeleton tests using only synthetic tool results."""

from __future__ import annotations

import unittest

from src.agentrec.domain import AgentState, PlanStatus, ShoppingRequirement
from src.agentrec.services import ShoppingPlanService
from src.agentrec.tools import RecommendationToolItem, RecommendationToolResult
from src.agentrec.workflows import (
    ShoppingWorkflowState,
    WorkflowRoute,
    build_shopping_workflow,
)


class FakeRecommendationTool:
    def __init__(self, prices: dict[str, float], *, empty_category: str | None = None):
        self.prices = prices
        self.empty_category = empty_category
        self.calls = []

    def recommend(self, *, user_id, args, excluded_parent_asins=()):
        self.calls.append((user_id, args, excluded_parent_asins))
        if args.category == self.empty_category:
            return RecommendationToolResult(
                personalization_status="personalized",
                fallback_reason=None,
                returned_count=0,
                items=(),
            )
        price = self.prices[args.category]
        return RecommendationToolResult(
            personalization_status="personalized",
            fallback_reason=None,
            returned_count=1,
            items=(RecommendationToolItem(
                rank=1,
                parent_asin=f"P-{args.category.upper()}",
                title=f"Test {args.category}",
                price=price,
                score=0.9,
                score_source="hybrid",
            ),),
        )


def initial_state() -> ShoppingWorkflowState:
    service = ShoppingPlanService()
    plan = service.create_plan(
        plan_id="workflow-demo",
        user_id="user-1",
        currency="USD",
        total_budget=500,
        requirements=(
            ShoppingRequirement(
                requirement_id="dock", category="Dock", max_budget=120,
                required_features=("HDMI",),
            ),
            ShoppingRequirement(
                requirement_id="mouse", category="Mouse", max_budget=30,
            ),
            ShoppingRequirement(requirement_id="headphones", category="Headphones"),
        ),
    )
    return ShoppingWorkflowState(
        agent_state=AgentState(
            user_id="user-1",
            session_id="session-1",
            shopping_plan=plan,
        )
    )


def run(prices: dict[str, float], *, empty_category: str | None = None):
    tool = FakeRecommendationTool(prices, empty_category=empty_category)
    graph = build_shopping_workflow(tool, ShoppingPlanService())
    output = graph.invoke(initial_state(), config={"recursion_limit": 50})
    return ShoppingWorkflowState.model_validate(output), tool


class ShoppingWorkflowTests(unittest.TestCase):
    def test_graph_compiles_and_dependencies_are_not_state_fields(self) -> None:
        tool = FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180})
        graph = build_shopping_workflow(tool, ShoppingPlanService())
        self.assertTrue(callable(graph.invoke))
        fields = set(ShoppingWorkflowState.model_fields)
        self.assertTrue({"agent_state", "last_tool_result", "evaluation"} <= fields)
        self.assertTrue({"service", "recommendation_tool", "tensor"}.isdisjoint(fields))

    def test_golden_demo_and_argument_policy(self) -> None:
        final, tool = run({"Dock": 120, "Mouse": 30, "Headphones": 180})
        plan = final.agent_state.shopping_plan
        self.assertEqual(plan.status, PlanStatus.READY)
        self.assertEqual(plan.total_spent, 330)
        self.assertEqual(plan.remaining_budget, 170)
        self.assertEqual(plan.version, 3)
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual([call[1].top_k for call in tool.calls], [5, 5, 5])
        self.assertEqual([call[1].category for call in tool.calls], ["Dock", "Mouse", "Headphones"])
        self.assertEqual([call[1].max_price for call in tool.calls], [120, 30, None])
        self.assertEqual(tool.calls[0][1].required_features, ("HDMI",))
        self.assertEqual([item.parent_asin for item in plan.selected_items], ["P-DOCK", "P-MOUSE", "P-HEADPHONES"])
        self.assertEqual(
            [item.selected_reason for item in plan.selected_items],
            ["selected_by_rank_policy"] * 3,
        )

    def test_conflict_demo(self) -> None:
        final, _ = run({"Dock": 120, "Mouse": 30, "Headphones": 400})
        plan = final.agent_state.shopping_plan
        self.assertEqual(plan.status, PlanStatus.CONFLICT)
        self.assertEqual(plan.total_spent, 550)
        self.assertEqual(final.route, WorkflowRoute.CONFLICT)
        self.assertTrue(final.evaluation.has_hard_constraint_conflict)

    def test_empty_recommendation_routes_to_conflict_without_mutation(self) -> None:
        final, tool = run(
            {"Dock": 120, "Mouse": 30, "Headphones": 180},
            empty_category="Mouse",
        )
        self.assertEqual(final.route, WorkflowRoute.CONFLICT)
        self.assertEqual(final.agent_state.error_state, "no_recommendation_candidates")
        self.assertEqual(final.agent_state.shopping_plan.total_spent, 120)
        self.assertEqual(final.agent_state.shopping_plan.version, 1)
        self.assertEqual(len(tool.calls), 2)

    def test_rank_one_candidate_is_selected(self) -> None:
        final, _ = run({"Dock": 120, "Mouse": 30, "Headphones": 180})
        self.assertEqual(final.agent_state.shopping_plan.selected_items[0].parent_asin, "P-DOCK")

    def test_same_input_has_same_logical_output(self) -> None:
        first, _ = run({"Dock": 120, "Mouse": 30, "Headphones": 180})
        second, _ = run({"Dock": 120, "Mouse": 30, "Headphones": 180})
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))

    def test_state_json_round_trip_and_runtime_dependency_absence(self) -> None:
        state = initial_state()
        restored = ShoppingWorkflowState.model_validate_json(state.model_dump_json())
        self.assertEqual(restored, state)
        payload = state.model_dump_json()
        self.assertNotIn("FakeRecommendationTool", payload)
        self.assertNotIn("ShoppingPlanService", payload)


if __name__ == "__main__":
    unittest.main()
