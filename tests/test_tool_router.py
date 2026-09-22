"""Unit tests for deterministic DecisionPlan-to-ToolRequest routing."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.decision import DecisionExecutionPlan
from src.agentrec.domain import ShoppingRequirement
from src.agentrec.tool import (
    RECOMMENDATION_TOOL_NAME,
    ToolRouter,
    ToolRoutingContext,
)
from src.agentrec.tools import RecommendationToolArgs


def requirement() -> ShoppingRequirement:
    return ShoppingRequirement(
        requirement_id="dock",
        category="Dock",
        quantity=1,
        max_budget=200,
        required_features=("HDMI", "USB-C"),
        soft_preferences=("compact",),
        priority=2,
    )


def context(
    plan: DecisionExecutionPlan,
    **overrides: object,
) -> ToolRoutingContext:
    values: dict[str, object] = {
        "decision_plan": plan,
        "requirement": requirement(),
        "top_k": 5,
    }
    values.update(overrides)
    return ToolRoutingContext.model_validate(values)


class ToolRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = ToolRouter()

    def test_recommendation_routing_matches_real_adapter_contract(self) -> None:
        requests = self.router.route(context(DecisionExecutionPlan(
            retrieve_products=True,
            verify_evidence=True,
            generate_response=True,
        )))

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.tool_name, RECOMMENDATION_TOOL_NAME)
        self.assertEqual(request.tool_name, "recommend_products")
        self.assertEqual(
            set(request.arguments),
            {"top_k", "category", "max_price", "required_features"},
        )
        self.assertEqual(
            RecommendationToolArgs.model_validate(request.arguments),
            RecommendationToolArgs(
                top_k=5,
                category="Dock",
                max_price=200,
                required_features=("HDMI", "USB-C"),
            ),
        )

    def test_retrieve_products_false_produces_no_request(self) -> None:
        requests = self.router.route(context(DecisionExecutionPlan(
            use_memory=True,
            verify_evidence=True,
            generate_response=True,
            stop=True,
        )))
        self.assertEqual(requests, ())

    def test_orchestration_only_actions_do_not_become_tools(self) -> None:
        for plan in (
            DecisionExecutionPlan(use_memory=True),
            DecisionExecutionPlan(ask_clarification=True),
            DecisionExecutionPlan(verify_evidence=True),
            DecisionExecutionPlan(generate_response=True),
            DecisionExecutionPlan(stop=True),
        ):
            with self.subTest(plan=plan):
                self.assertEqual(self.router.route(context(plan)), ())

    def test_routing_is_deterministic(self) -> None:
        routing_context = context(DecisionExecutionPlan(retrieve_products=True))
        first = self.router.route(routing_context)
        second = self.router.route(routing_context)
        self.assertEqual(first, second)
        self.assertEqual(tuple(value.tool_name for value in first), (
            RECOMMENDATION_TOOL_NAME,
        ))

    def test_context_is_frozen(self) -> None:
        routing_context = context(DecisionExecutionPlan(retrieve_products=True))
        with self.assertRaises(ValidationError):
            routing_context.top_k = 10

    def test_context_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            context(
                DecisionExecutionPlan(retrieve_products=True),
                model="internal",
            )

    def test_current_requirement_facts_are_preserved_exactly(self) -> None:
        current = requirement()
        request = self.router.route(context(
            DecisionExecutionPlan(retrieve_products=True),
            requirement=current,
            top_k=10,
        ))[0]
        args = RecommendationToolArgs.model_validate(request.arguments)
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.category, current.category)
        self.assertEqual(args.max_price, current.max_budget)
        self.assertEqual(args.required_features, current.required_features)

    def test_request_contains_no_internal_field_leakage(self) -> None:
        request = self.router.route(context(
            DecisionExecutionPlan(retrieve_products=True),
        ))[0]
        payload = request.model_dump(mode="json")
        forbidden = {
            "item_index",
            "embedding",
            "similarity",
            "artifact_path",
            "model",
            "backend",
            "gpu",
        }

        def keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key
                    yield from keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from keys(nested)

        self.assertTrue(forbidden.isdisjoint(set(keys(payload))))


if __name__ == "__main__":
    unittest.main()
