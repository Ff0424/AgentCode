"""Integration tests for generic execution through RecommendationToolAdapter."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.decision import DecisionExecutionPlan
from src.agentrec.domain import ShoppingRequirement
from src.agentrec.tool import (
    RECOMMENDATION_TOOL_DEFINITION,
    RECOMMENDATION_TOOL_NAME,
    RecommendationToolHandler,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolRequest,
    ToolRouter,
    ToolRoutingContext,
    ToolStatus,
)
from src.agentrec.tools import RecommendationToolItem, RecommendationToolResult


class FakeRecommendationAdapter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def recommend(self, *, user_id, args, excluded_parent_asins=()):
        self.calls.append((user_id, args, excluded_parent_asins))
        if self.fail:
            raise RuntimeError("private adapter exception and path")
        return RecommendationToolResult(
            personalization_status="personalized",
            fallback_reason=None,
            returned_count=1,
            items=(RecommendationToolItem(
                rank=1,
                item_index=42,
                parent_asin="P-DOCK",
                title="USB-C Dock",
                price=129.0,
                score=0.91,
                score_source="hybrid",
            ),),
        )


def executor(adapter):
    registry = ToolRegistry()
    registry.register(RECOMMENDATION_TOOL_DEFINITION)
    return ToolExecutor(
        registry=registry,
        handlers={RECOMMENDATION_TOOL_NAME: RecommendationToolHandler(adapter)},
    )


def request() -> ToolRequest:
    return ToolRequest(
        tool_name=RECOMMENDATION_TOOL_NAME,
        arguments={
            "top_k": 5,
            "category": "Dock",
            "max_price": 200,
            "required_features": ("HDMI", "USB-C"),
        },
    )


def execution_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        user_id="system-user",
        excluded_parent_asins=("P-OLD",),
    )


class RecommendationToolExecutionTests(unittest.TestCase):
    def test_execution_context_is_frozen_and_forbids_extra_fields(self) -> None:
        context = execution_context()
        with self.assertRaises(ValidationError):
            context.user_id = "changed"
        with self.assertRaises(ValidationError):
            ToolExecutionContext.model_validate({
                "user_id": "system-user",
                "model": "internal",
            })

    def test_request_adapter_success(self) -> None:
        adapter = FakeRecommendationAdapter()
        result = executor(adapter).execute(request(), context=execution_context())
        self.assertIs(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.tool_name, RECOMMENDATION_TOOL_NAME)
        self.assertEqual(result.output["returned_count"], 1)

    def test_recommendation_args_are_mapped_correctly(self) -> None:
        adapter = FakeRecommendationAdapter()
        executor(adapter).execute(request(), context=execution_context())
        args = adapter.calls[0][1]
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.category, "Dock")
        self.assertEqual(args.max_price, 200)
        self.assertEqual(args.required_features, ("HDMI", "USB-C"))

    def test_trusted_context_is_mapped_separately(self) -> None:
        adapter = FakeRecommendationAdapter()
        executor(adapter).execute(request(), context=execution_context())
        user_id, _args, excluded = adapter.calls[0]
        self.assertEqual(user_id, "system-user")
        self.assertEqual(excluded, ("P-OLD",))
        self.assertNotIn("user_id", request().arguments)
        self.assertNotIn("excluded_parent_asins", request().arguments)

    def test_request_cannot_spoof_system_user_id(self) -> None:
        adapter = FakeRecommendationAdapter()
        values = dict(request().arguments)
        values["user_id"] = "attacker"
        spoofed = ToolRequest(
            tool_name=RECOMMENDATION_TOOL_NAME,
            arguments=values,
        )
        result = executor(adapter).execute(
            spoofed, context=execution_context()
        )
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "tool_execution_failed")
        self.assertEqual(adapter.calls, [])

    def test_adapter_failure_is_safe(self) -> None:
        result = executor(FakeRecommendationAdapter(fail=True)).execute(
            request(), context=execution_context()
        )
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "tool_execution_failed")
        serialized = result.model_dump_json()
        self.assertNotIn("private", serialized)
        self.assertNotIn("path", serialized)

    def test_safe_output_allowlist(self) -> None:
        result = executor(FakeRecommendationAdapter()).execute(
            request(), context=execution_context()
        )
        self.assertEqual(
            set(result.output),
            {"personalization_status", "fallback_reason", "returned_count", "items"},
        )
        self.assertEqual(
            set(result.output["items"][0]),
            {"rank", "parent_asin", "title", "price"},
        )
        serialized = result.model_dump_json().casefold()
        for forbidden in (
            "item_index", "score", "similarity", "embedding", "vector",
            "artifact", "backend", "model", "gpu",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_router_to_executor_integration(self) -> None:
        routing_context = ToolRoutingContext(
            decision_plan=DecisionExecutionPlan(retrieve_products=True),
            requirement=ShoppingRequirement(
                requirement_id="dock",
                category="Dock",
                max_budget=200,
                required_features=("HDMI", "USB-C"),
            ),
            top_k=5,
        )
        requests = ToolRouter().route(routing_context)
        adapter = FakeRecommendationAdapter()
        result = executor(adapter).execute(
            requests[0], context=execution_context()
        )
        self.assertIs(result.status, ToolStatus.SUCCESS)
        self.assertEqual(adapter.calls[0][0], "system-user")


if __name__ == "__main__":
    unittest.main()
