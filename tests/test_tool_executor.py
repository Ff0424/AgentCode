"""Unit tests for safe deterministic tool request execution."""

from __future__ import annotations

import unittest

from src.agentrec.tool import (
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolRequest,
    ToolStatus,
)


def registry_with(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(ToolDefinition(
            name=name,
            description=f"Definition for {name}.",
        ))
    return registry


class ToolExecutorTests(unittest.TestCase):
    def test_successful_execution(self) -> None:
        executor = ToolExecutor(
            registry=registry_with("recommend_products"),
            handlers={
                "recommend_products": lambda category: {
                    "category": category,
                    "product_ids": ["P1", "P2"],
                }
            },
        )

        result = executor.execute(ToolRequest(
            tool_name="recommend_products",
            arguments={"category": "Dock"},
        ))

        self.assertIs(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output["category"], "Dock")
        self.assertIsNone(result.error_message)

    def test_unknown_tool(self) -> None:
        executor = ToolExecutor(registry=ToolRegistry(), handlers={})
        result = executor.execute(ToolRequest(tool_name="missing", arguments={}))
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "tool_not_found")
        self.assertEqual(result.output, {})

    def test_missing_handler(self) -> None:
        executor = ToolExecutor(
            registry=registry_with("recommend_products"),
            handlers={},
        )
        result = executor.execute(ToolRequest(
            tool_name="recommend_products",
            arguments={},
        ))
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "handler_not_found")

    def test_handler_failure_hides_exception_details(self) -> None:
        def failing_handler() -> dict:
            raise RuntimeError("secret path C:/private and stack details")

        executor = ToolExecutor(
            registry=registry_with("recommend_products"),
            handlers={"recommend_products": failing_handler},
        )
        result = executor.execute(ToolRequest(
            tool_name="recommend_products",
            arguments={},
        ))
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "tool_execution_failed")
        self.assertNotIn("secret", result.model_dump_json())
        self.assertNotIn("private", result.model_dump_json())

    def test_sensitive_output_is_rejected_safely(self) -> None:
        executor = ToolExecutor(
            registry=registry_with("recommend_products"),
            handlers={
                "recommend_products": lambda: {
                    "product": {"embedding": [0.1, 0.2]}
                }
            },
        )
        result = executor.execute(ToolRequest(
            tool_name="recommend_products",
            arguments={},
        ))
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "tool_execution_failed")
        self.assertEqual(result.output, {})

    def test_execution_is_deterministic(self) -> None:
        executor = ToolExecutor(
            registry=registry_with("recommend_products"),
            handlers={"recommend_products": lambda category: {"category": category}},
        )
        request = ToolRequest(
            tool_name="recommend_products",
            arguments={"category": "Dock"},
        )
        first = executor.execute(request)
        second = executor.execute(request)
        self.assertEqual(first, second)

    def test_execution_does_not_change_registry(self) -> None:
        registry = registry_with("recommend_products", "verify_evidence")
        before = registry.list_tools()
        executor = ToolExecutor(
            registry=registry,
            handlers={"recommend_products": lambda: {"product_ids": []}},
        )
        executor.execute(ToolRequest(
            tool_name="recommend_products",
            arguments={},
        ))
        self.assertEqual(registry.list_tools(), before)


if __name__ == "__main__":
    unittest.main()
