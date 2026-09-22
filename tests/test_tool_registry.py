"""Unit tests for deterministic ToolDefinition registration and lookup."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.tool import ToolDefinition, ToolRegistry


def definition(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"Description for {name}.")


class ToolRegistryTests(unittest.TestCase):
    def test_register_success(self) -> None:
        registry = ToolRegistry()
        tool = definition("recommend_products")
        registry.register(tool)
        self.assertTrue(registry.contains("recommend_products"))

    def test_duplicate_name_is_rejected_without_overwrite(self) -> None:
        registry = ToolRegistry()
        original = definition("recommend_products")
        registry.register(original)
        with self.assertRaises(ValueError):
            registry.register(ToolDefinition(
                name="recommend_products",
                description="A different definition.",
            ))
        self.assertIs(registry.get("recommend_products"), original)

    def test_get_existing(self) -> None:
        registry = ToolRegistry()
        tool = definition("search_products")
        registry.register(tool)
        self.assertIs(registry.get("search_products"), tool)

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(ToolRegistry().get("missing"))

    def test_list_tools_is_sorted_by_name(self) -> None:
        registry = ToolRegistry()
        for name in ("verify_evidence", "recommend_products", "ask_clarification"):
            registry.register(definition(name))
        self.assertEqual(
            tuple(tool.name for tool in registry.list_tools()),
            ("ask_clarification", "recommend_products", "verify_evidence"),
        )

    def test_remove_behavior(self) -> None:
        registry = ToolRegistry()
        registry.register(definition("recommend_products"))
        self.assertTrue(registry.remove("recommend_products"))
        self.assertFalse(registry.contains("recommend_products"))
        self.assertFalse(registry.remove("recommend_products"))

    def test_registered_definition_remains_immutable(self) -> None:
        registry = ToolRegistry()
        registry.register(definition("recommend_products"))
        stored = registry.get("recommend_products")
        with self.assertRaises(ValidationError):
            stored.name = "changed"


if __name__ == "__main__":
    unittest.main()
