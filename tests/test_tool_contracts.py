"""Unit tests for immutable AgentRec tool-calling contracts."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.tool import ToolDefinition, ToolRequest, ToolResult, ToolStatus


class ToolContractTests(unittest.TestCase):
    def test_valid_tool_definition_trims_strings(self) -> None:
        definition = ToolDefinition(
            name="  recommend_products  ",
            description="  Return product candidates.  ",
        )
        self.assertEqual(definition.name, "recommend_products")
        self.assertEqual(definition.description, "Return product candidates.")

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ToolDefinition(name="   ", description="Description")

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ToolDefinition.model_validate({
                "name": "recommend_products",
                "description": "Return candidates.",
                "version": "1",
            })

    def test_tool_request_validation(self) -> None:
        request = ToolRequest(
            tool_name="  recommend_products ",
            arguments={"category": "Dock", "features": ["HDMI", "USB-C"]},
        )
        self.assertEqual(request.tool_name, "recommend_products")
        self.assertEqual(request.arguments["category"], "Dock")
        with self.assertRaises(ValidationError):
            ToolRequest(tool_name=" ", arguments={})
        with self.assertRaises(ValidationError):
            ToolRequest.model_validate({"tool_name": "recommend_products"})
        with self.assertRaises(ValidationError):
            ToolRequest(
                tool_name="recommend_products",
                arguments={"filters": {"embedding": [0.1, 0.2]}},
            )

    def test_tool_result_success(self) -> None:
        result = ToolResult(
            status=ToolStatus.SUCCESS,
            tool_name="recommend_products",
            output={"product_ids": ["P1", "P2"]},
        )
        self.assertIs(result.status, ToolStatus.SUCCESS)
        self.assertIsNone(result.error_message)
        with self.assertRaises(ValidationError):
            ToolResult(
                status=ToolStatus.SUCCESS,
                tool_name="recommend_products",
                output={},
                error_message="unexpected",
            )

    def test_tool_result_failure(self) -> None:
        result = ToolResult(
            status=ToolStatus.FAILED,
            tool_name="recommend_products",
            output={},
            error_message="Tool execution failed.",
        )
        self.assertIs(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_message, "Tool execution failed.")
        with self.assertRaises(ValidationError):
            ToolResult(
                status=ToolStatus.FAILED,
                tool_name="recommend_products",
                output={"backend": "internal"},
            )

    def test_contracts_are_frozen(self) -> None:
        definition = ToolDefinition(name="tool", description="Description")
        request = ToolRequest(tool_name="tool", arguments={})
        result = ToolResult(status=ToolStatus.SUCCESS, tool_name="tool", output={})
        with self.assertRaises(ValidationError):
            definition.name = "changed"
        with self.assertRaises(ValidationError):
            request.tool_name = "changed"
        with self.assertRaises(ValidationError):
            result.status = ToolStatus.FAILED


if __name__ == "__main__":
    unittest.main()
