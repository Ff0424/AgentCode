"""Unit tests for immutable agent decision contracts."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.decision import AgentIntent, DecisionAction, DecisionDirective


class DecisionContractTests(unittest.TestCase):
    def test_valid_directive_preserves_action_order(self) -> None:
        directive = DecisionDirective(
            intent=AgentIntent.RECOMMENDATION,
            actions=(
                DecisionAction.USE_MEMORY,
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            ),
        )
        self.assertIs(directive.intent, AgentIntent.RECOMMENDATION)
        self.assertEqual(
            directive.actions,
            (
                DecisionAction.USE_MEMORY,
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            ),
        )

    def test_empty_actions_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DecisionDirective(intent=AgentIntent.INFORMATION, actions=())

    def test_duplicate_actions_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DecisionDirective(
                intent=AgentIntent.RECOMMENDATION,
                actions=(
                    DecisionAction.RETRIEVE_PRODUCTS,
                    DecisionAction.RETRIEVE_PRODUCTS,
                ),
            )

    def test_invalid_enum_values_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DecisionDirective(
                intent="unsupported",  # type: ignore[arg-type]
                actions=(DecisionAction.STOP,),
            )
        with self.assertRaises(ValidationError):
            DecisionDirective(
                intent=AgentIntent.FOLLOW_UP,
                actions=("unsupported",),  # type: ignore[arg-type]
            )

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DecisionDirective(
                intent=AgentIntent.CLARIFICATION,
                actions=(DecisionAction.ASK_CLARIFICATION,),
                score=0.9,  # type: ignore[call-arg]
            )

    def test_directive_is_frozen(self) -> None:
        directive = DecisionDirective(
            intent=AgentIntent.INFORMATION,
            actions=(DecisionAction.GENERATE_RESPONSE,),
        )
        with self.assertRaises(ValidationError):
            directive.intent = AgentIntent.RECOMMENDATION


if __name__ == "__main__":
    unittest.main()
