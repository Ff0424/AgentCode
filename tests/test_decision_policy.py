"""Unit tests for the deterministic agent decision policy."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.decision import (
    AgentIntent,
    DecisionAction,
    DecisionContext,
    DeterministicDecisionPolicy,
)


class DeterministicDecisionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DeterministicDecisionPolicy()

    @staticmethod
    def make_context(intent: AgentIntent, **overrides: object) -> DecisionContext:
        values: dict[str, object] = {
            "intent": intent,
            "memory_available": False,
            "requirement_complete": True,
            "candidate_available": False,
            "verification_required": True,
            "response_required": True,
        }
        values.update(overrides)
        return DecisionContext.model_validate(values)

    def test_information_routing(self) -> None:
        directive = self.policy.decide(self.make_context(AgentIntent.INFORMATION))
        self.assertEqual(
            directive.actions,
            (DecisionAction.GENERATE_RESPONSE, DecisionAction.STOP),
        )

    def test_incomplete_recommendation_asks_clarification(self) -> None:
        directive = self.policy.decide(
            self.make_context(
                AgentIntent.RECOMMENDATION,
                requirement_complete=False,
                memory_available=True,
            )
        )
        self.assertEqual(
            directive.actions,
            (DecisionAction.ASK_CLARIFICATION, DecisionAction.STOP),
        )

    def test_recommendation_with_memory(self) -> None:
        directive = self.policy.decide(
            self.make_context(AgentIntent.RECOMMENDATION, memory_available=True)
        )
        self.assertEqual(
            directive.actions,
            (
                DecisionAction.USE_MEMORY,
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            ),
        )

    def test_recommendation_without_memory(self) -> None:
        directive = self.policy.decide(
            self.make_context(AgentIntent.RECOMMENDATION, memory_available=False)
        )
        self.assertEqual(
            directive.actions,
            (
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            ),
        )

    def test_follow_up_routing(self) -> None:
        directive = self.policy.decide(self.make_context(AgentIntent.FOLLOW_UP))
        self.assertEqual(
            directive.actions,
            (DecisionAction.GENERATE_RESPONSE, DecisionAction.STOP),
        )

    def test_clarification_routing(self) -> None:
        directive = self.policy.decide(self.make_context(AgentIntent.CLARIFICATION))
        self.assertEqual(
            directive.actions,
            (DecisionAction.ASK_CLARIFICATION, DecisionAction.STOP),
        )

    def test_context_is_frozen(self) -> None:
        context = self.make_context(AgentIntent.INFORMATION)
        with self.assertRaises(ValidationError):
            context.memory_available = True

    def test_context_rejects_extra_fields(self) -> None:
        values = self.make_context(AgentIntent.INFORMATION).model_dump()
        values["unexpected"] = True
        with self.assertRaises(ValidationError):
            DecisionContext.model_validate(values)


if __name__ == "__main__":
    unittest.main()
