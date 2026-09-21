"""Unit tests for pure decision action resolution."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.decision import (
    AgentIntent,
    DecisionAction,
    DecisionDirective,
    DecisionExecutionPlan,
    DecisionExecutor,
)


class DecisionExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = DecisionExecutor()

    def test_recommendation_full_actions_mapping(self) -> None:
        directive = DecisionDirective(
            intent=AgentIntent.RECOMMENDATION,
            actions=(
                DecisionAction.USE_MEMORY,
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            ),
        )

        plan = self.executor.execute(directive)

        self.assertEqual(
            plan,
            DecisionExecutionPlan(
                use_memory=True,
                retrieve_products=True,
                verify_evidence=True,
                generate_response=True,
            ),
        )

    def test_clarification_mapping(self) -> None:
        directive = DecisionDirective(
            intent=AgentIntent.CLARIFICATION,
            actions=(DecisionAction.ASK_CLARIFICATION, DecisionAction.STOP),
        )

        plan = self.executor.execute(directive)

        self.assertTrue(plan.ask_clarification)
        self.assertTrue(plan.stop)
        self.assertFalse(plan.generate_response)
        self.assertFalse(plan.retrieve_products)

    def test_information_mapping(self) -> None:
        directive = DecisionDirective(
            intent=AgentIntent.INFORMATION,
            actions=(DecisionAction.GENERATE_RESPONSE, DecisionAction.STOP),
        )

        plan = self.executor.execute(directive)

        self.assertTrue(plan.generate_response)
        self.assertTrue(plan.stop)
        self.assertFalse(plan.use_memory)
        self.assertFalse(plan.verify_evidence)

    def test_empty_plan_flags_default_to_false(self) -> None:
        plan = DecisionExecutionPlan()
        self.assertEqual(
            plan.model_dump(),
            {
                "use_memory": False,
                "retrieve_products": False,
                "ask_clarification": False,
                "verify_evidence": False,
                "generate_response": False,
                "stop": False,
            },
        )

    def test_execution_plan_is_frozen(self) -> None:
        plan = DecisionExecutionPlan()
        with self.assertRaises(ValidationError):
            plan.stop = True

    def test_execution_plan_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            DecisionExecutionPlan.model_validate({"unsupported": True})


if __name__ == "__main__":
    unittest.main()
