"""Deterministic V2-10.5a Goal Extractor tests without network calls."""

from __future__ import annotations

import json
import unittest

from src.agentrec.planning import (
    AllocationPreferenceType,
    FakeGoalExtractor,
    GoalExtractor,
    PlannerProviderError,
    PlannerSchemaError,
    PlannerTimeoutError,
    ShoppingGoalExtractionDecision,
    StructuredGoalExtractor,
)


class FakeTextProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, messages, timeout_seconds):
        self.calls.append((messages, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def golden_payload(**updates):
    value = {
        "total_budget": 500,
        "requirement_proposals": [
            {
                "category": "Docking Stations",
                "quantity": 1,
                "max_budget": None,
                "required_features": ["HDMI"],
                "soft_preferences": [],
                "priority": 3,
            },
            {
                "category": "Mouse",
                "quantity": 1,
                "max_budget": None,
                "required_features": [],
                "soft_preferences": [],
                "priority": 3,
            },
            {
                "category": "Headphones",
                "quantity": 1,
                "max_budget": None,
                "required_features": [],
                "soft_preferences": [],
                "priority": 3,
            },
        ],
        "allocation_preferences": [
            {"target_index": 1, "preference": "save_more"},
            {"target_index": 2, "preference": "allocate_more"},
        ],
        "clarification_needed": False,
        "clarification_question": None,
    }
    value.update(updates)
    return value


class GoalExtractionTests(unittest.TestCase):
    def test_golden_demo_valid_json_and_order(self) -> None:
        provider = FakeTextProvider([json.dumps(golden_payload())])
        extractor = StructuredGoalExtractor(provider)
        decision = extractor.extract(user_request=(
            "I need a dock, mouse, and headphones for business travel under $500. "
            "The dock must support HDMI, save on the mouse, allocate more to headphones, "
            "and use my purchase history."
        ))
        self.assertIsInstance(extractor, GoalExtractor)
        self.assertEqual(decision.total_budget, 500)
        self.assertEqual(
            tuple(value.category for value in decision.requirement_proposals),
            ("Docking Stations", "Mouse", "Headphones"),
        )
        self.assertEqual(
            tuple(value.max_budget for value in decision.requirement_proposals),
            (None, None, None),
        )
        self.assertEqual(decision.requirement_proposals[0].required_features, ("HDMI",))
        self.assertTrue(
            all(not value.soft_preferences for value in decision.requirement_proposals)
        )
        self.assertEqual(
            tuple((value.target_index, value.preference) for value in decision.allocation_preferences),
            (
                (1, AllocationPreferenceType.SAVE_MORE),
                (2, AllocationPreferenceType.ALLOCATE_MORE),
            ),
        )
        self.assertFalse(decision.clarification_needed)
        self.assertIsNone(decision.clarification_question)

    def test_total_and_explicit_requirement_budget_remain_separate(self) -> None:
        payload = golden_payload()
        payload["requirement_proposals"][1]["max_budget"] = 30
        decision = StructuredGoalExtractor(
            FakeTextProvider([json.dumps(payload)])
        ).extract(user_request="Total budget $500; mouse at most $30.")
        self.assertEqual(decision.total_budget, 500)
        self.assertEqual(
            tuple(value.max_budget for value in decision.requirement_proposals),
            (None, 30, None),
        )

    def test_missing_total_budget_is_clarification_without_guessing(self) -> None:
        payload = golden_payload(
            total_budget=None,
            clarification_needed=True,
            clarification_question="What is your total budget?",
        )
        decision = StructuredGoalExtractor(
            FakeTextProvider([json.dumps(payload)])
        ).extract(user_request="Buy a dock, mouse, and headphones.")
        self.assertIsNone(decision.total_budget)
        self.assertTrue(decision.clarification_needed)
        self.assertEqual(decision.clarification_question, "What is your total budget?")

    def test_malformed_schema_invalid_and_extra_fields_fail(self) -> None:
        invalid_values = (
            "not-json",
            json.dumps(golden_payload(total_budget=0)),
            json.dumps(golden_payload(extra="forbidden")),
        )
        for raw in invalid_values:
            with self.subTest(raw=raw):
                extractor = StructuredGoalExtractor(
                    FakeTextProvider([raw]), schema_retries=0
                )
                with self.assertRaises(PlannerSchemaError):
                    extractor.extract(user_request="Build a shopping bundle for $500.")

    def test_schema_retry_success_and_exhaustion(self) -> None:
        provider = FakeTextProvider(["bad-json", json.dumps(golden_payload())])
        decision = StructuredGoalExtractor(provider, schema_retries=1).extract(
            user_request="Build a shopping bundle for $500."
        )
        self.assertEqual(decision.total_budget, 500)
        self.assertEqual(len(provider.calls), 2)

        exhausted = FakeTextProvider(["bad-json", "still-bad"])
        with self.assertRaises(PlannerSchemaError):
            StructuredGoalExtractor(exhausted, schema_retries=1).extract(
                user_request="Build a shopping bundle for $500."
            )
        self.assertEqual(len(exhausted.calls), 2)

    def test_timeout_and_provider_errors_are_sanitized(self) -> None:
        with self.assertRaises(PlannerTimeoutError):
            StructuredGoalExtractor(FakeTextProvider([TimeoutError("secret")])).extract(
                user_request="Build a shopping bundle for $500."
            )
        with self.assertRaises(PlannerProviderError):
            StructuredGoalExtractor(FakeTextProvider([RuntimeError("secret")])).extract(
                user_request="Build a shopping bundle for $500."
            )
        expected = PlannerProviderError("provider failure")
        with self.assertRaises(PlannerProviderError) as raised:
            StructuredGoalExtractor(FakeTextProvider([expected])).extract(
                user_request="Build a shopping bundle for $500."
            )
        self.assertIs(raised.exception, expected)

    def test_blank_request_and_invalid_configuration_are_rejected(self) -> None:
        provider = FakeTextProvider([json.dumps(golden_payload())])
        extractor = StructuredGoalExtractor(provider)
        for value in ("", "   "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                extractor.extract(user_request=value)
        with self.assertRaises(TypeError):
            extractor.extract(user_request=123)
        with self.assertRaises(TypeError):
            StructuredGoalExtractor(None)
        for timeout in (0, -1, float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                StructuredGoalExtractor(provider, timeout_seconds=timeout)
        for retries in (-1, 2, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                StructuredGoalExtractor(provider, schema_retries=retries)

    def test_prompt_contains_frozen_semantics_without_exact_text_coupling(self) -> None:
        provider = FakeTextProvider([json.dumps(golden_payload())])
        StructuredGoalExtractor(provider).extract(user_request="Goal with $500 total.")
        system_prompt = provider.calls[0][0][0]["content"].casefold()
        for semantic in (
            "total budget",
            "do not duplicate",
            "preserve the order",
            "save_more",
            "allocate_more",
            "do not invent allocation amounts",
            "do not invent historical preferences",
            "no total budget",
            "clarification_needed",
        ):
            with self.subTest(semantic=semantic):
                self.assertIn(semantic, system_prompt)

    def test_fake_extractor_fixed_keyed_deterministic_and_unknown(self) -> None:
        decision = ShoppingGoalExtractionDecision.model_validate(golden_payload())
        fixed = FakeGoalExtractor(decision)
        self.assertIsInstance(fixed, GoalExtractor)
        self.assertEqual(fixed.extract(user_request="first"), decision)
        self.assertEqual(fixed.extract(user_request="second"), decision)

        keyed = FakeGoalExtractor(decisions_by_request={" demo ": golden_payload()})
        first = keyed.extract(user_request="demo")
        second = keyed.extract(user_request=" demo ")
        self.assertEqual(first, second)
        with self.assertRaisesRegex(KeyError, "No fake goal extraction"):
            keyed.extract(user_request="unknown")
        with self.assertRaises(ValueError):
            keyed.extract(user_request=" ")
        with self.assertRaises(ValueError):
            FakeGoalExtractor(decisions_by_request={" ": golden_payload()})


if __name__ == "__main__":
    unittest.main()
