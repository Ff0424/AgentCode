"""Deterministic V2-08.5 requirement extraction tests without network calls."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from src.agentrec.domain import ShoppingRequirement
from src.agentrec.planning import (
    FakeRequirementExtractor,
    PlannerSchemaError,
    RequirementClarificationRequired,
    RequirementExtractionDecision,
    StructuredRequirementExtractor,
)


class FakeExtractionProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, messages, timeout_seconds):
        self.calls.append((messages, timeout_seconds))
        return self.responses.pop(0)


def payload(**updates):
    value = {
        "category": "Headphones",
        "quantity": 1,
        "max_budget": 180,
        "required_features": ["active   noise cancellation"],
        "soft_preferences": ["black color", "long battery life"],
        "priority": 5,
        "clarification_needed": False,
        "reason": "Explicit shopping requirement extracted from the request.",
    }
    value.update(updates)
    return value


class RequirementExtractionTests(unittest.TestCase):
    def test_normal_extraction_and_domain_conversion(self) -> None:
        provider = FakeExtractionProvider([json.dumps(payload())])
        decision = StructuredRequirementExtractor(provider).extract(
            user_request="Buy noise cancelling headphones under $180; black preferred."
        )
        self.assertEqual(decision.required_features, ("active noise cancellation",))
        self.assertEqual(decision.soft_preferences, ("black color", "long battery life"))
        requirement = decision.to_shopping_requirement(requirement_id="headphones")
        self.assertIsInstance(requirement, ShoppingRequirement)
        self.assertEqual(requirement.max_budget, 180)
        self.assertEqual(requirement.required_features, ("active noise cancellation",))

    def test_missing_budget_requires_clarification_without_guessing(self) -> None:
        decision = RequirementExtractionDecision.model_validate(payload(
            max_budget=None,
            clarification_needed=True,
            reason="The user did not provide a budget.",
        ))
        self.assertIsNone(decision.max_budget)
        self.assertTrue(decision.clarification_needed)
        with self.assertRaises(RequirementClarificationRequired):
            decision.to_shopping_requirement(requirement_id="headphones")
        with self.assertRaises(ValidationError):
            RequirementExtractionDecision.model_validate(payload(
                max_budget=None, clarification_needed=False
            ))

    def test_invalid_budget_and_quantity_are_rejected(self) -> None:
        for budget in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(budget=budget), self.assertRaises((ValidationError, TypeError)):
                RequirementExtractionDecision.model_validate(payload(max_budget=budget))
        for quantity in (0, -1, 1.5, True):
            with self.subTest(quantity=quantity), self.assertRaises(ValidationError):
                RequirementExtractionDecision.model_validate(payload(quantity=quantity))

    def test_empty_category_extra_field_and_long_reason_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RequirementExtractionDecision.model_validate(payload(category=" "))
        with self.assertRaises(ValidationError):
            RequirementExtractionDecision.model_validate(payload(unknown="value"))
        with self.assertRaises(ValidationError):
            RequirementExtractionDecision.model_validate(payload(reason="x" * 501))

    def test_hard_and_soft_constraints_remain_separate(self) -> None:
        decision = RequirementExtractionDecision.model_validate(payload(
            required_features=["USB-C", "HDMI"],
            soft_preferences=["Apple preferred", "silver color", "travel use"],
        ))
        requirement = decision.to_shopping_requirement(requirement_id="dock")
        self.assertEqual(requirement.required_features, ("USB-C", "HDMI"))
        self.assertEqual(
            requirement.soft_preferences,
            ("Apple preferred", "silver color", "travel use"),
        )

    def test_invalid_provider_json_and_schema_are_rejected(self) -> None:
        for raw in ("not-json", json.dumps(payload(category=""))):
            extractor = StructuredRequirementExtractor(
                FakeExtractionProvider([raw]), schema_retries=0
            )
            with self.assertRaises(PlannerSchemaError):
                extractor.extract(user_request="buy headphones")

    def test_json_round_trip_and_fake_extractor_are_deterministic(self) -> None:
        decision = RequirementExtractionDecision.model_validate(payload())
        restored = RequirementExtractionDecision.model_validate_json(
            decision.model_dump_json()
        )
        self.assertEqual(restored, decision)
        fake = FakeRequirementExtractor(decision)
        self.assertEqual(fake.extract(user_request="first"), decision)
        self.assertEqual(fake.extract(user_request="second"), decision)


if __name__ == "__main__":
    unittest.main()
