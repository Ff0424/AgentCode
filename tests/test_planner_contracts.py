"""Deterministic tests for V2-08.4a provider-neutral planner contracts."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.planning import (
    FakePlanner,
    Planner,
    PlannerAction,
    ReplanProposalDecision,
    RequestRecommendationDecision,
    RequestUserConfirmationDecision,
    SelectCandidateDecision,
    SelectRequirementDecision,
    validate_planner_decision,
)


class PlannerContractTests(unittest.TestCase):
    def test_all_decision_schemas_validate_and_normalize(self) -> None:
        decisions = (
            SelectRequirementDecision(plan_id="p", plan_version=0, requirement_id=" dock ", reason=" first "),
            RequestRecommendationDecision(
                plan_id="p", plan_version=0, requirement_id="dock", top_k=5, reason="retrieve candidates"
            ),
            SelectCandidateDecision(
                plan_id="p", plan_version=0, requirement_id="dock", parent_asin=" B000TEST ", reason="best fit"
            ),
            ReplanProposalDecision(
                plan_id="p", plan_version=0,
                requirement_id="dock",
                operation="increase_requirement_budget",
                proposed_value=150.0,
                reason="no valid candidates",
            ),
            RequestUserConfirmationDecision(
                plan_id="p", plan_version=0,
                confirmation_id="confirm-budget",
                prompt="Increase the budget?",
                reason="The hard limit blocks all candidates.",
            ),
        )
        self.assertEqual(decisions[0].requirement_id, "dock")
        self.assertEqual(decisions[2].parent_asin, "B000TEST")
        self.assertTrue(decisions[3].requires_confirmation)
        for decision in decisions:
            restored = validate_planner_decision(decision.model_dump(mode="json"))
            self.assertEqual(restored, decision)

    def test_discriminator_selects_the_correct_model(self) -> None:
        decision = validate_planner_decision({
            "action": "select_candidate",
            "plan_id": "p",
            "plan_version": 0,
            "requirement_id": "headphones",
            "parent_asin": "B01",
            "reason": "preferred option",
        })
        self.assertIsInstance(decision, SelectCandidateDecision)
        self.assertEqual(decision.action, PlannerAction.SELECT_CANDIDATE)

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SelectRequirementDecision(
                plan_id="p", plan_version=0, requirement_id="dock", reason="first", price=1
            )

    def test_models_are_frozen(self) -> None:
        decision = SelectRequirementDecision(plan_id="p", plan_version=0, requirement_id="dock", reason="first")
        with self.assertRaises(ValidationError):
            decision.requirement_id = "mouse"

    def test_invalid_or_mismatched_action_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_planner_decision({
                "action": "delete_plan",
                "plan_id": "p",
                "plan_version": 0,
                "requirement_id": "dock",
                "reason": "invalid",
            })
        with self.assertRaises(ValidationError):
            SelectCandidateDecision(
                action="select_requirement",
                plan_id="p",
                plan_version=0,
                requirement_id="dock",
                parent_asin="B01",
                reason="invalid action",
            )

    def test_field_constraints_reject_invalid_values(self) -> None:
        with self.assertRaises(ValidationError):
            RequestRecommendationDecision(
                plan_id="p", plan_version=0, requirement_id="dock", top_k=0, reason="retrieve"
            )
        with self.assertRaises(ValidationError):
            SelectCandidateDecision(
                plan_id="p", plan_version=0, requirement_id="dock", parent_asin=" ", reason="candidate"
            )
        with self.assertRaises(ValidationError):
            ReplanProposalDecision(
                plan_id="p", plan_version=0,
                operation="increase_budget",
                proposed_value=float("nan"),
                reason="blocked",
            )
        with self.assertRaises(ValidationError):
            ReplanProposalDecision(
                plan_id="p", plan_version=0,
                operation="increase_budget",
                reason="blocked",
                requires_confirmation=False,
            )

    def test_fake_planner_is_protocol_compatible_and_deterministic(self) -> None:
        source = {
            "action": "select_requirement",
            "plan_id": "p",
            "plan_version": 0,
            "requirement_id": "dock",
            "reason": "highest priority",
        }
        planner = FakePlanner(source)
        self.assertIsInstance(planner, Planner)
        first = planner.decide(context={"plan_version": 1})
        second = planner.decide(context={"plan_version": 999})
        self.assertEqual(first, second)
        self.assertEqual(first.model_dump(mode="json"), second.model_dump(mode="json"))
        with self.assertRaises(TypeError):
            planner.decide(context=["not", "a", "mapping"])


if __name__ == "__main__":
    unittest.main()
