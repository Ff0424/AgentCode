"""Deterministic tests for V2-10.2 goal extraction contracts."""

from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from src.agentrec.planning import (
    AllocationPreferenceType,
    GoalAllocationPreference,
    GoalRequirementProposal,
    ShoppingGoalExtractionDecision,
)


def proposal(category: str = "Docking Stations", **values) -> GoalRequirementProposal:
    return GoalRequirementProposal(category=category, **values)


class GoalExtractionContractTests(unittest.TestCase):
    def test_valid_multi_requirement_goal(self) -> None:
        decision = ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(proposal(), proposal("Mouse"), proposal("Headphones")),
            allocation_preferences=(GoalAllocationPreference(
                target_index=1,
                preference=AllocationPreferenceType.SAVE_MORE,
            ),),
        )
        self.assertEqual(len(decision.requirement_proposals), 3)

    def test_single_requirement_goal_is_valid(self) -> None:
        decision = ShoppingGoalExtractionDecision(
            total_budget=100,
            requirement_proposals=(proposal("Mouse"),),
        )
        self.assertEqual(decision.requirement_proposals[0].category, "Mouse")

    def test_empty_and_too_many_proposals_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(total_budget=500, requirement_proposals=())
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=tuple(proposal(f"Category {i}") for i in range(11)),
            )

    def test_invalid_total_budget_is_rejected(self) -> None:
        for value in (0, -1, math.inf, math.nan, True, "500"):
            with self.subTest(value=value), self.assertRaises((ValidationError, TypeError)):
                ShoppingGoalExtractionDecision(
                    total_budget=value,
                    requirement_proposals=(proposal(),),
                )

    def test_invalid_quantity_and_priority_are_rejected(self) -> None:
        for value in (0, 101, True, 1.5):
            with self.subTest(quantity=value), self.assertRaises(ValidationError):
                proposal(quantity=value)
        for value in (0, 6, True, 1.5):
            with self.subTest(priority=value), self.assertRaises(ValidationError):
                proposal(priority=value)

    def test_feature_and_soft_preference_normalization_and_deduplication(self) -> None:
        value = proposal(
            required_features=("  USB-C  ", "USB-C", "usb-c", " HDMI  output "),
            soft_preferences=("  prefer   cheaper ", "prefer cheaper", "Quiet"),
        )
        self.assertEqual(value.required_features, ("USB-C", "HDMI output"))
        self.assertEqual(value.soft_preferences, ("prefer cheaper", "Quiet"))

    def test_negative_and_out_of_range_allocation_targets_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GoalAllocationPreference(
                target_index=-1,
                preference=AllocationPreferenceType.SAVE_MORE,
            )
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(proposal(),),
                allocation_preferences=(GoalAllocationPreference(
                    target_index=1,
                    preference=AllocationPreferenceType.ALLOCATE_MORE,
                ),),
            )

    def test_duplicate_allocation_preference_is_rejected(self) -> None:
        duplicate = GoalAllocationPreference(
            target_index=0,
            preference=AllocationPreferenceType.SAVE_MORE,
        )
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(proposal(),),
                allocation_preferences=(duplicate, duplicate),
            )

    def test_conflicting_allocation_preferences_for_one_target_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(proposal(), proposal("Mouse")),
                allocation_preferences=(
                    GoalAllocationPreference(
                        target_index=1,
                        preference=AllocationPreferenceType.SAVE_MORE,
                    ),
                    GoalAllocationPreference(
                        target_index=1,
                        preference=AllocationPreferenceType.ALLOCATE_MORE,
                    ),
                ),
            )

    def test_clarification_invariant(self) -> None:
        valid = ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(proposal(),),
            clarification_needed=True,
            clarification_question="  Which mouse style do you prefer?  ",
        )
        self.assertEqual(
            valid.clarification_question,
            "Which mouse style do you prefer?",
        )
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(proposal(),),
                clarification_needed=True,
            )
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(proposal(),),
                clarification_question="Not allowed",
            )

    def test_clarification_can_represent_missing_budget_or_requirements(self) -> None:
        missing_budget = ShoppingGoalExtractionDecision(
            total_budget=None,
            requirement_proposals=(proposal(), proposal("Mouse"), proposal("Headphones")),
            clarification_needed=True,
            clarification_question="What is your total budget?",
        )
        self.assertIsNone(missing_budget.total_budget)

        missing_requirements = ShoppingGoalExtractionDecision(
            total_budget=None,
            requirement_proposals=(),
            clarification_needed=True,
            clarification_question="Which product categories do you need?",
        )
        self.assertEqual(missing_requirements.requirement_proposals, ())

        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=None,
                requirement_proposals=(proposal(),),
            )
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(),
            )

    def test_requirement_proposal_order_is_preserved(self) -> None:
        decision = ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(
                proposal("Docking Stations"),
                proposal("Mouse"),
                proposal("Headphones"),
            ),
            allocation_preferences=(
                GoalAllocationPreference(
                    target_index=1,
                    preference=AllocationPreferenceType.SAVE_MORE,
                ),
                GoalAllocationPreference(
                    target_index=2,
                    preference=AllocationPreferenceType.ALLOCATE_MORE,
                ),
            ),
        )
        self.assertEqual(
            tuple(value.category for value in decision.requirement_proposals),
            ("Docking Stations", "Mouse", "Headphones"),
        )
        self.assertEqual(
            tuple(value.target_index for value in decision.allocation_preferences),
            (1, 2),
        )

    def test_contracts_are_frozen(self) -> None:
        requirement = proposal()
        allocation = GoalAllocationPreference(
            target_index=0,
            preference=AllocationPreferenceType.SAVE_MORE,
        )
        decision = ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(requirement,),
        )
        for value, field, replacement in (
            (requirement, "category", "Mouse"),
            (allocation, "target_index", 1),
            (decision, "total_budget", 600),
        ):
            with self.subTest(type=type(value).__name__), self.assertRaises(ValidationError):
                setattr(value, field, replacement)

    def test_identity_and_other_extra_fields_are_rejected(self) -> None:
        for field in ("requirement_id", "user_id", "plan_id", "session_id"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                GoalRequirementProposal.model_validate({
                    "category": "Mouse",
                    field: "forbidden",
                })
        with self.assertRaises(ValidationError):
            ShoppingGoalExtractionDecision(
                total_budget=500,
                requirement_proposals=(proposal(),),
                user_id="forbidden",
            )

    def test_golden_demo_total_budget_is_not_duplicated(self) -> None:
        decision = ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(
                proposal("Docking Stations", required_features=("HDMI",)),
                proposal("Mouse", soft_preferences=("prefer cheaper",)),
                proposal("Headphones"),
            ),
            allocation_preferences=(
                GoalAllocationPreference(
                    target_index=1,
                    preference=AllocationPreferenceType.SAVE_MORE,
                ),
                GoalAllocationPreference(
                    target_index=2,
                    preference=AllocationPreferenceType.ALLOCATE_MORE,
                ),
            ),
        )
        self.assertEqual(decision.total_budget, 500)
        self.assertEqual(
            tuple(value.max_budget for value in decision.requirement_proposals),
            (None, None, None),
        )


if __name__ == "__main__":
    unittest.main()
