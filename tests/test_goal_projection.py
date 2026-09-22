"""Deterministic tests for V2-10.3 goal-to-requirement projection."""

from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from src.agentrec.domain import RequirementStatus, ShoppingRequirement
from src.agentrec.planning import (
    AllocationPreferenceType,
    GoalAllocationPreference,
    GoalRequirementProjection,
    GoalRequirementProposal,
    GoalToRequirementProjector,
    RequirementAllocationPreference,
    ShoppingGoalExtractionDecision,
)


def requirement(requirement_id: str = "req-001") -> ShoppingRequirement:
    return ShoppingRequirement(requirement_id=requirement_id, category="Dock")


def golden_decision() -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        total_budget=500,
        requirement_proposals=(
            GoalRequirementProposal(
                category="Dock",
                required_features=("HDMI",),
                priority=5,
            ),
            GoalRequirementProposal(
                category="Mouse",
                soft_preferences=("prefer cheaper",),
                priority=1,
            ),
            GoalRequirementProposal(category="Headphones", priority=4),
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


class GoalProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = GoalToRequirementProjector()

    def test_golden_projection_and_total_budget_invariant(self) -> None:
        result = self.projector.project(golden_decision())
        self.assertEqual(result.total_budget, 500)
        self.assertEqual(
            tuple(value.max_budget for value in result.requirements),
            (None, None, None),
        )
        self.assertEqual(
            tuple((value.requirement_id, value.category) for value in result.requirements),
            (("req-001", "Dock"), ("req-002", "Mouse"), ("req-003", "Headphones")),
        )
        self.assertTrue(
            all(value.status is RequirementStatus.PENDING for value in result.requirements)
        )

    def test_exact_field_mapping(self) -> None:
        proposal = GoalRequirementProposal(
            category="Docking Stations",
            quantity=2,
            max_budget=250,
            required_features=("HDMI", "USB-C"),
            soft_preferences=("compact",),
            priority=4,
        )
        result = self.projector.project(ShoppingGoalExtractionDecision(
            total_budget=500,
            requirement_proposals=(proposal,),
        ))
        projected = result.requirements[0]
        self.assertEqual(projected.requirement_id, "req-001")
        self.assertEqual(projected.category, proposal.category)
        self.assertEqual(projected.quantity, proposal.quantity)
        self.assertEqual(projected.max_budget, proposal.max_budget)
        self.assertEqual(projected.required_features, proposal.required_features)
        self.assertEqual(projected.soft_preferences, proposal.soft_preferences)
        self.assertEqual(projected.priority, proposal.priority)
        self.assertIs(projected.status, RequirementStatus.PENDING)

    def test_allocation_indexes_become_requirement_identities(self) -> None:
        result = self.projector.project(golden_decision())
        self.assertEqual(
            result.allocation_preferences,
            (
                RequirementAllocationPreference(
                    requirement_id="req-002",
                    preference=AllocationPreferenceType.SAVE_MORE,
                ),
                RequirementAllocationPreference(
                    requirement_id="req-003",
                    preference=AllocationPreferenceType.ALLOCATE_MORE,
                ),
            ),
        )
        self.assertNotIn(
            "target_index",
            result.model_dump_json(),
        )

    def test_projection_is_deterministic(self) -> None:
        decision = golden_decision()
        first = self.projector.project(decision)
        second = self.projector.project(decision)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(value.requirement_id for value in first.requirements),
            tuple(value.requirement_id for value in second.requirements),
        )
        self.assertEqual(first.allocation_preferences, second.allocation_preferences)

    def test_proposal_order_is_preserved_even_when_priorities_differ(self) -> None:
        result = self.projector.project(golden_decision())
        self.assertEqual(
            tuple((value.requirement_id, value.category) for value in result.requirements),
            (("req-001", "Dock"), ("req-002", "Mouse"), ("req-003", "Headphones")),
        )
        self.assertEqual(tuple(value.priority for value in result.requirements), (5, 1, 4))

    def test_clarification_with_empty_or_nonempty_proposals_is_rejected(self) -> None:
        decisions = (
            ShoppingGoalExtractionDecision(
                clarification_needed=True,
                clarification_question="What products and budget do you need?",
            ),
            ShoppingGoalExtractionDecision(
                requirement_proposals=(GoalRequirementProposal(category="Dock"),),
                clarification_needed=True,
                clarification_question="What is your total budget?",
            ),
        )
        for decision in decisions:
            with self.subTest(proposals=len(decision.requirement_proposals)):
                with self.assertRaisesRegex(
                    ValueError,
                    "Goal clarification must be resolved before projection.",
                ):
                    self.projector.project(decision)

    def test_requirement_allocation_contract_validation(self) -> None:
        value = RequirementAllocationPreference(
            requirement_id=" req-001 ",
            preference=AllocationPreferenceType.SAVE_MORE,
        )
        self.assertEqual(value.requirement_id, "req-001")
        with self.assertRaises(ValidationError):
            RequirementAllocationPreference(
                requirement_id=" ",
                preference=AllocationPreferenceType.SAVE_MORE,
            )
        with self.assertRaises(ValidationError):
            RequirementAllocationPreference(
                requirement_id="req-001",
                preference=AllocationPreferenceType.SAVE_MORE,
                target_index=0,
            )
        with self.assertRaises(ValidationError):
            value.requirement_id = "req-002"

    def test_projection_contract_rejects_invalid_shape_and_is_frozen(self) -> None:
        for budget in (0, -1, math.inf, math.nan, True, "500"):
            with self.subTest(budget=budget), self.assertRaises(
                (ValidationError, TypeError)
            ):
                GoalRequirementProjection(total_budget=budget, requirements=(requirement(),))
        with self.assertRaises(ValidationError):
            GoalRequirementProjection(total_budget=500, requirements=())
        with self.assertRaises(ValidationError):
            GoalRequirementProjection(
                total_budget=500,
                requirements=(requirement(),),
                unexpected=True,
            )
        value = GoalRequirementProjection(total_budget=500, requirements=(requirement(),))
        with self.assertRaises(ValidationError):
            value.total_budget = 600

    def test_projection_contract_rejects_identity_errors(self) -> None:
        with self.assertRaises(ValidationError):
            GoalRequirementProjection(
                total_budget=500,
                requirements=(requirement(), requirement()),
            )
        with self.assertRaises(ValidationError):
            GoalRequirementProjection(
                total_budget=500,
                requirements=(requirement(),),
                allocation_preferences=(RequirementAllocationPreference(
                    requirement_id="req-999",
                    preference=AllocationPreferenceType.SAVE_MORE,
                ),),
            )
        with self.assertRaises(ValidationError):
            GoalRequirementProjection(
                total_budget=500,
                requirements=(requirement(),),
                allocation_preferences=(
                    RequirementAllocationPreference(
                        requirement_id="req-001",
                        preference=AllocationPreferenceType.SAVE_MORE,
                    ),
                    RequirementAllocationPreference(
                        requirement_id="req-001",
                        preference=AllocationPreferenceType.ALLOCATE_MORE,
                    ),
                ),
            )

    def test_projector_rejects_wrong_input_type(self) -> None:
        with self.assertRaises(TypeError):
            self.projector.project({})


if __name__ == "__main__":
    unittest.main()
