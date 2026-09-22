"""Deterministic tests for V2-10.4 goal budget allocation."""

from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from src.agentrec.domain import ShoppingRequirement
from src.agentrec.planning import (
    AllocationPreferenceType,
    DeterministicBudgetAllocator,
    GoalBudgetAllocation,
    GoalRequirementProjection,
    RequirementAllocationPreference,
    RequirementBudgetAllocation,
)


def projection(
    total: float,
    requirements: tuple[ShoppingRequirement, ...],
    preferences: tuple[RequirementAllocationPreference, ...] = (),
) -> GoalRequirementProjection:
    return GoalRequirementProjection(
        total_budget=total,
        requirements=requirements,
        allocation_preferences=preferences,
    )


def requirement(
    identifier: str,
    category: str,
    *,
    max_budget: float | None = None,
    quantity: int = 1,
    priority: int = 3,
) -> ShoppingRequirement:
    return ShoppingRequirement(
        requirement_id=identifier,
        category=category,
        max_budget=max_budget,
        quantity=quantity,
        priority=priority,
    )


class BudgetAllocationContractTests(unittest.TestCase):
    def test_requirement_allocation_contract_is_strict_and_frozen(self) -> None:
        value = RequirementBudgetAllocation(
            requirement_id=" req-001 ", allocated_budget=10.126
        )
        self.assertEqual(value.requirement_id, "req-001")
        self.assertEqual(value.allocated_budget, 10.13)
        with self.assertRaises(ValidationError):
            value.allocated_budget = 20
        with self.assertRaises(ValidationError):
            RequirementBudgetAllocation(
                requirement_id="req-001", allocated_budget=10, extra=True
            )

    def test_requirement_allocation_rejects_invalid_values(self) -> None:
        for identifier in ("", "   "):
            with self.subTest(identifier=identifier), self.assertRaises(ValidationError):
                RequirementBudgetAllocation(
                    requirement_id=identifier, allocated_budget=0
                )
        for amount in (-1, math.nan, math.inf, True, "1"):
            with self.subTest(amount=amount), self.assertRaises(
                (ValidationError, TypeError)
            ):
                RequirementBudgetAllocation(
                    requirement_id="req-001", allocated_budget=amount
                )

    def test_goal_allocation_validates_shape_and_conservation(self) -> None:
        first = RequirementBudgetAllocation(
            requirement_id="req-001", allocated_budget=60
        )
        second = RequirementBudgetAllocation(
            requirement_id="req-002", allocated_budget=30
        )
        value = GoalBudgetAllocation(
            total_budget=100,
            allocations=(first, second),
            unallocated_budget=10,
        )
        with self.assertRaises(ValidationError):
            value.total_budget = 200
        with self.assertRaises(ValidationError):
            GoalBudgetAllocation(
                total_budget=100,
                allocations=(first, second),
                unallocated_budget=10,
                extra=True,
            )
        with self.assertRaises(ValidationError):
            GoalBudgetAllocation(
                total_budget=100,
                allocations=(first, first),
                unallocated_budget=0,
            )
        with self.assertRaises(ValidationError):
            GoalBudgetAllocation(
                total_budget=100,
                allocations=(first,),
                unallocated_budget=-1,
            )
        with self.assertRaises(ValidationError):
            GoalBudgetAllocation(
                total_budget=100,
                allocations=(first,),
                unallocated_budget=0,
            )
        for invalid in (0, -1, math.nan, math.inf, True, "100"):
            with self.subTest(total=invalid), self.assertRaises(
                (ValidationError, TypeError)
            ):
                GoalBudgetAllocation(
                    total_budget=invalid,
                    allocations=(first,),
                    unallocated_budget=40,
                )


class DeterministicBudgetAllocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = DeterministicBudgetAllocator()

    @staticmethod
    def golden(max_dock: float | None = None) -> GoalRequirementProjection:
        return projection(
            500,
            (
                requirement("req-001", "Dock", max_budget=max_dock),
                requirement("req-002", "Mouse"),
                requirement("req-003", "Headphones"),
            ),
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

    def test_golden_weighted_allocation(self) -> None:
        source = self.golden()
        before = source.model_dump()
        result = self.allocator.allocate(source)
        self.assertEqual(
            tuple((value.requirement_id, value.allocated_budget) for value in result.allocations),
            (("req-001", 166.67), ("req-002", 83.33), ("req-003", 250.0)),
        )
        self.assertEqual(result.unallocated_budget, 0.0)
        self.assertEqual(tuple(value.max_budget for value in source.requirements), (None,) * 3)
        self.assertEqual(source.model_dump(), before)

    def test_explicit_cap_is_a_ceiling_and_redistributes(self) -> None:
        source = self.golden(120)
        result = self.allocator.allocate(source)
        self.assertEqual(
            tuple(value.allocated_budget for value in result.allocations),
            (120.0, 95.0, 285.0),
        )
        self.assertEqual(result.unallocated_budget, 0.0)
        self.assertEqual(source.requirements[0].max_budget, 120)

    def test_cap_sum_greater_than_total_is_not_a_conflict(self) -> None:
        result = self.allocator.allocate(projection(
            100,
            (
                requirement("req-a", "Accessory", max_budget=80),
                requirement("req-b", "Accessory", max_budget=80),
            ),
        ))
        self.assertEqual(
            tuple(value.allocated_budget for value in result.allocations),
            (50.0, 50.0),
        )

    def test_cap_sum_less_than_total_becomes_unallocated(self) -> None:
        result = self.allocator.allocate(projection(
            500,
            (
                requirement("req-a", "A", max_budget=100),
                requirement("req-b", "B", max_budget=150),
            ),
        ))
        self.assertEqual(
            tuple(value.allocated_budget for value in result.allocations),
            (100.0, 150.0),
        )
        self.assertEqual(result.unallocated_budget, 250.0)

    def test_single_requirement_with_and_without_cap(self) -> None:
        uncapped = self.allocator.allocate(projection(
            500, (requirement("req-001", "Dock"),)
        ))
        capped = self.allocator.allocate(projection(
            500, (requirement("req-001", "Dock", max_budget=120),)
        ))
        self.assertEqual(uncapped.allocations[0].allocated_budget, 500)
        self.assertEqual(uncapped.unallocated_budget, 0)
        self.assertEqual(capped.allocations[0].allocated_budget, 120)
        self.assertEqual(capped.unallocated_budget, 380)

    def test_small_budget_uses_stable_remainder_order(self) -> None:
        result = self.allocator.allocate(projection(
            0.02,
            tuple(requirement(f"req-{index}", "Same") for index in range(1, 5)),
        ))
        self.assertEqual(
            tuple(value.allocated_budget for value in result.allocations),
            (0.01, 0.01, 0.0, 0.0),
        )

    def test_duplicate_categories_remain_identity_distinct(self) -> None:
        result = self.allocator.allocate(projection(
            10,
            (
                requirement("req-001", "Mouse"),
                requirement("req-002", "Mouse"),
            ),
        ))
        self.assertEqual(
            tuple(value.requirement_id for value in result.allocations),
            ("req-001", "req-002"),
        )
        self.assertEqual(
            tuple(value.allocated_budget for value in result.allocations),
            (5.0, 5.0),
        )

    def test_quantity_and_priority_do_not_change_allocation(self) -> None:
        result = self.allocator.allocate(projection(
            200,
            (
                requirement("req-001", "Mouse", quantity=2, priority=5),
                requirement("req-002", "Keyboard", quantity=1, priority=1),
            ),
        ))
        self.assertEqual(
            tuple(value.allocated_budget for value in result.allocations),
            (100.0, 100.0),
        )

    def test_determinism_order_and_input_immutability(self) -> None:
        source = self.golden(120)
        snapshot = source.model_dump_json()
        first = self.allocator.allocate(source)
        second = self.allocator.allocate(source)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(value.requirement_id for value in first.allocations),
            tuple(value.requirement_id for value in source.requirements),
        )
        self.assertEqual(source.model_dump_json(), snapshot)

    def test_wrong_input_type_and_sub_cent_total_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.allocator.allocate({})
        source = projection(0.001, (requirement("req-001", "A"),))
        with self.assertRaisesRegex(ValueError, "at least 0.01"):
            self.allocator.allocate(source)


if __name__ == "__main__":
    unittest.main()
