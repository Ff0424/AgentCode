"""Deterministic goal-level budget allocation for AgentRec V2.

The allocator converts a completed ``GoalRequirementProjection`` into an
immutable execution-budget allocation.  Explicit ``ShoppingRequirement``
``max_budget`` values remain user-owned hard subtotal ceilings; derived
allocations are stored separately and never mutate the projection.
"""

from __future__ import annotations

import math
import numbers
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .goal_contracts import AllocationPreferenceType
from .goal_projection import GoalRequirementProjection


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_CENT = Decimal("0.01")
_HUNDRED = Decimal(100)
_WEIGHTS = {
    None: Decimal("1.0"),
    AllocationPreferenceType.SAVE_MORE: Decimal("0.5"),
    AllocationPreferenceType.ALLOCATE_MORE: Decimal("1.5"),
}


def _money(value: object, name: str, *, positive: bool) -> float:
    """Validate and normalize one monetary value to P0 cent precision."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        qualifier = "positive" if positive else "non-negative"
        raise TypeError(f"{name} must be a finite {qualifier} number.")
    numeric = float(value)
    if not math.isfinite(numeric) or (numeric <= 0 if positive else numeric < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    normalized = Decimal(str(numeric)).quantize(_CENT, rounding=ROUND_HALF_UP)
    if positive and normalized <= 0:
        raise ValueError(f"{name} must be at least 0.01 at P0 monetary precision.")
    return float(normalized)


def _to_cents(value: float) -> int:
    normalized = Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)
    return int(normalized * _HUNDRED)


def _from_cents(value: int) -> float:
    return float(Decimal(value) / _HUNDRED)


class RequirementBudgetAllocation(BaseModel):
    """One system-derived execution budget bound to a requirement identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    allocated_budget: float

    @field_validator("allocated_budget", mode="before")
    @classmethod
    def validate_allocated_budget(cls, value: object) -> float:
        return _money(value, "allocated_budget", positive=False)


class GoalBudgetAllocation(BaseModel):
    """A conserved, ordered allocation of one goal-level total budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_budget: float
    allocations: Annotated[
        tuple[RequirementBudgetAllocation, ...], Field(min_length=1)
    ]
    unallocated_budget: float

    @field_validator("total_budget", mode="before")
    @classmethod
    def validate_total_budget(cls, value: object) -> float:
        return _money(value, "total_budget", positive=True)

    @field_validator("unallocated_budget", mode="before")
    @classmethod
    def validate_unallocated_budget(cls, value: object) -> float:
        return _money(value, "unallocated_budget", positive=False)

    @model_validator(mode="after")
    def validate_identity_and_conservation(self) -> "GoalBudgetAllocation":
        identifiers = tuple(value.requirement_id for value in self.allocations)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Allocation requirement_id values must be unique.")
        allocated_cents = sum(
            _to_cents(value.allocated_budget) for value in self.allocations
        )
        if allocated_cents + _to_cents(self.unallocated_budget) != _to_cents(
            self.total_budget
        ):
            raise ValueError(
                "Allocated and unallocated budgets must conserve total_budget "
                "at 0.01 precision."
            )
        return self


class DeterministicBudgetAllocator:
    """Allocate a goal budget with fixed weights and hard-cap water-filling."""

    def allocate(self, projection: GoalRequirementProjection) -> GoalBudgetAllocation:
        if not isinstance(projection, GoalRequirementProjection):
            raise TypeError("projection must be a GoalRequirementProjection.")

        total_cents = _to_cents(projection.total_budget)
        if total_cents <= 0:
            raise ValueError("projection.total_budget must round to at least 0.01.")

        preference_by_id = {
            value.requirement_id: value.preference
            for value in projection.allocation_preferences
        }
        weights = tuple(
            _WEIGHTS[preference_by_id.get(requirement.requirement_id)]
            for requirement in projection.requirements
        )
        caps = tuple(
            None
            if requirement.max_budget is None
            else _to_cents(requirement.max_budget)
            for requirement in projection.requirements
        )

        allocated = [0] * len(projection.requirements)
        active = list(range(len(projection.requirements)))
        remaining = total_cents

        # Repeatedly remove requirements whose proportional share exceeds their
        # explicit hard ceiling, then redistribute the remaining cents.
        while active and remaining:
            total_weight = sum(weights[index] for index in active)
            capped = [
                index
                for index in active
                if caps[index] is not None
                and Decimal(remaining) * weights[index] / total_weight
                > caps[index]
            ]
            if not capped:
                self._allocate_largest_remainders(
                    active=active,
                    remaining=remaining,
                    weights=weights,
                    caps=caps,
                    allocated=allocated,
                )
                remaining = 0
                break

            for index in capped:
                cap = caps[index]
                assert cap is not None  # established by the capped predicate
                allocated[index] = cap
                remaining -= cap
            capped_set = set(capped)
            active = [index for index in active if index not in capped_set]

        return GoalBudgetAllocation(
            total_budget=_from_cents(total_cents),
            allocations=tuple(
                RequirementBudgetAllocation(
                    requirement_id=requirement.requirement_id,
                    allocated_budget=_from_cents(allocated[index]),
                )
                for index, requirement in enumerate(projection.requirements)
            ),
            unallocated_budget=_from_cents(remaining),
        )

    @staticmethod
    def _allocate_largest_remainders(
        *,
        active: list[int],
        remaining: int,
        weights: tuple[Decimal, ...],
        caps: tuple[int | None, ...],
        allocated: list[int],
    ) -> None:
        """Allocate all remaining cents with stable largest-remainder ties."""

        total_weight = sum(weights[index] for index in active)
        exact = {
            index: Decimal(remaining) * weights[index] / total_weight
            for index in active
        }
        floors = {
            index: int(exact[index].to_integral_value(rounding=ROUND_FLOOR))
            for index in active
        }
        for index, amount in floors.items():
            allocated[index] = amount

        remainder = remaining - sum(floors.values())
        order = sorted(
            active,
            key=lambda index: (-(exact[index] - floors[index]), index),
        )
        for index in order:
            if remainder == 0:
                break
            cap = caps[index]
            if cap is None or allocated[index] < cap:
                allocated[index] += 1
                remainder -= 1
        if remainder:
            raise RuntimeError("Largest-remainder allocation could not conserve cents.")
