"""Cent-precise Goal execution-budget derivation for shopping workflows.

The helpers in this module translate one requirement-level subtotal allocation
into the conservative per-item ceiling required by RecommendationToolArgs. They
are pure: no workflow, plan, requirement, or allocation object is mutated.
"""

from __future__ import annotations

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

from ..domain import ShoppingPlan, ShoppingRequirement
from ..planning import GoalBudgetAllocation


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_CENT = Decimal("0.01")


class GoalBudgetDerivationError(ValueError):
    """Base failure for an invalid Goal execution-budget context."""


class GoalAllocationExhaustedError(GoalBudgetDerivationError):
    """The requirement still needs units but has no positive per-unit budget."""


def _money(value: float) -> Decimal:
    return Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP)


def _positive_money(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number.")
    result = _money(float(value))
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be positive at 0.01 precision.")
    return float(result)


def _nonnegative_money(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    result = _money(float(value))
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be non-negative at 0.01 precision.")
    return float(result)


class RecommendationBudgetProvenance(BaseModel):
    """Immutable inputs and output of one Goal recommendation budget decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    plan_version_before_call: Annotated[int, Field(strict=True, ge=0)]
    allocated_budget: float
    explicit_max_budget: float | None
    effective_subtotal_budget: float
    retained_subtotal_before_call: float
    remaining_quantity_before_call: Annotated[int, Field(strict=True, ge=1)]
    derived_unit_max_price: float

    @field_validator(
        "allocated_budget",
        "effective_subtotal_budget",
        "derived_unit_max_price",
        mode="before",
    )
    @classmethod
    def validate_positive_money(cls, value: object, info) -> float:
        return _positive_money(value, info.field_name)

    @field_validator("retained_subtotal_before_call", mode="before")
    @classmethod
    def validate_retained_subtotal(cls, value: object) -> float:
        return _nonnegative_money(value, "retained_subtotal_before_call")

    @field_validator("explicit_max_budget", mode="before")
    @classmethod
    def validate_explicit_cap(cls, value: object) -> object:
        if value is None:
            return None
        return _positive_money(value, "explicit_max_budget")

    @model_validator(mode="after")
    def validate_money_relationships(self) -> "RecommendationBudgetProvenance":
        allocated = _money(self.allocated_budget)
        effective = _money(self.effective_subtotal_budget)
        retained = _money(self.retained_subtotal_before_call)
        if effective > allocated:
            raise ValueError("effective subtotal cannot exceed allocated budget.")
        if self.explicit_max_budget is not None and effective > _money(
            self.explicit_max_budget
        ):
            raise ValueError("effective subtotal cannot exceed explicit max budget.")
        remaining = effective - retained
        if remaining <= 0:
            raise ValueError("executable provenance requires positive remaining budget.")
        expected = (remaining / self.remaining_quantity_before_call).quantize(
            _CENT, rounding=ROUND_FLOOR
        )
        if expected <= 0 or _money(self.derived_unit_max_price) != expected:
            raise ValueError("derived unit max price is inconsistent with budget inputs.")
        return self


def derive_goal_recommendation_budget(
    *,
    plan: ShoppingPlan,
    requirement: ShoppingRequirement,
    allocation: GoalBudgetAllocation,
) -> RecommendationBudgetProvenance:
    """Derive one conservative per-item cap from immutable Goal runtime facts."""

    if not isinstance(plan, ShoppingPlan):
        raise TypeError("plan must be a ShoppingPlan.")
    if not isinstance(requirement, ShoppingRequirement):
        raise TypeError("requirement must be a ShoppingRequirement.")
    if not isinstance(allocation, GoalBudgetAllocation):
        raise TypeError("allocation must be a GoalBudgetAllocation.")
    plan_matches = tuple(
        value
        for value in plan.requirements
        if value.requirement_id == requirement.requirement_id
    )
    if len(plan_matches) != 1 or plan_matches[0] != requirement:
        raise GoalBudgetDerivationError(
            "requirement must exactly match one ShoppingPlan requirement."
        )
    allocation_matches = tuple(
        value
        for value in allocation.allocations
        if value.requirement_id == requirement.requirement_id
    )
    if len(allocation_matches) != 1:
        raise GoalBudgetDerivationError(
            "current requirement must have exactly one Goal budget allocation."
        )

    allocated = _money(allocation_matches[0].allocated_budget)
    explicit = (
        None if requirement.max_budget is None else _money(requirement.max_budget)
    )
    effective = allocated if explicit is None else min(allocated, explicit)
    selected = tuple(
        item
        for item in plan.selected_items
        if item.requirement_id == requirement.requirement_id
    )
    retained = sum(
        (Decimal(str(item.price)) * item.quantity for item in selected),
        Decimal(0),
    ).quantize(_CENT, rounding=ROUND_HALF_UP)
    selected_quantity = sum(item.quantity for item in selected)
    remaining_quantity = requirement.quantity - selected_quantity
    if remaining_quantity == 0:
        raise GoalBudgetDerivationError(
            "requirement is already satisfied and cannot be recommended again."
        )
    if remaining_quantity < 0:
        raise GoalBudgetDerivationError(
            "selected quantity exceeds the current requirement quantity."
        )

    remaining_budget = effective - retained
    if remaining_budget <= 0:
        raise GoalAllocationExhaustedError(
            "Goal execution allocation is exhausted for the current requirement."
        )
    unit_cap = (remaining_budget / remaining_quantity).quantize(
        _CENT, rounding=ROUND_FLOOR
    )
    if unit_cap <= 0:
        raise GoalAllocationExhaustedError(
            "Goal execution allocation cannot fund one remaining unit."
        )
    return RecommendationBudgetProvenance(
        requirement_id=requirement.requirement_id,
        plan_version_before_call=plan.version,
        allocated_budget=float(allocated),
        explicit_max_budget=None if explicit is None else float(explicit),
        effective_subtotal_budget=float(effective),
        retained_subtotal_before_call=float(retained),
        remaining_quantity_before_call=remaining_quantity,
        derived_unit_max_price=float(unit_cap),
    )
