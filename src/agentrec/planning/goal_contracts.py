"""Immutable contracts for multi-requirement shopping-goal extraction.

These models represent a validated intermediate result produced from one
natural-language shopping goal. They do not own system identities, create a
ShoppingPlan, allocate money, or execute recommendation and workflow logic.
"""

from __future__ import annotations

import math
import numbers
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ClarificationText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]


def _finite_positive(value: object, field_name: str) -> float:
    """Validate money without accepting bool as an integer-like number."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field_name} must be a finite positive number.")
    amount = float(value)
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"{field_name} must be a finite positive number.")
    return amount


def _normalize_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    """Collapse whitespace and preserve the first occurrence in stable order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(value.split())
        key = item.casefold()
        if key not in seen:
            normalized.append(item)
            seen.add(key)
    return tuple(normalized)


class AllocationPreferenceType(str, Enum):
    """Minimal relative budget-allocation intent across requirements."""

    SAVE_MORE = "save_more"
    ALLOCATE_MORE = "allocate_more"


class GoalRequirementProposal(BaseModel):
    """One identity-free requirement proposed during goal extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: NonEmptyText
    quantity: Annotated[int, Field(strict=True, ge=1, le=100)] = 1
    max_budget: float | None = None
    required_features: Annotated[
        tuple[NonEmptyText, ...], Field(max_length=20)
    ] = ()
    soft_preferences: Annotated[
        tuple[NonEmptyText, ...], Field(max_length=20)
    ] = ()
    priority: Annotated[int, Field(strict=True, ge=1, le=5)] = 3

    @field_validator("max_budget", mode="before")
    @classmethod
    def validate_max_budget(cls, value: object) -> object:
        if value is None:
            return None
        return _finite_positive(value, "max_budget")

    @field_validator("required_features", "soft_preferences", mode="after")
    @classmethod
    def normalize_and_deduplicate(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_unique(values)


class GoalAllocationPreference(BaseModel):
    """One soft relative allocation preference targeting a proposal index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_index: Annotated[int, Field(strict=True, ge=0)]
    preference: AllocationPreferenceType


class ShoppingGoalExtractionDecision(BaseModel):
    """Validated multi-requirement extraction result before domain projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_budget: float | None = None
    requirement_proposals: Annotated[
        tuple[GoalRequirementProposal, ...], Field(max_length=10)
    ] = ()
    allocation_preferences: tuple[GoalAllocationPreference, ...] = ()
    clarification_needed: Annotated[bool, Field(strict=True)] = False
    clarification_question: ClarificationText | None = None

    @field_validator("total_budget", mode="before")
    @classmethod
    def validate_total_budget(cls, value: object) -> object:
        if value is None:
            return None
        return _finite_positive(value, "total_budget")

    @model_validator(mode="after")
    def validate_cross_field_invariants(self) -> "ShoppingGoalExtractionDecision":
        proposal_count = len(self.requirement_proposals)
        for preference in self.allocation_preferences:
            if preference.target_index >= proposal_count:
                raise ValueError(
                    "Allocation preference target_index must reference an existing "
                    "requirement proposal."
                )

        allocation_targets = tuple(
            value.target_index for value in self.allocation_preferences
        )
        if len(set(allocation_targets)) != len(allocation_targets):
            raise ValueError(
                "Each requirement proposal may have at most one allocation preference."
            )

        if self.clarification_needed and self.clarification_question is None:
            raise ValueError(
                "clarification_question is required when clarification_needed=true."
            )
        if not self.clarification_needed and self.clarification_question is not None:
            raise ValueError(
                "clarification_question must be None when clarification_needed=false."
            )
        if not self.clarification_needed and self.total_budget is None:
            raise ValueError(
                "total_budget is required when clarification_needed=false."
            )
        if not self.clarification_needed and not self.requirement_proposals:
            raise ValueError(
                "At least one requirement proposal is required when "
                "clarification_needed=false."
            )
        return self
