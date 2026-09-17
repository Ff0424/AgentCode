"""Immutable Shopping Plan domain models for AgentRec V2.

The models contain business facts only. Controlled operations return a new
versioned plan, while spend, remaining budget, conflicts, and readiness are
derived to avoid duplicated state. No runtime service, model, or persistence
dependency is stored here.
"""

from __future__ import annotations

import math
import numbers
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RequirementStatus(str, Enum):
    PENDING = "pending"
    CANDIDATE_SELECTED = "candidate_selected"
    SATISFIED = "satisfied"
    CONFLICT = "conflict"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    CONFLICT = "conflict"
    COMPLETED = "completed"


class ItemSource(str, Enum):
    HYBRID = "hybrid"
    POPULARITY_FALLBACK = "popularity_fallback"
    MANUAL = "manual"
    REPLACEMENT = "replacement"


class ShoppingRequirement(BaseModel):
    """One required purchase slot with separated hard and soft constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
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
    status: RequirementStatus = RequirementStatus.PENDING

    @field_validator("max_budget", mode="before")
    @classmethod
    def validate_max_budget(cls, value: object) -> object:
        if value is None:
            return None
        return _positive_money(value, "max_budget")


class ShoppingPlanItem(BaseModel):
    """One selected product with immutable identity and selection provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: NonEmptyText
    requirement_id: NonEmptyText
    category: NonEmptyText
    parent_asin: NonEmptyText
    title: NonEmptyText
    price: float
    quantity: Annotated[int, Field(strict=True, ge=1, le=100)] = 1
    selected_reason: NonEmptyText
    constraints_satisfied: Annotated[bool, Field(strict=True)]
    source: ItemSource
    recommendation_score: float | None = None

    @field_validator("price", mode="before")
    @classmethod
    def validate_price(cls, value: object) -> float:
        return _positive_money(value, "price")

    @field_validator("recommendation_score", mode="before")
    @classmethod
    def validate_recommendation_score(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("recommendation_score must be a finite number.")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("recommendation_score must be a finite number.")
        return score


def _positive_money(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite positive number.")
    amount = float(value)
    if not math.isfinite(amount) or amount <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return amount


def _normalized_category(value: str) -> str:
    return " ".join(value.split()).casefold()


class ShoppingPlan(BaseModel):
    """Versioned shopping task whose totals and conflicts are always derived."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyText
    user_id: NonEmptyText
    currency: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=3, to_upper=True)
    ] = "USD"
    total_budget: float
    requirements: Annotated[
        tuple[ShoppingRequirement, ...], Field(min_length=1, max_length=100)
    ]
    selected_items: tuple[ShoppingPlanItem, ...] = ()
    status: PlanStatus = PlanStatus.DRAFT
    version: Annotated[int, Field(strict=True, ge=0)] = 0

    @field_validator("total_budget", mode="before")
    @classmethod
    def validate_total_budget(cls, value: object) -> float:
        return _positive_money(value, "total_budget")

    @model_validator(mode="after")
    def validate_domain_invariants(self) -> "ShoppingPlan":
        requirements = {value.requirement_id: value for value in self.requirements}
        if len(requirements) != len(self.requirements):
            raise ValueError("requirement_id values must be unique.")
        if len({item.slot_id for item in self.selected_items}) != len(self.selected_items):
            raise ValueError("slot_id values must be unique.")
        if len({item.parent_asin for item in self.selected_items}) != len(self.selected_items):
            raise ValueError("A parent_asin cannot appear in multiple selected item records.")
        quantities = {key: 0 for key in requirements}
        for item in self.selected_items:
            requirement = requirements.get(item.requirement_id)
            if requirement is None:
                raise ValueError(
                    f"Selected item references unknown requirement_id={item.requirement_id!r}."
                )
            if _normalized_category(item.category) != _normalized_category(requirement.category):
                raise ValueError(
                    f"Selected item category does not match requirement {item.requirement_id!r}."
                )
            quantities[item.requirement_id] += item.quantity
            if quantities[item.requirement_id] > requirement.quantity:
                raise ValueError(
                    f"Selected quantity exceeds requirement {item.requirement_id!r}."
                )
        expected_requirement_statuses: dict[str, RequirementStatus] = {}
        for requirement in self.requirements:
            selected = tuple(
                item
                for item in self.selected_items
                if item.requirement_id == requirement.requirement_id
            )
            subtotal = math.fsum(item.price * item.quantity for item in selected)
            conflict = any(not item.constraints_satisfied for item in selected) or (
                requirement.max_budget is not None and subtotal > requirement.max_budget
            )
            if conflict:
                expected = RequirementStatus.CONFLICT
            elif quantities[requirement.requirement_id] == requirement.quantity:
                expected = RequirementStatus.SATISFIED
            elif quantities[requirement.requirement_id]:
                expected = RequirementStatus.CANDIDATE_SELECTED
            else:
                expected = RequirementStatus.PENDING
            expected_requirement_statuses[requirement.requirement_id] = expected
            if requirement.status != expected:
                raise ValueError(
                    f"Requirement {requirement.requirement_id!r} status is inconsistent."
                )
        conflict = self.total_spent > self.total_budget or any(
            status == RequirementStatus.CONFLICT
            for status in expected_requirement_statuses.values()
        )
        if conflict:
            expected_plan_status = PlanStatus.CONFLICT
        elif all(
            status == RequirementStatus.SATISFIED
            for status in expected_requirement_statuses.values()
        ):
            expected_plan_status = PlanStatus.READY
        elif self.selected_items:
            expected_plan_status = PlanStatus.IN_PROGRESS
        else:
            expected_plan_status = PlanStatus.DRAFT
        if self.status == PlanStatus.COMPLETED:
            if expected_plan_status != PlanStatus.READY:
                raise ValueError("Only a valid ready plan can have completed status.")
        elif self.status != expected_plan_status:
            raise ValueError("ShoppingPlan status is inconsistent with its business facts.")
        return self

    @property
    def total_spent(self) -> float:
        return math.fsum(item.price * item.quantity for item in self.selected_items)

    @property
    def remaining_budget(self) -> float:
        return self.total_budget - self.total_spent

    @property
    def is_over_budget(self) -> bool:
        return self.total_spent > self.total_budget

    @property
    def has_hard_constraint_conflict(self) -> bool:
        if self.is_over_budget or any(
            not item.constraints_satisfied for item in self.selected_items
        ):
            return True
        for requirement in self.requirements:
            if requirement.max_budget is None:
                continue
            subtotal = math.fsum(
                item.price * item.quantity
                for item in self.selected_items
                if item.requirement_id == requirement.requirement_id
            )
            if subtotal > requirement.max_budget:
                return True
        return False

    @property
    def is_valid(self) -> bool:
        return self.status in {PlanStatus.READY, PlanStatus.COMPLETED} and not (
            self.has_hard_constraint_conflict
        )

    def _synchronized_requirements(
        self, items: tuple[ShoppingPlanItem, ...]
    ) -> tuple[ShoppingRequirement, ...]:
        synchronized = []
        for requirement in self.requirements:
            selected = tuple(
                item for item in items if item.requirement_id == requirement.requirement_id
            )
            quantity = sum(item.quantity for item in selected)
            subtotal = math.fsum(item.price * item.quantity for item in selected)
            conflict = any(not item.constraints_satisfied for item in selected) or (
                requirement.max_budget is not None and subtotal > requirement.max_budget
            )
            if conflict:
                status = RequirementStatus.CONFLICT
            elif quantity == requirement.quantity:
                status = RequirementStatus.SATISFIED
            elif quantity:
                status = RequirementStatus.CANDIDATE_SELECTED
            else:
                status = RequirementStatus.PENDING
            synchronized.append(requirement.model_copy(update={"status": status}))
        return tuple(synchronized)

    def _mutated(self, items: tuple[ShoppingPlanItem, ...]) -> "ShoppingPlan":
        requirements = self._synchronized_requirements(items)
        total = math.fsum(item.price * item.quantity for item in items)
        conflict = total > self.total_budget or any(
            requirement.status == RequirementStatus.CONFLICT
            for requirement in requirements
        )
        if conflict:
            status = PlanStatus.CONFLICT
        elif all(
            requirement.status == RequirementStatus.SATISFIED
            for requirement in requirements
        ):
            status = PlanStatus.READY
        elif items:
            status = PlanStatus.IN_PROGRESS
        else:
            status = PlanStatus.DRAFT
        return self.model_copy(
            update={
                "requirements": requirements,
                "selected_items": items,
                "status": status,
                "version": self.version + 1,
            }
        )

    def add_item(self, item: ShoppingPlanItem) -> "ShoppingPlan":
        if not isinstance(item, ShoppingPlanItem):
            raise TypeError("item must be a ShoppingPlanItem.")
        if self.status == PlanStatus.COMPLETED:
            raise ValueError("A completed ShoppingPlan cannot be mutated.")
        # Full model validation on the returned plan enforces identity/category/quantity.
        candidate = self._mutated((*self.selected_items, item))
        return type(self).model_validate(candidate.model_dump())

    def replace_item(self, slot_id: str, replacement: ShoppingPlanItem) -> "ShoppingPlan":
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("slot_id must be a non-empty string.")
        if not isinstance(replacement, ShoppingPlanItem):
            raise TypeError("replacement must be a ShoppingPlanItem.")
        if self.status == PlanStatus.COMPLETED:
            raise ValueError("A completed ShoppingPlan cannot be mutated.")
        matches = [index for index, item in enumerate(self.selected_items) if item.slot_id == slot_id.strip()]
        if len(matches) != 1:
            raise KeyError(f"Selected slot_id={slot_id.strip()!r} does not exist.")
        items = list(self.selected_items)
        items[matches[0]] = replacement
        candidate = self._mutated(tuple(items))
        return type(self).model_validate(candidate.model_dump())

    def remove_item(self, slot_id: str) -> "ShoppingPlan":
        if not isinstance(slot_id, str) or not slot_id.strip():
            raise ValueError("slot_id must be a non-empty string.")
        if self.status == PlanStatus.COMPLETED:
            raise ValueError("A completed ShoppingPlan cannot be mutated.")
        normalized = slot_id.strip()
        items = tuple(item for item in self.selected_items if item.slot_id != normalized)
        if len(items) == len(self.selected_items):
            raise KeyError(f"Selected slot_id={normalized!r} does not exist.")
        candidate = self._mutated(items)
        return type(self).model_validate(candidate.model_dump())

    def mark_completed(self) -> "ShoppingPlan":
        if self.status != PlanStatus.READY:
            raise ValueError("Only a ready ShoppingPlan can be completed.")
        return self.model_copy(
            update={"status": PlanStatus.COMPLETED, "version": self.version + 1}
        )

    def to_domain_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible stored fields only."""

        return self.model_dump(mode="json")

    @classmethod
    def from_domain_dict(cls, value: dict[str, Any]) -> "ShoppingPlan":
        return cls.model_validate(value)
