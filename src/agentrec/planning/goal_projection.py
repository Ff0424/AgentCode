"""Deterministic projection from goal extraction into domain requirements.

This planning-layer bridge binds proposal indexes to stable plan-local
requirement identities. It does not create a ShoppingPlan, allocate money,
invoke runtime services, or resolve clarification.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..domain import ShoppingRequirement
from .constraint_grounding import DeterministicConstraintGrounder
from .goal_contracts import (
    AllocationPreferenceType,
    ShoppingGoalExtractionDecision,
    _finite_positive,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RequirementAllocationPreference(BaseModel):
    """Identity-bound soft allocation preference produced by trusted projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: NonEmptyText
    preference: AllocationPreferenceType


class GoalRequirementProjection(BaseModel):
    """Immutable requirements and goal-level facts ready for later plan creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_budget: float
    requirements: Annotated[
        tuple[ShoppingRequirement, ...], Field(min_length=1, max_length=10)
    ]
    allocation_preferences: tuple[RequirementAllocationPreference, ...] = ()

    @field_validator("total_budget", mode="before")
    @classmethod
    def validate_total_budget(cls, value: object) -> float:
        return _finite_positive(value, "total_budget")

    @model_validator(mode="after")
    def validate_identity_bindings(self) -> "GoalRequirementProjection":
        requirement_ids = tuple(value.requirement_id for value in self.requirements)
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("Projected requirement_id values must be unique.")

        allocation_ids = tuple(
            value.requirement_id for value in self.allocation_preferences
        )
        if any(value not in set(requirement_ids) for value in allocation_ids):
            raise ValueError(
                "Every allocation preference must reference a projected requirement."
            )
        if len(set(allocation_ids)) != len(allocation_ids):
            raise ValueError(
                "Each requirement may have at most one allocation preference."
            )
        return self


class GoalToRequirementProjector:
    """Project one complete goal decision without side effects or randomness."""

    def __init__(
        self,
        constraint_grounder: DeterministicConstraintGrounder | None = None,
    ) -> None:
        self._constraint_grounder = (
            constraint_grounder or DeterministicConstraintGrounder()
        )

    def project(
        self,
        decision: ShoppingGoalExtractionDecision,
    ) -> GoalRequirementProjection:
        if not isinstance(decision, ShoppingGoalExtractionDecision):
            raise TypeError("decision must be a ShoppingGoalExtractionDecision.")
        if decision.clarification_needed:
            raise ValueError("Goal clarification must be resolved before projection.")
        # Complete decisions are contract-validated to contain both values.
        if decision.total_budget is None or not decision.requirement_proposals:
            raise ValueError("Goal decision is incomplete and cannot be projected.")

        grounded_features = tuple(
            self._constraint_grounder.ground_required_features(
                proposal.required_features
            )
            for proposal in decision.requirement_proposals
        )
        requirements = tuple(
            ShoppingRequirement(
                requirement_id=f"req-{index + 1:03d}",
                category=proposal.category,
                quantity=proposal.quantity,
                max_budget=proposal.max_budget,
                required_features=grounded_features[index],
                soft_preferences=proposal.soft_preferences,
                priority=proposal.priority,
            )
            for index, proposal in enumerate(decision.requirement_proposals)
        )
        allocations = tuple(
            RequirementAllocationPreference(
                requirement_id=requirements[value.target_index].requirement_id,
                preference=value.preference,
            )
            for value in decision.allocation_preferences
        )
        return GoalRequirementProjection(
            total_budget=decision.total_budget,
            requirements=requirements,
            allocation_preferences=allocations,
        )
