"""Strict provider-neutral decision schemas for the future LLM planner.

These models describe proposals only. They cannot mutate ShoppingPlan, invoke
tools, or establish product and budget facts. Later workflow integration must
still perform permission, candidate, and business validation before execution.
"""

from __future__ import annotations

import math
import numbers
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ReasonText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]


class PlannerAction(str, Enum):
    """Closed action vocabulary emitted by a Planner implementation."""

    SELECT_REQUIREMENT = "select_requirement"
    REQUEST_RECOMMENDATION = "request_recommendation"
    SELECT_CANDIDATE = "select_candidate"
    PROPOSE_REPLAN = "propose_replan"
    REQUEST_USER_CONFIRMATION = "request_user_confirmation"


class _DecisionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Every proposal is bound to the exact immutable plan snapshot it observed.
    plan_id: NonEmptyText
    plan_version: Annotated[int, Field(strict=True, ge=0)]


class SelectRequirementDecision(_DecisionBase):
    """Propose which existing unfinished requirement to process next."""

    action: Literal[PlannerAction.SELECT_REQUIREMENT] = PlannerAction.SELECT_REQUIREMENT
    requirement_id: NonEmptyText
    reason: ReasonText


class RequestRecommendationDecision(_DecisionBase):
    """Request candidates for an existing requirement.

    Category, price, required features, user identity, and exclusions remain
    system-controlled and deliberately do not appear in this decision.
    """

    action: Literal[PlannerAction.REQUEST_RECOMMENDATION] = (
        PlannerAction.REQUEST_RECOMMENDATION
    )
    requirement_id: NonEmptyText
    top_k: Annotated[int, Field(strict=True, ge=1, le=50)] = 5
    reason: ReasonText


class SelectCandidateDecision(_DecisionBase):
    """Propose one product identity from the current trusted Tool Result."""

    action: Literal[PlannerAction.SELECT_CANDIDATE] = PlannerAction.SELECT_CANDIDATE
    requirement_id: NonEmptyText
    parent_asin: NonEmptyText
    reason: ReasonText


class ReplanProposalDecision(_DecisionBase):
    """Propose, but never execute, one conflict-resolution operation."""

    action: Literal[PlannerAction.PROPOSE_REPLAN] = PlannerAction.PROPOSE_REPLAN
    requirement_id: NonEmptyText | None = None
    operation: NonEmptyText
    proposed_value: str | int | float | bool | None = None
    reason: ReasonText
    requires_confirmation: Literal[True] = True

    @field_validator("proposed_value", mode="before")
    @classmethod
    def validate_proposed_value(cls, value: object) -> object:
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError("Numeric proposed_value must be finite.")
        return value


class RequestUserConfirmationDecision(_DecisionBase):
    """Pause execution and request an explicit user decision."""

    action: Literal[PlannerAction.REQUEST_USER_CONFIRMATION] = (
        PlannerAction.REQUEST_USER_CONFIRMATION
    )
    confirmation_id: NonEmptyText
    prompt: ReasonText
    reason: ReasonText


PlannerDecision: TypeAlias = Annotated[
    SelectRequirementDecision
    | RequestRecommendationDecision
    | SelectCandidateDecision
    | ReplanProposalDecision
    | RequestUserConfirmationDecision,
    Field(discriminator="action"),
]

_PLANNER_DECISION_ADAPTER = TypeAdapter(PlannerDecision)


def validate_planner_decision(value: Any) -> PlannerDecision:
    """Parse an untrusted value through the discriminated decision union."""

    return _PLANNER_DECISION_ADAPTER.validate_python(value)
