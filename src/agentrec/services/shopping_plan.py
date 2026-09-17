"""Deterministic Shopping Plan application service for AgentRec V2.

The service is the trust boundary between recommendation-tool output and the
immutable Shopping Plan domain. It validates tool-call provenance and workflow
inputs, then delegates state transitions and all budget/status calculations to
the domain models. It performs no recommendation, LLM, routing, persistence,
or API work.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from ..domain.shopping import (
    ItemSource,
    PlanStatus,
    RequirementStatus,
    ShoppingPlan,
    ShoppingPlanItem,
    ShoppingRequirement,
)
from ..tools.recommendation import (
    RecommendationToolArgs,
    RecommendationToolItem,
    RecommendationToolResult,
)


class ShoppingPlanServiceError(Exception):
    """Base class for expected ShoppingPlanService boundary failures."""


class InvalidRequirementError(ShoppingPlanServiceError):
    pass


class CategoryMismatchError(ShoppingPlanServiceError):
    pass


class InvalidToolItemError(ShoppingPlanServiceError):
    pass


class DuplicateItemError(ShoppingPlanServiceError):
    pass


class QuantityExceededError(ShoppingPlanServiceError):
    pass


class CompletedPlanMutationError(ShoppingPlanServiceError):
    pass


class CandidateNotFoundError(ShoppingPlanServiceError):
    pass


class RequirementConstraintError(ShoppingPlanServiceError):
    pass


class PlanNotReadyError(ShoppingPlanServiceError):
    pass


class PlanIssueCode(str, Enum):
    MISSING_REQUIRED_ITEM = "missing_required_item"
    REQUIREMENT_QUANTITY_INCOMPLETE = "requirement_quantity_incomplete"
    REQUIREMENT_BUDGET_EXCEEDED = "requirement_budget_exceeded"
    TOTAL_BUDGET_EXCEEDED = "total_budget_exceeded"
    ITEM_CONSTRAINT_FAILURE = "item_constraint_failure"


class PlanIssue(BaseModel):
    """Stable machine-readable projection of one domain conflict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: PlanIssueCode
    requirement_id: str | None = None
    parent_asin: str | None = None


class PlanEvaluation(BaseModel):
    """Read-only evaluation assembled exclusively from existing domain facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    plan_version: Annotated[int, Field(strict=True, ge=0)]
    status: PlanStatus
    total_spent: float
    remaining_budget: float
    is_over_budget: bool
    has_hard_constraint_conflict: bool
    requirement_count: Annotated[int, Field(strict=True, ge=1)]
    satisfied_requirement_count: Annotated[int, Field(strict=True, ge=0)]
    pending_requirement_count: Annotated[int, Field(strict=True, ge=0)]
    conflict_requirement_count: Annotated[int, Field(strict=True, ge=0)]
    is_ready: bool
    issues: tuple[PlanIssue, ...] = ()


_SOURCE_MAP = {
    "hybrid": ItemSource.HYBRID,
    "popularity_fallback": ItemSource.POPULARITY_FALLBACK,
}


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


class ShoppingPlanService:
    """Validate selection commands and invoke immutable domain transitions."""

    def create_plan(
        self,
        *,
        plan_id: str,
        user_id: str,
        currency: str,
        total_budget: float,
        requirements: tuple[ShoppingRequirement, ...],
    ) -> ShoppingPlan:
        if not isinstance(requirements, tuple):
            raise TypeError("requirements must be a tuple of ShoppingRequirement values.")
        if any(not isinstance(value, ShoppingRequirement) for value in requirements):
            raise TypeError("Every requirement must be a ShoppingRequirement.")
        if any(value.status != RequirementStatus.PENDING for value in requirements):
            raise InvalidRequirementError("A new plan may contain only pending requirements.")
        return ShoppingPlan(
            plan_id=plan_id,
            user_id=user_id,
            currency=currency,
            total_budget=total_budget,
            requirements=requirements,
        )

    def select_item(
        self,
        plan: ShoppingPlan,
        *,
        requirement_id: str,
        recommendation_result: RecommendationToolResult,
        recommendation_args: RecommendationToolArgs,
        selected_parent_asin: str,
        quantity: int = 1,
        selected_reason: str,
    ) -> ShoppingPlan:
        self._validate_mutable_plan(plan)
        requirement = self._requirement(plan, requirement_id)
        candidate = self._candidate(recommendation_result, selected_parent_asin)
        self._validate_selection(
            plan, requirement, candidate, recommendation_args, quantity, ignored_slot=None
        )
        item = self._build_item(
            requirement=requirement,
            candidate=candidate,
            slot_id=self._next_slot_id(plan, requirement.requirement_id),
            quantity=quantity,
            selected_reason=selected_reason,
        )
        return plan.add_item(item)

    def replace_item(
        self,
        plan: ShoppingPlan,
        *,
        slot_id: str,
        recommendation_result: RecommendationToolResult,
        recommendation_args: RecommendationToolArgs,
        selected_parent_asin: str,
        quantity: int = 1,
        selected_reason: str,
    ) -> ShoppingPlan:
        self._validate_mutable_plan(plan)
        normalized_slot = _non_empty(slot_id, "slot_id")
        old_items = tuple(item for item in plan.selected_items if item.slot_id == normalized_slot)
        if len(old_items) != 1:
            raise InvalidRequirementError(f"Selected slot_id={normalized_slot!r} does not exist.")
        old_item = old_items[0]
        requirement = self._requirement(plan, old_item.requirement_id)
        candidate = self._candidate(recommendation_result, selected_parent_asin)
        self._validate_selection(
            plan,
            requirement,
            candidate,
            recommendation_args,
            quantity,
            ignored_slot=normalized_slot,
        )
        replacement = self._build_item(
            requirement=requirement,
            candidate=candidate,
            slot_id=normalized_slot,
            quantity=quantity,
            selected_reason=selected_reason,
        )
        # Replacement is an action; provenance remains hybrid or popularity fallback.
        return plan.replace_item(normalized_slot, replacement)

    def remove_item(self, plan: ShoppingPlan, *, slot_id: str) -> ShoppingPlan:
        self._validate_mutable_plan(plan)
        normalized_slot = _non_empty(slot_id, "slot_id")
        if not any(item.slot_id == normalized_slot for item in plan.selected_items):
            raise InvalidRequirementError(f"Selected slot_id={normalized_slot!r} does not exist.")
        return plan.remove_item(normalized_slot)

    def evaluate_plan(self, plan: ShoppingPlan) -> PlanEvaluation:
        if not isinstance(plan, ShoppingPlan):
            raise TypeError("plan must be a ShoppingPlan.")
        issues: list[PlanIssue] = []
        for requirement in plan.requirements:
            selected = tuple(
                item for item in plan.selected_items
                if item.requirement_id == requirement.requirement_id
            )
            selected_quantity = sum(item.quantity for item in selected)
            if not selected:
                issues.append(PlanIssue(
                    code=PlanIssueCode.MISSING_REQUIRED_ITEM,
                    requirement_id=requirement.requirement_id,
                ))
            elif selected_quantity < requirement.quantity:
                issues.append(PlanIssue(
                    code=PlanIssueCode.REQUIREMENT_QUANTITY_INCOMPLETE,
                    requirement_id=requirement.requirement_id,
                ))
            for item in selected:
                if not item.constraints_satisfied:
                    issues.append(PlanIssue(
                        code=PlanIssueCode.ITEM_CONSTRAINT_FAILURE,
                        requirement_id=requirement.requirement_id,
                        parent_asin=item.parent_asin,
                    ))
            # Status is the domain-owned determination of any requirement conflict.
            if (
                requirement.status == RequirementStatus.CONFLICT
                and not any(not item.constraints_satisfied for item in selected)
            ):
                issues.append(PlanIssue(
                    code=PlanIssueCode.REQUIREMENT_BUDGET_EXCEEDED,
                    requirement_id=requirement.requirement_id,
                ))
        if plan.is_over_budget:
            issues.append(PlanIssue(code=PlanIssueCode.TOTAL_BUDGET_EXCEEDED))

        statuses = tuple(value.status for value in plan.requirements)
        return PlanEvaluation(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            status=plan.status,
            total_spent=plan.total_spent,
            remaining_budget=plan.remaining_budget,
            is_over_budget=plan.is_over_budget,
            has_hard_constraint_conflict=plan.has_hard_constraint_conflict,
            requirement_count=len(statuses),
            satisfied_requirement_count=statuses.count(RequirementStatus.SATISFIED),
            pending_requirement_count=sum(
                value in {RequirementStatus.PENDING, RequirementStatus.CANDIDATE_SELECTED}
                for value in statuses
            ),
            conflict_requirement_count=statuses.count(RequirementStatus.CONFLICT),
            is_ready=plan.status == PlanStatus.READY and plan.is_valid,
            issues=tuple(issues),
        )

    def mark_completed(self, plan: ShoppingPlan) -> ShoppingPlan:
        if not isinstance(plan, ShoppingPlan):
            raise TypeError("plan must be a ShoppingPlan.")
        if not self.evaluate_plan(plan).is_ready:
            raise PlanNotReadyError("Only a ready plan without hard conflicts can be completed.")
        return plan.mark_completed()

    @staticmethod
    def _validate_mutable_plan(plan: ShoppingPlan) -> None:
        if not isinstance(plan, ShoppingPlan):
            raise TypeError("plan must be a ShoppingPlan.")
        if plan.status == PlanStatus.COMPLETED:
            raise CompletedPlanMutationError("A completed ShoppingPlan cannot be mutated.")

    @staticmethod
    def _requirement(plan: ShoppingPlan, requirement_id: str) -> ShoppingRequirement:
        normalized = _non_empty(requirement_id, "requirement_id")
        matches = tuple(
            value for value in plan.requirements if value.requirement_id == normalized
        )
        if len(matches) != 1:
            raise InvalidRequirementError(f"requirement_id={normalized!r} does not exist.")
        return matches[0]

    @staticmethod
    def _candidate(
        result: RecommendationToolResult, parent_asin: str
    ) -> RecommendationToolItem:
        if not isinstance(result, RecommendationToolResult):
            raise TypeError("recommendation_result must be a RecommendationToolResult.")
        normalized = _non_empty(parent_asin, "selected_parent_asin")
        matches = tuple(item for item in result.items if item.parent_asin == normalized)
        if len(matches) != 1:
            raise CandidateNotFoundError(
                f"Exactly one candidate with parent_asin={normalized!r} is required."
            )
        return matches[0]

    def _validate_selection(
        self,
        plan: ShoppingPlan,
        requirement: ShoppingRequirement,
        candidate: RecommendationToolItem,
        args: RecommendationToolArgs,
        quantity: int,
        *,
        ignored_slot: str | None,
    ) -> None:
        if not isinstance(args, RecommendationToolArgs):
            raise TypeError("recommendation_args must be a RecommendationToolArgs instance.")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise QuantityExceededError("quantity must be a positive integer and cannot be bool.")
        if args.category is None or _normalized_text(args.category) != _normalized_text(
            requirement.category
        ):
            raise CategoryMismatchError(
                "Recommendation category must exactly match the target requirement."
            )
        requested_features = {_normalized_text(value) for value in args.required_features}
        required_features = {_normalized_text(value) for value in requirement.required_features}
        if not required_features.issubset(requested_features):
            raise RequirementConstraintError(
                "Recommendation arguments do not preserve all required features."
            )
        if requirement.max_budget is not None:
            if args.max_price is None:
                raise RequirementConstraintError(
                    "max_price is required when the ShoppingRequirement has max_budget."
                )
            if args.max_price > requirement.max_budget:
                raise RequirementConstraintError(
                    "Recommendation max_price cannot exceed requirement max_budget."
                )
        if candidate.title is None or not candidate.title.strip():
            raise InvalidToolItemError("Recommendation candidate title must be non-empty.")
        if candidate.price is None or not math.isfinite(candidate.price) or candidate.price <= 0:
            raise InvalidToolItemError("Recommendation candidate price must be finite and positive.")
        if candidate.score_source not in _SOURCE_MAP:
            raise InvalidToolItemError(
                f"Unsupported recommendation score_source={candidate.score_source!r}."
            )
        retained = tuple(
            item for item in plan.selected_items if item.slot_id != ignored_slot
        )
        if any(item.parent_asin == candidate.parent_asin for item in retained):
            raise DuplicateItemError(
                f"parent_asin={candidate.parent_asin!r} is already selected."
            )
        used_quantity = sum(
            item.quantity for item in retained
            if item.requirement_id == requirement.requirement_id
        )
        if used_quantity + quantity > requirement.quantity:
            raise QuantityExceededError(
                f"Selected quantity exceeds requirement {requirement.requirement_id!r}."
            )
        # max_budget is a hard selection-boundary constraint, not a later warning.
        retained_subtotal = math.fsum(
            item.price * item.quantity for item in retained
            if item.requirement_id == requirement.requirement_id
        )
        if (
            requirement.max_budget is not None
            and retained_subtotal + candidate.price * quantity > requirement.max_budget
        ):
            raise RequirementConstraintError(
                f"Candidate subtotal exceeds max_budget for requirement "
                f"{requirement.requirement_id!r}."
            )

    @staticmethod
    def _build_item(
        *,
        requirement: ShoppingRequirement,
        candidate: RecommendationToolItem,
        slot_id: str,
        quantity: int,
        selected_reason: str,
    ) -> ShoppingPlanItem:
        reason = _non_empty(selected_reason, "selected_reason")
        return ShoppingPlanItem(
            slot_id=slot_id,
            requirement_id=requirement.requirement_id,
            category=requirement.category,
            parent_asin=candidate.parent_asin,
            title=candidate.title,
            price=candidate.price,
            quantity=quantity,
            selected_reason=reason,
            constraints_satisfied=True,
            source=_SOURCE_MAP[candidate.score_source],
            recommendation_score=candidate.score,
        )

    @staticmethod
    def _next_slot_id(plan: ShoppingPlan, requirement_id: str) -> str:
        existing = {item.slot_id for item in plan.selected_items}
        ordinal = 1
        while f"{requirement_id}:{ordinal}" in existing:
            ordinal += 1
        return f"{requirement_id}:{ordinal}"
