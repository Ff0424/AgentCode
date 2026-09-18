"""Agent-facing schema and adapter for the runtime RecommendationService.

Only shopping constraints that an LLM may choose are present in
``RecommendationToolArgs``. User identity and previously selected/excluded
products are injected by trusted system and workflow state respectively.
The adapter performs no scoring, reranking, fallback, or natural-language work.
"""

from __future__ import annotations

import math
import numbers
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..recommendation.contracts import (
    RecommendationRequest,
    RecommendationResult,
    RecommendedProduct,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RecommendationToolArgs(BaseModel):
    """Strict LLM-controlled arguments for one recommendation tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: Annotated[int, Field(strict=True, ge=1, le=50)] = 10
    category: NonEmptyText | None = None
    max_price: float | None = None
    required_features: Annotated[
        tuple[NonEmptyText, ...], Field(max_length=20)
    ] = ()

    @field_validator("max_price", mode="before")
    @classmethod
    def validate_max_price(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("max_price must be a finite number greater than zero.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("max_price must be a finite number greater than zero.")
        return numeric


class RecommendationToolItem(BaseModel):
    """Compact product projection safe to return across the Agent boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Annotated[int, Field(strict=True, ge=1)]
    # Canonical identity is projected by the trusted adapter, never supplied by the LLM.
    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    title: NonEmptyText | None
    price: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    score: Annotated[float, Field(allow_inf_nan=False)]
    score_source: NonEmptyText


class RecommendationToolResult(BaseModel):
    """Structured tool result preserving personalization and fallback status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    personalization_status: NonEmptyText
    fallback_reason: NonEmptyText | None
    returned_count: Annotated[int, Field(strict=True, ge=0)]
    items: tuple[RecommendationToolItem, ...] = ()

    @model_validator(mode="after")
    def validate_result_alignment(self) -> "RecommendationToolResult":
        if self.returned_count != len(self.items):
            raise ValueError("returned_count must equal the number of tool items.")
        expected_ranks = tuple(range(1, self.returned_count + 1))
        if tuple(item.rank for item in self.items) != expected_ranks:
            raise ValueError("Tool item ranks must be exactly 1..returned_count.")
        if len({item.item_index for item in self.items}) != len(self.items):
            raise ValueError("Tool result item_index values must be unique.")
        if len({item.parent_asin for item in self.items}) != len(self.items):
            raise ValueError("Tool result parent_asin values must be unique.")
        return self


def _validated_system_user_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("System user_id must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("System user_id must be non-empty.")
    return normalized


def _validated_exclusions(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError("excluded_parent_asins must be a tuple of strings.")
    if len(values) > 100:
        raise ValueError("excluded_parent_asins cannot contain more than 100 values.")
    normalized: list[str] = []
    for position, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"excluded_parent_asins[{position}] must be a string.")
        item = value.strip()
        if not item:
            raise ValueError(f"excluded_parent_asins[{position}] must be non-empty.")
        normalized.append(item)
    return tuple(normalized)


class RecommendationToolAdapter:
    """Validate boundary inputs and project RecommendationService output."""

    def __init__(self, service: Any) -> None:
        if service is None or not callable(getattr(service, "recommend", None)):
            raise TypeError("service must provide a callable recommend(request) method.")
        # The already initialized singleton/service instance is retained as-is.
        self._service = service

    @staticmethod
    def _project_item(item: RecommendedProduct) -> RecommendationToolItem:
        return RecommendationToolItem(
            rank=item.rank,
            item_index=item.item_index,
            parent_asin=item.parent_asin,
            title=item.title,
            price=item.price,
            score=item.score,
            score_source=item.score_source,
        )

    @classmethod
    def _project_result(cls, result: RecommendationResult) -> RecommendationToolResult:
        if not isinstance(result, RecommendationResult):
            raise TypeError("RecommendationService returned an invalid result type.")
        return RecommendationToolResult(
            personalization_status=result.personalization_status,
            fallback_reason=result.fallback_reason,
            returned_count=result.returned_count,
            items=tuple(cls._project_item(item) for item in result.items),
        )

    def recommend(
        self,
        *,
        user_id: str,
        args: RecommendationToolArgs,
        excluded_parent_asins: tuple[str, ...] = (),
    ) -> RecommendationToolResult:
        """Merge trusted identity/workflow state with validated LLM arguments."""

        system_user_id = _validated_system_user_id(user_id)
        if not isinstance(args, RecommendationToolArgs):
            raise TypeError("args must be a RecommendationToolArgs instance.")
        workflow_exclusions = _validated_exclusions(excluded_parent_asins)
        request = RecommendationRequest(
            user_id=system_user_id,
            top_k=args.top_k,
            category=args.category,
            max_price=args.max_price,
            required_features=args.required_features,
            excluded_parent_asins=workflow_exclusions,
        )
        # Service execution errors intentionally propagate; they are not empty results.
        return self._project_result(self._service.recommend(request))
