"""Trusted execution binding for the existing RecommendationToolAdapter."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..tools import (
    RecommendationToolAdapter,
    RecommendationToolArgs,
    RecommendationToolResult,
)
from .contracts import ToolDefinition, ToolRequest


RECOMMENDATION_TOOL_NAME = "recommend_products"
RECOMMENDATION_TOOL_DEFINITION = ToolDefinition(
    name=RECOMMENDATION_TOOL_NAME,
    description="Return product recommendations for validated shopping constraints.",
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ToolExecutionContext(BaseModel):
    """Trusted system/workflow values that cannot be supplied by tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: NonEmptyText
    excluded_parent_asins: Annotated[
        tuple[NonEmptyText, ...], Field(max_length=100)
    ] = ()


class RecommendationToolHandler:
    """Execute one generic request through the existing recommendation adapter."""

    def __init__(self, adapter: Any) -> None:
        if adapter is None or not callable(getattr(adapter, "recommend", None)):
            raise TypeError("adapter must provide recommend().")
        self._adapter = adapter

    @staticmethod
    def _safe_output(result: RecommendationToolResult) -> dict[str, Any]:
        if not isinstance(result, RecommendationToolResult):
            raise TypeError("Recommendation adapter returned an invalid result type.")
        return {
            "personalization_status": result.personalization_status,
            "fallback_reason": result.fallback_reason,
            "returned_count": result.returned_count,
            "items": [
                {
                    "rank": item.rank,
                    "parent_asin": item.parent_asin,
                    "title": item.title,
                    "price": item.price,
                }
                for item in result.items
            ],
        }

    def execute(
        self,
        *,
        request: ToolRequest,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        if not isinstance(request, ToolRequest):
            raise TypeError("request must be a ToolRequest.")
        if request.tool_name != RECOMMENDATION_TOOL_NAME:
            raise ValueError("RecommendationToolHandler received another tool name.")
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("Recommendation execution requires ToolExecutionContext.")

        args = RecommendationToolArgs.model_validate(request.arguments)
        result = self._adapter.recommend(
            user_id=context.user_id,
            args=args,
            excluded_parent_asins=context.excluded_parent_asins,
        )
        return self._safe_output(result)
