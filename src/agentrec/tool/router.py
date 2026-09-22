"""Deterministic request construction for AgentRec tool calling.

The router converts a decision execution plan plus trusted business context
into validated ToolRequest values. It never executes tools or runtime services.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from ..decision import DecisionExecutionPlan
from ..domain import ShoppingRequirement
from ..tools import RecommendationToolArgs
from .contracts import ToolRequest
from .recommendation import RECOMMENDATION_TOOL_NAME


class ToolRoutingContext(BaseModel):
    """Immutable trusted facts required to construct tool requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_plan: DecisionExecutionPlan
    requirement: ShoppingRequirement
    top_k: Annotated[int, Field(strict=True, ge=1, le=50)] = 5


class ToolRouter:
    """Map actual tool actions to requests without business execution."""

    def route(self, context: ToolRoutingContext) -> tuple[ToolRequest, ...]:
        """Return requests in stable routing order for the supplied context."""

        if not isinstance(context, ToolRoutingContext):
            raise TypeError("context must be a ToolRoutingContext.")
        if not context.decision_plan.retrieve_products:
            return ()

        requirement = context.requirement
        recommendation_args = RecommendationToolArgs(
            top_k=context.top_k,
            category=requirement.category,
            max_price=requirement.max_budget,
            required_features=requirement.required_features,
        )
        return (
            ToolRequest(
                tool_name=RECOMMENDATION_TOOL_NAME,
                arguments=recommendation_args.model_dump(mode="python"),
            ),
        )
