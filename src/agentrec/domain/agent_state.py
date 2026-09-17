"""Minimal typed workflow state for the future LangGraph shopping agent."""

from __future__ import annotations

from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .shopping import ShoppingPlan


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AgentState(BaseModel):
    """Business/workflow state only; runtime dependencies are deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: NonEmptyText
    session_id: NonEmptyText
    shopping_plan: ShoppingPlan
    current_requirement_id: NonEmptyText | None = None
    conversation_turn: Annotated[int, Field(strict=True, ge=0)] = 0
    last_tool_result: dict[str, Any] | None = None
    pending_action: NonEmptyText | None = None
    error_state: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_state_identity(self) -> "AgentState":
        if self.user_id != self.shopping_plan.user_id:
            raise ValueError("AgentState user_id must match ShoppingPlan user_id.")
        if self.current_requirement_id is not None and self.current_requirement_id not in {
            requirement.requirement_id for requirement in self.shopping_plan.requirements
        }:
            raise ValueError("current_requirement_id is not present in ShoppingPlan.")
        return self
