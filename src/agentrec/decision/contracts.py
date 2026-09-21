"""Immutable contracts for the AgentRec decision layer.

These contracts describe validated intent and ordered action directives only.
They do not execute tools, mutate workflow state, or access runtime services.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentIntent(str, Enum):
    """Closed vocabulary for the purpose of one agent decision."""

    RECOMMENDATION = "recommendation"
    CLARIFICATION = "clarification"
    INFORMATION = "information"
    FOLLOW_UP = "follow_up"


class DecisionAction(str, Enum):
    """Closed vocabulary of actions that a decision may direct."""

    USE_MEMORY = "use_memory"
    RETRIEVE_PRODUCTS = "retrieve_products"
    ASK_CLARIFICATION = "ask_clarification"
    VERIFY_EVIDENCE = "verify_evidence"
    GENERATE_RESPONSE = "generate_response"
    STOP = "stop"


class DecisionDirective(BaseModel):
    """One immutable intent with a non-empty deterministic action sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: AgentIntent
    actions: Annotated[tuple[DecisionAction, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_actions(self) -> "DecisionDirective":
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must not contain duplicate values.")
        return self
