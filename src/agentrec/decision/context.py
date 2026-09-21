"""Immutable input context for deterministic agent decision policies."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .contracts import AgentIntent


class DecisionContext(BaseModel):
    """Validated facts used to select an ordered decision directive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: AgentIntent
    memory_available: bool
    requirement_complete: bool
    candidate_available: bool
    verification_required: bool
    response_required: bool
