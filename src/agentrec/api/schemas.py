"""Pydantic request and response schemas for the AgentRec HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    """One user chat request received by the API boundary."""

    user_id: NonEmptyText | None = None
    session_id: NonEmptyText
    query: NonEmptyText


class ClarificationResponse(BaseModel):
    """One user-facing question required before execution can begin."""

    needed: bool
    question: NonEmptyText


class PlanResponse(BaseModel):
    """Web-safe monetary and status facts from ShoppingPlan."""

    status: str
    currency: str
    total_budget: float
    total_spent: float
    remaining_budget: float


class RequirementResponse(BaseModel):
    """Web-safe projection of one planned shopping requirement."""

    requirement_id: str
    category: str
    quantity: int
    max_budget: float | None
    allocated_budget: float | None
    required_features: list[str]
    soft_preferences: list[str]
    status: str


class VerifiedRequirementResponse(BaseModel):
    """A constraint already supported by the grounded response projector."""

    constraint: str
    status: Literal["supported"]


class ProductResponse(BaseModel):
    """Web-safe selected product facts without model or retrieval internals."""

    requirement_id: str
    category: str
    title: str
    parent_asin: str
    price: float
    quantity: int
    subtotal: float
    source: str
    verified_requirements: list[VerifiedRequirementResponse]


class ConflictResponse(BaseModel):
    """Grounded conflict facts safe for direct Web presentation."""

    reason: str
    category: str
    required_features: list[str]
    failed_constraints: list[str]
    unknown_constraints: list[str]
    contradicted_constraints: list[str]
    replan_attempts_performed: int
    user_action_required: bool


class ChatResponse(BaseModel):
    """Stable response envelope for the chat endpoint."""

    status: str
    response: str | None
    clarification: ClarificationResponse | None
    plan: PlanResponse | None
    requirements: list[RequirementResponse]
    products: list[ProductResponse]
    conflict: ConflictResponse | None
