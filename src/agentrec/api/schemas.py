"""Pydantic request and response schemas for the AgentRec HTTP API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    """One user chat request received by the API boundary."""

    user_id: NonEmptyText
    session_id: NonEmptyText
    query: NonEmptyText


class ProductResponse(BaseModel):
    """Minimal product projection reserved for future Agent responses."""

    category: str
    title: str
    parent_asin: str
    price: float


class ChatResponse(BaseModel):
    """Stable response envelope for the chat endpoint."""

    status: str
    response: str | None
    products: list[ProductResponse]
    workflow_status: str
    remaining_budget: float | None
