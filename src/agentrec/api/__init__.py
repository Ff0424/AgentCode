"""Public HTTP API surface for AgentRec."""

from .app import app
from .schemas import ChatRequest, ChatResponse, ProductResponse

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ProductResponse",
    "app",
]
