"""Public contracts for AgentRec tool calling."""

from .contracts import ToolDefinition, ToolRequest, ToolResult, ToolStatus
from .executor import ToolExecutor
from .recommendation import (
    RECOMMENDATION_TOOL_DEFINITION,
    RECOMMENDATION_TOOL_NAME,
    RecommendationToolHandler,
    ToolExecutionContext,
)
from .registry import ToolRegistry
from .router import ToolRouter, ToolRoutingContext

__all__ = [
    "RECOMMENDATION_TOOL_DEFINITION",
    "RECOMMENDATION_TOOL_NAME",
    "RecommendationToolHandler",
    "ToolDefinition",
    "ToolExecutor",
    "ToolExecutionContext",
    "ToolRequest",
    "ToolResult",
    "ToolRegistry",
    "ToolRouter",
    "ToolRoutingContext",
    "ToolStatus",
]
