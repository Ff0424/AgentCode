"""Synchronous execution boundary for registered AgentRec tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import ToolRequest, ToolResult, ToolStatus
from .registry import ToolRegistry


class ToolExecutor:
    """Dispatch validated requests to injected handlers with safe failures."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        handlers: Mapping[str, Any],
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry.")
        if not isinstance(handlers, Mapping):
            raise TypeError("handlers must be a mapping.")
        copied_handlers: dict[str, Any] = {}
        for name, handler in handlers.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("handler names must be non-empty strings.")
            if not callable(handler) and not callable(getattr(handler, "execute", None)):
                raise TypeError(
                    f"Handler for {name!r} must be callable or provide execute()."
                )
            copied_handlers[name] = handler

        self._registry = registry
        self._handlers = copied_handlers

    def execute(self, request: ToolRequest, *, context: Any | None = None) -> ToolResult:
        """Execute one request without exposing handler failure details."""

        if not isinstance(request, ToolRequest):
            raise TypeError("request must be a ToolRequest.")

        tool_name = request.tool_name
        if self._registry.get(tool_name) is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                tool_name=tool_name,
                output={},
                error_message="tool_not_found",
            )

        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                status=ToolStatus.FAILED,
                tool_name=tool_name,
                output={},
                error_message="handler_not_found",
            )

        try:
            contextual_execute = getattr(handler, "execute", None)
            if callable(contextual_execute):
                output = contextual_execute(request=request, context=context)
            else:
                # Backward-compatible V2-09.9.3 handler contract.
                output = handler(**request.arguments)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=tool_name,
                output=output,
            )
        except Exception:
            # Handler and output-validation details must not cross this boundary.
            return ToolResult(
                status=ToolStatus.FAILED,
                tool_name=tool_name,
                output={},
                error_message="tool_execution_failed",
            )
