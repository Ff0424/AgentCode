"""Deterministic in-memory registry for AgentRec tool definitions."""

from __future__ import annotations

from .contracts import ToolDefinition


class ToolRegistry:
    """Register and query immutable tool definitions without executing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register one definition and reject duplicate names."""

        if not isinstance(tool, ToolDefinition):
            raise TypeError("tool must be a ToolDefinition.")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Return the exact-name definition, or None when it is absent."""

        return self._tools.get(name)

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        """Return all definitions in deterministic ascending name order."""

        return tuple(self._tools[name] for name in sorted(self._tools))

    def contains(self, name: str) -> bool:
        """Report whether an exact tool name is registered."""

        return name in self._tools

    def remove(self, name: str) -> bool:
        """Remove an exact-name definition and report whether it existed."""

        if name not in self._tools:
            return False
        del self._tools[name]
        return True
