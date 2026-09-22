"""Immutable, provider-neutral contracts for AgentRec tool calling.

This module defines data boundaries only. It does not register, dispatch, or
execute tools and does not access any AgentRec runtime service.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_FORBIDDEN_PAYLOAD_FIELDS = frozenset({
    "embedding",
    "vector",
    "score",
    "similarity",
    "artifact",
    "backend",
    "model",
    "gpu",
})


def _validate_safe_payload(value: Any, *, path: str) -> Any:
    """Reject internal implementation fields anywhere in a tool payload."""

    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings.")
            normalized_key = key.strip().casefold()
            if normalized_key in _FORBIDDEN_PAYLOAD_FIELDS:
                raise ValueError(f"{path} contains forbidden field {key!r}.")
            _validate_safe_payload(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_safe_payload(nested, path=f"{path}[{index}]")
    return value


class ToolStatus(str, Enum):
    """Closed status vocabulary for one tool call result."""

    SUCCESS = "success"
    FAILED = "failed"
    INVALID_ARGUMENT = "invalid_argument"


class ToolDefinition(BaseModel):
    """Minimal immutable description of an available tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyText
    description: NonEmptyText


class ToolRequest(BaseModel):
    """Validated request to a named tool with explicit arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: NonEmptyText
    arguments: dict[str, Any]

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_safe_payload(value, path="arguments")


class ToolResult(BaseModel):
    """Immutable structured outcome from a tool execution boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolStatus
    tool_name: NonEmptyText
    output: dict[str, Any]
    error_message: NonEmptyText | None = None

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_safe_payload(value, path="output")

    @model_validator(mode="after")
    def validate_status_error_alignment(self) -> "ToolResult":
        if self.status is ToolStatus.SUCCESS and self.error_message is not None:
            raise ValueError("SUCCESS ToolResult must not include error_message.")
        return self
