"""Public end-to-end integration contracts for AgentRec V2."""

from .agent import (
    AgentExecutionResult,
    AgentExecutionStatus,
    AgentTaskRunner,
    RequirementInput,
)

__all__ = [
    "AgentExecutionResult",
    "AgentExecutionStatus",
    "AgentTaskRunner",
    "RequirementInput",
]
