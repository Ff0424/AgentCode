"""Public end-to-end integration contracts for AgentRec V2."""

from .agent import (
    AgentExecutionResult,
    AgentExecutionStatus,
    AgentTaskRunner,
    RequirementInput,
)
from .goal import (
    GoalExecutionResult,
    GoalExecutionStatus,
    PreparedGoalExecution,
)

__all__ = [
    "AgentExecutionResult",
    "AgentExecutionStatus",
    "AgentTaskRunner",
    "GoalExecutionResult",
    "GoalExecutionStatus",
    "PreparedGoalExecution",
    "RequirementInput",
]
