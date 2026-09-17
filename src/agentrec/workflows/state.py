"""Serializable state passed between AgentRec shopping workflow nodes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from ..domain import AgentState
from ..planning import PlannerDecision
from ..services import PlanEvaluation
from ..tools import RecommendationToolArgs, RecommendationToolResult
from .routes import WorkflowRoute


class ShoppingWorkflowState(BaseModel):
    """Agent state plus compact, typed data needed by one workflow execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_state: AgentState
    recommendation_args: RecommendationToolArgs | None = None
    last_tool_result: RecommendationToolResult | None = None
    selected_parent_asin: str | None = None
    evaluation: PlanEvaluation | None = None
    planner_decision: PlannerDecision | None = None
    route: WorkflowRoute | None = None

    @model_validator(mode="after")
    def validate_step_alignment(self) -> "ShoppingWorkflowState":
        if self.selected_parent_asin is not None and self.last_tool_result is None:
            raise ValueError("selected_parent_asin requires last_tool_result.")
        if self.evaluation is not None:
            plan = self.agent_state.shopping_plan
            if (
                self.evaluation.plan_id != plan.plan_id
                or self.evaluation.plan_version != plan.version
            ):
                raise ValueError("evaluation must describe the current ShoppingPlan version.")
        return self
