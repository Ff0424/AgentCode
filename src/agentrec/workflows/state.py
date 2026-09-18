"""Serializable state passed between AgentRec shopping workflow nodes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from ..domain import AgentState
from ..evidence import RequirementEvidence, SelectedProductEvidence
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
    current_evidence: RequirementEvidence | None = None
    selected_evidence: tuple[SelectedProductEvidence, ...] = ()
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
        plan = self.agent_state.shopping_plan
        if self.current_evidence is not None:
            if (
                self.current_evidence.plan_id != plan.plan_id
                or self.current_evidence.retrieved_at_plan_version != plan.version
                or self.current_evidence.requirement_id
                != self.agent_state.current_requirement_id
            ):
                raise ValueError("current_evidence must match the current pre-mutation plan state.")
        if any(
            value.plan_id != plan.plan_id
            or value.selected_at_plan_version > plan.version
            for value in self.selected_evidence
        ):
            raise ValueError("selected_evidence is not aligned with the current plan history.")
        if len({value.requirement_id for value in self.selected_evidence}) != len(
            self.selected_evidence
        ):
            raise ValueError("selected_evidence may contain one current snapshot per requirement.")
        return self
