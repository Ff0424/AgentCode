"""Serializable state passed between AgentRec shopping workflow nodes."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import AgentState
from ..evidence import RequirementEvidence, SelectedProductEvidence
from ..planning import PlannerDecision
from ..replanning import FailureDiagnosis, ReplanDirective
from ..services import PlanEvaluation
from ..tools import RecommendationToolArgs, RecommendationToolResult
from ..verification import RequirementVerification, SelectedCandidateVerification
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
    current_verification: RequirementVerification | None = None
    selected_verifications: tuple[SelectedCandidateVerification, ...] = ()
    recommendation_top_k: Annotated[int, Field(strict=True, ge=1, le=50)] = 5
    replan_attempt: Annotated[int, Field(strict=True, ge=0, le=1)] = 0
    max_replan_attempts: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    current_failure_diagnosis: FailureDiagnosis | None = None
    current_replan_directive: ReplanDirective | None = None
    failure_history: tuple[FailureDiagnosis, ...] = ()
    replan_history: tuple[ReplanDirective, ...] = ()
    current_evidence_attempt: Annotated[int, Field(strict=True, ge=0, le=1)] | None = None
    current_verification_attempt: Annotated[int, Field(strict=True, ge=0, le=1)] | None = None
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
            if self.current_evidence_attempt != self.replan_attempt:
                raise ValueError("current_evidence attempt must match replan_attempt.")
        elif self.current_evidence_attempt is not None:
            raise ValueError("current_evidence_attempt requires current_evidence.")
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
        if self.current_verification is not None:
            if self.current_evidence is None:
                raise ValueError("current_verification requires current_evidence.")
            if (
                self.current_verification.plan_id != plan.plan_id
                or self.current_verification.requirement_id != self.agent_state.current_requirement_id
                or self.current_verification.verified_at_plan_version != plan.version
                or self.current_verification.verified_at_plan_version
                != self.current_evidence.retrieved_at_plan_version
            ):
                raise ValueError("current_verification must match current evidence and plan.")
            if (
                self.current_verification_attempt != self.replan_attempt
                or self.current_verification_attempt != self.current_evidence_attempt
            ):
                raise ValueError("current_verification attempt must match current evidence.")
        elif self.current_verification_attempt is not None:
            raise ValueError("current_verification_attempt requires current_verification.")
        if any(
            value.plan_id != plan.plan_id or value.selected_at_plan_version > plan.version
            for value in self.selected_verifications
        ):
            raise ValueError("selected_verifications are not aligned with plan history.")
        if len({value.requirement_id for value in self.selected_verifications}) != len(
            self.selected_verifications
        ):
            raise ValueError("selected_verifications may contain one snapshot per requirement.")
        if self.current_failure_diagnosis is not None and (
            self.current_failure_diagnosis.plan_id != plan.plan_id
            or self.current_failure_diagnosis.diagnosed_at_plan_version != plan.version
            or self.current_failure_diagnosis.requirement_id
            != self.agent_state.current_requirement_id
        ):
            raise ValueError("current_failure_diagnosis is stale or identity-mismatched.")
        if self.current_replan_directive is not None and (
            self.current_replan_directive.plan_id != plan.plan_id
            or self.current_replan_directive.source_plan_version != plan.version
            or self.current_replan_directive.requirement_id
            != self.agent_state.current_requirement_id
            or self.current_replan_directive.attempt
            not in {self.replan_attempt, self.replan_attempt + 1}
        ):
            raise ValueError("current_replan_directive is stale or identity-mismatched.")
        return self
