"""Pure action resolution for validated agent decision directives."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .contracts import DecisionAction, DecisionDirective


class DecisionExecutionPlan(BaseModel):
    """Immutable boolean projection of the actions selected by a directive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    use_memory: bool = False
    retrieve_products: bool = False
    ask_clarification: bool = False
    verify_evidence: bool = False
    generate_response: bool = False
    stop: bool = False


class DecisionExecutor:
    """Resolve a directive without executing any runtime business operation."""

    def execute(self, directive: DecisionDirective) -> DecisionExecutionPlan:
        """Project ordered directive actions onto independent execution flags."""

        actions = set(directive.actions)
        return DecisionExecutionPlan(
            use_memory=DecisionAction.USE_MEMORY in actions,
            retrieve_products=DecisionAction.RETRIEVE_PRODUCTS in actions,
            ask_clarification=DecisionAction.ASK_CLARIFICATION in actions,
            verify_evidence=DecisionAction.VERIFY_EVIDENCE in actions,
            generate_response=DecisionAction.GENERATE_RESPONSE in actions,
            stop=DecisionAction.STOP in actions,
        )
