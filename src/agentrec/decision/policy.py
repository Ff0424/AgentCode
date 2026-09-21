"""Deterministic decision policy with no runtime service dependencies."""

from __future__ import annotations

from .context import DecisionContext
from .contracts import AgentIntent, DecisionAction, DecisionDirective


class DeterministicDecisionPolicy:
    """Map validated decision facts to a stable ordered action sequence."""

    def decide(self, context: DecisionContext) -> DecisionDirective:
        """Return the directive defined by the frozen V2-09.8.2 rules."""

        if context.intent is AgentIntent.INFORMATION:
            actions = (
                DecisionAction.GENERATE_RESPONSE,
                DecisionAction.STOP,
            )
        elif context.intent is AgentIntent.FOLLOW_UP:
            actions = (
                DecisionAction.GENERATE_RESPONSE,
                DecisionAction.STOP,
            )
        elif context.intent is AgentIntent.CLARIFICATION:
            actions = (
                DecisionAction.ASK_CLARIFICATION,
                DecisionAction.STOP,
            )
        elif not context.requirement_complete:
            actions = (
                DecisionAction.ASK_CLARIFICATION,
                DecisionAction.STOP,
            )
        elif context.memory_available:
            actions = (
                DecisionAction.USE_MEMORY,
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            )
        else:
            actions = (
                DecisionAction.RETRIEVE_PRODUCTS,
                DecisionAction.VERIFY_EVIDENCE,
                DecisionAction.GENERATE_RESPONSE,
            )

        return DecisionDirective(intent=context.intent, actions=actions)
