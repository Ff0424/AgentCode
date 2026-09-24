"""Shared execution boundary for the deterministic shopping LangGraph.

The runner owns only graph construction, invocation, and output validation.
Extraction, memory, Goal preparation, status mapping, and response projection
remain responsibilities of their existing integration layers.
"""

from __future__ import annotations

from typing import Any

from ..workflows import ShoppingWorkflowState, build_shopping_workflow


class WorkflowRunner:
    """Build and execute one validated shopping workflow."""

    def __init__(
        self,
        *,
        planner: Any,
        recommendation_tool: Any,
        evidence_service: Any,
        verification_service: Any,
        shopping_plan_service: Any,
    ) -> None:
        dependencies = (
            (planner, "decide", "planner"),
            (recommendation_tool, "recommend", "recommendation_tool"),
            (evidence_service, "retrieve", "evidence_service"),
            (verification_service, "verify", "verification_service"),
            (shopping_plan_service, "create_plan", "shopping_plan_service"),
        )
        for dependency, method, name in dependencies:
            if dependency is None or not callable(getattr(dependency, method, None)):
                raise TypeError(f"{name} must provide {method}().")

        self._planner = planner
        self._recommendation_tool = recommendation_tool
        self._evidence_service = evidence_service
        self._verification_service = verification_service
        self._plan_service = shopping_plan_service

    def execute(
        self,
        initial_state: ShoppingWorkflowState,
        *,
        recursion_limit: int = 50,
    ) -> ShoppingWorkflowState:
        """Run the graph and validate its terminal serialized state."""

        if not isinstance(initial_state, ShoppingWorkflowState):
            raise TypeError("initial_state must be a ShoppingWorkflowState.")
        if (
            isinstance(recursion_limit, bool)
            or not isinstance(recursion_limit, int)
            or recursion_limit < 1
        ):
            raise ValueError("recursion_limit must be a positive integer.")

        graph = build_shopping_workflow(
            self._recommendation_tool,
            self._plan_service,
            planner=self._planner,
            evidence_service=self._evidence_service,
            verification_service=self._verification_service,
        )
        return ShoppingWorkflowState.model_validate(
            graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
        )
