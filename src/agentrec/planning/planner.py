"""Planner interface and deterministic test implementation.

No provider SDK or real LLM is imported here. Production adapters can later
implement the same protocol after converting provider output into a validated
PlannerDecision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .contracts import PlannerDecision, validate_planner_decision


@runtime_checkable
class Planner(Protocol):
    """Provider-neutral interface consumed by future planner workflow nodes."""

    def decide(self, *, context: Mapping[str, Any]) -> PlannerDecision:
        """Return one validated proposal without executing it."""


class FakePlanner:
    """Return a fixed or context-keyed decision without mutable call state."""

    def __init__(
        self,
        decision: PlannerDecision | Mapping[str, Any] | None = None,
        *,
        decisions_by_key: Mapping[str, PlannerDecision | Mapping[str, Any]] | None = None,
    ) -> None:
        if (decision is None) == (decisions_by_key is None):
            raise ValueError("Provide exactly one of decision or decisions_by_key.")
        self._decision = (
            None if decision is None else validate_planner_decision(decision)
        )
        self._decisions_by_key = (
            None
            if decisions_by_key is None
            else {
                key: validate_planner_decision(value)
                for key, value in decisions_by_key.items()
            }
        )
        if self._decisions_by_key is not None and (
            not self._decisions_by_key
            or any(not isinstance(key, str) or not key for key in self._decisions_by_key)
        ):
            raise ValueError("decisions_by_key must contain non-empty string keys.")

    def decide(self, *, context: Mapping[str, Any]) -> PlannerDecision:
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping.")
        if self._decision is not None:
            return self._decision
        decision_key = context.get("decision_key")
        if not isinstance(decision_key, str) or not decision_key:
            raise ValueError("Context must provide a non-empty decision_key.")
        try:
            # Models are frozen and lookup is stateless, so repeated runs are stable.
            return self._decisions_by_key[decision_key]  # type: ignore[index]
        except KeyError as exc:
            raise KeyError(f"No fake decision for key={decision_key!r}.") from exc
