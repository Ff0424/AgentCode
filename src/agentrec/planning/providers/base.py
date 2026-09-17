"""Low-level text-provider protocol and stable Planner provider errors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class PlannerProviderError(RuntimeError):
    """A provider request or response-envelope failure."""


class PlannerTimeoutError(PlannerProviderError):
    """The provider did not complete within the configured timeout."""


class PlannerSchemaError(PlannerProviderError):
    """Provider text was not valid JSON matching PlannerDecision."""


@runtime_checkable
class PlannerTextProvider(Protocol):
    """Minimal provider interface; it knows nothing about AgentRec domain state."""

    def complete(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        timeout_seconds: float,
    ) -> str:
        """Return raw assistant text for later JSON/schema validation."""
