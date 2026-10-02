"""Runtime-independent adapter routing for AgentRec evaluation cases.

Concrete Track adapters are intentionally outside this foundation module. The
router only validates Track identity and delegates an immutable EvaluationCase
to an injected adapter implementing the small protocol below.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import EvaluationCase, EvaluationResult


TRACK_A = "track_a"
TRACK_B = "track_b"
TRACK_C = "track_c"
EVALUATION_TRACKS = (TRACK_A, TRACK_B, TRACK_C)


@runtime_checkable
class EvaluationAdapter(Protocol):
    """Execution boundary implemented by a future track-specific adapter."""

    def execute(self, case: EvaluationCase) -> EvaluationResult:
        """Execute one validated evaluation case and return its result."""

        ...


def resolve_track(case: EvaluationCase) -> str:
    """Return the case's single Track tag or reject ambiguous attribution."""

    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase.")
    matches = tuple(track for track in EVALUATION_TRACKS if track in case.tags)
    if len(matches) != 1:
        raise ValueError(
            "EvaluationCase tags must contain exactly one of "
            f"{EVALUATION_TRACKS!r}; found {matches!r}."
        )
    return matches[0]


class EvaluationAdapterRouter:
    """Register one adapter per Track and deterministically dispatch cases."""

    def __init__(self) -> None:
        self._adapters: dict[str, EvaluationAdapter] = {}

    def register(self, track: str, adapter: EvaluationAdapter) -> None:
        """Register an adapter without silently replacing existing behavior."""

        if track not in EVALUATION_TRACKS:
            raise ValueError(f"Unsupported evaluation track: {track!r}.")
        if not isinstance(adapter, EvaluationAdapter):
            raise TypeError("adapter must implement execute(case).")
        if track in self._adapters:
            raise ValueError(f"Adapter already registered for {track!r}.")
        self._adapters[track] = adapter

    def execute(self, case: EvaluationCase) -> EvaluationResult:
        """Resolve the case Track and execute its registered adapter."""

        track = resolve_track(case)
        adapter = self._adapters.get(track)
        if adapter is None:
            raise ValueError(f"No evaluation adapter registered for {track!r}.")
        result = adapter.execute(case)
        if not isinstance(result, EvaluationResult):
            raise TypeError("Evaluation adapter must return an EvaluationResult.")
        if result.case_id != case.case_id:
            raise ValueError("EvaluationResult case_id must match the input case.")
        return result


__all__ = [
    "EVALUATION_TRACKS",
    "TRACK_A",
    "TRACK_B",
    "TRACK_C",
    "EvaluationAdapter",
    "EvaluationAdapterRouter",
    "resolve_track",
]
