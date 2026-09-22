"""Provider-neutral structured extraction of multi-requirement shopping goals.

The extractor reuses the existing text-provider boundary and validates raw JSON
as ``ShoppingGoalExtractionDecision``.  It does not project requirements,
allocate budget, read memory, create plans, or invoke runtime services.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from .goal_contracts import ShoppingGoalExtractionDecision
from .providers import (
    PlannerProviderError,
    PlannerSchemaError,
    PlannerTextProvider,
    PlannerTimeoutError,
)


_SYSTEM_PROMPT = """You extract one multi-requirement shopping goal from user text.
Return one JSON object only with total_budget, ordered requirement_proposals,
allocation_preferences, clarification_needed, and clarification_question.

Each requirement proposal may contain only category, quantity, max_budget,
required_features, soft_preferences, and priority. Preserve the order in which
the user mentions requirements; allocation preference target_index values refer
to that unchanged order. Use quantity=1 when no quantity is explicitly stated.
Use priority=3 in P0; do not infer a new priority scale or reorder requirements.

The goal total budget and a requirement explicit maximum budget are different
concepts. Put an overall bundle budget only in total_budget. Do not duplicate
the total budget into proposal max_budget fields. Set proposal max_budget only
when the user explicitly gives that requirement its own maximum amount.

Hard constraints such as "must support HDMI" belong in required_features.
Preferences such as "prefer HDMI" remain soft_preferences. "Save on" or
"as cheap as possible" maps only to save_more. "Allocate more" maps only to
allocate_more. Do not invent allocation amounts, weights, percentages, or
per-requirement budgets. Ordinary requirements have no allocation preference.

Do not invent historical preferences. A request to use purchase history does
not authorize guessed brands, features, colors, styles, or usage preferences;
only preferences explicitly present in the current user request may be output.
Do not query or describe memory.

Never invent a total budget. If the user provides no total budget, set
total_budget to null, clarification_needed to true, and provide one concise
non-empty budget clarification_question. You may retain only requirement
proposals explicitly stated by the user. Otherwise set clarification_needed to
false and clarification_question to null. Do not create products or facts."""


@runtime_checkable
class GoalExtractor(Protocol):
    """Provider-neutral interface for one validated shopping-goal proposal."""

    def extract(self, *, user_request: str) -> ShoppingGoalExtractionDecision:
        """Extract one structured multi-requirement shopping goal."""


class StructuredGoalExtractor:
    """Decode provider JSON into a strict ShoppingGoalExtractionDecision."""

    def __init__(
        self,
        provider: PlannerTextProvider,
        *,
        timeout_seconds: float = 30.0,
        schema_retries: int = 1,
    ) -> None:
        if provider is None or not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must provide complete().")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, numbers.Real)
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number.")
        if isinstance(schema_retries, bool) or schema_retries not in {0, 1}:
            raise ValueError("schema_retries must be 0 or 1.")
        self._provider = provider
        self._timeout_seconds = float(timeout_seconds)
        self._schema_retries = schema_retries

    def extract(self, *, user_request: str) -> ShoppingGoalExtractionDecision:
        if not isinstance(user_request, str):
            raise TypeError("user_request must be a string.")
        request = user_request.strip()
        if not request:
            raise ValueError("user_request must be non-empty.")
        messages = (
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": request},
        )
        last_error: Exception | None = None
        for _attempt in range(self._schema_retries + 1):
            try:
                raw = self._provider.complete(
                    messages=messages,
                    timeout_seconds=self._timeout_seconds,
                )
            except PlannerProviderError:
                raise
            except TimeoutError as exc:
                raise PlannerTimeoutError("Goal extraction request timed out.") from exc
            except Exception as exc:
                raise PlannerProviderError("Goal extraction provider failed.") from exc
            try:
                payload = json.loads(raw)
                return ShoppingGoalExtractionDecision.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                last_error = exc
        raise PlannerSchemaError(
            "Goal extraction failed JSON or schema validation."
        ) from last_error


class FakeGoalExtractor:
    """Fixed or request-keyed deterministic GoalExtractor for tests."""

    def __init__(
        self,
        decision: ShoppingGoalExtractionDecision | Mapping[str, object] | None = None,
        *,
        decisions_by_request: Mapping[
            str, ShoppingGoalExtractionDecision | Mapping[str, object]
        ]
        | None = None,
    ) -> None:
        if (decision is None) == (decisions_by_request is None):
            raise ValueError("Provide exactly one of decision or decisions_by_request.")
        self._decision = (
            None
            if decision is None
            else ShoppingGoalExtractionDecision.model_validate(decision)
        )
        if decisions_by_request is not None and (
            not decisions_by_request
            or any(
                not isinstance(request, str) or not request.strip()
                for request in decisions_by_request
            )
        ):
            raise ValueError("decisions_by_request requires non-empty string keys.")
        self._decisions_by_request = (
            None
            if decisions_by_request is None
            else {
                request.strip(): ShoppingGoalExtractionDecision.model_validate(value)
                for request, value in decisions_by_request.items()
            }
        )

    def extract(self, *, user_request: str) -> ShoppingGoalExtractionDecision:
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError("user_request must be a non-empty string.")
        if self._decision is not None:
            return self._decision
        request = user_request.strip()
        try:
            return self._decisions_by_request[request]  # type: ignore[index]
        except KeyError as exc:
            raise KeyError(f"No fake goal extraction for request={request!r}.") from exc
