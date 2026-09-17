"""Provider-neutral natural-language requirement extraction for AgentRec V2.

The LLM may only propose structured requirement fields. Conversion to the
domain model is explicit and always runs ShoppingRequirement validation. This
module does not access recommendation, workflow, persistence, or product data.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from typing import Annotated, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from ..domain import ShoppingRequirement
from .providers import (
    PlannerProviderError,
    PlannerSchemaError,
    PlannerTextProvider,
    PlannerTimeoutError,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ReasonText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]

_SYSTEM_PROMPT = """You extract one shopping requirement from user text.
Return one JSON object only with category, quantity, max_budget,
required_features, soft_preferences, priority, clarification_needed, and
reason. Hard constraints are category, quantity, explicit maximum budget, and
explicit required features. Brand, style, color, and usage preferences remain
soft unless the user explicitly states they are mandatory. Never invent a
budget. If no budget is stated, set max_budget to null and
clarification_needed to true. Do not create products or return product facts."""


class RequirementExtractionDecision(BaseModel):
    """Strict LLM proposal before ShoppingRequirement domain conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: NonEmptyText
    quantity: Annotated[int, Field(strict=True, ge=1, le=100)] = 1
    max_budget: float | None = None
    required_features: Annotated[tuple[NonEmptyText, ...], Field(max_length=20)] = ()
    soft_preferences: Annotated[tuple[NonEmptyText, ...], Field(max_length=20)] = ()
    priority: Annotated[int, Field(strict=True, ge=1, le=5)] = 3
    clarification_needed: Annotated[bool, Field(strict=True)] = False
    reason: ReasonText

    @field_validator("max_budget", mode="before")
    @classmethod
    def validate_budget(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("max_budget must be a finite positive number or None.")
        budget = float(value)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("max_budget must be a finite positive number or None.")
        return budget

    @field_validator("required_features", "soft_preferences", mode="after")
    @classmethod
    def normalize_list_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        # Normalize whitespace only; semantic classification remains the LLM proposal.
        return tuple(" ".join(value.split()) for value in values)

    @model_validator(mode="after")
    def require_clarification_for_missing_budget(self) -> "RequirementExtractionDecision":
        if self.max_budget is None and not self.clarification_needed:
            raise ValueError("Missing max_budget requires clarification_needed=true.")
        return self

    def to_shopping_requirement(self, *, requirement_id: str) -> ShoppingRequirement:
        """Create a domain-validated requirement only when clarification is complete."""

        if self.clarification_needed:
            raise RequirementClarificationRequired(
                "Requirement needs clarification before domain conversion."
            )
        return ShoppingRequirement(
            requirement_id=requirement_id,
            category=self.category,
            quantity=self.quantity,
            max_budget=self.max_budget,
            required_features=self.required_features,
            soft_preferences=self.soft_preferences,
            priority=self.priority,
        )


class RequirementClarificationRequired(ValueError):
    """A valid proposal is incomplete and must not enter ShoppingPlan yet."""


@runtime_checkable
class RequirementExtractor(Protocol):
    def extract(self, *, user_request: str) -> RequirementExtractionDecision:
        """Return one validated extraction proposal."""


class StructuredRequirementExtractor:
    """Decode provider JSON into a strict RequirementExtractionDecision."""

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

    def extract(self, *, user_request: str) -> RequirementExtractionDecision:
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
                raise PlannerTimeoutError("Requirement extraction request timed out.") from exc
            except Exception as exc:
                raise PlannerProviderError("Requirement extraction provider failed.") from exc
            try:
                payload = json.loads(raw)
                return RequirementExtractionDecision.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                last_error = exc
        raise PlannerSchemaError(
            "Requirement extraction failed JSON or schema validation."
        ) from last_error


class FakeRequirementExtractor:
    """Fixed or request-keyed extraction implementation for deterministic tests."""

    def __init__(
        self,
        decision: RequirementExtractionDecision | Mapping[str, object] | None = None,
        *,
        decisions_by_request: Mapping[
            str, RequirementExtractionDecision | Mapping[str, object]
        ] | None = None,
    ) -> None:
        if (decision is None) == (decisions_by_request is None):
            raise ValueError("Provide exactly one of decision or decisions_by_request.")
        self._decision = (
            None if decision is None else RequirementExtractionDecision.model_validate(decision)
        )
        self._decisions_by_request = (
            None
            if decisions_by_request is None
            else {
                request: RequirementExtractionDecision.model_validate(value)
                for request, value in decisions_by_request.items()
            }
        )
        if self._decisions_by_request is not None and (
            not self._decisions_by_request
            or any(
                not isinstance(request, str) or not request.strip()
                for request in self._decisions_by_request
            )
        ):
            raise ValueError("decisions_by_request requires non-empty string keys.")

    def extract(self, *, user_request: str) -> RequirementExtractionDecision:
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError("user_request must be a non-empty string.")
        if self._decision is not None:
            return self._decision
        try:
            return self._decisions_by_request[user_request.strip()]  # type: ignore[index]
        except KeyError as exc:
            raise KeyError(f"No fake extraction for request={user_request.strip()!r}.") from exc
