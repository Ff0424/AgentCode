"""Deterministic grounding of natural-language hard product constraints."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from ..feature_vocabulary import FeatureAliasRegistry, normalize_feature_text


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConstraintGroundingStatus(str, Enum):
    GROUNDED = "grounded"
    UNRESOLVED = "unresolved"


class ConstraintGroundingResult(BaseModel):
    """Immutable, score-free result for one required feature string."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_constraint: NonEmptyText
    canonical_constraint: NonEmptyText | None = None
    status: ConstraintGroundingStatus

    @model_validator(mode="after")
    def validate_status_payload(self) -> "ConstraintGroundingResult":
        if (
            self.status is ConstraintGroundingStatus.GROUNDED
            and self.canonical_constraint is None
        ):
            raise ValueError("GROUNDED requires canonical_constraint.")
        if (
            self.status is ConstraintGroundingStatus.UNRESOLVED
            and self.canonical_constraint is not None
        ):
            raise ValueError("UNRESOLVED cannot contain canonical_constraint.")
        return self


class UnresolvedConstraintGroundingError(ValueError):
    """Raised before Domain projection when a hard constraint is unresolved."""

    def __init__(self, results: tuple[ConstraintGroundingResult, ...]) -> None:
        self.results = results
        unresolved = tuple(
            value.original_constraint
            for value in results
            if value.status is ConstraintGroundingStatus.UNRESOLVED
        )
        super().__init__(
            "Required feature grounding is unresolved: " + ", ".join(unresolved)
        )


_CONSTRAINT_WRAPPERS = tuple(
    sorted(
        (
            "必须支持",
            "必须有",
            "需要支持",
            "需要",
            "支持",
            "要有",
            "must support",
            "must have",
            "needs to support",
            "need to support",
            "requires",
            "require",
        ),
        key=len,
        reverse=True,
    )
)


def _strip_registered_wrapper(value: str) -> str:
    """Remove one exact high-confidence wrapper, never an arbitrary substring."""

    normalized = normalize_feature_text(value)
    for raw_wrapper in _CONSTRAINT_WRAPPERS:
        wrapper = normalize_feature_text(raw_wrapper)
        if wrapper[-1].isascii() and wrapper[-1].isalnum():
            prefix = wrapper + " "
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip()
        elif normalized.startswith(wrapper):
            return normalized[len(wrapper) :].strip()
    return normalized


class DeterministicConstraintGrounder:
    """Ground registered concepts after removing one explicit wrapper."""

    def __init__(self, registry: FeatureAliasRegistry | None = None) -> None:
        self._registry = registry or FeatureAliasRegistry()

    def ground(self, constraint: str) -> ConstraintGroundingResult:
        if not isinstance(constraint, str) or not constraint.strip():
            raise ValueError("constraint must be a non-empty string.")
        original = constraint.strip()
        candidate = _strip_registered_wrapper(original)
        resolved = self._registry.resolve_registered(candidate) if candidate else None
        if resolved is None:
            return ConstraintGroundingResult(
                original_constraint=original,
                canonical_constraint=None,
                status=ConstraintGroundingStatus.UNRESOLVED,
            )
        canonical, _aliases = resolved
        return ConstraintGroundingResult(
            original_constraint=original,
            canonical_constraint=canonical,
            status=ConstraintGroundingStatus.GROUNDED,
        )

    def ground_required_features(self, values: tuple[str, ...]) -> tuple[str, ...]:
        results = tuple(self.ground(value) for value in values)
        if any(
            value.status is ConstraintGroundingStatus.UNRESOLVED
            for value in results
        ):
            raise UnresolvedConstraintGroundingError(results)
        # Canonical aliases may collapse multiple natural-language forms.
        return tuple(
            dict.fromkeys(
                value.canonical_constraint
                for value in results
                if value.canonical_constraint is not None
            )
        )
