"""Immutable, provider-neutral contracts for Recommendation Serving.

These types contain only structured recommendation inputs and compact product
outputs. Natural-language parsing, model scoring, persistence, and Agent state
mutation deliberately remain outside this module.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


MAX_TOP_K = 100


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _normalized_text_tuple(
    values: object,
    name: str,
    *,
    deduplicate: bool,
    drop_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of strings.")
    normalized: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{position}] must be a string.")
        text = value.strip()
        if not text:
            if drop_empty:
                continue
            raise ValueError(f"{name}[{position}] must be non-empty.")
        if deduplicate:
            if text in seen:
                continue
            seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


@dataclass(frozen=True)
class RecommendationRequest:
    """Validated request passed from the Agent tool boundary to recommendation."""

    user_id: str
    top_k: int = 10
    category: str | None = None
    max_price: float | None = None
    required_features: tuple[str, ...] = ()
    excluded_parent_asins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _nonempty_text(self.user_id, "user_id"))
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, numbers.Integral):
            raise TypeError("top_k must be an integer.")
        top_k = int(self.top_k)
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
        object.__setattr__(self, "top_k", top_k)
        if self.category is not None:
            object.__setattr__(
                self, "category", _nonempty_text(self.category, "category")
            )
        if self.max_price is not None:
            price = _finite_float(self.max_price, "max_price")
            if price <= 0.0:
                raise ValueError("max_price must be greater than zero.")
            object.__setattr__(self, "max_price", price)
        object.__setattr__(
            self,
            "required_features",
            _normalized_text_tuple(
                self.required_features,
                "required_features",
                deduplicate=False,
                drop_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "excluded_parent_asins",
            _normalized_text_tuple(
                self.excluded_parent_asins,
                "excluded_parent_asins",
                deduplicate=True,
            ),
        )


@dataclass(frozen=True)
class RecommendedProduct:
    """Compact Agent-facing product result; no description or embedding is exposed."""

    rank: int
    item_index: int
    parent_asin: str
    title: str | None
    price: float | None
    categories: tuple[str, ...]
    features: tuple[str, ...]
    score: float
    score_components: Mapping[str, float]
    score_source: str

    def __post_init__(self) -> None:
        for name in ("rank", "item_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer.")
            integer = int(value)
            if integer < (1 if name == "rank" else 0):
                raise ValueError(f"{name} is outside its valid range.")
            object.__setattr__(self, name, integer)
        object.__setattr__(
            self, "parent_asin", _nonempty_text(self.parent_asin, "parent_asin")
        )
        if self.title is not None:
            object.__setattr__(self, "title", _nonempty_text(self.title, "title"))
        if self.price is not None:
            price = _finite_float(self.price, "price")
            if price < 0.0:
                raise ValueError("price cannot be negative.")
            object.__setattr__(self, "price", price)
        object.__setattr__(
            self,
            "categories",
            _normalized_text_tuple(self.categories, "categories", deduplicate=False),
        )
        object.__setattr__(
            self,
            "features",
            _normalized_text_tuple(self.features, "features", deduplicate=False),
        )
        object.__setattr__(self, "score", _finite_float(self.score, "score"))
        if not isinstance(self.score_components, Mapping):
            raise TypeError("score_components must be a mapping.")
        components = {
            _nonempty_text(key, "score_components key"): _finite_float(
                value, f"score_components[{key!r}]"
            )
            for key, value in self.score_components.items()
        }
        object.__setattr__(self, "score_components", MappingProxyType(components))
        object.__setattr__(
            self, "score_source", _nonempty_text(self.score_source, "score_source")
        )


@dataclass(frozen=True)
class RecommendationResult:
    """One complete recommendation response with explicit identity provenance."""

    canonical_user_index: int | None
    model_user_row: int | None
    personalization_status: str
    applied_constraints: Mapping[str, object]
    artifact_version: str
    items: tuple[RecommendedProduct, ...] = field(default_factory=tuple)
    fallback_reason: str | None = None
    returned_count: int = field(init=False)

    def __post_init__(self) -> None:
        for name in ("canonical_user_index", "model_user_row"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be None or a non-negative integer.")
            integer = int(value)
            if integer < 0:
                raise ValueError(f"{name} cannot be negative.")
            object.__setattr__(self, name, integer)
        object.__setattr__(
            self,
            "personalization_status",
            _nonempty_text(self.personalization_status, "personalization_status"),
        )
        object.__setattr__(
            self, "artifact_version", _nonempty_text(self.artifact_version, "artifact_version")
        )
        if not isinstance(self.applied_constraints, Mapping):
            raise TypeError("applied_constraints must be a mapping.")
        object.__setattr__(
            self,
            "applied_constraints",
            _freeze(dict(self.applied_constraints)),
        )
        if not isinstance(self.items, tuple) or any(
            not isinstance(item, RecommendedProduct) for item in self.items
        ):
            raise TypeError("items must be a tuple of RecommendedProduct values.")
        expected_ranks = tuple(range(1, len(self.items) + 1))
        if tuple(item.rank for item in self.items) != expected_ranks:
            raise ValueError("items ranks must be exactly 1..returned_count.")
        if self.fallback_reason is not None:
            object.__setattr__(
                self,
                "fallback_reason",
                _nonempty_text(self.fallback_reason, "fallback_reason"),
            )
        object.__setattr__(self, "returned_count", len(self.items))
