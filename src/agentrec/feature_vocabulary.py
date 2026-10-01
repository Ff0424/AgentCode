"""Shared deterministic vocabulary for canonical product features.

This module contains no planning, recommendation, or verification policy.  It
only owns exact normalization and registered aliases so every layer resolves
the same canonical feature names.
"""

from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType
from typing import Mapping


DEFAULT_FEATURE_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "HDMI": ("hdmi",),
        "USB-C": ("usb-c", "usb c", "type-c", "type c"),
    }
)


def normalize_feature_text(value: str) -> str:
    """Normalize textual form without inferring an unregistered concept."""

    if not isinstance(value, str):
        raise TypeError("Feature text must be a string.")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FeatureAliasRegistry:
    """Immutable exact alias registry shared across trusted boundaries."""

    def __init__(self, aliases: Mapping[str, tuple[str, ...]] | None = None) -> None:
        source = aliases or DEFAULT_FEATURE_ALIASES
        normalized: dict[str, tuple[str, ...]] = {}
        for canonical, values in source.items():
            if not isinstance(canonical, str) or not canonical.strip() or not values:
                raise ValueError("Alias registry entries must be non-empty.")
            members = tuple(
                dict.fromkeys(normalize_feature_text(value) for value in values)
            )
            if any(not value for value in members):
                raise ValueError("Feature aliases must be non-empty.")
            normalized[canonical.strip()] = members
        self._aliases = MappingProxyType(normalized)

    def resolve_registered(self, constraint: str) -> tuple[str, tuple[str, ...]] | None:
        """Resolve only exact registered aliases; return ``None`` if unknown."""

        normalized = normalize_feature_text(constraint)
        if not normalized:
            raise ValueError("constraint must be non-empty.")
        for canonical, aliases in self._aliases.items():
            if normalized in aliases or normalized == normalize_feature_text(canonical):
                return canonical, aliases
        return None

    def resolve(self, constraint: str) -> tuple[str, tuple[str, ...]]:
        """Preserve the verifier's fail-closed fallback for unknown constraints."""

        normalized = normalize_feature_text(constraint)
        if not normalized:
            raise ValueError("constraint must be non-empty.")
        registered = self.resolve_registered(constraint)
        return registered if registered is not None else (constraint.strip(), (normalized,))
