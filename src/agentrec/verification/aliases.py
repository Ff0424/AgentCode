"""Conservative canonical feature aliases used by the deterministic verifier."""

from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType
from typing import Mapping


def normalize_feature_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Feature text must be a string.")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FeatureAliasRegistry:
    """Immutable high-confidence aliases; it performs no common-sense inference."""

    def __init__(self, aliases: Mapping[str, tuple[str, ...]] | None = None) -> None:
        source = aliases or {
            "HDMI": ("hdmi",),
            "USB-C": ("usb-c", "usb c", "type-c", "type c"),
        }
        normalized: dict[str, tuple[str, ...]] = {}
        for canonical, values in source.items():
            if not isinstance(canonical, str) or not canonical.strip() or not values:
                raise ValueError("Alias registry entries must be non-empty.")
            members = tuple(dict.fromkeys(normalize_feature_text(value) for value in values))
            if any(not value for value in members):
                raise ValueError("Feature aliases must be non-empty.")
            normalized[canonical.strip()] = members
        self._aliases = MappingProxyType(normalized)

    def resolve(self, constraint: str) -> tuple[str, tuple[str, ...]]:
        normalized = normalize_feature_text(constraint)
        if not normalized:
            raise ValueError("constraint must be non-empty.")
        for canonical, aliases in self._aliases.items():
            if normalized in aliases or normalized == normalize_feature_text(canonical):
                return canonical, aliases
        return constraint.strip(), (normalized,)
