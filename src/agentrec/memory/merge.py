"""Deterministic merge of confirmed memory into ShoppingRequirement.

Current requirement facts always win. Confirmed memory may only add missing
supported facts; candidate and rejected preferences never affect the result.
"""

from __future__ import annotations

import math

from ..domain import ShoppingRequirement
from .contracts import PreferenceItem, PreferenceStatus, PreferenceType


class RequirementMemoryMerger:
    """Create a new validated requirement from explicit facts plus memory."""

    # ShoppingRequirement has no brand field. Keeping this explicit prevents a
    # future caller from assuming that BRAND memory was silently applied.
    unsupported_preference_types = (PreferenceType.BRAND,)

    def merge(
        self,
        requirement: ShoppingRequirement,
        preferences: tuple[PreferenceItem, ...],
    ) -> ShoppingRequirement:
        """Return a new requirement with supported confirmed preferences merged."""

        if not isinstance(requirement, ShoppingRequirement):
            raise TypeError("requirement must be a ShoppingRequirement.")
        if not isinstance(preferences, tuple):
            raise TypeError("preferences must be a tuple of PreferenceItem values.")
        if any(not isinstance(value, PreferenceItem) for value in preferences):
            raise TypeError("Every preferences entry must be a PreferenceItem.")

        confirmed = tuple(
            value
            for value in preferences
            if value.status is PreferenceStatus.CONFIRMED
        )

        features = list(requirement.required_features)
        known_features = set(features)
        for preference in confirmed:
            if (
                preference.type is PreferenceType.FEATURE
                and preference.value not in known_features
            ):
                features.append(preference.value)
                known_features.add(preference.value)

        # A valid ShoppingRequirement currently always has a non-empty category.
        # This conditional preserves the frozen precedence rule if the Domain
        # later permits an unspecified category, without overriding current input.
        category = requirement.category
        if not category:
            category = self._first_value(confirmed, PreferenceType.CATEGORY)

        max_budget = requirement.max_budget
        if max_budget is None:
            budget_value = self._first_value(confirmed, PreferenceType.BUDGET)
            if budget_value is not None:
                max_budget = self._parse_budget(budget_value)

        values = requirement.model_dump()
        values.update(
            category=category,
            max_budget=max_budget,
            required_features=tuple(features),
        )
        # Revalidate rather than model_copy so all frozen Domain constraints remain
        # authoritative after memory contributes values.
        return ShoppingRequirement.model_validate(values)

    @staticmethod
    def _first_value(
        preferences: tuple[PreferenceItem, ...],
        preference_type: PreferenceType,
    ) -> str | None:
        return next(
            (
                value.value
                for value in preferences
                if value.type is preference_type
            ),
            None,
        )

    @staticmethod
    def _parse_budget(value: str) -> float:
        try:
            budget = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Confirmed BUDGET preference must be a finite positive number: {value!r}."
            ) from exc
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError(
                f"Confirmed BUDGET preference must be a finite positive number: {value!r}."
            )
        return budget
