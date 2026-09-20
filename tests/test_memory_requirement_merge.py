"""Tests for deterministic confirmed-memory requirement merging."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.agentrec.domain import ShoppingRequirement
from src.agentrec.memory import (
    MemoryScope,
    PreferenceItem,
    PreferenceStatus,
    PreferenceType,
    RequirementMemoryMerger,
)


NOW = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
MERGER = RequirementMemoryMerger()


def preference(
    preference_id: str,
    preference_type: PreferenceType,
    value: str,
    *,
    status: PreferenceStatus = PreferenceStatus.CONFIRMED,
) -> PreferenceItem:
    return PreferenceItem(
        id=preference_id,
        user_id="user-1",
        scope=MemoryScope.USER_PROFILE,
        type=preference_type,
        value=value,
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def requirement(**updates) -> ShoppingRequirement:
    values = {
        "requirement_id": "dock",
        "category": "Dock",
        "max_budget": None,
        "required_features": ("HDMI",),
    }
    values.update(updates)
    return ShoppingRequirement(**values)


class RequirementMemoryMergeTests(unittest.TestCase):
    def test_confirmed_feature_is_appended(self) -> None:
        merged = MERGER.merge(requirement(), (
            preference("feature-1", PreferenceType.FEATURE, "USB-C"),
        ))
        self.assertEqual(merged.required_features, ("HDMI", "USB-C"))

    def test_candidate_feature_is_ignored(self) -> None:
        original = requirement()
        merged = MERGER.merge(original, (
            preference(
                "feature-1",
                PreferenceType.FEATURE,
                "USB-C",
                status=PreferenceStatus.CANDIDATE,
            ),
        ))
        self.assertEqual(merged, original)

    def test_rejected_feature_is_ignored(self) -> None:
        original = requirement()
        merged = MERGER.merge(original, (
            preference(
                "feature-1",
                PreferenceType.FEATURE,
                "USB-C",
                status=PreferenceStatus.REJECTED,
            ),
        ))
        self.assertEqual(merged, original)

    def test_duplicate_feature_is_not_added(self) -> None:
        merged = MERGER.merge(requirement(required_features=("USB-C",)), (
            preference("feature-1", PreferenceType.FEATURE, "USB-C"),
        ))
        self.assertEqual(merged.required_features, ("USB-C",))

    def test_current_requirement_has_priority_over_memory(self) -> None:
        original = requirement(category="Lenovo", max_budget=100)
        merged = MERGER.merge(original, (
            preference("category", PreferenceType.CATEGORY, "Apple"),
            preference("budget", PreferenceType.BUDGET, "300"),
        ))
        self.assertEqual(merged.category, "Lenovo")
        self.assertEqual(merged.max_budget, 100)

    def test_confirmed_budget_fills_missing_budget(self) -> None:
        merged = MERGER.merge(requirement(max_budget=None), (
            preference("budget", PreferenceType.BUDGET, "300.50"),
        ))
        self.assertEqual(merged.max_budget, 300.5)

    def test_original_requirement_is_unchanged(self) -> None:
        original = requirement()
        before = original.model_dump(mode="json")
        merged = MERGER.merge(original, (
            preference("feature-1", PreferenceType.FEATURE, "USB-C"),
        ))
        self.assertEqual(original.model_dump(mode="json"), before)
        self.assertIsNot(merged, original)

    def test_memory_feature_order_is_deterministic(self) -> None:
        preferences = (
            preference("feature-2", PreferenceType.FEATURE, "Ethernet"),
            preference("feature-1", PreferenceType.FEATURE, "USB-C"),
            preference("feature-3", PreferenceType.FEATURE, "Ethernet"),
        )
        first = MERGER.merge(requirement(), preferences)
        second = MERGER.merge(requirement(), preferences)
        self.assertEqual(first, second)
        self.assertEqual(
            first.required_features,
            ("HDMI", "Ethernet", "USB-C"),
        )

    def test_brand_is_explicitly_unsupported_and_ignored(self) -> None:
        original = requirement()
        merged = MERGER.merge(original, (
            preference("brand", PreferenceType.BRAND, "Apple"),
        ))
        self.assertEqual(merged, original)
        self.assertIn(
            PreferenceType.BRAND,
            RequirementMemoryMerger.unsupported_preference_types,
        )

    def test_invalid_confirmed_budget_fails_closed(self) -> None:
        for value in ("not-a-number", "0", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MERGER.merge(requirement(), (
                    preference("budget", PreferenceType.BUDGET, value),
                ))


if __name__ == "__main__":
    unittest.main()
