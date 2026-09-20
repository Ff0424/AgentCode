"""Unit tests for immutable AgentRec preference memory contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from pydantic import ValidationError

from src.agentrec.memory import (
    MemoryScope,
    PreferenceItem,
    PreferenceStatus,
    PreferenceType,
)


CREATED_AT = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
UPDATED_AT = CREATED_AT + timedelta(minutes=5)


def preference(**updates) -> PreferenceItem:
    values = {
        "id": "preference-1",
        "user_id": "user-1",
        "scope": MemoryScope.SESSION,
        "type": PreferenceType.FEATURE,
        "value": "USB-C",
        "status": PreferenceStatus.CANDIDATE,
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
    }
    values.update(updates)
    return PreferenceItem(**values)


class MemoryContractTests(unittest.TestCase):
    def test_session_feature_candidate_is_created_and_trimmed(self) -> None:
        item = preference(id=" preference-1 ", user_id=" user-1 ", value=" USB-C ")
        self.assertEqual(item.id, "preference-1")
        self.assertEqual(item.user_id, "user-1")
        self.assertEqual(item.value, "USB-C")
        self.assertIs(item.scope, MemoryScope.SESSION)
        self.assertIs(item.type, PreferenceType.FEATURE)
        self.assertIs(item.status, PreferenceStatus.CANDIDATE)

    def test_confirmed_user_profile_preference_is_valid(self) -> None:
        item = preference(
            scope=MemoryScope.USER_PROFILE,
            status=PreferenceStatus.CONFIRMED,
            value="lightweight",
        )
        self.assertIs(item.scope, MemoryScope.USER_PROFILE)
        self.assertIs(item.status, PreferenceStatus.CONFIRMED)

    def test_empty_value_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            preference(value="   ")

    def test_empty_user_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            preference(user_id="")

    def test_invalid_enum_values_are_rejected(self) -> None:
        for field, value in (
            ("scope", "unknown"),
            ("type", "unknown"),
            ("status", "UNKNOWN"),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                preference(**{field: value})

    def test_internal_or_other_extra_fields_are_rejected(self) -> None:
        forbidden = (
            "embedding",
            "vector",
            "score",
            "similarity",
            "chunk_id",
            "artifact_path",
            "model_name",
        )
        for field in forbidden:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                preference(**{field: [1, 2, 3]})

    def test_preference_item_is_immutable(self) -> None:
        item = preference()
        with self.assertRaises(ValidationError):
            item.value = "HDMI"

    def test_both_memory_scopes_are_supported(self) -> None:
        session = preference(scope=MemoryScope.SESSION)
        profile = preference(scope=MemoryScope.USER_PROFILE)
        self.assertIs(session.scope, MemoryScope.SESSION)
        self.assertIs(profile.scope, MemoryScope.USER_PROFILE)

    def test_timestamps_must_be_timezone_aware_and_ordered(self) -> None:
        with self.assertRaises(ValidationError):
            preference(created_at=datetime(2026, 9, 20, 8, 0))
        with self.assertRaises(ValidationError):
            preference(updated_at=CREATED_AT - timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
