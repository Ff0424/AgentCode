"""Deterministic unit tests for the in-memory preference store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from pydantic import ValidationError

from src.agentrec.memory import (
    MemoryScope,
    MemoryStore,
    PreferenceItem,
    PreferenceStatus,
    PreferenceType,
)


BASE_TIME = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)


def preference(
    preference_id: str,
    *,
    user_id: str = "user-1",
    scope: MemoryScope = MemoryScope.SESSION,
    status: PreferenceStatus = PreferenceStatus.CANDIDATE,
    created_offset: int = 0,
) -> PreferenceItem:
    created_at = BASE_TIME + timedelta(seconds=created_offset)
    return PreferenceItem(
        id=preference_id,
        user_id=user_id,
        scope=scope,
        type=PreferenceType.FEATURE,
        value="USB-C",
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


class MemoryStoreTests(unittest.TestCase):
    def test_add_and_get(self) -> None:
        store = MemoryStore()
        item = preference("pref-1")
        store.add(item)
        self.assertIs(store.get("pref-1"), item)
        self.assertIs(store.get(" pref-1 "), item)

    def test_duplicate_id_is_rejected_without_overwrite(self) -> None:
        store = MemoryStore()
        original = preference("pref-1", user_id="user-1")
        store.add(original)
        with self.assertRaises(ValueError):
            store.add(preference("pref-1", user_id="user-2"))
        self.assertIs(store.get("pref-1"), original)

    def test_user_isolation(self) -> None:
        store = MemoryStore()
        first = preference("pref-1", user_id="user-1")
        second = preference("pref-2", user_id="user-2")
        store.add(first)
        store.add(second)
        self.assertEqual(store.get_user_preferences("user-1"), (first,))
        self.assertEqual(store.get_user_preferences("user-2"), (second,))

    def test_scope_filtering(self) -> None:
        store = MemoryStore()
        session = preference("session", scope=MemoryScope.SESSION)
        profile = preference("profile", scope=MemoryScope.USER_PROFILE)
        store.add(profile)
        store.add(session)
        self.assertEqual(
            store.get_user_preferences("user-1", MemoryScope.SESSION),
            (session,),
        )
        self.assertEqual(
            store.get_user_preferences("user-1", MemoryScope.USER_PROFILE),
            (profile,),
        )

    def test_confirmed_filtering(self) -> None:
        store = MemoryStore()
        candidate = preference("candidate", status=PreferenceStatus.CANDIDATE)
        confirmed = preference("confirmed", status=PreferenceStatus.CONFIRMED)
        rejected = preference("rejected", status=PreferenceStatus.REJECTED)
        for item in (candidate, confirmed, rejected):
            store.add(item)
        self.assertEqual(store.get_confirmed_preferences("user-1"), (confirmed,))

    def test_deterministic_created_at_then_id_ordering(self) -> None:
        store = MemoryStore()
        later = preference("z-later", created_offset=5)
        same_b = preference("b-same")
        same_a = preference("a-same")
        for item in (later, same_b, same_a):
            store.add(item)
        self.assertEqual(
            tuple(value.id for value in store.get_user_preferences("user-1")),
            ("a-same", "b-same", "z-later"),
        )

    def test_returned_preferences_remain_immutable(self) -> None:
        store = MemoryStore()
        store.add(preference("pref-1"))
        returned = store.get_user_preferences("user-1")
        self.assertIsInstance(returned, tuple)
        with self.assertRaises(ValidationError):
            returned[0].value = "HDMI"

    def test_empty_store_and_unknown_id(self) -> None:
        store = MemoryStore()
        self.assertEqual(store.get_user_preferences("user-1"), ())
        self.assertEqual(store.get_confirmed_preferences("user-1"), ())
        self.assertIsNone(store.get("unknown"))

    def test_session_operations_fail_explicitly_until_contract_extension(self) -> None:
        store = MemoryStore()
        with self.assertRaises(NotImplementedError):
            store.get_session_preferences("session-1")
        with self.assertRaises(NotImplementedError):
            store.clear_session("session-1")


if __name__ == "__main__":
    unittest.main()
