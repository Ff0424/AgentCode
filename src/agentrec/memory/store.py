"""Deterministic in-memory storage for immutable preference contracts.

The store owns no persistence, requirement merge, workflow integration, or
model behavior. PreferenceItem instances are retained and returned unchanged.
"""

from __future__ import annotations

from .contracts import MemoryScope, PreferenceItem, PreferenceStatus


def _non_empty_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


class MemoryStore:
    """Process-local store keyed by the immutable PreferenceItem identity."""

    def __init__(self) -> None:
        self._internal_preferences: dict[str, PreferenceItem] = {}

    def add(self, preference: PreferenceItem) -> None:
        """Add one preference without permitting silent identity overwrite."""

        if not isinstance(preference, PreferenceItem):
            raise TypeError("preference must be a PreferenceItem.")
        if preference.id in self._internal_preferences:
            raise ValueError(f"Preference id already exists: {preference.id!r}.")
        self._internal_preferences[preference.id] = preference

    def get(self, preference_id: str) -> PreferenceItem | None:
        """Return one immutable preference, or None when its ID is unknown."""

        normalized = _non_empty_identifier(
            preference_id, field_name="preference_id"
        )
        return self._internal_preferences.get(normalized)

    def get_user_preferences(
        self,
        user_id: str,
        scope: MemoryScope | None = None,
    ) -> tuple[PreferenceItem, ...]:
        """Return one user's preferences in deterministic creation order."""

        normalized_user_id = _non_empty_identifier(user_id, field_name="user_id")
        if scope is not None and not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope or None.")
        matches = (
            preference
            for preference in self._internal_preferences.values()
            if preference.user_id == normalized_user_id
            and (scope is None or preference.scope is scope)
        )
        return tuple(sorted(matches, key=lambda value: (value.created_at, value.id)))

    def get_confirmed_preferences(
        self,
        user_id: str,
    ) -> tuple[PreferenceItem, ...]:
        """Return only explicitly confirmed preferences for one user."""

        return tuple(
            preference
            for preference in self.get_user_preferences(user_id)
            if preference.status is PreferenceStatus.CONFIRMED
        )

    def get_session_preferences(
        self,
        session_id: str,
    ) -> tuple[PreferenceItem, ...]:
        """Reject unsupported session lookup until the contract has session_id."""

        _non_empty_identifier(session_id, field_name="session_id")
        raise NotImplementedError(
            "PreferenceItem has no session_id; session lookup requires a future "
            "memory contract extension."
        )

    def clear_session(self, session_id: str) -> int:
        """Reject unsupported session cleanup rather than guessing membership."""

        _non_empty_identifier(session_id, field_name="session_id")
        raise NotImplementedError(
            "PreferenceItem has no session_id; session cleanup requires a future "
            "memory contract extension."
        )
