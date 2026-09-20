"""Immutable contracts for future session and user-profile preference memory.

These contracts describe validated preference facts only. They do not store
model internals, persist data, merge requirements, or affect recommendation.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MemoryScope(str, Enum):
    """Lifetime boundary for one preference memory item."""

    SESSION = "session"
    USER_PROFILE = "user_profile"


class PreferenceType(str, Enum):
    """Closed vocabulary of preference meanings supported by the contract."""

    CATEGORY = "category"
    FEATURE = "feature"
    BUDGET = "budget"
    BRAND = "brand"


class PreferenceStatus(str, Enum):
    """Confirmation lifecycle for a preference expressed by a user."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class PreferenceItem(BaseModel):
    """One immutable, model-independent user preference memory fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyText
    user_id: NonEmptyText
    scope: MemoryScope
    type: PreferenceType
    value: NonEmptyText
    status: PreferenceStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_timestamps(self) -> "PreferenceItem":
        """Reject memory records whose update precedes their creation."""

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at.")
        return self
