"""Strict immutable schemas for the AgentRec product knowledge layer."""

from __future__ import annotations

import math
import numbers
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ProductDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    title: NonEmptyText | None = None
    categories: tuple[NonEmptyText, ...] = ()
    price: float | None = None
    average_rating: float | None = None
    rating_number: Annotated[int, Field(strict=True, ge=0)] | None = None
    features: tuple[NonEmptyText, ...] = ()
    description: NonEmptyText | None = None

    @field_validator("price", mode="before")
    @classmethod
    def validate_price(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("price must be a finite positive number or None.")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("price must be a finite positive number or None.")
        return numeric

    @field_validator("average_rating", mode="before")
    @classmethod
    def validate_rating(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("average_rating must be a finite number in [0, 5] or None.")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric <= 5:
            raise ValueError("average_rating must be a finite number in [0, 5] or None.")
        return numeric


class KnowledgeChunkType(str, Enum):
    SUMMARY = "summary"
    FEATURES = "features"
    DESCRIPTION = "description"


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyText
    chunk_index: Annotated[int, Field(strict=True, ge=0)]
    item_index: Annotated[int, Field(strict=True, ge=0)]
    parent_asin: NonEmptyText
    chunk_type: KnowledgeChunkType
    part_index: Annotated[int, Field(strict=True, ge=0)]
    text: NonEmptyText
