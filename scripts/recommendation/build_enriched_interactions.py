"""
Build and validate the AgentRec V2 enriched 10-core interaction dataset.

Inputs:
    --interactions
        Frozen interactions_10core.npz containing aligned user_ids/item_ids.
    --user-mapping
        Canonical integer user index -> Amazon user_id JSON mapping.
    --item-mapping
        Canonical integer item index -> Amazon parent_asin JSON mapping.
    --parent-asins
        Ordered parent_asins_10core.json identity scope.
    --reviews
        Original Amazon Reviews'23 Electronics JSONL or JSONL.GZ file.

Outputs:
    --output
        enriched_interactions.npz with user_indices, item_indices, ratings,
        and timestamps in the exact frozen interaction row order.
    --report
        Markdown validation report for the V2-05.2 release record.

The original reviews and all frozen Recommendation artifacts are read-only.
Both outputs are first written to temporary files and published only after the
complete source scan and strict artifact validation succeed.
"""

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np


# ============================================================
# 1. Frozen dataset contract and input helpers
# ============================================================

EXPECTED_INTERACTION_COUNT = 6_173_972
MAPPING_READ_CHUNK_CHARS = 1 << 20
MAX_MAPPING_TOKEN_CHARS = 4 << 20


def _require_file(path: Path, label: str) -> None:
    """Fail early when a required input is absent or not a regular file."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def _load_json(path: Path, label: str):
    """Load a regular UTF-8 JSON file with contextual errors."""

    _require_file(path, label)
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} file {path}: {exc}") from exc
    except UnicodeError as exc:
        raise ValueError(f"{label} file is not valid UTF-8: {path}") from exc


def load_frozen_interactions(
    path: Path,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the immutable ordered 10-core identity pairs."""

    _require_file(path, "frozen interactions")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required_keys = {"user_ids", "item_ids"}
            missing_keys = required_keys.difference(archive.files)
            if missing_keys:
                raise ValueError(
                    "Frozen interactions archive is missing arrays: "
                    f"{sorted(missing_keys)}."
                )
            user_ids = np.asarray(archive["user_ids"])
            item_ids = np.asarray(archive["item_ids"])
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Unable to load frozen interactions archive {path}: {exc}"
        ) from exc

    if user_ids.ndim != 1 or item_ids.ndim != 1:
        raise ValueError("Frozen user_ids and item_ids must be one-dimensional.")
    if user_ids.shape != item_ids.shape:
        raise ValueError(
            "Frozen user_ids and item_ids shapes differ: "
            f"{user_ids.shape} != {item_ids.shape}."
        )
    if user_ids.shape[0] != expected_count:
        raise ValueError(
            f"Frozen interaction count is {user_ids.shape[0]:,}; expected "
            f"{expected_count:,}."
        )
    if user_ids.dtype.kind not in "iu" or item_ids.dtype.kind not in "iu":
        raise TypeError("Frozen user_ids and item_ids must use integer dtypes.")
    if np.any(user_ids < 0) or np.any(item_ids < 0):
        raise ValueError("Frozen canonical user/item indices must be non-negative.")
    if user_ids.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Frozen user indices do not fit in int32.")
    if item_ids.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Frozen item indices do not fit in int32.")

    # Preserve the exact upstream values while fixing the published dtypes.
    return (
        np.ascontiguousarray(user_ids, dtype=np.int32),
        np.ascontiguousarray(item_ids, dtype=np.int32),
    )


# ============================================================
# 2. Memory-bounded canonical mapping reader
# ============================================================

def iter_flat_json_object(path: Path, label: str) -> Iterator[tuple[object, object]]:
    """Stream key/value pairs from a flat JSON object using only stdlib JSON.

    The canonical user mapping is hundreds of MiB. Loading the full object and
    reversing it would consume unnecessary memory, so this reader incrementally
    decodes one JSON key and value at a time.
    """

    _require_file(path, label)
    decoder = json.JSONDecoder()

    try:
        with path.open("r", encoding="utf-8") as input_file:
            buffer = ""
            position = 0
            eof = False

            def read_more() -> None:
                nonlocal buffer, position, eof
                if position:
                    buffer = buffer[position:]
                    position = 0
                chunk = input_file.read(MAPPING_READ_CHUNK_CHARS)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            def next_non_whitespace() -> str:
                nonlocal position
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        return buffer[position]
                    if eof:
                        raise ValueError(f"Unexpected end of {label} JSON object.")
                    read_more()

            def decode_value():
                nonlocal position
                while True:
                    next_non_whitespace()
                    try:
                        value, end_position = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError as exc:
                        remaining = len(buffer) - position
                        if eof or remaining > MAX_MAPPING_TOKEN_CHARS:
                            raise ValueError(
                                f"Invalid or oversized JSON token in {label} "
                                f"file {path}: {exc}."
                            ) from exc
                        read_more()
                        continue
                    position = end_position
                    return value

            read_more()
            if next_non_whitespace() != "{":
                raise ValueError(f"{label} must contain a JSON object.")
            position += 1

            if next_non_whitespace() == "}":
                position += 1
            else:
                while True:
                    key = decode_value()
                    if not isinstance(key, str):
                        raise ValueError(f"Every {label} JSON key must be a string.")
                    if next_non_whitespace() != ":":
                        raise ValueError(f"Missing ':' after a key in {label}.")
                    position += 1
                    value = decode_value()
                    yield key, value

                    delimiter = next_non_whitespace()
                    if delimiter == ",":
                        position += 1
                        continue
                    if delimiter == "}":
                        position += 1
                        break
                    raise ValueError(
                        f"Expected ',' or '}}' after a value in {label}."
                    )

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    raise ValueError(f"Unexpected trailing content in {label}.")
                if eof:
                    break
                read_more()
    except UnicodeError as exc:
        raise ValueError(f"{label} file is not valid UTF-8: {path}") from exc


def load_target_raw_identity_map(
    path: Path,
    target_indices: set[int],
    label: str,
    raw_identity_label: str,
) -> dict[str, int]:
    """Load only raw identities belonging to frozen target integer indices."""

    raw_to_index: dict[str, int] = {}
    found_indices: set[int] = set()

    for raw_index, raw_identity in iter_flat_json_object(path, label):
        if not raw_index.isdecimal():
            raise ValueError(
                f"{label} key must be a non-negative integer string: "
                f"{raw_index!r}."
            )
        canonical_index = int(raw_index)
        if canonical_index not in target_indices:
            continue
        if canonical_index in found_indices:
            raise ValueError(
                f"Duplicate target canonical index in {label}: {canonical_index}."
            )
        if not isinstance(raw_identity, str) or not raw_identity.strip():
            raise ValueError(
                f"{raw_identity_label} for canonical index {canonical_index} "
                "must be a non-empty string."
            )
        if raw_identity in raw_to_index:
            raise ValueError(
                f"Duplicate target {raw_identity_label} in {label}: "
                f"{raw_identity!r}."
            )

        raw_to_index[raw_identity] = canonical_index
        found_indices.add(canonical_index)
        if len(found_indices) == len(target_indices):
            break

    missing_indices = target_indices.difference(found_indices)
    if missing_indices:
        raise ValueError(
            f"{label} is missing {len(missing_indices):,} frozen canonical "
            f"indices; examples: {sorted(missing_indices)[:10]}."
        )

    return raw_to_index


def load_and_validate_item_identities(
    item_mapping_path: Path,
    parent_asins_path: Path,
    target_item_indices: np.ndarray,
) -> dict[str, int]:
    """Validate ordered 10-core ASINs against the canonical item mapping."""

    raw_parent_asins = _load_json(parent_asins_path, "10-core parent ASIN")
    if not isinstance(raw_parent_asins, list):
        raise TypeError("10-core parent ASIN file must contain a JSON list.")
    if len(raw_parent_asins) != len(target_item_indices):
        raise ValueError(
            "10-core parent ASIN count does not match unique frozen item count: "
            f"{len(raw_parent_asins):,} != {len(target_item_indices):,}."
        )

    expected_by_index: dict[int, str] = {}
    seen_parent_asins: set[str] = set()
    for position, (item_index, parent_asin) in enumerate(
        zip(target_item_indices.tolist(), raw_parent_asins)
    ):
        if not isinstance(parent_asin, str) or not parent_asin.strip():
            raise ValueError(
                f"parent_asins[{position}] must be a non-empty string."
            )
        if parent_asin in seen_parent_asins:
            raise ValueError(
                f"Duplicate 10-core parent_asin at position {position}: "
                f"{parent_asin!r}."
            )
        seen_parent_asins.add(parent_asin)
        expected_by_index[int(item_index)] = parent_asin

    raw_to_item = load_target_raw_identity_map(
        item_mapping_path,
        set(expected_by_index),
        "item mapping",
        "parent_asin",
    )
    actual_by_index = {index: raw_id for raw_id, index in raw_to_item.items()}
    for item_index, expected_parent_asin in expected_by_index.items():
        actual_parent_asin = actual_by_index[item_index]
        if actual_parent_asin != expected_parent_asin:
            raise ValueError(
                "10-core parent ASIN order disagrees with item_mapping.json at "
                f"item_index {item_index}: {expected_parent_asin!r} != "
                f"{actual_parent_asin!r}."
            )

    return raw_to_item


# ============================================================
# 3. Original review streaming and exact row reconstruction
# ============================================================

def _open_reviews(path: Path) -> TextIO:
    """Open plain or gzip-compressed review JSONL as UTF-8 text."""

    _require_file(path, "Amazon reviews")
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _validate_rating(value, source_line: int) -> float:
    """Validate one matched Amazon rating and return a Python float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"rating at source line {source_line:,} must be numeric, got "
            f"{type(value).__name__}."
        )
    rating = float(value)
    if not math.isfinite(rating) or not 1.0 <= rating <= 5.0:
        raise ValueError(
            f"rating at source line {source_line:,} is outside [1, 5]: "
            f"{rating!r}."
        )
    return rating


def _validate_timestamp(value, source_line: int) -> int:
    """Validate one matched Unix timestamp expressed in milliseconds."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"timestamp at source line {source_line:,} must be an integer, got "
            f"{type(value).__name__}."
        )
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError(
            f"timestamp at source line {source_line:,} is outside int64 "
            f"milliseconds: {value!r}."
        )
    try:
        datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(
            f"timestamp at source line {source_line:,} cannot be interpreted "
            f"as Unix milliseconds: {value!r}."
        ) from exc
    return value


def reconstruct_enriched_rows(
    reviews_path: Path,
    frozen_users: np.ndarray,
    frozen_items: np.ndarray,
    raw_to_user: dict[str, int],
    raw_to_item: dict[str, int],
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Recover rating/timestamp while enforcing exact frozen row order."""

    interaction_count = frozen_users.shape[0]
    output_users = np.empty(interaction_count, dtype=np.int32)
    output_items = np.empty(interaction_count, dtype=np.int32)
    ratings = np.empty(interaction_count, dtype=np.float32)
    timestamps = np.empty(interaction_count, dtype=np.int64)

    matched_count = 0
    source_count = 0
    rating_counts: Counter[float] = Counter()
    timestamp_min = None
    timestamp_max = None

    try:
        with _open_reviews(reviews_path) as review_file:
            for source_count, line in enumerate(review_file, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty review record at source line {source_count:,}."
                    )
                try:
                    review = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid review JSON at source line {source_count:,}: "
                        f"{exc}."
                    ) from exc
                if not isinstance(review, dict):
                    raise TypeError(
                        f"Review at source line {source_count:,} must be an object."
                    )

                raw_user_id = review.get("user_id")
                parent_asin = review.get("parent_asin")
                if not isinstance(raw_user_id, str) or not raw_user_id:
                    raise ValueError(
                        f"user_id at source line {source_count:,} must be a "
                        "non-empty string."
                    )
                if not isinstance(parent_asin, str) or not parent_asin:
                    raise ValueError(
                        f"parent_asin at source line {source_count:,} must be a "
                        "non-empty string."
                    )

                user_index = raw_to_user.get(raw_user_id)
                item_index = raw_to_item.get(parent_asin)
                if user_index is None or item_index is None:
                    continue

                if matched_count >= interaction_count:
                    raise ValueError(
                        "Original reviews contain more target user-item rows than "
                        f"the frozen dataset; first extra match is source line "
                        f"{source_count:,}."
                    )

                expected_user = int(frozen_users[matched_count])
                expected_item = int(frozen_items[matched_count])
                if user_index != expected_user or item_index != expected_item:
                    raise ValueError(
                        "Frozen row order/identity mismatch at enriched row "
                        f"{matched_count:,} (source line {source_count:,}): "
                        f"expected ({expected_user}, {expected_item}), got "
                        f"({user_index}, {item_index})."
                    )

                rating = _validate_rating(review.get("rating"), source_count)
                timestamp = _validate_timestamp(
                    review.get("timestamp"), source_count
                )

                output_users[matched_count] = user_index
                output_items[matched_count] = item_index
                ratings[matched_count] = rating
                timestamps[matched_count] = timestamp
                matched_count += 1

                rating_counts[rating] += 1
                timestamp_min = (
                    timestamp
                    if timestamp_min is None
                    else min(timestamp_min, timestamp)
                )
                timestamp_max = (
                    timestamp
                    if timestamp_max is None
                    else max(timestamp_max, timestamp)
                )

                if progress_every and matched_count % progress_every == 0:
                    print(
                        f"Matched {matched_count:,}/{interaction_count:,} "
                        f"enriched rows after scanning {source_count:,} reviews."
                    )
    except UnicodeError as exc:
        raise ValueError(f"Amazon reviews file is not valid UTF-8: {reviews_path}") from exc

    if matched_count != interaction_count:
        raise ValueError(
            f"Matched {matched_count:,} target interactions; expected "
            f"{interaction_count:,}."
        )

    arrays = {
        "user_indices": output_users,
        "item_indices": output_items,
        "ratings": ratings,
        "timestamps": timestamps,
    }
    statistics = {
        "source_review_count": source_count,
        "interaction_count": matched_count,
        "identity_match_count": matched_count,
        "identity_mismatch_count": 0,
        "rating_counts": dict(sorted(rating_counts.items())),
        "rating_min": float(ratings.min()),
        "rating_max": float(ratings.max()),
        "rating_mean": float(ratings.mean(dtype=np.float64)),
        "timestamp_min": int(timestamp_min),
        "timestamp_max": int(timestamp_max),
    }
    return arrays, statistics


# ============================================================
# 4. Safe NPZ publication and post-write validation
# ============================================================

def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def write_npz_temporary(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ to an exact temporary path and fsync it."""

    with path.open("xb") as output_file:
        np.savez(
            output_file,
            user_indices=arrays["user_indices"],
            item_indices=arrays["item_indices"],
            ratings=arrays["ratings"],
            timestamps=arrays["timestamps"],
        )
        output_file.flush()
        os.fsync(output_file.fileno())


def validate_written_npz(
    path: Path,
    frozen_users: np.ndarray,
    frozen_items: np.ndarray,
) -> dict:
    """Reload the temporary artifact and verify its full public contract."""

    expected_keys = (
        "user_indices",
        "item_indices",
        "ratings",
        "timestamps",
    )
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != expected_keys:
            raise ValueError(
                f"Temporary enriched artifact keys are {archive.files}; expected "
                f"{list(expected_keys)}."
            )

        user_indices = archive["user_indices"]
        item_indices = archive["item_indices"]
        ratings = archive["ratings"]
        timestamps = archive["timestamps"]
        expected_shape = frozen_users.shape

        for name, array, dtype in (
            ("user_indices", user_indices, np.dtype("int32")),
            ("item_indices", item_indices, np.dtype("int32")),
            ("ratings", ratings, np.dtype("float32")),
            ("timestamps", timestamps, np.dtype("int64")),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} shape is {array.shape}; expected {expected_shape}."
                )
            if array.dtype != dtype:
                raise TypeError(
                    f"{name} dtype is {array.dtype}; expected {dtype}."
                )

        if not np.array_equal(user_indices, frozen_users):
            raise ValueError(
                "Published user_indices do not exactly match frozen row order."
            )
        if not np.array_equal(item_indices, frozen_items):
            raise ValueError(
                "Published item_indices do not exactly match frozen row order."
            )
        if not np.isfinite(ratings).all() or np.any((ratings < 1) | (ratings > 5)):
            raise ValueError("Published ratings contain invalid values.")
        if np.any(timestamps < 0):
            raise ValueError("Published timestamps contain invalid values.")

        return {
            "keys": list(archive.files),
            "shape": list(expected_shape),
            "dtypes": {
                "user_indices": str(user_indices.dtype),
                "item_indices": str(item_indices.dtype),
                "ratings": str(ratings.dtype),
                "timestamps": str(timestamps.dtype),
            },
            "order_exact": True,
            "ratings_valid": True,
            "timestamps_valid": True,
        }


# ============================================================
# 5. Validation report
# ============================================================

def _utc_iso_from_milliseconds(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()


def build_report(
    interactions_path: Path,
    user_mapping_path: Path,
    item_mapping_path: Path,
    parent_asins_path: Path,
    reviews_path: Path,
    output_path: Path,
    statistics: dict,
    validation: dict,
    unique_user_count: int,
    unique_item_count: int,
) -> str:
    """Create the human-readable V2-05.2 validation record."""

    interaction_count = statistics["interaction_count"]
    rating_rows = []
    for rating, count in statistics["rating_counts"].items():
        percentage = count / interaction_count * 100.0
        rating_rows.append(
            f"| {rating:g} | {count:,} | {percentage:.4f}% |"
        )

    return "\n".join(
        [
            "# Enriched Interaction Validation Report",
            "",
            "## Validation Purpose",
            "",
            "Validate that rating and timestamp values recovered from the original "
            "Amazon Reviews'23 Electronics stream are aligned exactly with the "
            "frozen 10-core canonical `(user_index, item_index)` row order.",
            "",
            "## Inputs and Output",
            "",
            f"- Frozen interactions: `{interactions_path.as_posix()}`",
            f"- Canonical user mapping: `{user_mapping_path.as_posix()}`",
            f"- Canonical item mapping: `{item_mapping_path.as_posix()}`",
            f"- 10-core product identities: `{parent_asins_path.as_posix()}`",
            f"- Original reviews: `{reviews_path.as_posix()}`",
            f"- Enriched artifact: `{output_path.as_posix()}`",
            "",
            "## Dataset Overview",
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            f"| Source reviews scanned | {statistics['source_review_count']:,} |",
            f"| Enriched interactions | {interaction_count:,} |",
            f"| Unique canonical users | {unique_user_count:,} |",
            f"| Unique canonical items | {unique_item_count:,} |",
            "",
            "## Identity and Order Validation",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| User/item identity matches | {statistics['identity_match_count']:,} / {interaction_count:,} (100.0000%) |",
            f"| User/item identity mismatches | {statistics['identity_mismatch_count']:,} |",
            "| Row order vs. `interactions_10core.npz` | PASS (exact full-array equality) |",
            "| `parent_asins_10core.json` vs. `item_mapping.json` | PASS |",
            "",
            "## Artifact Schema",
            "",
            "| Array | Shape | dtype |",
            "| --- | --- | --- |",
            f"| `user_indices` | `({interaction_count},)` | `{validation['dtypes']['user_indices']}` |",
            f"| `item_indices` | `({interaction_count},)` | `{validation['dtypes']['item_indices']}` |",
            f"| `ratings` | `({interaction_count},)` | `{validation['dtypes']['ratings']}` |",
            f"| `timestamps` | `({interaction_count},)` | `{validation['dtypes']['timestamps']}` |",
            "",
            "## Rating Validation",
            "",
            f"- Valid finite range: **PASS** (`{statistics['rating_min']:g}` to `{statistics['rating_max']:g}`)",
            f"- Mean rating: `{statistics['rating_mean']:.6f}`",
            "",
            "| Rating | Count | Share |",
            "| ---: | ---: | ---: |",
            *rating_rows,
            "",
            "## Timestamp Validation",
            "",
            "- Integer Unix milliseconds, non-negative and UTC-convertible: **PASS**",
            f"- Minimum: `{statistics['timestamp_min']}` ({_utc_iso_from_milliseconds(statistics['timestamp_min'])})",
            f"- Maximum: `{statistics['timestamp_max']}` ({_utc_iso_from_milliseconds(statistics['timestamp_max'])})",
            "",
            "## Final Conclusion",
            "",
            "**PASS** — `enriched_interactions.npz` contains exactly 6,173,972 "
            "validated interactions. Canonical user/item identities and complete "
            "row order match the frozen 10-core artifact exactly; every recovered "
            "rating and timestamp satisfies the V2-05.2 data contract. The dataset "
            "is ready for deterministic positive-feedback derivation and temporal "
            "splitting in the next Recommendation experiment stage.",
            "",
        ]
    )


def write_text_temporary(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


# ============================================================
# 6. End-to-end build orchestration and CLI
# ============================================================

def build_enriched_interactions(
    interactions_path: Path,
    user_mapping_path: Path,
    item_mapping_path: Path,
    parent_asins_path: Path,
    reviews_path: Path,
    output_path: Path,
    report_path: Path,
    expected_count: int,
    progress_every: int,
) -> dict:
    """Build, validate, report, and atomically publish V2-05.2 outputs."""

    if output_path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"Report already exists; refusing to overwrite: {report_path}")

    frozen_users, frozen_items = load_frozen_interactions(
        interactions_path, expected_count
    )
    unique_users = np.unique(frozen_users)
    unique_items = np.unique(frozen_items)

    print(
        f"Frozen interactions: {expected_count:,} | "
        f"users: {len(unique_users):,} | items: {len(unique_items):,}"
    )
    print("Loading target canonical user identities (streaming mapping)...")
    raw_to_user = load_target_raw_identity_map(
        user_mapping_path,
        set(map(int, unique_users)),
        "user mapping",
        "user_id",
    )
    print(f"Resolved target users: {len(raw_to_user):,}")

    print("Loading and validating target canonical item identities...")
    raw_to_item = load_and_validate_item_identities(
        item_mapping_path,
        parent_asins_path,
        unique_items,
    )
    print(f"Resolved target items: {len(raw_to_item):,}")

    print("Scanning original reviews and reconstructing enriched rows...")
    arrays, statistics = reconstruct_enriched_rows(
        reviews_path,
        frozen_users,
        frozen_items,
        raw_to_user,
        raw_to_item,
        progress_every,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_path(output_path)
    temporary_report = _temporary_path(report_path)
    for temporary_path in (temporary_output, temporary_report):
        if temporary_path.exists():
            raise FileExistsError(
                "Temporary output already exists; inspect or remove it before "
                f"retrying: {temporary_path}"
            )

    try:
        print("Writing and reloading temporary enriched artifact...")
        write_npz_temporary(temporary_output, arrays)
        validation = validate_written_npz(
            temporary_output, frozen_users, frozen_items
        )
        report = build_report(
            interactions_path,
            user_mapping_path,
            item_mapping_path,
            parent_asins_path,
            reviews_path,
            output_path,
            statistics,
            validation,
            len(unique_users),
            len(unique_items),
        )
        write_text_temporary(temporary_report, report)

        # Publication occurs only after the full source and temporary artifact pass.
        os.replace(temporary_output, output_path)
        os.replace(temporary_report, report_path)
    except Exception:
        for temporary_path in (temporary_output, temporary_report):
            if temporary_path.exists():
                temporary_path.unlink()
        raise

    return {
        **statistics,
        "unique_user_count": len(unique_users),
        "unique_item_count": len(unique_items),
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the V2-05.2 enriched interaction artifact from frozen "
            "canonical identities and original Amazon reviews."
        )
    )
    parser.add_argument(
        "--interactions",
        required=True,
        type=Path,
        help="Path to frozen interactions_10core.npz.",
    )
    parser.add_argument(
        "--user-mapping",
        required=True,
        type=Path,
        help="Path to canonical user_mapping.json.",
    )
    parser.add_argument(
        "--item-mapping",
        required=True,
        type=Path,
        help="Path to canonical item_mapping.json.",
    )
    parser.add_argument(
        "--parent-asins",
        required=True,
        type=Path,
        help="Path to ordered parent_asins_10core.json.",
    )
    parser.add_argument(
        "--reviews",
        required=True,
        type=Path,
        help="Path to Amazon Electronics reviews JSONL or JSONL.GZ.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination enriched_interactions.npz path.",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Destination validation Markdown report path.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_INTERACTION_COUNT,
        help=(
            "Required frozen interaction count "
            f"(default: {EXPECTED_INTERACTION_COUNT})."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress after this many matched rows; 0 disables progress.",
    )
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected-count must be a positive integer.")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = build_enriched_interactions(
            interactions_path=args.interactions,
            user_mapping_path=args.user_mapping,
            item_mapping_path=args.item_mapping,
            parent_asins_path=args.parent_asins,
            reviews_path=args.reviews,
            output_path=args.output,
            report_path=args.report,
            expected_count=args.expected_count,
            progress_every=args.progress_every,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n========== Enriched Interaction Validation ==========")
    print(f"Interactions: {result['interaction_count']:,}")
    print(f"Identity matches: {result['identity_match_count']:,}")
    print(f"Identity mismatches: {result['identity_mismatch_count']:,}")
    print(f"Unique users: {result['unique_user_count']:,}")
    print(f"Unique items: {result['unique_item_count']:,}")
    print(
        "Rating min/mean/max: "
        f"{result['rating_min']:g} / {result['rating_mean']:.6f} / "
        f"{result['rating_max']:g}"
    )
    print(
        "Timestamp range: "
        f"{result['timestamp_min']} .. {result['timestamp_max']}"
    )
    print("Frozen row order: PASS")
    print(f"Artifact: {args.output}")
    print(f"Report: {args.report}")
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
