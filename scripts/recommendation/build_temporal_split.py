"""
Build the AgentRec V2 implicit-feedback temporal recommendation split.

Input:
    --input
        enriched_interactions.npz containing aligned user_indices,
        item_indices, ratings, and Unix-millisecond timestamps.

Outputs under --output-dir:
    train.npz
    valid.npz
    test.npz

Each output contains int32 user_indices and item_indices. The construction is
fully deterministic and never uses random splitting:

1. Keep interactions with rating >= 4 as positive implicit feedback.
2. Deduplicate each (user_index, item_index), retaining its latest timestamp.
3. Sort each user's unique positives by (timestamp, item_index).
4. For users with at least three positives, reserve the latest interaction for
   test and the second latest for validation; keep the remainder in train.
5. Keep all positives from users with fewer than three in train so evaluation
   never introduces users that lack training history.

Example:
    python scripts/recommendation/build_temporal_split.py \
        --input data/processed/recommendation/experiments/enriched_interactions.npz \
        --output-dir data/processed/recommendation/experiments

The source artifact is read-only. All three outputs are written to temporary
files, reloaded and validated, then atomically published without overwriting
existing formal split artifacts.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np


# ============================================================
# 1. Frozen split contract and input validation
# ============================================================

EXPECTED_INTERACTION_COUNT = 6_173_972
POSITIVE_RATING_THRESHOLD = 4.0
OUTPUT_NAMES = ("train.npz", "valid.npz", "test.npz")
OUTPUT_KEYS = ("user_indices", "item_indices")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def load_enriched_interactions(
    path: Path,
    expected_count: int,
) -> dict[str, np.ndarray]:
    """Load and strictly validate the V2-05.2 enriched interaction contract."""

    _require_file(path, "enriched interactions")
    expected_keys = (
        "user_indices",
        "item_indices",
        "ratings",
        "timestamps",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != expected_keys:
                raise ValueError(
                    f"Enriched artifact arrays are {archive.files}; expected "
                    f"{list(expected_keys)}."
                )
            arrays = {name: np.asarray(archive[name]) for name in expected_keys}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load enriched artifact {path}: {exc}") from exc

    expected_shape = (expected_count,)
    for name, expected_dtype in (
        ("user_indices", np.dtype("int32")),
        ("item_indices", np.dtype("int32")),
        ("ratings", np.dtype("float32")),
        ("timestamps", np.dtype("int64")),
    ):
        array = arrays[name]
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape is {array.shape}; expected {expected_shape}."
            )
        if array.dtype != expected_dtype:
            raise TypeError(
                f"{name} dtype is {array.dtype}; expected {expected_dtype}."
            )

    if np.any(arrays["user_indices"] < 0):
        raise ValueError("user_indices contains negative canonical indices.")
    if np.any(arrays["item_indices"] < 0):
        raise ValueError("item_indices contains negative canonical indices.")
    ratings = arrays["ratings"]
    if not np.isfinite(ratings).all() or np.any((ratings < 1.0) | (ratings > 5.0)):
        raise ValueError("ratings must be finite values in the inclusive range [1, 5].")
    if np.any(arrays["timestamps"] < 0):
        raise ValueError("timestamps contains negative Unix-millisecond values.")

    return arrays


# ============================================================
# 2. Positive implicit-feedback derivation and deduplication
# ============================================================

def derive_unique_positive_interactions(arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Keep rating >= 4 events and retain each user-item pair's latest event."""

    positive_mask = arrays["ratings"] >= POSITIVE_RATING_THRESHOLD
    positive_source_rows = np.flatnonzero(positive_mask)
    positive_users = arrays["user_indices"][positive_mask]
    positive_items = arrays["item_indices"][positive_mask]
    positive_timestamps = arrays["timestamps"][positive_mask]
    positive_event_count = int(positive_users.size)
    if positive_event_count == 0:
        raise ValueError("No positive interactions satisfy rating >= 4.")

    # The final key makes duplicate source events deterministic when their
    # user, item, and timestamp are identical. The last event in each group is
    # the authoritative latest observation for that user-item pair.
    pair_order = np.lexsort(
        (
            positive_source_rows,
            positive_timestamps,
            positive_items,
            positive_users,
        )
    )
    ordered_users = positive_users[pair_order]
    ordered_items = positive_items[pair_order]

    keep_latest = np.ones(positive_event_count, dtype=bool)
    if positive_event_count > 1:
        keep_latest[:-1] = (
            (ordered_users[:-1] != ordered_users[1:])
            | (ordered_items[:-1] != ordered_items[1:])
        )
    latest_source_rows = pair_order[keep_latest]

    unique_positive = {
        "user_indices": np.ascontiguousarray(
            positive_users[latest_source_rows], dtype=np.int32
        ),
        "item_indices": np.ascontiguousarray(
            positive_items[latest_source_rows], dtype=np.int32
        ),
        "timestamps": np.ascontiguousarray(
            positive_timestamps[latest_source_rows], dtype=np.int64
        ),
    }
    unique_count = int(unique_positive["user_indices"].size)
    statistics = {
        "source_interaction_count": int(arrays["ratings"].size),
        "positive_event_count": positive_event_count,
        "non_positive_event_count": int(arrays["ratings"].size - positive_event_count),
        "unique_positive_count": unique_count,
        "duplicate_positive_events_removed": positive_event_count - unique_count,
    }
    return unique_positive, statistics


# ============================================================
# 3. Deterministic per-user temporal leave-one-out split
# ============================================================

def build_temporal_split(unique_positive: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
    """Create train/validation/test masks after deterministic temporal sorting."""

    users = unique_positive["user_indices"]
    items = unique_positive["item_indices"]
    timestamps = unique_positive["timestamps"]

    # item_index is the deterministic tie-break when two unique items share a
    # timestamp. No random state participates in the split.
    temporal_order = np.lexsort((items, timestamps, users))
    users = np.ascontiguousarray(users[temporal_order], dtype=np.int32)
    items = np.ascontiguousarray(items[temporal_order], dtype=np.int32)
    timestamps = np.ascontiguousarray(timestamps[temporal_order], dtype=np.int64)

    user_starts = np.flatnonzero(
        np.concatenate(([True], users[1:] != users[:-1]))
    )
    user_ends = np.concatenate((user_starts[1:], [users.size]))
    user_counts = user_ends - user_starts
    evaluation_user_mask = user_counts >= 3
    evaluation_user_count = int(evaluation_user_mask.sum())

    validation_positions = user_ends[evaluation_user_mask] - 2
    test_positions = user_ends[evaluation_user_mask] - 1
    last_train_positions = user_ends[evaluation_user_mask] - 3

    train_mask = np.ones(users.size, dtype=bool)
    train_mask[validation_positions] = False
    train_mask[test_positions] = False
    valid_mask = np.zeros(users.size, dtype=bool)
    valid_mask[validation_positions] = True
    test_mask = np.zeros(users.size, dtype=bool)
    test_mask[test_positions] = True

    if not np.all(
        timestamps[last_train_positions] <= timestamps[validation_positions]
    ):
        raise ValueError("A validation interaction occurs before its user's train history.")
    if not np.all(timestamps[validation_positions] <= timestamps[test_positions]):
        raise ValueError("A test interaction occurs before its user's validation interaction.")

    def select(mask: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "user_indices": np.ascontiguousarray(users[mask], dtype=np.int32),
            "item_indices": np.ascontiguousarray(items[mask], dtype=np.int32),
        }

    splits = {
        "train": select(train_mask),
        "valid": select(valid_mask),
        "test": select(test_mask),
    }
    metadata = {
        "temporally_sorted_users": users,
        "temporally_sorted_items": items,
        "temporally_sorted_timestamps": timestamps,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "test_mask": test_mask,
    }
    statistics = {
        "positive_user_count": int(user_starts.size),
        "evaluation_user_count": evaluation_user_count,
        "train_only_user_count": int((user_counts < 3).sum()),
        "min_positive_count_per_user": int(user_counts.min()),
        "max_positive_count_per_user": int(user_counts.max()),
    }
    return splits, metadata, statistics


# ============================================================
# 4. Leakage, identity-range, and temporal-order validation
# ============================================================

def _pair_keys(split: dict[str, np.ndarray]) -> np.ndarray:
    """Encode non-negative int32 user-item pairs as exact uint64 keys."""

    users = split["user_indices"].astype(np.uint64, copy=False)
    items = split["item_indices"].astype(np.uint64, copy=False)
    return (users << np.uint64(32)) | items


def validate_split(
    splits: dict[str, dict[str, np.ndarray]],
    temporal_metadata: dict,
    source_users: np.ndarray,
    source_items: np.ndarray,
    expected_unique_positive_count: int,
) -> dict:
    """Validate no interaction/future leakage and canonical identity coverage."""

    counts = {name: int(split["user_indices"].size) for name, split in splits.items()}
    if sum(counts.values()) != expected_unique_positive_count:
        raise ValueError(
            "Split interaction counts do not reconstruct all unique positives: "
            f"{sum(counts.values()):,} != {expected_unique_positive_count:,}."
        )

    source_user_set = np.unique(source_users)
    source_item_set = np.unique(source_items)
    pair_keys = {}
    for name, split in splits.items():
        users = split["user_indices"]
        items = split["item_indices"]
        if users.ndim != 1 or items.ndim != 1 or users.shape != items.shape:
            raise ValueError(f"{name} user/item arrays must be aligned one-dimensional arrays.")
        if users.dtype != np.dtype("int32") or items.dtype != np.dtype("int32"):
            raise TypeError(f"{name} user/item arrays must use int32.")
        if np.any(users < 0) or np.any(items < 0):
            raise ValueError(f"{name} contains negative canonical indices.")
        if np.setdiff1d(np.unique(users), source_user_set, assume_unique=True).size:
            raise ValueError(f"{name} contains user indices absent from the enriched source.")
        if np.setdiff1d(np.unique(items), source_item_set, assume_unique=True).size:
            raise ValueError(f"{name} contains item indices absent from the enriched source.")

        keys = _pair_keys(split)
        unique_keys = np.unique(keys)
        if unique_keys.size != keys.size:
            raise ValueError(f"{name} contains duplicate user-item interactions.")
        pair_keys[name] = unique_keys

    for left_name, right_name in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = np.intersect1d(
            pair_keys[left_name], pair_keys[right_name], assume_unique=True
        )
        if overlap.size:
            raise ValueError(
                f"{left_name}/{right_name} contain {overlap.size:,} overlapping "
                "user-item interactions."
            )

    train_users = np.unique(splits["train"]["user_indices"])
    valid_users = np.unique(splits["valid"]["user_indices"])
    test_users = np.unique(splits["test"]["user_indices"])
    if not np.array_equal(valid_users, test_users):
        raise ValueError("Validation and test evaluation-user sets differ.")
    if np.setdiff1d(valid_users, train_users, assume_unique=True).size:
        raise ValueError("Validation contains users with no training history.")
    if np.setdiff1d(test_users, train_users, assume_unique=True).size:
        raise ValueError("Test contains users with no training history.")

    ordered_users = temporal_metadata["temporally_sorted_users"]
    ordered_timestamps = temporal_metadata["temporally_sorted_timestamps"]
    train_mask = temporal_metadata["train_mask"]
    valid_mask = temporal_metadata["valid_mask"]
    test_mask = temporal_metadata["test_mask"]
    evaluation_users = ordered_users[valid_mask]

    # For every evaluation user, all training timestamps must be no later than
    # validation, and validation must be no later than test.
    change_positions = np.flatnonzero(
        np.concatenate(([True], ordered_users[1:] != ordered_users[:-1]))
    )
    end_positions = np.concatenate((change_positions[1:], [ordered_users.size]))
    counts_per_user = end_positions - change_positions
    eval_ends = end_positions[counts_per_user >= 3]
    if not np.array_equal(evaluation_users, ordered_users[eval_ends - 2]):
        raise ValueError("Validation rows are not each evaluation user's penultimate event.")
    if not np.array_equal(evaluation_users, ordered_users[eval_ends - 1]):
        raise ValueError("Test rows are not each evaluation user's final event.")
    if not np.all(ordered_timestamps[eval_ends - 3] <= ordered_timestamps[eval_ends - 2]):
        raise ValueError("Temporal train-to-validation order validation failed.")
    if not np.all(ordered_timestamps[eval_ends - 2] <= ordered_timestamps[eval_ends - 1]):
        raise ValueError("Temporal validation-to-test order validation failed.")
    if not np.all(train_mask | valid_mask | test_mask):
        raise ValueError("At least one unique positive interaction was not assigned.")
    if np.any(train_mask & valid_mask) or np.any(train_mask & test_mask) or np.any(valid_mask & test_mask):
        raise ValueError("Internal temporal split masks overlap.")

    return {
        "counts": counts,
        "user_counts": {
            name: int(np.unique(split["user_indices"]).size)
            for name, split in splits.items()
        },
        "item_counts": {
            name: int(np.unique(split["item_indices"]).size)
            for name, split in splits.items()
        },
        "interaction_overlap_count": 0,
        "test_in_train_count": 0,
        "evaluation_users_have_train_history": True,
        "canonical_index_ranges_valid": True,
        "temporal_order_valid": True,
    }


# ============================================================
# 5. Safe artifact writing and post-write verification
# ============================================================

def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def write_split_temporary(path: Path, split: dict[str, np.ndarray]) -> None:
    with path.open("xb") as output_file:
        np.savez(
            output_file,
            user_indices=split["user_indices"],
            item_indices=split["item_indices"],
        )
        output_file.flush()
        os.fsync(output_file.fileno())


def validate_temporary_split(path: Path, expected: dict[str, np.ndarray], label: str) -> None:
    """Reload one temporary NPZ and require exact array equality."""

    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != OUTPUT_KEYS:
            raise ValueError(
                f"Temporary {label} arrays are {archive.files}; expected "
                f"{list(OUTPUT_KEYS)}."
            )
        for name in OUTPUT_KEYS:
            actual = archive[name]
            if actual.dtype != np.dtype("int32"):
                raise TypeError(f"Temporary {label} {name} must use int32.")
            if not np.array_equal(actual, expected[name]):
                raise ValueError(
                    f"Temporary {label} {name} differs from the validated split."
                )


def publish_splits(output_dir: Path, splits: dict[str, dict[str, np.ndarray]]) -> dict[str, Path]:
    """Write, reload, and atomically publish all formal split artifacts."""

    output_paths = {
        "train": output_dir / "train.npz",
        "valid": output_dir / "valid.npz",
        "test": output_dir / "test.npz",
    }
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: _temporary_path(path) for name, path in output_paths.items()
    }
    for path in temporary_paths.values():
        if path.exists():
            raise FileExistsError(
                f"Temporary output already exists; inspect or remove it first: {path}"
            )

    try:
        for name in ("train", "valid", "test"):
            write_split_temporary(temporary_paths[name], splits[name])
            validate_temporary_split(temporary_paths[name], splits[name], name)

        # All three temporary artifacts must pass before any formal path appears.
        for name in ("train", "valid", "test"):
            os.replace(temporary_paths[name], output_paths[name])
    except Exception:
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        raise

    return output_paths


# ============================================================
# 6. End-to-end orchestration and command-line interface
# ============================================================

def run_build(input_path: Path, output_dir: Path, expected_count: int) -> dict:
    arrays = load_enriched_interactions(input_path, expected_count)
    print(f"Loaded enriched interactions: {expected_count:,}")

    unique_positive, positive_statistics = derive_unique_positive_interactions(arrays)
    print(
        f"Positive events (rating >= {POSITIVE_RATING_THRESHOLD:g}): "
        f"{positive_statistics['positive_event_count']:,}"
    )
    print(
        "Unique positive user-item interactions: "
        f"{positive_statistics['unique_positive_count']:,}"
    )

    splits, temporal_metadata, split_statistics = build_temporal_split(unique_positive)
    validation = validate_split(
        splits,
        temporal_metadata,
        arrays["user_indices"],
        arrays["item_indices"],
        positive_statistics["unique_positive_count"],
    )
    output_paths = publish_splits(output_dir, splits)

    return {
        **positive_statistics,
        **split_statistics,
        "validation": validation,
        "output_paths": output_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic implicit-feedback train/validation/test "
            "splits from enriched AgentRec interactions."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to enriched_interactions.npz.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for train.npz, valid.npz, and test.npz.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_INTERACTION_COUNT,
        help=(
            "Required enriched interaction count "
            f"(default: {EXPECTED_INTERACTION_COUNT})."
        ),
    )
    args = parser.parse_args()
    if isinstance(args.expected_count, bool) or args.expected_count <= 0:
        parser.error("--expected-count must be a positive integer.")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run_build(args.input, args.output_dir, args.expected_count)
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    validation = result["validation"]
    print("\n========== Recommendation Temporal Split ==========")
    print(f"Source interactions: {result['source_interaction_count']:,}")
    print(f"Positive events: {result['positive_event_count']:,}")
    print(
        "Duplicate positive events removed: "
        f"{result['duplicate_positive_events_removed']:,}"
    )
    print(f"Unique positive interactions: {result['unique_positive_count']:,}")
    print(f"Positive users: {result['positive_user_count']:,}")
    print(f"Evaluation users: {result['evaluation_user_count']:,}")
    print(f"Train-only users (<3 positives): {result['train_only_user_count']:,}")

    for display_name, split_name in (("Train", "train"), ("Valid", "valid"), ("Test", "test")):
        print(f"\n{display_name}:")
        print(f"  Users: {validation['user_counts'][split_name]:,}")
        print(f"  Items: {validation['item_counts'][split_name]:,}")
        print(f"  Interactions: {validation['counts'][split_name]:,}")

    print("\nValidation:")
    print("  Cross-split interaction overlap: 0")
    print("  Test interactions found in train: 0")
    print("  Evaluation users have train history: PASS")
    print("  Canonical user/item index ranges: PASS")
    print("  Per-user temporal order: PASS")
    print("  Random split used: NO")
    for name in ("train", "valid", "test"):
        print(f"  {name}: {result['output_paths'][name]}")
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
