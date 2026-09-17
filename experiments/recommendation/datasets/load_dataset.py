"""
Load the frozen AgentRec V2 recommendation experiment splits.

Input directory:
    train.npz
    valid.npz
    test.npz

Each archive must contain aligned one-dimensional int32 arrays named
``user_indices`` and ``item_indices``. ``load_dataset()`` exposes them through
a typed ``RecommendationDataset`` with the stable field names required by all
future Popularity, LightGCN, Semantic, and Hybrid experiment runners:

    train_user, train_item,
    valid_user, valid_item,
    test_user, test_item

The loader is read-only and preserves canonical sparse user/item indices. A
model that requires compact embedding-table indices must create an explicit
model-local mapping rather than changing these frozen arrays.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# 1. Public dataset contract
# ============================================================

SPLIT_FILES = {
    "train": "train.npz",
    "valid": "valid.npz",
    "test": "test.npz",
}
EXPECTED_KEYS = ("user_indices", "item_indices")
EXPECTED_DTYPE = np.dtype("int32")


@dataclass(frozen=True)
class RecommendationDataset:
    """Aligned canonical interaction arrays for all three temporal splits."""

    train_user: np.ndarray
    train_item: np.ndarray
    valid_user: np.ndarray
    valid_item: np.ndarray
    test_user: np.ndarray
    test_item: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the six-array interface with explicit stable field names."""

        return {
            "train_user": self.train_user,
            "train_item": self.train_item,
            "valid_user": self.valid_user,
            "valid_item": self.valid_item,
            "test_user": self.test_user,
            "test_item": self.test_item,
        }


# ============================================================
# 2. Strict NPZ loading
# ============================================================

def _load_split(path: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split and enforce its shape, dtype, and index invariants."""

    if not path.is_file():
        raise FileNotFoundError(f"{split_name} split file not found: {path}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != EXPECTED_KEYS:
                raise ValueError(
                    f"{split_name} arrays are {archive.files}; expected "
                    f"{list(EXPECTED_KEYS)}."
                )
            user_indices = np.asarray(archive["user_indices"])
            item_indices = np.asarray(archive["item_indices"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load {split_name} split {path}: {exc}") from exc

    if user_indices.ndim != 1 or item_indices.ndim != 1:
        raise ValueError(
            f"{split_name} user/item arrays must be one-dimensional."
        )
    if user_indices.shape != item_indices.shape:
        raise ValueError(
            f"{split_name} user/item shapes differ: "
            f"{user_indices.shape} != {item_indices.shape}."
        )
    if user_indices.dtype != EXPECTED_DTYPE or item_indices.dtype != EXPECTED_DTYPE:
        raise TypeError(
            f"{split_name} user/item arrays must both use int32; got "
            f"{user_indices.dtype} and {item_indices.dtype}."
        )
    if np.any(user_indices < 0) or np.any(item_indices < 0):
        raise ValueError(f"{split_name} contains negative canonical indices.")

    # np.load materializes these arrays before the archive is closed. Mark them
    # read-only to protect the in-memory representation from accidental edits.
    user_indices.setflags(write=False)
    item_indices.setflags(write=False)
    return user_indices, item_indices


def load_dataset(dataset_path: str | Path) -> RecommendationDataset:
    """Load train/valid/test into the unified six-array experiment interface.

    Args:
        dataset_path: Directory containing train.npz, valid.npz, and test.npz.

    Returns:
        An immutable dataclass containing six read-only NumPy arrays.

    Raises:
        FileNotFoundError: When the dataset directory or a split is missing.
        TypeError: When a split does not use the frozen int32 dtype.
        ValueError: When archive keys, shapes, or canonical indices are invalid.
    """

    path = Path(dataset_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    loaded = {
        name: _load_split(path / filename, name)
        for name, filename in SPLIT_FILES.items()
    }
    return RecommendationDataset(
        train_user=loaded["train"][0],
        train_item=loaded["train"][1],
        valid_user=loaded["valid"][0],
        valid_item=loaded["valid"][1],
        test_user=loaded["test"][0],
        test_item=loaded["test"][1],
    )


# ============================================================
# 3. Minimal inspection CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect frozen AgentRec recommendation splits."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Directory containing train.npz, valid.npz, and test.npz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = load_dataset(args.dataset_path)
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Recommendation dataset loaded successfully.")
    for split_name in ("train", "valid", "test"):
        users = getattr(dataset, f"{split_name}_user")
        items = getattr(dataset, f"{split_name}_item")
        print(
            f"{split_name}: interactions={users.size:,}, "
            f"users={np.unique(users).size:,}, items={np.unique(items).size:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
