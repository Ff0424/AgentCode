"""Export the V2-06 models into a validated V2-07.2 serving bundle.

Inputs:
    - frozen train/valid/test splits and canonical user/item mappings;
    - the selected LightGCN checkpoint;
    - the existing Semantic item embedding and metadata artifacts;
    - the frozen Product Catalog.

Outputs under ``artifacts/recommendation/serving``:
    propagated LightGCN embeddings, compact model identities, train-seen CSR,
    train-observed mask, deterministic popularity order, and a checksummed
    serving manifest. The semantic embedding is referenced, not copied.

This is a one-time GPU export. It does not train a model, regenerate BGE-M3
embeddings, mutate frozen data, or implement online recommendation scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.recommendation.datasets.canonical_items import (
    load_canonical_item_universe,
    map_canonical_items,
)
from experiments.recommendation.datasets.load_dataset import load_dataset
from experiments.recommendation.datasets.semantic_artifacts import (
    sha256_file,
    validate_embedding_artifacts,
)
from experiments.recommendation.datasets.semantic_config import load_semantic_config
from experiments.recommendation.models.popularity import PopularityRecommender
from experiments.recommendation.runners.build_semantic_item_embeddings import (
    _select_cuda_device,
)
from scripts.recommendation.build_enriched_interactions import (
    load_target_raw_identity_map,
)
from src.agentrec.recommendation.artifacts import (
    SCHEMA_VERSION,
    load_serving_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/recommendation/serving"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/recommendation/lightgcn_best.pt"
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/recommendation/experiments"
DEFAULT_USER_MAPPING = PROJECT_ROOT / "data/processed/recommendation/user_mapping.json"
DEFAULT_ITEM_MAPPING = PROJECT_ROOT / "data/processed/recommendation/item_mapping.json"
DEFAULT_PARENT_ASINS = (
    PROJECT_ROOT / "data/processed/recommendation/parent_asins_10core.json"
)
DEFAULT_CATALOG = PROJECT_ROOT / "data/processed/recommendation/product_catalog.jsonl"
DEFAULT_SEMANTIC_CONFIG = PROJECT_ROOT / "experiments/recommendation/configs/semantic.yaml"
EXPECTED_MODEL_USERS = 357_974
EXPECTED_CANONICAL_ITEMS = 125_762
EXPECTED_TRAIN_OBSERVED_ITEMS = 125_684


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the V2-07.2 serving bundle.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--user-mapping", type=Path, default=DEFAULT_USER_MAPPING)
    parser.add_argument("--item-mapping", type=Path, default=DEFAULT_ITEM_MAPPING)
    parser.add_argument("--parent-asins", type=Path, default=DEFAULT_PARENT_ASINS)
    parser.add_argument("--product-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--semantic-config", type=Path, default=DEFAULT_SEMANTIC_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _file_spec(path: Path) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }


def _compact_users(dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    users = np.unique(
        np.concatenate((dataset.train_user, dataset.valid_user, dataset.test_user))
    ).astype(np.int32, copy=False)
    if users.size != EXPECTED_MODEL_USERS:
        raise ValueError(
            f"Model user count={users.size:,}; expected {EXPECTED_MODEL_USERS:,}."
        )
    mapped: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        canonical = getattr(dataset, f"{split}_user")
        rows = np.searchsorted(users, canonical)
        if np.any(rows >= users.size) or not np.array_equal(users[rows], canonical):
            raise ValueError(f"Unable to map every {split} user to a model row.")
        mapped[split] = np.ascontiguousarray(rows, dtype=np.int32)
    return np.ascontiguousarray(users), mapped


def build_train_seen_csr(
    train_user_rows: np.ndarray,
    train_item_rows: np.ndarray,
    num_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return user-row CSR with sorted, duplicate-free serving item rows."""

    order = np.lexsort((train_item_rows, train_user_rows))
    users = train_user_rows[order]
    items = train_item_rows[order]
    if np.any((users[1:] == users[:-1]) & (items[1:] == items[:-1])):
        raise ValueError("Frozen train split contains duplicate user-item pairs.")
    counts = np.bincount(users, minlength=num_users)
    if np.any(counts == 0):
        raise ValueError("Every model-known user must have train history.")
    offsets = np.empty(num_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, np.ascontiguousarray(items, dtype=np.int32)


def _validate_csr_round_trip(
    offsets: np.ndarray,
    items: np.ndarray,
    train_users: np.ndarray,
    train_items: np.ndarray,
) -> None:
    counts = np.diff(offsets)
    rebuilt_users = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
    expected_order = np.lexsort((train_items, train_users))
    if not np.array_equal(rebuilt_users, train_users[expected_order]) or not np.array_equal(
        items, train_items[expected_order]
    ):
        raise ValueError("Train-seen CSR does not round-trip to frozen train pairs.")


def _model_user_ids(user_mapping: Path, model_users: np.ndarray) -> list[str]:
    targets = {int(value) for value in model_users}
    raw_to_index = load_target_raw_identity_map(
        user_mapping,
        targets,
        "canonical user mapping",
        "raw reviewer ID",
    )
    by_index = {canonical: raw_id for raw_id, canonical in raw_to_index.items()}
    if len(by_index) != model_users.size:
        raise ValueError("Unable to resolve every model user to one raw reviewer ID.")
    ordered = [by_index[int(value)] for value in model_users]
    if len(set(ordered)) != len(ordered):
        raise ValueError("Raw reviewer IDs are not unique for model-known users.")
    return ordered


def _validate_catalog(path: Path, universe) -> None:
    by_item: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Catalog line {line_number} is not an object.")
            item_index = record.get("item_index")
            parent_asin = record.get("parent_asin")
            if isinstance(item_index, bool) or not isinstance(item_index, int):
                raise ValueError(f"Invalid item_index at Catalog line {line_number}.")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"Invalid parent_asin at Catalog line {line_number}.")
            if item_index in by_item:
                raise ValueError(f"Duplicate Catalog item_index={item_index}.")
            by_item[item_index] = parent_asin
    for row, item_index in enumerate(universe.item_indices):
        if by_item.get(int(item_index)) != universe.parent_asins[row]:
            raise ValueError(
                f"Catalog identity mismatch for canonical item_index={int(item_index)}."
            )
    if len(by_item) != universe.item_indices.size:
        raise ValueError("Catalog contains identities outside the canonical universe.")


def _manifest(
    output: Path,
    *,
    checkpoint: Path,
    train_split: Path,
    user_mapping: Path,
    item_mapping: Path,
    parent_asins: Path,
    product_catalog: Path,
    semantic_config,
    semantic_metadata: dict,
    universe,
    num_users: int,
    embedding_dim: int,
    layers: int,
    observed_count: int,
) -> dict:
    files = {name: _file_spec(output / f"{name}.npy") for name in (
        "lightgcn_user_embeddings",
        "lightgcn_item_embeddings",
        "model_user_indices",
        "canonical_item_indices",
        "train_seen_offsets",
        "train_seen_items",
        "train_observed_mask",
        "popular_item_indices",
    )}
    user_ids_path = output / "model_user_ids.json"
    files["model_user_ids"] = {
        "path": user_ids_path.name,
        "sha256": sha256_file(user_ids_path),
        "count": num_users,
    }
    source_hashes = {
        "lightgcn_checkpoint": (checkpoint, sha256_file(checkpoint)),
        "train_split": (train_split, sha256_file(train_split)),
        "user_mapping": (user_mapping, sha256_file(user_mapping)),
        "item_mapping": (item_mapping, sha256_file(item_mapping)),
        "parent_asins": (parent_asins, sha256_file(parent_asins)),
        "product_catalog": (product_catalog, sha256_file(product_catalog)),
    }
    fingerprint_material = "\n".join(
        [source_hashes["lightgcn_checkpoint"][1], source_hashes["train_split"][1],
         universe.identity_sha256, sha256_file(semantic_config.item_embedding_path)]
    )
    import hashlib
    artifact_version = "serving-v1-" + hashlib.sha256(
        fingerprint_material.encode("ascii")
    ).hexdigest()[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_version": artifact_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_model_users": num_users,
        "num_canonical_items": int(universe.item_indices.size),
        "num_train_observed_items": observed_count,
        "num_train_unseen_items": int(universe.item_indices.size - observed_count),
        "embedding_dim": embedding_dim,
        "lightgcn_layers": layers,
        "canonical_identity_sha256": universe.identity_sha256,
        "files": files,
        "external_artifacts": {
            "semantic_item_embeddings": {
                "path": _project_relative(semantic_config.item_embedding_path),
                "sha256": sha256_file(semantic_config.item_embedding_path),
            },
            "semantic_metadata": {
                "path": _project_relative(semantic_config.item_embedding_metadata_path),
                "sha256": sha256_file(semantic_config.item_embedding_metadata_path),
            },
        },
        "sources": {
            name: {"path": _project_relative(path), "sha256": digest}
            for name, (path, digest) in source_hashes.items()
        },
        "row_mappings": {
            "lightgcn_user_embeddings": "row i -> model_user_indices[i] -> canonical user_index",
            "model_user_ids": "row i -> raw/API user_id for model_user_indices[i]",
            "lightgcn_item_embeddings": "row i -> canonical_item_indices[i] -> canonical item_index",
            "train_seen_csr": "model user row u -> serving item rows in offsets[u]:offsets[u+1]",
            "semantic_item_embeddings": "row i -> canonical_item_indices[i] -> canonical item_index",
            "popular_item_indices": "ranked canonical item_index values",
        },
    }


def export(args: argparse.Namespace) -> Path:
    """Build all artifacts in a temporary directory, validate, then publish."""

    # Heavy imports are delayed so local contract/static checks do not require PyTorch.
    import torch

    from experiments.recommendation.models.lightgcn import LightGCN
    from experiments.recommendation.runners.run_lightgcn import build_normalized_adjacency

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Serving output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        dataset = load_dataset(args.dataset_path.resolve())
        universe = load_canonical_item_universe(
            args.item_mapping.resolve(), args.parent_asins.resolve()
        )
        if universe.item_indices.size != EXPECTED_CANONICAL_ITEMS:
            raise ValueError("Canonical item universe is not 125,762.")
        _validate_catalog(args.product_catalog.resolve(), universe)
        model_users, user_rows = _compact_users(dataset)
        item_rows = {
            split: map_canonical_items(
                getattr(dataset, f"{split}_item"), universe.item_indices, f"{split} item"
            ).astype(np.int32)
            for split in ("train", "valid", "test")
        }
        offsets, seen_items = build_train_seen_csr(
            user_rows["train"], item_rows["train"], model_users.size
        )
        _validate_csr_round_trip(
            offsets, seen_items, user_rows["train"], item_rows["train"]
        )
        observed = np.zeros(universe.item_indices.size, dtype=bool)
        observed[np.unique(item_rows["train"])] = True
        if int(observed.sum()) != EXPECTED_TRAIN_OBSERVED_ITEMS:
            raise ValueError(
                f"Train-observed items={int(observed.sum()):,}; expected 125,684."
            )
        popularity = PopularityRecommender().fit(dataset.train_user, dataset.train_item)
        if popularity.popular_items.size != EXPECTED_TRAIN_OBSERVED_ITEMS:
            raise ValueError("Popularity artifact must contain every train-observed item.")
        raw_user_ids = _model_user_ids(args.user_mapping.resolve(), model_users)

        semantic_config = load_semantic_config(args.semantic_config.resolve())
        if semantic_config.item_mapping_path != args.item_mapping.resolve():
            raise ValueError("Semantic config item mapping differs from export input.")
        semantic_array, semantic_metadata = validate_embedding_artifacts(
            semantic_config, universe
        )
        if semantic_array.shape != (EXPECTED_CANONICAL_ITEMS, 1024):
            raise ValueError("Semantic artifact shape is not (125762, 1024).")

        device_index = _select_cuda_device(torch, args.device)
        device = torch.device("cuda", device_index)
        checkpoint = torch.load(
            args.checkpoint.resolve(), map_location=device, weights_only=True
        )
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("LightGCN checkpoint lacks model_config.")
        embedding_dim = int(model_config["embedding_dim"])
        layers = int(model_config["num_layers"])
        adjacency = build_normalized_adjacency(
            user_rows["train"], item_rows["train"], model_users.size,
            universe.item_indices.size, device,
        )
        model = LightGCN(
            model_users.size, universe.item_indices.size,
            embedding_dim, layers, adjacency,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            final_users, final_items = model.propagate()
        # Export the layer-mean propagated representations, never ego embeddings.
        user_embeddings = final_users.detach().to("cpu", dtype=torch.float32).numpy()
        item_embeddings = final_items.detach().to("cpu", dtype=torch.float32).numpy()

        arrays = {
            "lightgcn_user_embeddings": user_embeddings,
            "lightgcn_item_embeddings": item_embeddings,
            "model_user_indices": model_users,
            "canonical_item_indices": universe.item_indices,
            "train_seen_offsets": offsets,
            "train_seen_items": seen_items,
            "train_observed_mask": observed,
            "popular_item_indices": popularity.popular_items,
        }
        for name, array in arrays.items():
            _write_npy(temporary / f"{name}.npy", array)
        _write_json(temporary / "model_user_ids.json", raw_user_ids)
        manifest = _manifest(
            temporary,
            checkpoint=args.checkpoint.resolve(),
            train_split=args.dataset_path.resolve() / "train.npz",
            user_mapping=args.user_mapping.resolve(),
            item_mapping=args.item_mapping.resolve(),
            parent_asins=args.parent_asins.resolve(),
            product_catalog=args.product_catalog.resolve(),
            semantic_config=semantic_config,
            semantic_metadata=semantic_metadata,
            universe=universe,
            num_users=model_users.size,
            embedding_dim=embedding_dim,
            layers=layers,
            observed_count=int(observed.sum()),
        )
        _write_json(temporary / "serving_manifest.json", manifest)
        load_serving_artifacts(
            temporary, project_root=PROJECT_ROOT, validate_catalog=True
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    args = parse_args()
    try:
        output = export(args)
        artifacts = load_serving_artifacts(output, project_root=PROJECT_ROOT)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Serving bundle exported and validated: {output}")
    for name in (
        "lightgcn_user_embeddings", "lightgcn_item_embeddings", "model_user_indices",
        "canonical_item_indices", "train_seen_offsets", "train_seen_items",
        "train_observed_mask", "popular_item_indices",
    ):
        array = getattr(artifacts, name)
        spec = artifacts.manifest["files"][name]
        print(
            f"{name}.npy: shape={array.shape}, dtype={array.dtype}, "
            f"sha256={spec['sha256']}"
        )
    user_spec = artifacts.manifest["files"]["model_user_ids"]
    print(
        "model_user_ids.json: "
        f"count={len(artifacts.model_user_ids)}, sha256={user_spec['sha256']}"
    )
    print(f"serving_manifest.json: sha256={sha256_file(output / 'serving_manifest.json')}")
    print(f"artifact_version={artifacts.manifest['artifact_version']}")
    print("BUNDLE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
