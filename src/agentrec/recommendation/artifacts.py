"""Strict read-only loader for the AgentRec recommendation serving bundle.

The loader never repairs, regenerates, or trains artifacts. Every bundle file
is checksum-verified before opening, NumPy arrays use ``mmap_mode='r'``, and
cross-file identity/CSR invariants fail fast during process initialization.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "1.0"
SCHEMA_VERSION_V2 = "2.0"
EXPECTED_CANONICAL_ITEMS = 125_762
EXPECTED_TRAIN_OBSERVED_ITEMS = 125_684
ARRAY_FILES = (
    "lightgcn_user_embeddings",
    "lightgcn_item_embeddings",
    "model_user_indices",
    "canonical_item_indices",
    "train_seen_offsets",
    "train_seen_items",
    "train_observed_mask",
    "popular_item_indices",
)
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "artifact_version",
    "created_at",
    "num_model_users",
    "num_canonical_items",
    "num_train_observed_items",
    "num_train_unseen_items",
    "embedding_dim",
    "lightgcn_layers",
    "canonical_identity_sha256",
    "files",
    "external_artifacts",
    "sources",
    "row_mappings",
}
REQUIRED_TOP_LEVEL_V2 = REQUIRED_TOP_LEVEL | {
    "num_canonical_users",
    "canonical_user_lookup",
}
REQUIRED_SOURCES = {
    "lightgcn_checkpoint",
    "train_split",
    "user_mapping",
    "item_mapping",
    "parent_asins",
    "product_catalog",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON in {label} {path}: {exc}") from exc


def _freeze(value: Any) -> Any:
    """Recursively remove mutable containers from public manifest state."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _require_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _validate_file_spec(spec: object, name: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError(f"Manifest files.{name} must be an object.")
    required = {"path", "sha256", "shape", "dtype"}
    if set(spec) != required:
        raise ValueError(f"Manifest files.{name} keys must be {sorted(required)}.")
    if not isinstance(spec["path"], str) or Path(spec["path"]).name != spec["path"]:
        raise ValueError(f"Manifest files.{name}.path must be a bundle-local filename.")
    if not isinstance(spec["sha256"], str) or re.fullmatch(
        r"[0-9a-f]{64}", spec["sha256"]
    ) is None:
        raise ValueError(f"Manifest files.{name}.sha256 must be a SHA-256 hex digest.")
    if not isinstance(spec["shape"], list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in spec["shape"]
    ):
        raise ValueError(f"Manifest files.{name}.shape must be non-negative integers.")
    try:
        np.dtype(spec["dtype"])
    except TypeError as exc:
        raise ValueError(f"Manifest files.{name}.dtype is invalid.") from exc
    return spec


def validate_manifest_schema(manifest: object) -> dict[str, Any]:
    """Validate the complete serving manifest structure without touching files."""

    if not isinstance(manifest, dict):
        raise TypeError("Serving manifest must contain a JSON object.")
    schema_version = manifest.get("schema_version")
    required_top_level = (
        REQUIRED_TOP_LEVEL_V2
        if schema_version == SCHEMA_VERSION_V2
        else REQUIRED_TOP_LEVEL
    )
    if set(manifest) != required_top_level:
        raise ValueError(
            "Serving manifest keys mismatch; "
            f"missing={sorted(required_top_level-set(manifest))}, "
            f"extra={sorted(set(manifest)-required_top_level)}."
        )
    if schema_version not in {SCHEMA_VERSION, SCHEMA_VERSION_V2}:
        raise ValueError(
            f"Unsupported serving schema_version={schema_version!r}; "
            f"expected {SCHEMA_VERSION!r} or {SCHEMA_VERSION_V2!r}."
        )
    for key in ("artifact_version", "created_at", "canonical_identity_sha256"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"Manifest {key} must be a non-empty string.")
    if re.fullmatch(r"[0-9a-f]{64}", manifest["canonical_identity_sha256"]) is None:
        raise ValueError("canonical_identity_sha256 must be a SHA-256 hex digest.")
    for key in (
        "num_model_users",
        "num_canonical_items",
        "num_train_observed_items",
        "num_train_unseen_items",
        "embedding_dim",
        "lightgcn_layers",
    ):
        _require_int(manifest[key], f"Manifest {key}", 0)
    if manifest["num_model_users"] == 0 or manifest["num_canonical_items"] == 0:
        raise ValueError("Serving bundle must contain users and canonical items.")
    if manifest["embedding_dim"] == 0:
        raise ValueError("embedding_dim must be positive.")
    if (
        manifest["num_train_observed_items"] + manifest["num_train_unseen_items"]
        != manifest["num_canonical_items"]
    ):
        raise ValueError("Observed and unseen item counts do not cover the catalog.")
    files = manifest["files"]
    expected_files = set(ARRAY_FILES) | {"model_user_ids"}
    if schema_version == SCHEMA_VERSION_V2:
        expected_files |= {
            "canonical_user_id_bytes",
            "canonical_user_id_offsets",
            "canonical_user_indices",
        }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("Manifest files section does not match the serving bundle schema.")
    for name in ARRAY_FILES:
        _validate_file_spec(files[name], name)
    user_ids = files["model_user_ids"]
    if not isinstance(user_ids, dict) or set(user_ids) != {"path", "sha256", "count"}:
        raise ValueError("Manifest files.model_user_ids schema is invalid.")
    if (
        user_ids["path"] != "model_user_ids.json"
        or not isinstance(user_ids["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", user_ids["sha256"]) is None
    ):
        raise ValueError("Manifest model_user_ids identity is invalid.")
    if user_ids["count"] != manifest["num_model_users"]:
        raise ValueError("model_user_ids count does not match num_model_users.")
    external = manifest["external_artifacts"]
    if not isinstance(external, dict) or set(external) != {
        "semantic_item_embeddings", "semantic_metadata"
    }:
        raise ValueError("Manifest external_artifacts schema is invalid.")
    for name in external:
        spec = external[name]
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise ValueError(f"External artifact {name} schema is invalid.")
        if not all(isinstance(spec[key], str) and spec[key] for key in spec):
            raise ValueError(f"External artifact {name} values must be non-empty strings.")
        if re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is None:
            raise ValueError(f"External artifact {name} SHA-256 is invalid.")
    if not isinstance(manifest["sources"], dict) or set(manifest["sources"]) != REQUIRED_SOURCES:
        raise ValueError("Manifest sources do not match the serving provenance schema.")
    for name, spec in manifest["sources"].items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("Every source entry must be a named object.")
        if set(spec) != {"path", "sha256"}:
            raise ValueError(f"Source {name!r} schema is invalid.")
        if (
            not isinstance(spec["path"], str)
            or not spec["path"].strip()
            or not isinstance(spec["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is None
        ):
            raise ValueError(f"Source {name!r} path or SHA-256 is invalid.")
    if schema_version == SCHEMA_VERSION_V2:
        _validate_canonical_user_lookup_manifest(manifest)
    mappings = manifest["row_mappings"]
    expected_mappings = {
        "lightgcn_user_embeddings": "row i -> model_user_indices[i] -> canonical user_index",
        "model_user_ids": "row i -> raw/API user_id for model_user_indices[i]",
        "lightgcn_item_embeddings": "row i -> canonical_item_indices[i] -> canonical item_index",
        "train_seen_csr": "model user row u -> serving item rows in offsets[u]:offsets[u+1]",
        "semantic_item_embeddings": "row i -> canonical_item_indices[i] -> canonical item_index",
        "popular_item_indices": "ranked canonical item_index values",
    }
    if schema_version == SCHEMA_VERSION_V2:
        expected_mappings["canonical_user_lookup"] = (
            "sorted raw user_id bytes -> canonical user_index via exact binary search"
        )
    if mappings != expected_mappings:
        raise ValueError("Manifest row_mappings contract is invalid.")
    return manifest


def _validate_canonical_user_lookup_manifest(manifest: dict[str, Any]) -> None:
    """Validate the v2 compact canonical-user lookup contract."""

    count = _require_int(manifest["num_canonical_users"], "num_canonical_users", 1)
    lookup = manifest["canonical_user_lookup"]
    expected_lookup = {
        "encoding",
        "sorting",
        "lookup",
        "source_user_mapping_sha256",
    }
    if not isinstance(lookup, dict) or set(lookup) != expected_lookup:
        raise ValueError("canonical_user_lookup manifest contract is invalid.")
    if lookup["encoding"] != "utf-8-concatenated-bytes-with-uint64-offsets":
        raise ValueError("Unsupported canonical user ID encoding.")
    if lookup["sorting"] != "strict-ascending-utf8-bytewise":
        raise ValueError("Unsupported canonical user sorting contract.")
    if lookup["lookup"] != "binary-search-exact-bytes":
        raise ValueError("Unsupported canonical user lookup contract.")
    source_hash = lookup["source_user_mapping_sha256"]
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise ValueError("canonical user source fingerprint is invalid.")
    if source_hash != manifest["sources"]["user_mapping"]["sha256"]:
        raise ValueError("Canonical user lookup source fingerprint mismatch.")

    files = manifest["files"]
    bytes_spec = files["canonical_user_id_bytes"]
    if not isinstance(bytes_spec, dict) or set(bytes_spec) != {"path", "sha256", "size_bytes", "format"}:
        raise ValueError("canonical_user_id_bytes file contract is invalid.")
    if (
        bytes_spec["path"] != "canonical_user_id_bytes.bin"
        or bytes_spec["format"] != "concatenated-utf8"
        or not isinstance(bytes_spec["size_bytes"], int)
        or isinstance(bytes_spec["size_bytes"], bool)
        or bytes_spec["size_bytes"] <= 0
        or not isinstance(bytes_spec["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", bytes_spec["sha256"]) is None
    ):
        raise ValueError("canonical_user_id_bytes file metadata is invalid.")
    for name, shape, dtype in (
        ("canonical_user_id_offsets", [count + 1], "uint64"),
        ("canonical_user_indices", [count], "int32"),
    ):
        spec = _validate_file_spec(files[name], name)
        if spec["path"] != f"{name}.npy" or spec["shape"] != shape or spec["dtype"] != dtype:
            raise ValueError(f"{name} shape/dtype contract is invalid.")


def _resolve_project_path(project_root: Path, raw_path: str, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} path must be non-empty.")
    path = Path(raw_path)
    return (path if path.is_absolute() else project_root / path).resolve()


def _verified_array(bundle_dir: Path, spec: dict[str, Any], name: str) -> np.ndarray:
    path = bundle_dir / spec["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Serving artifact {name} not found: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Serving artifact checksum mismatch: {name}.")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to open serving artifact {name}: {exc}") from exc
    if list(array.shape) != spec["shape"] or array.dtype != np.dtype(spec["dtype"]):
        raise ValueError(
            f"Serving artifact {name} is {array.shape}/{array.dtype}; expected "
            f"{tuple(spec['shape'])}/{spec['dtype']}."
        )
    if array.flags.writeable:
        raise ValueError(f"Serving artifact {name} unexpectedly opened writable.")
    return array


def _validate_catalog_identity(
    path: Path,
    canonical_items: np.ndarray,
    expected_identity_sha256: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Product Catalog not found: {path}")
    found: dict[int, str] = {}
    found_parents: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid Product Catalog JSON at line {line_number}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Product Catalog line {line_number} is not an object.")
            item_index = record.get("item_index")
            parent_asin = record.get("parent_asin")
            if (
                isinstance(item_index, bool)
                or not isinstance(item_index, int)
                or not isinstance(parent_asin, str)
                or not parent_asin.strip()
            ):
                raise ValueError(f"Invalid Product Catalog identity at line {line_number}.")
            if item_index in found:
                raise ValueError(f"Duplicate Product Catalog item_index={item_index}.")
            if parent_asin in found_parents:
                raise ValueError(f"Duplicate Product Catalog parent_asin={parent_asin!r}.")
            found[item_index] = parent_asin
            found_parents.add(parent_asin)
    if len(found) != canonical_items.size or not np.array_equal(
        np.asarray(sorted(found), dtype=canonical_items.dtype), canonical_items
    ):
        raise ValueError("Product Catalog item identities do not match canonical item rows.")
    identity = hashlib.sha256()
    for item_index in canonical_items:
        identity.update(f"{int(item_index)}\t{found[int(item_index)]}\n".encode("utf-8"))
    if identity.hexdigest() != expected_identity_sha256:
        raise ValueError("Product Catalog parent-ASIN mapping fingerprint mismatch.")


class CanonicalUserResolver:
    """Exact read-only lookup over sorted concatenated UTF-8 user IDs."""

    def __init__(
        self,
        bytes_path: Path,
        offsets: np.ndarray,
        canonical_indices: np.ndarray,
    ) -> None:
        self._bytes_path = bytes_path
        self._offsets = offsets
        self._canonical_indices = canonical_indices
        self._bytes_file = bytes_path.open("rb")
        self._bytes = mmap.mmap(self._bytes_file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def count(self) -> int:
        return int(self._canonical_indices.size)

    def _entry_bytes(self, row: int) -> bytes:
        start = int(self._offsets[row])
        stop = int(self._offsets[row + 1])
        return self._bytes[start:stop]

    def resolve_canonical_user(self, raw_user_id: str) -> int | None:
        """Return the exact canonical user index, or ``None`` when absent."""

        if not isinstance(raw_user_id, str):
            raise TypeError("raw_user_id must be a string.")
        normalized = raw_user_id.strip()
        if not normalized:
            raise ValueError("raw_user_id must be non-empty.")
        target = normalized.encode("utf-8")
        low = 0
        high = self.count
        while low < high:
            middle = (low + high) // 2
            candidate = self._entry_bytes(middle)
            if candidate < target:
                low = middle + 1
            else:
                high = middle
        if low < self.count and self._entry_bytes(low) == target:
            return int(self._canonical_indices[low])
        return None

    def close(self) -> None:
        if not self._bytes.closed:
            self._bytes.close()
        if not self._bytes_file.closed:
            self._bytes_file.close()
        for array in (self._offsets, self._canonical_indices):
            mapped = getattr(array, "_mmap", None)
            if mapped is not None and not mapped.closed:
                mapped.close()

    def __del__(self) -> None:
        # Best-effort cleanup for application shutdown; callers may close explicitly.
        try:
            self.close()
        except (AttributeError, BufferError, OSError):
            pass


def _load_canonical_user_resolver(
    bundle: Path,
    manifest: dict[str, Any],
) -> CanonicalUserResolver:
    files = manifest["files"]
    bytes_spec = files["canonical_user_id_bytes"]
    bytes_path = bundle / bytes_spec["path"]
    if not bytes_path.is_file():
        raise FileNotFoundError(f"Canonical user ID bytes not found: {bytes_path}")
    if bytes_path.stat().st_size != bytes_spec["size_bytes"]:
        raise ValueError("Canonical user ID bytes length mismatch.")
    if sha256_file(bytes_path) != bytes_spec["sha256"]:
        raise ValueError("Serving artifact checksum mismatch: canonical_user_id_bytes.")
    offsets = _verified_array(
        bundle, files["canonical_user_id_offsets"], "canonical_user_id_offsets"
    )
    indices = _verified_array(
        bundle, files["canonical_user_indices"], "canonical_user_indices"
    )
    count = manifest["num_canonical_users"]
    if offsets.shape != (count + 1,) or offsets.dtype != np.dtype("uint64"):
        raise ValueError("Canonical user offsets shape/dtype mismatch.")
    if indices.shape != (count,) or indices.dtype != np.dtype("int32"):
        raise ValueError("Canonical user indices shape/dtype mismatch.")
    if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("Canonical user offsets must be strictly increasing from zero.")
    if int(offsets[-1]) != bytes_path.stat().st_size:
        raise ValueError("Canonical user offsets do not cover the bytes file.")
    if np.any(indices < 0):
        raise ValueError("Canonical user indices contain a negative value.")
    ordered_indices = np.sort(np.asarray(indices))
    if not np.array_equal(ordered_indices, np.arange(count, dtype=np.int32)):
        raise ValueError("Canonical user indices must be a permutation of 0..count-1.")

    resolver = CanonicalUserResolver(bytes_path, offsets, indices)
    try:
        previous: bytes | None = None
        for row in range(count):
            current = resolver._entry_bytes(row)
            try:
                current.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Canonical user ID row {row} is not valid UTF-8.") from exc
            if previous is not None and current <= previous:
                raise ValueError("Canonical raw user IDs are duplicate or unsorted.")
            previous = current
    except Exception:
        resolver.close()
        raise
    return resolver


@dataclass(frozen=True)
class ServingArtifacts:
    """Validated mmap-backed arrays and compact serving identity mappings."""

    manifest: Mapping[str, Any]
    lightgcn_user_embeddings: np.ndarray
    lightgcn_item_embeddings: np.ndarray
    model_user_indices: np.ndarray
    model_user_ids: tuple[str, ...]
    canonical_item_indices: np.ndarray
    train_seen_offsets: np.ndarray
    train_seen_items: np.ndarray
    train_observed_mask: np.ndarray
    popular_item_indices: np.ndarray
    semantic_item_embeddings: np.ndarray
    canonical_user_resolver: CanonicalUserResolver | None = None

    def resolve_canonical_user(self, raw_user_id: str) -> int | None:
        if self.canonical_user_resolver is None:
            raise RuntimeError("Serving bundle does not contain canonical user lookup artifacts.")
        return self.canonical_user_resolver.resolve_canonical_user(raw_user_id)

    def resolve_model_user_row(self, canonical_user_index: int) -> int | None:
        if isinstance(canonical_user_index, bool) or not isinstance(
            canonical_user_index, (int, np.integer)
        ):
            raise TypeError("canonical_user_index must be an integer.")
        canonical = int(canonical_user_index)
        if canonical < 0:
            raise ValueError("canonical_user_index must be non-negative.")
        row = int(np.searchsorted(self.model_user_indices, canonical))
        if row < self.model_user_indices.size and int(self.model_user_indices[row]) == canonical:
            return row
        return None

    def close(self) -> None:
        """Release mmap handles explicitly, primarily for tests and atomic publication."""

        if self.canonical_user_resolver is not None:
            self.canonical_user_resolver.close()
        for array in (
            self.lightgcn_user_embeddings,
            self.lightgcn_item_embeddings,
            self.model_user_indices,
            self.canonical_item_indices,
            self.train_seen_offsets,
            self.train_seen_items,
            self.train_observed_mask,
            self.popular_item_indices,
            self.semantic_item_embeddings,
        ):
            mapped = getattr(array, "_mmap", None)
            if mapped is not None and not mapped.closed:
                mapped.close()


def load_serving_artifacts(
    bundle_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    validate_catalog: bool = True,
) -> ServingArtifacts:
    """Load and cross-validate a serving bundle without modifying any artifact."""

    bundle = Path(bundle_dir).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"Serving bundle directory not found: {bundle}")
    root = Path(project_root).resolve() if project_root is not None else bundle.parents[2]
    manifest = validate_manifest_schema(_load_json(bundle / "serving_manifest.json", "manifest"))
    arrays = {
        name: _verified_array(bundle, manifest["files"][name], name)
        for name in ARRAY_FILES
    }
    user_id_spec = manifest["files"]["model_user_ids"]
    user_id_path = bundle / user_id_spec["path"]
    if sha256_file(user_id_path) != user_id_spec["sha256"]:
        raise ValueError("Serving artifact checksum mismatch: model_user_ids.")
    raw_user_ids = _load_json(user_id_path, "model user identities")
    if not isinstance(raw_user_ids, list) or len(raw_user_ids) != manifest["num_model_users"]:
        raise ValueError("model_user_ids.json count does not match the manifest.")
    if any(not isinstance(value, str) or not value.strip() for value in raw_user_ids):
        raise ValueError("Every model user ID must be a non-empty string.")
    if len(set(raw_user_ids)) != len(raw_user_ids):
        raise ValueError("model_user_ids.json contains duplicate raw user IDs.")

    users = manifest["num_model_users"]
    items = manifest["num_canonical_items"]
    dim = manifest["embedding_dim"]
    expected = {
        "lightgcn_user_embeddings": ((users, dim), np.dtype("float32")),
        "lightgcn_item_embeddings": ((items, dim), np.dtype("float32")),
        "model_user_indices": ((users,), np.dtype("int32")),
        "canonical_item_indices": ((items,), np.dtype("int32")),
        "train_seen_offsets": ((users + 1,), np.dtype("int64")),
        "train_seen_items": ((manifest["files"]["train_seen_items"]["shape"][0],), np.dtype("int32")),
        "train_observed_mask": ((items,), np.dtype("bool")),
        "popular_item_indices": ((manifest["num_train_observed_items"],), np.dtype("int32")),
    }
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Serving artifact {name} violates the cross-file shape/dtype contract.")

    model_users = arrays["model_user_indices"]
    canonical_items = arrays["canonical_item_indices"]
    if np.any(model_users[1:] <= model_users[:-1]):
        raise ValueError("model_user_indices must be unique and strictly increasing.")
    if np.any(canonical_items[1:] <= canonical_items[:-1]):
        raise ValueError("canonical_item_indices must be unique and strictly increasing.")

    offsets = arrays["train_seen_offsets"]
    seen = arrays["train_seen_items"]
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != seen.size:
        raise ValueError("train_seen CSR offsets are invalid.")
    if seen.size and (np.any(seen < 0) or np.any(seen >= items)):
        raise ValueError("train_seen_items contains an out-of-range serving item row.")
    for user_row in range(users):
        values = seen[int(offsets[user_row]) : int(offsets[user_row + 1])]
        if values.size == 0 or np.any(values[1:] <= values[:-1]):
            raise ValueError(f"Invalid or empty train history for model user row {user_row}.")

    observed = arrays["train_observed_mask"]
    observed_count = int(np.count_nonzero(observed))
    if observed_count != manifest["num_train_observed_items"]:
        raise ValueError("train_observed_mask count does not match the manifest.")
    if items == EXPECTED_CANONICAL_ITEMS and (
        observed_count != EXPECTED_TRAIN_OBSERVED_ITEMS
        or items - observed_count != 78
    ):
        raise ValueError("Production train-observed/cold counts are not 125,684/78.")
    expected_observed = np.zeros(items, dtype=bool)
    expected_observed[np.unique(seen)] = True
    if not np.array_equal(observed, expected_observed):
        raise ValueError("train_observed_mask disagrees with train_seen CSR.")

    popular = arrays["popular_item_indices"]
    if np.unique(popular).size != popular.size:
        raise ValueError("popular_item_indices contains duplicates.")
    popular_rows = np.searchsorted(canonical_items, popular)
    if np.any(popular_rows >= items) or not np.array_equal(canonical_items[popular_rows], popular):
        raise ValueError("popular_item_indices contains a non-canonical item.")
    if not np.all(observed[popular_rows]):
        raise ValueError("Popularity fallback includes a train-unseen item.")

    external = manifest["external_artifacts"]
    semantic_path = _resolve_project_path(
        root, external["semantic_item_embeddings"]["path"], "semantic embeddings"
    )
    metadata_path = _resolve_project_path(
        root, external["semantic_metadata"]["path"], "semantic metadata"
    )
    for path, name in ((semantic_path, "semantic embeddings"), (metadata_path, "semantic metadata")):
        if not path.is_file():
            raise FileNotFoundError(f"External {name} not found: {path}")
    if sha256_file(semantic_path) != external["semantic_item_embeddings"]["sha256"]:
        raise ValueError("Semantic embedding fingerprint mismatch.")
    if sha256_file(metadata_path) != external["semantic_metadata"]["sha256"]:
        raise ValueError("Semantic metadata fingerprint mismatch.")
    semantic_metadata = _load_json(metadata_path, "semantic metadata")
    semantic_provenance = (
        semantic_metadata.get("provenance")
        if isinstance(semantic_metadata, dict)
        else None
    )
    if not isinstance(semantic_provenance, dict) or (
        semantic_provenance.get("canonical_identity_sha256")
        != manifest["canonical_identity_sha256"]
    ):
        raise ValueError("Semantic metadata canonical identity mismatch.")
    semantic = np.load(semantic_path, mmap_mode="r", allow_pickle=False)
    if semantic.shape != (items, 1024) or semantic.dtype != np.dtype("float32"):
        raise ValueError("Semantic item embedding shape/dtype contract mismatch.")
    if semantic.flags.writeable:
        raise ValueError("Semantic embeddings unexpectedly opened writable.")

    if validate_catalog:
        catalog_source = manifest["sources"].get("product_catalog")
        if not isinstance(catalog_source, dict):
            raise ValueError("Manifest lacks Product Catalog provenance.")
        catalog_path = _resolve_project_path(root, catalog_source["path"], "Product Catalog")
        if sha256_file(catalog_path) != catalog_source["sha256"]:
            raise ValueError("Product Catalog fingerprint mismatch.")
        _validate_catalog_identity(
            catalog_path, canonical_items, manifest["canonical_identity_sha256"]
        )

    canonical_user_resolver = (
        _load_canonical_user_resolver(bundle, manifest)
        if manifest["schema_version"] == SCHEMA_VERSION_V2
        else None
    )

    return ServingArtifacts(
        manifest=_freeze(manifest),
        lightgcn_user_embeddings=arrays["lightgcn_user_embeddings"],
        lightgcn_item_embeddings=arrays["lightgcn_item_embeddings"],
        model_user_indices=arrays["model_user_indices"],
        model_user_ids=tuple(raw_user_ids),
        canonical_item_indices=arrays["canonical_item_indices"],
        train_seen_offsets=offsets,
        train_seen_items=seen,
        train_observed_mask=observed,
        popular_item_indices=popular,
        semantic_item_embeddings=semantic,
        canonical_user_resolver=canonical_user_resolver,
    )
