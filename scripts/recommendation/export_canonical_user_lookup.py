"""Publish a V2-07.2b serving bundle with compact canonical-user lookup.

The existing V2-07.2 bundle is treated as immutable input. The large canonical
``user_mapping.json`` is parsed incrementally, sorted with bounded-memory chunk
files, and emitted as concatenated UTF-8 bytes plus mmap-friendly offsets and
canonical indices. A complete v2 bundle is assembled in a temporary directory
and atomically renamed; existing output is never overwritten.

Inputs:
    artifacts/recommendation/serving
    data/processed/recommendation/user_mapping.json

Output:
    artifacts/recommendation/serving_v2
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import struct
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from scripts.recommendation.build_enriched_interactions import iter_flat_json_object
from src.agentrec.recommendation.artifacts import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    load_serving_artifacts,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_BUNDLE = PROJECT_ROOT / "artifacts/recommendation/serving"
DEFAULT_USER_MAPPING = PROJECT_ROOT / "data/processed/recommendation/user_mapping.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/recommendation/serving_v2"
DEFAULT_CHUNK_ENTRIES = 500_000
CHUNK_HEADER = struct.Struct("<II")  # UTF-8 byte length, canonical user index.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add the canonical-user lookup to a new serving v2 bundle."
    )
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--user-mapping", type=Path, default=DEFAULT_USER_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-entries", type=int, default=DEFAULT_CHUNK_ENTRIES)
    return parser.parse_args()


def _write_chunk(path: Path, entries: list[tuple[bytes, int]]) -> None:
    entries.sort(key=lambda entry: entry[0])
    with path.open("xb") as handle:
        for raw_id, canonical_index in entries:
            handle.write(CHUNK_HEADER.pack(len(raw_id), canonical_index))
            handle.write(raw_id)
        handle.flush()
        os.fsync(handle.fileno())


def _read_chunk_record(handle: BinaryIO) -> tuple[bytes, int] | None:
    header = handle.read(CHUNK_HEADER.size)
    if not header:
        return None
    if len(header) != CHUNK_HEADER.size:
        raise ValueError("Temporary canonical-user sort chunk has a truncated header.")
    length, canonical_index = CHUNK_HEADER.unpack(header)
    raw_id = handle.read(length)
    if len(raw_id) != length:
        raise ValueError("Temporary canonical-user sort chunk has a truncated user ID.")
    return raw_id, canonical_index


def _build_sorted_chunks(
    user_mapping: Path,
    temporary: Path,
    chunk_entries: int,
) -> tuple[list[Path], int]:
    if isinstance(chunk_entries, bool) or not isinstance(chunk_entries, int) or chunk_entries <= 0:
        raise ValueError("chunk_entries must be a positive integer.")
    chunks: list[Path] = []
    entries: list[tuple[bytes, int]] = []
    count = 0
    for raw_index, raw_user_id in iter_flat_json_object(
        user_mapping, "canonical user mapping"
    ):
        try:
            canonical_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid canonical user index {raw_index!r}.") from exc
        if (
            str(canonical_index) != raw_index
            or canonical_index < 0
            or canonical_index > np.iinfo(np.int32).max
        ):
            raise ValueError(f"Invalid canonical user index {raw_index!r}.")
        if not isinstance(raw_user_id, str) or not raw_user_id.strip():
            raise ValueError(f"Invalid raw user ID for canonical index {canonical_index}.")
        encoded = raw_user_id.encode("utf-8")
        if len(encoded) > np.iinfo(np.uint32).max:
            raise ValueError("Raw user ID exceeds the uint32 chunk record length.")
        entries.append((encoded, canonical_index))
        count += 1
        if len(entries) == chunk_entries:
            path = temporary / f"canonical-users-{len(chunks):05d}.chunk"
            _write_chunk(path, entries)
            chunks.append(path)
            entries = []
    if entries:
        path = temporary / f"canonical-users-{len(chunks):05d}.chunk"
        _write_chunk(path, entries)
        chunks.append(path)
    if count == 0:
        raise ValueError("Canonical user mapping is empty.")
    return chunks, count


def _merge_chunks(chunks: list[Path], output: Path, count: int) -> None:
    bytes_path = output / "canonical_user_id_bytes.bin"
    offsets_path = output / "canonical_user_id_offsets.npy"
    indices_path = output / "canonical_user_indices.npy"
    offsets = np.lib.format.open_memmap(
        offsets_path, mode="w+", dtype=np.uint64, shape=(count + 1,)
    )
    indices = np.lib.format.open_memmap(
        indices_path, mode="w+", dtype=np.int32, shape=(count,)
    )
    handles = [path.open("rb") for path in chunks]
    heap: list[tuple[bytes, int, int]] = []
    try:
        for source, handle in enumerate(handles):
            record = _read_chunk_record(handle)
            if record is not None:
                heapq.heappush(heap, (record[0], record[1], source))
        offsets[0] = 0
        position = 0
        previous: bytes | None = None
        with bytes_path.open("xb") as bytes_file:
            while heap:
                raw_id, canonical_index, source = heapq.heappop(heap)
                if previous is not None and raw_id <= previous:
                    raise ValueError("Canonical raw user IDs contain duplicates.")
                bytes_file.write(raw_id)
                indices[position] = canonical_index
                offsets[position + 1] = int(offsets[position]) + len(raw_id)
                position += 1
                previous = raw_id
                record = _read_chunk_record(handles[source])
                if record is not None:
                    heapq.heappush(heap, (record[0], record[1], source))
            bytes_file.flush()
            os.fsync(bytes_file.fileno())
        if position != count:
            raise ValueError(f"Merged canonical users={position:,}; expected {count:,}.")
        offsets.flush()
        indices.flush()
    finally:
        for handle in handles:
            handle.close()
        del offsets, indices


def _npy_spec(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": array.dtype.name,
        }
    finally:
        array._mmap.close()


def _copy_existing_bundle(source: Path, temporary: Path, manifest: dict) -> None:
    for spec in manifest["files"].values():
        source_path = source / spec["path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Source serving artifact not found: {source_path}")
        shutil.copyfile(source_path, temporary / source_path.name)


def _write_manifest(path: Path, manifest: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_canonical_user_lookup(
    source_bundle: Path,
    user_mapping: Path,
    output_dir: Path,
    *,
    project_root: Path,
    chunk_entries: int = DEFAULT_CHUNK_ENTRIES,
) -> Path:
    """Atomically publish a complete serving v2 bundle without changing v1."""

    source = source_bundle.resolve()
    mapping = user_mapping.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Serving v2 output already exists: {output}")
    if not mapping.is_file():
        raise FileNotFoundError(f"Canonical user mapping not found: {mapping}")
    source_artifacts = load_serving_artifacts(
        source, project_root=project_root, validate_catalog=True
    )
    try:
        if source_artifacts.manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("V2-07.2b source bundle must use serving schema 1.0.")
    finally:
        source_artifacts.close()
    source_manifest = json.loads(
        (source / "serving_manifest.json").read_text(encoding="utf-8")
    )
    mapping_sha256 = sha256_file(mapping)
    if mapping_sha256 != source_manifest["sources"]["user_mapping"]["sha256"]:
        raise ValueError("Current user_mapping fingerprint differs from the v1 manifest.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    chunks_dir = temporary / ".canonical-user-sort"
    chunks_dir.mkdir()
    try:
        chunks, count = _build_sorted_chunks(mapping, chunks_dir, chunk_entries)
        _copy_existing_bundle(source, temporary, source_manifest)
        _merge_chunks(chunks, temporary, count)
        shutil.rmtree(chunks_dir)

        manifest = source_manifest
        manifest["schema_version"] = SCHEMA_VERSION_V2
        manifest["num_canonical_users"] = count
        manifest["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lookup_hashes = [
            sha256_file(temporary / "canonical_user_id_bytes.bin"),
            sha256_file(temporary / "canonical_user_id_offsets.npy"),
            sha256_file(temporary / "canonical_user_indices.npy"),
        ]
        version_material = "\n".join(
            [source_manifest["artifact_version"], mapping_sha256, *lookup_hashes]
        )
        manifest["artifact_version"] = "serving-v2-" + hashlib.sha256(
            version_material.encode("ascii")
        ).hexdigest()[:16]
        bytes_path = temporary / "canonical_user_id_bytes.bin"
        manifest["files"]["canonical_user_id_bytes"] = {
            "path": bytes_path.name,
            "sha256": lookup_hashes[0],
            "size_bytes": bytes_path.stat().st_size,
            "format": "concatenated-utf8",
        }
        manifest["files"]["canonical_user_id_offsets"] = _npy_spec(
            temporary / "canonical_user_id_offsets.npy"
        )
        manifest["files"]["canonical_user_indices"] = _npy_spec(
            temporary / "canonical_user_indices.npy"
        )
        manifest["canonical_user_lookup"] = {
            "encoding": "utf-8-concatenated-bytes-with-uint64-offsets",
            "sorting": "strict-ascending-utf8-bytewise",
            "lookup": "binary-search-exact-bytes",
            "source_user_mapping_sha256": mapping_sha256,
        }
        manifest["row_mappings"]["canonical_user_lookup"] = (
            "sorted raw user_id bytes -> canonical user_index via exact binary search"
        )
        _write_manifest(temporary / "serving_manifest.json", manifest)
        validated = load_serving_artifacts(
            temporary, project_root=project_root, validate_catalog=True
        )
        if validated.canonical_user_resolver is None or (
            validated.canonical_user_resolver.count != count
        ):
            raise ValueError("Canonical user resolver did not validate the complete mapping.")
        # Windows cannot atomically rename a directory while its mmap files are open.
        validated.close()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    args = parse_args()
    try:
        output = export_canonical_user_lookup(
            args.source_bundle,
            args.user_mapping,
            args.output_dir,
            project_root=PROJECT_ROOT,
            chunk_entries=args.chunk_entries,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    artifacts = load_serving_artifacts(output, project_root=PROJECT_ROOT)
    resolver = artifacts.canonical_user_resolver
    assert resolver is not None
    files = artifacts.manifest["files"]
    print(f"Serving v2 bundle exported and validated: {output}")
    print(f"Canonical users: {resolver.count:,}")
    for name in (
        "canonical_user_id_bytes",
        "canonical_user_id_offsets",
        "canonical_user_indices",
    ):
        print(f"{name}: {files[name]['path']} sha256={files[name]['sha256']}")
    print("V2-07.2b CANONICAL USER LOOKUP EXPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
