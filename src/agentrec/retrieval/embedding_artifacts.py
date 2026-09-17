"""Build deterministic, row-aligned dense embeddings for knowledge chunks.

The artifact preserves this immutable identity chain:
``embedding row -> chunk_id -> parent_asin -> canonical item_index``.
The builder accepts an injected encoder so artifact logic can be tested without
loading BGE-M3 or requiring CUDA. The production CLI supplies BGEM3ChunkEncoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from src.agentrec.knowledge import KnowledgeChunk


ARTIFACT_VERSION = "agentrec-retrieval-v1"
EMBEDDING_FILE = "chunk_embeddings.npy"
METADATA_FILE = "chunk_metadata.json"
MANIFEST_FILE = "retrieval_manifest.json"
PROGRESS_FILE = "export_progress.json"


class DenseTextEncoder(Protocol):
    """Minimal provider-neutral contract required by the artifact builder."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one dense vector per input text."""


@dataclass(frozen=True)
class ChunkEmbeddingConfig:
    """Frozen embedding contract shared by the production builder and manifest."""

    model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    batch_size: int = 8
    max_length: int = 2048
    device: str = "cuda:0"
    use_fp16: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        for name in ("embedding_dimension", "batch_size", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


class BGEM3ChunkEncoder:
    """CUDA-only BGE-M3 dense encoder using the established AgentRec settings."""

    def __init__(self, model_path: str | Path, config: ChunkEmbeddingConfig) -> None:
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local BGE-M3 model directory not found: {path}")

        # Heavy runtime dependencies remain isolated from import/static-test paths.
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - depends on GPU environment
            raise RuntimeError("BGE-M3 runtime requires torch and FlagEmbedding.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for production chunk embedding.")
        if config.device != "cuda:0":
            raise ValueError("The frozen chunk embedding contract requires device='cuda:0'.")
        torch.cuda.set_device(0)
        self._config = config
        self._model = BGEM3FlagModel(
            str(path), use_fp16=config.use_fp16, devices=[config.device]
        )
        configured_devices = getattr(self._model, "devices", None)
        normalized_devices = (
            [configured_devices] if isinstance(configured_devices, str)
            else list(configured_devices) if configured_devices is not None else None
        )
        if normalized_devices is not None and normalized_devices != [config.device]:
            raise RuntimeError(
                "BGE-M3 did not retain the requested single-device configuration: "
                f"{configured_devices!r}."
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        result = self._model.encode(
            list(texts), batch_size=self._config.batch_size,
            max_length=self._config.max_length, return_dense=True,
            return_sparse=False, return_colbert_vecs=False,
        )
        return np.asarray(result["dense_vecs"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_chunk(line: str, line_number: int) -> KnowledgeChunk:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid chunk JSON at line {line_number}.") from exc
    try:
        return KnowledgeChunk.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid chunk schema at line {line_number}: {exc}") from exc


def _scan_chunk_identities(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_chunk_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            chunk = _read_chunk(line, row + 1)
            if chunk.chunk_index != row:
                raise ValueError(
                    f"Chunk row alignment mismatch at row {row}: "
                    f"chunk_index={chunk.chunk_index}."
                )
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id at row {row}: {chunk.chunk_id!r}.")
            seen_chunk_ids.add(chunk.chunk_id)
            rows.append({
                "source_row": row, "chunk_id": chunk.chunk_id,
                "parent_asin": chunk.parent_asin, "item_index": chunk.item_index,
            })
    if not rows:
        raise ValueError("chunks.jsonl contains no chunks.")
    return rows


def _normalize_batch(vectors: object, rows: int, dimension: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (rows, dimension):
        raise ValueError(
            f"Encoder returned shape {array.shape}; expected ({rows}, {dimension})."
        )
    if not np.isfinite(array).all():
        raise ValueError("Encoder returned NaN or Inf values.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Encoder returned a non-finite or zero-length vector.")
    # Normalization is explicit even if a provider currently returns unit vectors.
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist progress inside the unpublished temporary directory."""

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_chunk_embedding_artifacts(
    *, chunks_path: str | Path, output_dir: str | Path,
    encoder: DenseTextEncoder, config: ChunkEmbeddingConfig | None = None,
    start_row: int = 0, max_rows: int | None = None,
    progress_rows: int = 4096, diagnostic: bool = False,
    progress_callback: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Encode chunks incrementally and atomically publish one retrieval bundle."""

    source = Path(chunks_path).resolve()
    destination = Path(output_dir).resolve()
    settings = config or ChunkEmbeddingConfig()
    if not source.is_file():
        raise FileNotFoundError(f"Knowledge chunks not found: {source}")
    if destination.exists():
        raise FileExistsError(f"Retrieval output already exists: {destination}")
    for name, value in (("start_row", start_row), ("progress_rows", progress_rows)):
        minimum = 0 if name == "start_row" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0
    ):
        raise ValueError("max_rows must be None or a positive integer.")

    source_fingerprint = _sha256(source)
    all_identities = _scan_chunk_identities(source)
    total_source_rows = len(all_identities)
    if start_row >= total_source_rows:
        raise ValueError(
            f"start_row={start_row} is outside source row count {total_source_rows}."
        )
    end_row = min(
        total_source_rows,
        total_source_rows if max_rows is None else start_row + max_rows,
    )
    is_full_export = start_row == 0 and end_row == total_source_rows
    if not is_full_export and not diagnostic:
        raise ValueError("A partial row range requires diagnostic=True.")
    identities = [
        {"row": embedding_row, **identity}
        for embedding_row, identity in enumerate(all_identities[start_row:end_row])
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        embedding_path = temporary / EMBEDDING_FILE
        matrix = np.lib.format.open_memmap(
            embedding_path, mode="w+", dtype=np.float32,
            shape=(len(identities), settings.embedding_dimension),
        )
        next_row = 0
        texts: list[str] = []
        source_batch_start = start_row
        progress_path = temporary / PROGRESS_FILE
        progress = {
            "status": "encoding", "mode": "full" if is_full_export else "diagnostic",
            "source_start_row": start_row, "source_end_row_exclusive": end_row,
            "processed_rows": 0, "last_flushed_source_row_exclusive": start_row,
            "active_source_row_range": None,
        }
        _write_progress(progress_path, progress)
        next_flush_at = progress_rows

        def emit(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)

        def encode_pending() -> None:
            nonlocal next_row, source_batch_start, next_flush_at
            if not texts:
                return
            source_batch_end = source_batch_start + len(texts)
            progress["active_source_row_range"] = [source_batch_start, source_batch_end]
            progress["active_chunk_ids"] = [
                identity["chunk_id"]
                for identity in identities[next_row:next_row + len(texts)]
            ]
            progress["active_text_length_chars"] = {
                "min": min(map(len, texts)), "max": max(map(len, texts)),
            }
            _write_progress(progress_path, progress)
            emit(
                f"ENCODE START source_rows=[{source_batch_start},{source_batch_end}) "
                f"processed_rows={next_row} max_text_chars={max(map(len, texts))}"
            )
            batch = _normalize_batch(
                encoder.encode(tuple(texts)), len(texts), settings.embedding_dimension
            )
            matrix[next_row:next_row + len(texts)] = batch
            next_row += len(texts)
            progress["processed_rows"] = next_row
            progress["active_source_row_range"] = None
            progress["active_chunk_ids"] = None
            progress["active_text_length_chars"] = None
            emit(
                f"ENCODE DONE source_rows=[{source_batch_start},{source_batch_end}) "
                f"processed_rows={next_row}"
            )
            if next_row >= next_flush_at or next_row == len(identities):
                matrix.flush()
                progress["last_flushed_source_row_exclusive"] = source_batch_end
                _write_progress(progress_path, progress)
                emit(
                    f"CHECKPOINT processed_rows={next_row} "
                    f"last_flushed_source_row_exclusive={source_batch_end}"
                )
                while next_flush_at <= next_row:
                    next_flush_at += progress_rows
            texts.clear()
            source_batch_start = source_batch_end

        with source.open("r", encoding="utf-8") as handle:
            for source_row, line in enumerate(handle):
                if source_row < start_row:
                    continue
                if source_row >= end_row:
                    break
                line_number = source_row + 1
                texts.append(_read_chunk(line, line_number).text)
                if len(texts) == settings.batch_size:
                    encode_pending()
            encode_pending()
        if next_row != len(identities):
            raise RuntimeError("Chunk count changed between identity scan and encoding pass.")
        if _sha256(source) != source_fingerprint:
            raise RuntimeError("chunks.jsonl changed during embedding export.")
        matrix.flush()
        del matrix

        stored = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        norms = np.linalg.norm(stored, axis=1)
        if stored.shape != (len(identities), settings.embedding_dimension):
            raise RuntimeError("Stored embedding shape verification failed.")
        if stored.dtype != np.dtype(np.float32) or not np.isfinite(stored).all():
            raise RuntimeError("Stored embeddings failed dtype/finite verification.")
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Stored embeddings failed L2 normalization verification.")
        norm_statistics = {
            "count": int(norms.size), "min": float(norms.min()),
            "mean": float(norms.mean()), "max": float(norms.max()),
        }
        del stored

        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "row_count": len(identities),
            "embedding_dimension": settings.embedding_dimension,
            "embedding_dtype": "float32",
            "l2_normalized": True,
            "identity_contract": "embedding row -> chunk_id -> parent_asin -> item_index",
            "export_scope": {
                "mode": "full" if is_full_export else "diagnostic",
                "source_start_row": start_row,
                "source_end_row_exclusive": end_row,
                "total_source_rows": total_source_rows,
            },
            "rows": identities,
        }
        metadata_path = temporary / METADATA_FILE
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest: dict[str, object] = {
            "artifact_version": ARTIFACT_VERSION,
            "source": {
                "file": source.name, "sha256": source_fingerprint,
                "chunk_count": total_source_rows,
                "selected_row_range": [start_row, end_row],
            },
            "model": {
                "name": settings.model_name,
                "embedding_dimension": settings.embedding_dimension,
                "batch_size": settings.batch_size,
                "max_length": settings.max_length,
                "device": settings.device,
                "use_fp16": settings.use_fp16,
                "dense_only": True,
                "execution_contract": "single process, one configured CUDA device",
            },
            "embedding": {
                "shape": [len(identities), settings.embedding_dimension],
                "dtype": "float32", "contiguous": True, "l2_normalized": True,
                "norm_statistics": norm_statistics,
            },
            "outputs": {
                EMBEDDING_FILE: {"sha256": _sha256(embedding_path)},
                METADATA_FILE: {"sha256": _sha256(metadata_path)},
            },
        }
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        validate_chunk_embedding_artifacts(temporary, expected_chunks_path=source)
        progress_path.unlink()
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_chunk_embedding_artifacts(
    output_dir: str | Path, *, expected_chunks_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate checksums, row identity, shape, dtype, finiteness and norms."""

    root = Path(output_dir).resolve()
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Retrieval manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported retrieval artifact version.")
    for name in (EMBEDDING_FILE, METADATA_FILE):
        path = root / name
        expected = manifest.get("outputs", {}).get(name, {}).get("sha256")
        if not path.is_file() or not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"Retrieval artifact fingerprint mismatch: {name}")

    metadata = json.loads((root / METADATA_FILE).read_text(encoding="utf-8"))
    rows = metadata.get("rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or row.get("row") != index
        or not isinstance(row.get("source_row"), int)
        for index, row in enumerate(rows)
    ):
        raise ValueError("Chunk metadata row alignment is invalid.")
    chunk_ids = [row.get("chunk_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in chunk_ids):
        raise ValueError("Chunk metadata contains an invalid chunk_id.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk metadata contains duplicate chunk IDs.")

    array = np.load(root / EMBEDDING_FILE, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(manifest.get("embedding", {}).get("shape", ()))
    if array.shape != expected_shape or array.shape[0] != len(rows):
        raise ValueError("Embedding shape does not match manifest/metadata.")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise ValueError("Embeddings must be contiguous float32.")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or Inf values.")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Embeddings are not L2 normalized.")
    if expected_chunks_path is not None:
        source = Path(expected_chunks_path).resolve()
        expected_fingerprint = manifest.get("source", {}).get("sha256")
        if not source.is_file() or _sha256(source) != expected_fingerprint:
            raise ValueError("Source chunks fingerprint mismatch.")
    return manifest
