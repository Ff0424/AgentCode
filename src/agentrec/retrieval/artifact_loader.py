"""Strict or explicitly trusted loading of published chunk retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

import numpy as np

from src.agentrec.knowledge import KnowledgeChunk

from .embedding_artifacts import ARTIFACT_VERSION, EMBEDDING_FILE, METADATA_FILE, MANIFEST_FILE


class ValidationMode(str, Enum):
    STRICT = "strict"
    TRUSTED_PUBLISHED = "trusted_published"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ChunkRetrievalArtifacts:
    """Read-only embeddings, row metadata, offsets and item-to-row mapping."""

    def __init__(
        self, *, retrieval_dir: str | Path, knowledge_dir: str | Path,
        validation_mode: ValidationMode | str, expected_dimension: int = 1024,
    ) -> None:
        try:
            self.validation_mode = ValidationMode(validation_mode)
        except ValueError as exc:
            raise ValueError("validation_mode must be 'strict' or 'trusted_published'.") from exc
        self.retrieval_dir = Path(retrieval_dir).resolve()
        self.knowledge_dir = Path(knowledge_dir).resolve()
        self.chunks_path = self.knowledge_dir / "chunks.jsonl"
        self._expected_dimension = expected_dimension
        self.manifest, self.metadata, self.embeddings = self._load_core()
        self.offsets, self._item_rows = self._scan_knowledge_and_build_offsets()
        if self.validation_mode is ValidationMode.STRICT:
            self._run_strict_validation()

    def _read_json(self, path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc

    def _load_core(self) -> tuple[dict, dict, np.ndarray]:
        paths = {
            "manifest": self.retrieval_dir / MANIFEST_FILE,
            "embeddings": self.retrieval_dir / EMBEDDING_FILE,
            "metadata": self.retrieval_dir / METADATA_FILE,
            "chunks": self.chunks_path,
            "knowledge_manifest": self.knowledge_dir / "knowledge_manifest.json",
        }
        missing = [path for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing retrieval artifact: " + ", ".join(map(str, missing)))
        manifest = self._read_json(paths["manifest"])
        metadata = self._read_json(paths["metadata"])
        knowledge_manifest = self._read_json(paths["knowledge_manifest"])
        if not isinstance(manifest, dict) or manifest.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported retrieval artifact version.")
        if not isinstance(metadata, dict) or metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported chunk metadata artifact version.")
        if not isinstance(knowledge_manifest, dict):
            raise ValueError("knowledge_manifest.json must contain an object.")

        embeddings = np.load(paths["embeddings"], mmap_mode="r", allow_pickle=False)
        embedding_contract = manifest.get("embedding")
        source_contract = manifest.get("source")
        if not isinstance(embedding_contract, dict) or not isinstance(source_contract, dict):
            raise ValueError("Retrieval manifest is missing source/embedding contracts.")
        expected_shape = (source_contract.get("chunk_count"), self._expected_dimension)
        if embeddings.shape != expected_shape or embeddings.dtype != np.dtype(np.float32):
            raise ValueError(
                f"Embedding artifact is {embeddings.shape}/{embeddings.dtype}; "
                f"expected {expected_shape}/float32."
            )
        checks = {
            "manifest shape": embedding_contract.get("shape") == list(expected_shape),
            "manifest dtype": embedding_contract.get("dtype") == "float32",
            "manifest contiguous": embedding_contract.get("contiguous") is True,
            "manifest normalized": embedding_contract.get("l2_normalized") is True,
            "metadata row count": metadata.get("row_count") == expected_shape[0],
            "metadata dimension": metadata.get("embedding_dimension") == self._expected_dimension,
            "metadata dtype": metadata.get("embedding_dtype") == "float32",
            "metadata normalized": metadata.get("l2_normalized") is True,
            "formal scope": metadata.get("export_scope", {}).get("mode") == "full",
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("Artifact contract mismatch: " + ", ".join(failed))
        if not embeddings.flags.c_contiguous or embeddings.flags.writeable:
            raise ValueError("Embedding mmap must be contiguous and read-only.")

        rows = metadata.get("rows")
        if not isinstance(rows, list) or len(rows) != expected_shape[0]:
            raise ValueError("Chunk metadata rows do not match embedding row count.")
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Chunk metadata row {index} must be an object.")
            if row.get("row") != index or row.get("source_row") != index:
                raise ValueError(f"Chunk metadata row identity mismatch at row {index}.")
            chunk_id = row.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
                raise ValueError(f"Invalid or duplicate chunk_id at metadata row {index}.")
            if not isinstance(row.get("parent_asin"), str) or not row["parent_asin"]:
                raise ValueError(f"Invalid parent_asin at metadata row {index}.")
            item_index = row.get("item_index")
            if isinstance(item_index, bool) or not isinstance(item_index, int) or item_index < 0:
                raise ValueError(f"Invalid item_index at metadata row {index}.")
            seen.add(chunk_id)

        knowledge_sha = knowledge_manifest.get("outputs", {}).get("chunks.jsonl", {}).get("sha256")
        if source_contract.get("sha256") != knowledge_sha:
            raise ValueError("Retrieval source fingerprint does not match knowledge manifest.")
        metadata_expected_sha = manifest.get("outputs", {}).get(METADATA_FILE, {}).get("sha256")
        if not isinstance(metadata_expected_sha, str) or _sha256(paths["metadata"]) != metadata_expected_sha:
            raise ValueError("Chunk metadata checksum mismatch.")
        return manifest, metadata, embeddings

    def _scan_knowledge_and_build_offsets(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        rows = self.metadata["rows"]
        offsets: list[int] = []
        item_rows: dict[int, list[int]] = {}
        with self.chunks_path.open("rb") as handle:
            row = 0
            while line := handle.readline():
                offsets.append(handle.tell() - len(line))
                if row >= len(rows):
                    raise ValueError("Knowledge chunks contain more rows than metadata.")
                try:
                    chunk = KnowledgeChunk.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"Invalid KnowledgeChunk at row {row}: {exc}") from exc
                metadata = rows[row]
                if (
                    chunk.chunk_index != row
                    or chunk.chunk_id != metadata["chunk_id"]
                    or chunk.parent_asin != metadata["parent_asin"]
                    or chunk.item_index != metadata["item_index"]
                ):
                    raise ValueError(f"Knowledge/embedding identity mismatch at row {row}.")
                item_rows.setdefault(chunk.item_index, []).append(row)
                row += 1
            offsets.append(handle.tell())
        if row != len(rows):
            raise ValueError("Knowledge chunks contain fewer rows than metadata.")
        compact = {
            item: np.asarray(values, dtype=np.int64) for item, values in item_rows.items()
        }
        return np.asarray(offsets, dtype=np.uint64), compact

    def _run_strict_validation(self) -> None:
        output_contract = self.manifest.get("outputs", {})
        expected_embedding_sha = output_contract.get(EMBEDDING_FILE, {}).get("sha256")
        if not isinstance(expected_embedding_sha, str) or _sha256(
            self.retrieval_dir / EMBEDDING_FILE
        ) != expected_embedding_sha:
            raise ValueError("Chunk embedding checksum mismatch.")
        if _sha256(self.chunks_path) != self.manifest["source"]["sha256"]:
            raise ValueError("Knowledge chunks checksum mismatch.")
        for start in range(0, self.embeddings.shape[0], 4096):
            block = self.embeddings[start:start + 4096]
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite embedding values at rows {start}:{start+len(block)}.")
            norms = np.linalg.norm(block, axis=1)
            if np.any(norms <= 0) or not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Invalid normalized embeddings at rows {start}:{start+len(block)}.")

    @property
    def row_count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def rows_for_items(self, item_indices: tuple[int, ...]) -> np.ndarray:
        arrays = [self._item_rows[item] for item in item_indices if item in self._item_rows]
        if not arrays:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)

    def get_chunk(self, embedding_row: int) -> KnowledgeChunk:
        if isinstance(embedding_row, bool) or not isinstance(embedding_row, int):
            raise TypeError("embedding_row must be an integer.")
        if not 0 <= embedding_row < self.row_count:
            raise IndexError(f"embedding_row {embedding_row} is out of range.")
        begin = int(self.offsets[embedding_row])
        length = int(self.offsets[embedding_row + 1] - self.offsets[embedding_row])
        with self.chunks_path.open("rb") as handle:
            handle.seek(begin)
            line = handle.read(length)
        chunk = KnowledgeChunk.model_validate_json(line)
        metadata = self.metadata["rows"][embedding_row]
        if (
            chunk.chunk_index != embedding_row
            or chunk.chunk_id != metadata["chunk_id"]
            or chunk.parent_asin != metadata["parent_asin"]
            or chunk.item_index != metadata["item_index"]
        ):
            raise RuntimeError(f"Knowledge identity changed at embedding row {embedding_row}.")
        return chunk
