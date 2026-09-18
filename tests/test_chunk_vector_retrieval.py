"""Artifact, exact-search, identity and JSONL-offset tests for V2-09.2b."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.agentrec.retrieval import (
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from src.agentrec.retrieval.embedding_artifacts import ARTIFACT_VERSION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path]:
    knowledge = root / "knowledge"
    retrieval = root / "retrieval"
    knowledge.mkdir(parents=True)
    retrieval.mkdir(parents=True)
    chunks = [
        {"chunk_id": "10:summary:0", "chunk_index": 0, "item_index": 10,
         "parent_asin": "A10", "chunk_type": "summary", "part_index": 0,
         "text": "alpha exact"},
        {"chunk_id": "10:features:0", "chunk_index": 1, "item_index": 10,
         "parent_asin": "A10", "chunk_type": "features", "part_index": 0,
         "text": "alpha secondary"},
        {"chunk_id": "20:description:0", "chunk_index": 2, "item_index": 20,
         "parent_asin": "A20", "chunk_type": "description", "part_index": 0,
         "text": "beta product"},
    ]
    chunks_path = knowledge / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(value) + "\n" for value in chunks), encoding="utf-8"
    )
    (knowledge / "knowledge_manifest.json").write_text(json.dumps({
        "artifact_version": "agentrec-knowledge-v1",
        "outputs": {"chunks.jsonl": {"sha256": _sha(chunks_path)}},
    }), encoding="utf-8")
    matrix = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    np.save(retrieval / "chunk_embeddings.npy", matrix, allow_pickle=False)
    rows = [
        {"row": index, "source_row": index, "chunk_id": chunk["chunk_id"],
         "parent_asin": chunk["parent_asin"], "item_index": chunk["item_index"]}
        for index, chunk in enumerate(chunks)
    ]
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "row_count": 3,
        "embedding_dimension": 2, "embedding_dtype": "float32",
        "l2_normalized": True,
        "export_scope": {"mode": "full", "source_start_row": 0,
                         "source_end_row_exclusive": 3, "total_source_rows": 3},
        "rows": rows,
    }
    metadata_path = retrieval / "chunk_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    embedding_path = retrieval / "chunk_embeddings.npy"
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source": {"file": "chunks.jsonl", "sha256": _sha(chunks_path),
                   "chunk_count": 3, "selected_row_range": [0, 3]},
        "embedding": {"shape": [3, 2], "dtype": "float32",
                      "contiguous": True, "l2_normalized": True},
        "outputs": {
            "chunk_embeddings.npy": {"sha256": _sha(embedding_path)},
            "chunk_metadata.json": {"sha256": _sha(metadata_path)},
        },
    }
    (retrieval / "retrieval_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return retrieval, knowledge


class FakeEncoder:
    embedding_dimension = 2

    def encode(self, queries):
        return np.asarray([[1.0, 0.0] for _ in queries], dtype=np.float32)


class ChunkVectorRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.retrieval_dir, self.knowledge_dir = _fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _artifacts(self, mode=ValidationMode.STRICT) -> ChunkRetrievalArtifacts:
        return ChunkRetrievalArtifacts(
            retrieval_dir=self.retrieval_dir, knowledge_dir=self.knowledge_dir,
            validation_mode=mode, expected_dimension=2,
        )

    def test_strict_and_trusted_modes_offsets_and_identity_round_trip(self) -> None:
        for mode in (ValidationMode.STRICT, ValidationMode.TRUSTED_PUBLISHED):
            with self.subTest(mode=mode):
                artifacts = self._artifacts(mode)
                self.assertEqual(artifacts.offsets.dtype, np.uint64)
                self.assertEqual(artifacts.offsets.shape, (4,))
                chunk = artifacts.get_chunk(1)
                self.assertEqual(chunk.chunk_id, "10:features:0")
                self.assertEqual(chunk.text, "alpha secondary")
                np.testing.assert_array_equal(artifacts.rows_for_items((10,)), [0, 1])

    def test_artifact_shape_and_metadata_mismatch_refusal(self) -> None:
        np.save(
            self.retrieval_dir / "chunk_embeddings.npy",
            np.ones((2, 2), dtype=np.float32), allow_pickle=False,
        )
        with self.assertRaisesRegex(ValueError, "Embedding artifact"):
            self._artifacts(ValidationMode.TRUSTED_PUBLISHED)

        self.retrieval_dir, self.knowledge_dir = _fixture(self.root / "second")
        metadata_path = self.retrieval_dir / "chunk_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["rows"].pop()
        metadata_path.write_text(json.dumps(metadata))
        manifest_path = self.retrieval_dir / "retrieval_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["chunk_metadata.json"]["sha256"] = _sha(metadata_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "metadata rows"):
            self._artifacts(ValidationMode.TRUSTED_PUBLISHED)

    def test_knowledge_identity_mismatch_refusal(self) -> None:
        chunks_path = self.knowledge_dir / "chunks.jsonl"
        value = json.loads(chunks_path.read_text().splitlines()[0])
        value["parent_asin"] = "WRONG"
        lines = chunks_path.read_text().splitlines()
        lines[0] = json.dumps(value)
        chunks_path.write_text("\n".join(lines) + "\n")
        knowledge_manifest = self.knowledge_dir / "knowledge_manifest.json"
        payload = json.loads(knowledge_manifest.read_text())
        payload["outputs"]["chunks.jsonl"]["sha256"] = _sha(chunks_path)
        knowledge_manifest.write_text(json.dumps(payload))
        manifest_path = self.retrieval_dir / "retrieval_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source"]["sha256"] = _sha(chunks_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self._artifacts(ValidationMode.TRUSTED_PUBLISHED)

    def test_source_fingerprint_mismatch_is_always_rejected(self) -> None:
        manifest_path = self.retrieval_dir / "retrieval_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        for mode in (ValidationMode.STRICT, ValidationMode.TRUSTED_PUBLISHED):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "source fingerprint"
            ):
                self._artifacts(mode)

    def test_strict_checks_large_embedding_checksum_but_trusted_does_not(self) -> None:
        embedding_path = self.retrieval_dir / "chunk_embeddings.npy"
        writable = np.load(embedding_path, mmap_mode="r+")
        writable[0] = np.asarray([0.0, 1.0], dtype=np.float32)
        writable.flush()
        del writable

        # Published-artifact mode validates cheap contracts and trusts the release checksum.
        self._artifacts(ValidationMode.TRUSTED_PUBLISHED)
        with self.assertRaisesRegex(ValueError, "embedding checksum"):
            self._artifacts(ValidationMode.STRICT)

    def test_exact_top_k_and_global_rows(self) -> None:
        backend = NumPyExactBackend(self._artifacts().embeddings)
        result = backend.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=3)
        np.testing.assert_array_equal(result.rows, [0, 1, 2])
        np.testing.assert_allclose(result.scores, [1.0, 0.8, 0.0])
        restricted = backend.search(
            np.asarray([1.0, 0.0], dtype=np.float32), top_k=3,
            candidate_rows=np.asarray([2, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(restricted.rows, [1, 2])

    def test_deterministic_tie_breaking(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        backend = NumPyExactBackend(matrix)
        first = backend.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
        second = backend.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
        np.testing.assert_array_equal(first.rows, [0, 1])
        np.testing.assert_array_equal(first.rows, second.rows)

    def test_top_k_validation(self) -> None:
        backend = NumPyExactBackend(self._artifacts().embeddings)
        for value in (0, -1, True, 4):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=value)

    def test_chunk_retriever_and_candidate_item_restriction(self) -> None:
        artifacts = self._artifacts()
        retriever = ChunkRetriever(
            encoder=FakeEncoder(), backend=NumPyExactBackend(artifacts.embeddings),
            artifacts=artifacts,
        )
        result = retriever.search(" alpha query ", top_k=3, candidate_item_indices=[20, 10, 10])
        self.assertTrue(result.candidate_restricted)
        self.assertEqual(result.returned_count, 3)
        self.assertEqual([chunk.embedding_row for chunk in result.chunks], [0, 1, 2])
        self.assertEqual(result.chunks[0].chunk_id, "10:summary:0")
        empty = retriever.search("query", top_k=2, candidate_item_indices=[])
        self.assertEqual(empty.returned_count, 0)
        unknown = retriever.search("query", top_k=2, candidate_item_indices=[999])
        self.assertEqual(unknown.returned_count, 0)


if __name__ == "__main__":
    unittest.main()
