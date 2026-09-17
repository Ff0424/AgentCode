"""Deterministic tests for V2-09.2a without loading BGE-M3 or CUDA."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.agentrec.retrieval import (
    ChunkEmbeddingConfig,
    build_chunk_embedding_artifacts,
    validate_chunk_embedding_artifacts,
)


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([digest[index] + 1.0 for index in range(4)])
        return np.asarray(rows, dtype=np.float64)


class ZeroEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 4), dtype=np.float32)


class FailingEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated encoder failure")
        return super().encode(texts)


def _write_chunks(path: Path, *, duplicate: bool = False) -> None:
    values = [
        {"chunk_id": "0:summary:0", "chunk_index": 0, "item_index": 0,
         "parent_asin": "A0", "chunk_type": "summary", "part_index": 0,
         "text": "Title: USB-C hub"},
        {"chunk_id": "0:summary:0" if duplicate else "1:features:0",
         "chunk_index": 1, "item_index": 1, "parent_asin": "A1",
         "chunk_type": "features", "part_index": 0,
         "text": "HDMI and ethernet"},
        {"chunk_id": "2:description:0", "chunk_index": 2, "item_index": 2,
         "parent_asin": "A2", "chunk_type": "description", "part_index": 0,
         "text": "Compact laptop adapter"},
    ]
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _config() -> ChunkEmbeddingConfig:
    return ChunkEmbeddingConfig(
        model_name="fake-bge-m3", embedding_dimension=4,
        batch_size=2, max_length=32, device="cuda:0", use_fp16=True,
    )


class ChunkEmbeddingArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_preserves_identity_and_normalizes(self) -> None:
        chunks = self.root / "chunks.jsonl"
        output = self.root / "retrieval"
        _write_chunks(chunks)
        manifest = build_chunk_embedding_artifacts(
            chunks_path=chunks, output_dir=output, encoder=FakeEncoder(), config=_config()
        )
        array = np.load(output / "chunk_embeddings.npy", allow_pickle=False)
        metadata = json.loads((output / "chunk_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(array.shape, (3, 4))
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(array.flags.c_contiguous)
        np.testing.assert_allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-6)
        self.assertEqual(metadata["rows"], [
            {"row": 0, "source_row": 0, "chunk_id": "0:summary:0", "parent_asin": "A0", "item_index": 0},
            {"row": 1, "source_row": 1, "chunk_id": "1:features:0", "parent_asin": "A1", "item_index": 1},
            {"row": 2, "source_row": 2, "chunk_id": "2:description:0", "parent_asin": "A2", "item_index": 2},
        ])
        self.assertEqual(
            manifest["source"]["sha256"], hashlib.sha256(chunks.read_bytes()).hexdigest()
        )
        self.assertEqual(manifest["model"]["name"], "fake-bge-m3")
        validate_chunk_embedding_artifacts(output, expected_chunks_path=chunks)

    def test_export_is_deterministic(self) -> None:
        chunks = self.root / "chunks.jsonl"
        _write_chunks(chunks)
        manifests = [build_chunk_embedding_artifacts(
            chunks_path=chunks, output_dir=self.root / name,
            encoder=FakeEncoder(), config=_config(),
        ) for name in ("first", "second")]
        self.assertEqual(manifests[0], manifests[1])
        for filename in ("chunk_embeddings.npy", "chunk_metadata.json", "retrieval_manifest.json"):
            self.assertEqual(
                (self.root / "first" / filename).read_bytes(),
                (self.root / "second" / filename).read_bytes(),
            )

    def test_duplicate_chunk_identity_is_rejected(self) -> None:
        chunks = self.root / "chunks.jsonl"
        _write_chunks(chunks, duplicate=True)
        with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
            build_chunk_embedding_artifacts(
                chunks_path=chunks, output_dir=self.root / "output",
                encoder=FakeEncoder(), config=_config(),
            )

    def test_zero_vector_is_rejected_without_publication(self) -> None:
        chunks = self.root / "chunks.jsonl"
        output = self.root / "output"
        _write_chunks(chunks)
        with self.assertRaisesRegex(ValueError, "zero-length"):
            build_chunk_embedding_artifacts(
                chunks_path=chunks, output_dir=output,
                encoder=ZeroEncoder(), config=_config(),
            )
        self.assertFalse(output.exists())

    def test_source_fingerprint_mismatch_is_rejected(self) -> None:
        chunks = self.root / "chunks.jsonl"
        output = self.root / "output"
        _write_chunks(chunks)
        build_chunk_embedding_artifacts(
            chunks_path=chunks, output_dir=output,
            encoder=FakeEncoder(), config=_config()
        )
        chunks.write_text(chunks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source chunks fingerprint mismatch"):
            validate_chunk_embedding_artifacts(output, expected_chunks_path=chunks)

    def test_diagnostic_row_range_and_memmap_alignment(self) -> None:
        chunks = self.root / "chunks.jsonl"
        output = self.root / "diagnostic"
        messages: list[str] = []
        _write_chunks(chunks)
        manifest = build_chunk_embedding_artifacts(
            chunks_path=chunks, output_dir=output, encoder=FakeEncoder(),
            config=_config(), start_row=1, max_rows=2, progress_rows=1,
            diagnostic=True, progress_callback=messages.append,
        )
        metadata = json.loads((output / "chunk_metadata.json").read_text(encoding="utf-8"))
        matrix = np.load(output / "chunk_embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.assertEqual(matrix.shape, (2, 4))
        self.assertEqual([row["source_row"] for row in metadata["rows"]], [1, 2])
        self.assertEqual([row["row"] for row in metadata["rows"]], [0, 1])
        self.assertEqual(manifest["source"]["selected_row_range"], [1, 3])
        self.assertEqual(metadata["export_scope"]["mode"], "diagnostic")
        self.assertTrue(any("source_rows=[1,3)" in value for value in messages))
        self.assertTrue(any("CHECKPOINT processed_rows=2" in value for value in messages))
        self.assertFalse((output / "export_progress.json").exists())

    def test_partial_range_requires_diagnostic_mode(self) -> None:
        chunks = self.root / "chunks.jsonl"
        _write_chunks(chunks)
        with self.assertRaisesRegex(ValueError, "requires diagnostic=True"):
            build_chunk_embedding_artifacts(
                chunks_path=chunks, output_dir=self.root / "formal",
                encoder=FakeEncoder(), config=_config(), max_rows=1,
            )

    def test_encoder_failure_never_publishes_incomplete_artifact(self) -> None:
        chunks = self.root / "chunks.jsonl"
        output = self.root / "formal"
        _write_chunks(chunks)
        with self.assertRaisesRegex(RuntimeError, "simulated encoder failure"):
            build_chunk_embedding_artifacts(
                chunks_path=chunks, output_dir=output,
                encoder=FailingEncoder(), config=_config(), progress_rows=1,
                progress_callback=None,
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
