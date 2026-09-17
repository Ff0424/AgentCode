"""Small deterministic tests for the V2-07.2b canonical-user contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.recommendation.export_canonical_user_lookup import (
    _build_sorted_chunks,
    _merge_chunks,
    export_canonical_user_lookup,
)
from src.agentrec.recommendation.artifacts import (
    SCHEMA_VERSION_V2,
    load_serving_artifacts,
    sha256_file,
)
from tests.test_recommendation_serving_foundation import ServingFixture


class CanonicalUserLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = ServingFixture(self.root)
        self.raw_to_canonical = {
            "user-zero": 0,
            "user-one": 1,
            "user-two": 2,
            "user-three": 3,
            "user-four": 4,
            "user-five": 5,
            "user-six": 6,
            "user-seven": 7,
            "user-eight": 8,
            "user-nine": 9,
        }
        self._publish_lookup(self.raw_to_canonical)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish_lookup(self, values: dict[str, int]) -> None:
        ordered = sorted(values.items(), key=lambda entry: entry[0].encode("utf-8"))
        raw_values = [raw.encode("utf-8") for raw, _ in ordered]
        bytes_path = self.fixture.bundle / "canonical_user_id_bytes.bin"
        bytes_path.write_bytes(b"".join(raw_values))
        offsets = np.zeros(len(raw_values) + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum([len(raw) for raw in raw_values], dtype=np.uint64)
        indices = np.asarray([canonical for _, canonical in ordered], dtype=np.int32)
        offsets_path = self.fixture.bundle / "canonical_user_id_offsets.npy"
        indices_path = self.fixture.bundle / "canonical_user_indices.npy"
        np.save(offsets_path, offsets, allow_pickle=False)
        np.save(indices_path, indices, allow_pickle=False)

        manifest = self.fixture.manifest
        manifest["row_mappings"] = dict(manifest["row_mappings"])
        manifest["schema_version"] = SCHEMA_VERSION_V2
        manifest["num_canonical_users"] = len(values)
        source_hash = manifest["sources"]["user_mapping"]["sha256"]
        manifest["canonical_user_lookup"] = {
            "encoding": "utf-8-concatenated-bytes-with-uint64-offsets",
            "sorting": "strict-ascending-utf8-bytewise",
            "lookup": "binary-search-exact-bytes",
            "source_user_mapping_sha256": source_hash,
        }
        manifest["files"].update(
            {
                "canonical_user_id_bytes": {
                    "path": bytes_path.name,
                    "sha256": sha256_file(bytes_path),
                    "size_bytes": bytes_path.stat().st_size,
                    "format": "concatenated-utf8",
                },
                "canonical_user_id_offsets": self._array_spec(offsets_path),
                "canonical_user_indices": self._array_spec(indices_path),
            }
        )
        manifest["row_mappings"]["canonical_user_lookup"] = (
            "sorted raw user_id bytes -> canonical user_index via exact binary search"
        )
        self.fixture.write_manifest()

    @staticmethod
    def _array_spec(path: Path) -> dict[str, object]:
        array = np.load(path, allow_pickle=False)
        return {
            "path": path.name,
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": array.dtype.name,
        }

    def _load(self):
        return load_serving_artifacts(
            self.fixture.bundle, project_root=self.root, validate_catalog=False
        )

    def test_three_identity_classes_and_model_row(self) -> None:
        artifacts = self._load()
        self.assertEqual(artifacts.resolve_canonical_user("user-four"), 4)
        self.assertEqual(artifacts.resolve_model_user_row(4), 0)
        self.assertEqual(artifacts.resolve_canonical_user("user-three"), 3)
        self.assertIsNone(artifacts.resolve_model_user_row(3))
        self.assertIsNone(artifacts.resolve_canonical_user("not-in-canonical"))
        artifacts.canonical_user_resolver.close()

    def test_first_and_last_boundaries(self) -> None:
        artifacts = self._load()
        ordered = sorted(self.raw_to_canonical, key=lambda value: value.encode("utf-8"))
        self.assertEqual(
            artifacts.resolve_canonical_user(ordered[0]), self.raw_to_canonical[ordered[0]]
        )
        self.assertEqual(
            artifacts.resolve_canonical_user(ordered[-1]), self.raw_to_canonical[ordered[-1]]
        )
        artifacts.canonical_user_resolver.close()

    def test_duplicate_and_unsorted_raw_users_are_rejected(self) -> None:
        for raw_bytes, split in ((b"samesame", 4), (b"z-usera-user", 6)):
            with self.subTest(raw_bytes=raw_bytes):
                bytes_path = self.fixture.bundle / "canonical_user_id_bytes.bin"
                bytes_path.write_bytes(raw_bytes)
                np.save(
                    self.fixture.bundle / "canonical_user_id_offsets.npy",
                    np.array([0, split, len(raw_bytes)], dtype=np.uint64),
                    allow_pickle=False,
                )
                np.save(
                    self.fixture.bundle / "canonical_user_indices.npy",
                    np.array([0, 1], dtype=np.int32),
                    allow_pickle=False,
                )
                self.fixture.manifest["num_canonical_users"] = 2
                self._refresh_lookup_specs()
                with self.assertRaisesRegex(ValueError, "duplicate or unsorted"):
                    self._load()

    def _refresh_lookup_specs(self) -> None:
        bytes_path = self.fixture.bundle / "canonical_user_id_bytes.bin"
        self.fixture.manifest["files"]["canonical_user_id_bytes"].update(
            sha256=sha256_file(bytes_path), size_bytes=bytes_path.stat().st_size
        )
        for name in ("canonical_user_id_offsets", "canonical_user_indices"):
            self.fixture.manifest["files"][name] = self._array_spec(
                self.fixture.bundle / f"{name}.npy"
            )
        self.fixture.write_manifest()

    def test_offsets_corruption_is_rejected(self) -> None:
        path = self.fixture.bundle / "canonical_user_id_offsets.npy"
        np.save(path, np.zeros(11, dtype=np.uint64), allow_pickle=False)
        self._refresh_lookup_specs()
        with self.assertRaisesRegex(ValueError, "offsets"):
            self._load()

    def test_checksum_mismatch_is_rejected(self) -> None:
        self.fixture.manifest["files"]["canonical_user_id_bytes"]["sha256"] = "0" * 64
        self.fixture.write_manifest()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._load()

    def test_source_fingerprint_mismatch_is_rejected(self) -> None:
        self.fixture.manifest["canonical_user_lookup"][
            "source_user_mapping_sha256"
        ] = "0" * 64
        self.fixture.write_manifest()
        with self.assertRaisesRegex(ValueError, "source fingerprint mismatch"):
            self._load()

    def test_external_sort_export_is_deterministic(self) -> None:
        mapping = self.root / "users.json"
        mapping.write_text(
            '{"0":"z-user","1":"a-user","2":"m-user"}', encoding="utf-8"
        )
        hashes = []
        for run in range(2):
            output = self.root / f"output-{run}"
            chunks_dir = output / "chunks"
            output.mkdir()
            chunks_dir.mkdir()
            chunks, count = _build_sorted_chunks(mapping, chunks_dir, chunk_entries=2)
            _merge_chunks(chunks, output, count)
            hashes.append(
                tuple(
                    sha256_file(output / name)
                    for name in (
                        "canonical_user_id_bytes.bin",
                        "canonical_user_id_offsets.npy",
                        "canonical_user_indices.npy",
                    )
                )
            )
        self.assertEqual(hashes[0], hashes[1])

    def test_complete_v2_bundle_is_published_without_changing_v1(self) -> None:
        # Restore a clean v1 source bundle, then supplement it into a separate v2 path.
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = ServingFixture(self.root)
        mapping = self.root / "users.json"
        canonical = {str(index): f"user-{index}" for index in range(10)}
        canonical["4"] = "user-four"
        canonical["9"] = "user-nine"
        mapping.write_text(json.dumps(canonical, separators=(",", ":")), encoding="utf-8")
        self.fixture.manifest["sources"]["user_mapping"] = {
            "path": mapping.name,
            "sha256": sha256_file(mapping),
        }
        self.fixture.write_manifest()
        original_manifest_hash = sha256_file(
            self.fixture.bundle / "serving_manifest.json"
        )
        output = self.root / "serving-v2"
        export_canonical_user_lookup(
            self.fixture.bundle,
            mapping,
            output,
            project_root=self.root,
            chunk_entries=3,
        )
        self.assertEqual(
            sha256_file(self.fixture.bundle / "serving_manifest.json"),
            original_manifest_hash,
        )
        artifacts = load_serving_artifacts(
            output, project_root=self.root, validate_catalog=False
        )
        self.assertEqual(artifacts.manifest["schema_version"], SCHEMA_VERSION_V2)
        self.assertEqual(artifacts.resolve_canonical_user("user-four"), 4)
        self.assertEqual(artifacts.resolve_model_user_row(4), 0)
        artifacts.canonical_user_resolver.close()


if __name__ == "__main__":
    unittest.main()
