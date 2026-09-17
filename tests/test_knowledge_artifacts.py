"""Small deterministic V2-09.1 knowledge artifact tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.agentrec.knowledge import (
    KnowledgeChunk, ProductDocument, build_knowledge_artifacts,
    validate_knowledge_artifacts,
)


class KnowledgeArtifactTests(unittest.TestCase):
    def fixture(self, root: Path):
        mapping = root / "mapping.json"
        catalog = root / "catalog.jsonl"
        mapping.write_text(json.dumps({"2": "A2", "7": "A7"}), encoding="utf-8")
        rows = [
            {"item_index": 2, "parent_asin": "A2", "title": "Dock", "categories": ["Electronics"], "price": 99, "average_rating": 4.5, "rating_number": 10, "features": ["HDMI"], "description": "A useful dock."},
            {"item_index": 7, "parent_asin": "A7", "title": "Mouse", "categories": [], "price": None, "average_rating": None, "rating_number": None, "features": [], "description": None},
        ]
        catalog.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        return catalog, mapping

    def test_identity_round_trip_and_missing_description(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); catalog, mapping = self.fixture(root); output = root / "out"
            build_knowledge_artifacts(catalog_path=catalog, item_mapping_path=mapping, output_dir=output)
            documents = [ProductDocument.model_validate_json(x) for x in (output / "product_documents.jsonl").read_text(encoding="utf-8").splitlines()]
            chunks = [KnowledgeChunk.model_validate_json(x) for x in (output / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            identities = {(x.item_index, x.parent_asin) for x in documents}
            self.assertTrue(all((x.item_index, x.parent_asin) in identities for x in chunks))
            self.assertIsNone(documents[1].description)
            self.assertFalse(any(x.parent_asin == "A7" and x.chunk_type.value == "description" for x in chunks))

    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); catalog, mapping = self.fixture(root)
            first = build_knowledge_artifacts(catalog_path=catalog, item_mapping_path=mapping, output_dir=root / "a")
            second = build_knowledge_artifacts(catalog_path=catalog, item_mapping_path=mapping, output_dir=root / "b")
            self.assertEqual(first["outputs"], second["outputs"])

    def test_schema_and_forbidden_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ProductDocument(item_index=1, parent_asin="A", secret="x")
        with self.assertRaises(ValidationError):
            KnowledgeChunk(chunk_id="x", chunk_index=0, item_index=1, parent_asin="A", chunk_type="summary", part_index=0, text="x", score=1)

    def test_identity_mismatch_and_fingerprint_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); catalog, mapping = self.fixture(root)
            bad = root / "bad.jsonl"
            bad.write_text(catalog.read_text(encoding="utf-8").replace('"A2"', '"WRONG"', 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_knowledge_artifacts(catalog_path=bad, item_mapping_path=mapping, output_dir=root / "bad-out")
            output = root / "out"
            build_knowledge_artifacts(catalog_path=catalog, item_mapping_path=mapping, output_dir=output)
            validate_knowledge_artifacts(output)
            with (output / "chunks.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                validate_knowledge_artifacts(output)


if __name__ == "__main__":
    unittest.main()
