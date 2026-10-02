"""Tests for the runtime-independent evaluation dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agentrec.evaluation import (
    EvaluationCaseLoadError,
    EvaluationManifestError,
    load_cases,
    load_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data/evaluation/curated_v1_pilot"


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = DATASET_DIR / "manifest.json"
        cls.cases_path = DATASET_DIR / "cases.jsonl"
        cls.manifest = load_manifest(cls.manifest_path)
        cls.cases = load_cases(cls.cases_path)

    def test_pilot_case_count_matches_manifest(self) -> None:
        self.assertEqual(len(self.cases), 20)
        self.assertEqual(len(self.cases), self.manifest["case_count"])

    def test_case_ids_are_unique_and_source_order_is_preserved(self) -> None:
        case_ids = tuple(case.case_id for case in self.cases)
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertEqual(case_ids[0], "goal_zh_single_001")
        self.assertEqual(case_ids[-1], "e2e_fallback_001")

    def test_track_information_comes_from_tags(self) -> None:
        counts = {
            track: sum(track in case.tags for case in self.cases)
            for track in ("track_a", "track_b", "track_c")
        }
        self.assertEqual(counts, self.manifest["track_counts"])
        self.assertTrue(all("track_a" in case.tags for case in self.cases[:9]))

    def test_invalid_manifest_json_fails_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"case_count":', encoding="utf-8")
            with self.assertRaisesRegex(EvaluationManifestError, r"line 1, column"):
                load_manifest(path)

    def test_invalid_case_json_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n{not-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationCaseLoadError, r"line 2"):
                load_cases(path)

    def test_invalid_case_schema_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                json.dumps({"case_id": "missing-required-fields"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                EvaluationCaseLoadError,
                r"Invalid EvaluationCase schema.*line 1",
            ):
                load_cases(path)

    def test_duplicate_case_id_is_rejected(self) -> None:
        first_case = next(
            line
            for line in self.cases_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(first_case + "\n" + first_case + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                EvaluationCaseLoadError,
                r"Duplicate case_id.*line 2.*line 1",
            ):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
