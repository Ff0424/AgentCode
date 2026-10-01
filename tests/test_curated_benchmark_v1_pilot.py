"""Lightweight integrity tests for the Curated Benchmark V1 Pilot dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

from agentrec.evaluation import (
    AdjudicationStatus,
    EvaluationCase,
    EvaluationSplit,
    ExpectedGroundingStatus,
    ExpectedTerminalStatus,
    ReplanExpectation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPOSITORY_ROOT / "data" / "evaluation" / "curated_v1_pilot"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
CASES_PATH = DATASET_DIR / "cases.jsonl"

EXPECTED_CASE_IDS = {
    "goal_zh_single_001",
    "goal_en_multi_001",
    "goal_mixed_multi_001",
    "goal_zh_quantity_001",
    "goal_en_missing_budget_001",
    "goal_zh_ambiguous_budget_001",
    "goal_en_soft_pref_001",
    "goal_mixed_wrapper_001",
    "goal_zh_unresolved_001",
    "grounding_wrappers_001",
    "grounding_unresolved_001",
    "allocation_multi_001",
    "tool_args_budget_001",
    "pipeline_identity_001",
    "e2e_ready_001",
    "e2e_zero_candidate_001",
    "e2e_unknown_replan_001",
    "e2e_contradicted_001",
    "e2e_replan_recover_001",
    "e2e_fallback_001",
}


def _load_raw_lines() -> tuple[str, ...]:
    return tuple(
        line
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _load_cases() -> tuple[EvaluationCase, ...]:
    return tuple(EvaluationCase.model_validate_json(line) for line in _load_raw_lines())


class CuratedBenchmarkV1PilotTests(unittest.TestCase):
    def test_required_dataset_files_and_manifest(self) -> None:
        self.assertTrue(MANIFEST_PATH.is_file())
        self.assertTrue(CASES_PATH.is_file())
        self.assertTrue((DATASET_DIR / "README.md").is_file())
        self.assertTrue((DATASET_DIR / "annotation_guidelines.md").is_file())
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["case_count"], 20)
        self.assertEqual(manifest["split"], "development")
        self.assertEqual(
            manifest["track_counts"],
            {"track_a": 9, "track_b": 5, "track_c": 6},
        )
        self.assertFalse(manifest["blind_test"])

    def test_jsonl_has_exactly_approved_unique_cases(self) -> None:
        raw_lines = _load_raw_lines()
        parsed_objects = tuple(json.loads(line) for line in raw_lines)
        identifiers = tuple(value["case_id"] for value in parsed_objects)
        self.assertEqual(len(raw_lines), 20)
        self.assertEqual(len(set(identifiers)), 20)
        self.assertEqual(set(identifiers), EXPECTED_CASE_IDS)

    def test_every_case_validates_and_round_trips(self) -> None:
        for case in _load_cases():
            with self.subTest(case_id=case.case_id):
                restored = EvaluationCase.model_validate_json(case.model_dump_json())
                self.assertEqual(restored, case)

    def test_all_cases_are_pending_development_annotations(self) -> None:
        for case in _load_cases():
            with self.subTest(case_id=case.case_id):
                self.assertIs(case.split, EvaluationSplit.DEVELOPMENT)
                self.assertIs(
                    case.annotation_metadata.adjudication_status,
                    AdjudicationStatus.PENDING,
                )

    def test_track_counts_match_approved_matrix(self) -> None:
        counts = {"track_a": 0, "track_b": 0, "track_c": 0}
        for case in _load_cases():
            matching = tuple(tag for tag in case.tags if tag in counts)
            self.assertEqual(matching, matching[:1], case.case_id)
            self.assertEqual(len(matching), 1, case.case_id)
            counts[matching[0]] += 1
        self.assertEqual(counts, {"track_a": 9, "track_b": 5, "track_c": 6})

    def test_language_coverage_exists(self) -> None:
        tags = {tag for case in _load_cases() for tag in case.tags}
        self.assertTrue({"language_zh", "language_en", "language_mixed"} <= tags)

    def test_business_outcome_coverage_exists(self) -> None:
        statuses = {
            status
            for case in _load_cases()
            for status in case.expected_execution.allowed_terminal_statuses
        }
        self.assertTrue(
            {
                ExpectedTerminalStatus.READY,
                ExpectedTerminalStatus.CLARIFICATION_REQUIRED,
                ExpectedTerminalStatus.CONFLICT,
            }
            <= statuses
        )

    def test_quantity_unresolved_and_replan_coverage_exists(self) -> None:
        cases = _load_cases()
        self.assertTrue(
            any(
                proposal.quantity > 1
                for case in cases
                for proposal in case.expected_goal.ordered_requirement_proposals
            )
        )
        self.assertTrue(
            any(
                grounding.expected_status is ExpectedGroundingStatus.UNRESOLVED
                for case in cases
                for grounding in case.expected_grounding
            )
        )
        replans = {case.expected_execution.replan_expectation for case in cases}
        self.assertIn(ReplanExpectation.EXPECTED_TO_RECOVER, replans)
        self.assertIn(ReplanExpectation.EXPECTED_TO_EXHAUST, replans)

    def test_dataset_contains_no_secrets_or_machine_absolute_paths(self) -> None:
        contents = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                MANIFEST_PATH,
                CASES_PATH,
                DATASET_DIR / "README.md",
                DATASET_DIR / "annotation_guidelines.md",
            )
        )
        forbidden_patterns = (
            r"\bsk-[A-Za-z0-9_-]+",
            r"\bAKIA[A-Z0-9]+",
            r"(?i)api[_-]?key\s*[:=]",
            r"(?i)[a-z]:[\\/][^\s]+",
            r"(?<!\w)/(?:home|users|root|opt|srv|mnt)/[^\s]+",
        )
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, contents))

    def test_import_and_load_are_lightweight(self) -> None:
        code = (
            "import sys; from pathlib import Path; "
            "from agentrec.evaluation import EvaluationCase; "
            f"p=Path({str(CASES_PATH)!r}); "
            "cases=[EvaluationCase.model_validate_json(x) for x in "
            "p.read_text(encoding='utf-8').splitlines() if x.strip()]; "
            "assert len(cases)==20; "
            "blocked=('torch','transformers','FlagEmbedding','fastapi'); "
            "assert not any(name in sys.modules for name in blocked)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
