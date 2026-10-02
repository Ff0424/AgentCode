"""Tests for runtime-independent evaluation adapter routing."""

from __future__ import annotations

from pathlib import Path
import unittest

from agentrec.evaluation import (
    EvaluationAdapterRouter,
    EvaluationResult,
    ExpectedTerminalStatus,
    load_cases,
    resolve_track,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "data/evaluation/curated_v1_pilot/cases.jsonl"


class RecordingAdapter:
    def __init__(self) -> None:
        self.case_ids: list[str] = []

    def execute(self, case):
        self.case_ids.append(case.case_id)
        return EvaluationResult(
            case_id=case.case_id,
            execution_status=ExpectedTerminalStatus.READY,
        )


class EvaluationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(CASES_PATH)

    def _case_for_track(self, track: str):
        return next(case for case in self.cases if track in case.tags)

    def test_track_a_is_resolved(self) -> None:
        self.assertEqual(resolve_track(self._case_for_track("track_a")), "track_a")

    def test_track_b_is_resolved(self) -> None:
        self.assertEqual(resolve_track(self._case_for_track("track_b")), "track_b")

    def test_track_c_is_resolved(self) -> None:
        self.assertEqual(resolve_track(self._case_for_track("track_c")), "track_c")

    def test_missing_track_is_rejected(self) -> None:
        case = self._case_for_track("track_a").model_copy(
            update={"tags": ("domain_llm", "language_zh")}
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_track(case)

    def test_multiple_tracks_are_rejected(self) -> None:
        case = self._case_for_track("track_a").model_copy(
            update={"tags": ("track_a", "track_b")}
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_track(case)

    def test_router_dispatches_to_resolved_track(self) -> None:
        router = EvaluationAdapterRouter()
        track_a = RecordingAdapter()
        track_b = RecordingAdapter()
        router.register("track_a", track_a)
        router.register("track_b", track_b)

        case = self._case_for_track("track_b")
        result = router.execute(case)

        self.assertEqual(result.case_id, case.case_id)
        self.assertEqual(track_a.case_ids, [])
        self.assertEqual(track_b.case_ids, [case.case_id])


if __name__ == "__main__":
    unittest.main()
