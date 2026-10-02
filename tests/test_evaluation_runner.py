"""Tests for the minimal curated benchmark runner."""

from __future__ import annotations

from pathlib import Path
import unittest

from agentrec.evaluation import (
    CuratedBenchmarkRunner,
    EvaluationAdapterRouter,
    EvaluationLayer,
    EvaluationResult,
    ExpectedTerminalStatus,
    FailureCategory,
    FailureCode,
    load_cases,
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


class FailingAdapter:
    def execute(self, case):
        raise RuntimeError("private adapter failure")


class WrongCaseIdAdapter:
    def execute(self, case):
        return EvaluationResult(
            case_id=f"wrong-{case.case_id}",
            execution_status=ExpectedTerminalStatus.READY,
        )


class EvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(CASES_PATH)

    def _cases_for_track(self, track: str, count: int = 1):
        selected = tuple(case for case in self.cases if track in case.tags)
        self.assertGreaterEqual(len(selected), count)
        return selected[:count]

    def _runner(self, track: str, adapter) -> CuratedBenchmarkRunner:
        router = EvaluationAdapterRouter()
        router.register(track, adapter)
        return CuratedBenchmarkRunner(router)

    def test_single_case_executes_normally(self) -> None:
        case = self._cases_for_track("track_a")[0]
        adapter = RecordingAdapter()

        result = self._runner("track_a", adapter).run_case(case)

        self.assertEqual(result.case_id, case.case_id)
        self.assertIs(result.execution_status, ExpectedTerminalStatus.READY)
        self.assertEqual(adapter.case_ids, [case.case_id])

    def test_multiple_cases_preserve_input_order(self) -> None:
        cases = self._cases_for_track("track_a", count=3)
        adapter = RecordingAdapter()

        results = self._runner("track_a", adapter).run_cases(cases)

        expected_ids = tuple(case.case_id for case in cases)
        self.assertEqual(tuple(result.case_id for result in results), expected_ids)
        self.assertEqual(tuple(adapter.case_ids), expected_ids)

    def test_adapter_exception_becomes_error_result(self) -> None:
        case = self._cases_for_track("track_b")[0]

        result = self._runner("track_b", FailingAdapter()).run_case(case)

        self.assertEqual(result.case_id, case.case_id)
        self.assertIs(result.execution_status, ExpectedTerminalStatus.ERROR)
        self.assertIs(
            result.earliest_failure_layer,
            EvaluationLayer.L0_CONTRACT_VALIDITY,
        )
        self.assertIs(
            result.failure_category,
            FailureCategory.INFRASTRUCTURE_FAILURE,
        )
        self.assertIs(result.failure_code, FailureCode.INTERNAL_ERROR)
        self.assertEqual(result.component_outputs, ())
        self.assertEqual(result.metric_results, ())

    def test_wrong_case_id_becomes_error_result(self) -> None:
        case = self._cases_for_track("track_c")[0]

        result = self._runner("track_c", WrongCaseIdAdapter()).run_case(case)

        self.assertEqual(result.case_id, case.case_id)
        self.assertIs(result.execution_status, ExpectedTerminalStatus.ERROR)
        self.assertIs(result.failure_code, FailureCode.INTERNAL_ERROR)


if __name__ == "__main__":
    unittest.main()
