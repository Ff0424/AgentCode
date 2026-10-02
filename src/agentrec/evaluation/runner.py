"""Minimal ordered runner for curated AgentRec evaluation cases.

This module coordinates case execution only. Metric calculation, artifact
inspection, ground-truth mutation, and production runtime construction remain
outside the runner boundary.
"""

from __future__ import annotations

from .adapters import EvaluationAdapterRouter
from .contracts import (
    EvaluationCase,
    EvaluationLayer,
    EvaluationResult,
    ExpectedTerminalStatus,
    FailureCategory,
    FailureCode,
)


class CuratedBenchmarkRunner:
    """Execute evaluation cases in order through an injected adapter router."""

    def __init__(self, adapter_router: EvaluationAdapterRouter) -> None:
        if not isinstance(adapter_router, EvaluationAdapterRouter):
            raise TypeError("adapter_router must be an EvaluationAdapterRouter.")
        self._adapter_router = adapter_router

    def run_case(self, case: EvaluationCase) -> EvaluationResult:
        """Execute one case and isolate all adapter/router failures."""

        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase.")
        try:
            return self._adapter_router.execute(case)
        except Exception:
            # Do not persist exception text: evaluation results are safe,
            # stable contracts rather than diagnostic traceback containers.
            return EvaluationResult(
                case_id=case.case_id,
                execution_status=ExpectedTerminalStatus.ERROR,
                earliest_failure_layer=EvaluationLayer.L0_CONTRACT_VALIDITY,
                failure_category=FailureCategory.INFRASTRUCTURE_FAILURE,
                failure_code=FailureCode.INTERNAL_ERROR,
            )

    def run_cases(
        self,
        cases: tuple[EvaluationCase, ...],
    ) -> tuple[EvaluationResult, ...]:
        """Execute cases sequentially while preserving their input order."""

        if not isinstance(cases, tuple):
            raise TypeError("cases must be a tuple of EvaluationCase values.")
        return tuple(self.run_case(case) for case in cases)


__all__ = ["CuratedBenchmarkRunner"]
