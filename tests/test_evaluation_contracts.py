"""Deterministic tests for the standalone evaluation data contracts."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from pydantic import ValidationError

from agentrec.evaluation import (
    AdjudicationStatus,
    AllocationPreference,
    AnnotationMetadata,
    ClarificationExpectation,
    EvaluationCase,
    EvaluationLayer,
    EvaluationResult,
    EvaluationSplit,
    ExpectedAllocationPreference,
    ExpectedExecution,
    ExpectedGoal,
    ExpectedGroundingStatus,
    ExpectedRequirementProposal,
    ExpectedTerminalStatus,
    FailureCategory,
    FailureCode,
    GroundingExpectation,
    MetricResult,
    ReplanExpectation,
    ResponseRequirements,
    RuntimeProfile,
)


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        model_identifier="fixture-model",
        provider="fake-provider",
        adapter="structured-goal",
        user_identity_policy="fixed-evaluation-user",
    )


def _metadata() -> AnnotationMetadata:
    return AnnotationMetadata(
        annotator="human-reviewer",
        annotation_version="1.0",
        adjudication_status=AdjudicationStatus.APPROVED,
        provenance="manual independent annotation",
    )


def _minimal_case(**updates) -> EvaluationCase:
    values = {
        "case_id": "case-001",
        "schema_version": "1.0",
        "split": EvaluationSplit.TEST,
        "locale": "en-US",
        "tags": ("single",),
        "user_input": "I need a dock with a budget of 200 dollars.",
        "runtime_profile": _runtime_profile(),
        "expected_goal": ExpectedGoal(
            total_budget=200,
            ordered_requirement_proposals=(
                ExpectedRequirementProposal(category="Docking Stations"),
            ),
            clarification=ClarificationExpectation(needed=False),
        ),
        "expected_grounding": (),
        "expected_execution": ExpectedExecution(
            allowed_terminal_statuses=(ExpectedTerminalStatus.READY,)
        ),
        "response_requirements": ResponseRequirements(
            required_facts=("plan ready",),
            forbidden_leakage=("item_index",),
        ),
        "annotation_metadata": _metadata(),
    }
    values.update(updates)
    return EvaluationCase(**values)


class EvaluationContractTests(unittest.TestCase):
    def test_minimal_valid_case(self) -> None:
        case = _minimal_case()
        self.assertEqual(case.case_id, "case-001")
        self.assertEqual(case.expected_goal.total_budget, 200.0)

    def test_multi_requirement_order_and_quantity_are_preserved(self) -> None:
        goal = ExpectedGoal(
            total_budget=500,
            ordered_requirement_proposals=(
                ExpectedRequirementProposal(category="Docking Stations"),
                ExpectedRequirementProposal(category="Mice", quantity=2),
                ExpectedRequirementProposal(category="Headphones"),
            ),
            allocation_preferences=(
                ExpectedAllocationPreference(
                    target_index=1,
                    preference=AllocationPreference.SAVE_MORE,
                ),
                ExpectedAllocationPreference(
                    target_index=2,
                    preference=AllocationPreference.ALLOCATE_MORE,
                ),
            ),
            clarification=ClarificationExpectation(needed=False),
        )
        case = _minimal_case(expected_goal=goal)
        self.assertEqual(
            tuple(value.category for value in case.expected_goal.ordered_requirement_proposals),
            ("Docking Stations", "Mice", "Headphones"),
        )
        self.assertEqual(case.expected_goal.ordered_requirement_proposals[1].quantity, 2)
        self.assertEqual(case.expected_goal.allocation_preferences[0].target_index, 1)

    def test_clarification_case_can_omit_budget_and_proposals(self) -> None:
        case = _minimal_case(
            expected_goal=ExpectedGoal(
                clarification=ClarificationExpectation(
                    needed=True,
                    question="What is your total budget?",
                )
            ),
            expected_execution=ExpectedExecution(
                allowed_terminal_statuses=(
                    ExpectedTerminalStatus.CLARIFICATION_REQUIRED,
                )
            ),
        )
        self.assertIsNone(case.expected_goal.total_budget)
        self.assertFalse(case.expected_goal.ordered_requirement_proposals)

    def test_grounded_constraint_requires_canonical_value(self) -> None:
        result = GroundingExpectation(
            original_constraint="supports HDMI",
            expected_canonical_constraint="HDMI",
            expected_status=ExpectedGroundingStatus.GROUNDED,
        )
        self.assertEqual(result.expected_canonical_constraint, "HDMI")
        with self.assertRaises(ValidationError):
            GroundingExpectation(
                original_constraint="supports HDMI",
                expected_status=ExpectedGroundingStatus.GROUNDED,
            )

    def test_unresolved_constraint_forbids_canonical_value(self) -> None:
        result = GroundingExpectation(
            original_constraint="quantum transport",
            expected_status=ExpectedGroundingStatus.UNRESOLVED,
        )
        self.assertIsNone(result.expected_canonical_constraint)
        with self.assertRaises(ValidationError):
            GroundingExpectation(
                original_constraint="quantum transport",
                expected_canonical_constraint="Quantum",
                expected_status=ExpectedGroundingStatus.UNRESOLVED,
            )

    def test_invalid_quantity_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExpectedRequirementProposal(category="Mice", quantity=0)

    def test_invalid_money_is_rejected(self) -> None:
        for value in (-1, 0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ExpectedGoal(
                    total_budget=value,
                    ordered_requirement_proposals=(
                        ExpectedRequirementProposal(category="Mice"),
                    ),
                    clarification=ClarificationExpectation(needed=False),
                )

    def test_duplicate_tags_are_stably_deduplicated(self) -> None:
        case = _minimal_case(tags=("P0", "p0", "budget"))
        self.assertEqual(case.tags, ("P0", "budget"))

    def test_evaluation_result_failure_taxonomy(self) -> None:
        result = EvaluationResult(
            case_id="case-001",
            execution_status=ExpectedTerminalStatus.CLARIFICATION_REQUIRED,
            earliest_failure_layer=EvaluationLayer.L1_GOAL_UNDERSTANDING_GROUNDING,
            failure_category=FailureCategory.BUSINESS_CLARIFICATION,
            failure_code=FailureCode.GROUNDING_UNRESOLVED,
        )
        self.assertEqual(result.failure_category, FailureCategory.BUSINESS_CLARIFICATION)
        with self.assertRaises(ValidationError):
            EvaluationResult(
                case_id="case-001",
                execution_status=ExpectedTerminalStatus.ERROR,
                failure_code=FailureCode.INTERNAL_ERROR,
            )

    def test_metric_result_and_invalid_ratios(self) -> None:
        metric = MetricResult(
            metric_name="budget_compliance_rate",
            value=0.75,
            numerator=3,
            denominator=4,
            evaluation_scope="test",
        )
        self.assertEqual(metric.denominator, 4.0)
        invalid_values = (
            {"numerator": 1, "denominator": 0},
            {"numerator": -1, "denominator": 2},
            {"numerator": 3, "denominator": 2},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                MetricResult(
                    metric_name="metric",
                    value=0,
                    evaluation_scope="test",
                    **values,
                )

    def test_serialization_round_trip(self) -> None:
        original = _minimal_case(
            expected_grounding=(
                GroundingExpectation(
                    original_constraint="HDMI",
                    expected_canonical_constraint="HDMI",
                    expected_status=ExpectedGroundingStatus.GROUNDED,
                ),
            )
        )
        restored = EvaluationCase.model_validate_json(original.model_dump_json())
        self.assertEqual(restored, original)
        self.assertEqual(json.loads(restored.model_dump_json())["split"], "test")

    def test_contracts_are_frozen_and_forbid_extra_fields(self) -> None:
        case = _minimal_case()
        with self.assertRaises(ValidationError):
            case.case_id = "changed"
        with self.assertRaises(ValidationError):
            ExpectedRequirementProposal(category="Mice", prediction="not-truth")

    def test_import_is_lightweight(self) -> None:
        code = (
            "import sys; import agentrec.evaluation; "
            "blocked=('torch','transformers','FlagEmbedding','fastapi'); "
            "assert not any(name in sys.modules for name in blocked)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_replan_expectation_is_explicit(self) -> None:
        execution = ExpectedExecution(
            allowed_terminal_statuses=(ExpectedTerminalStatus.CONFLICT,),
            replan_expectation=ReplanExpectation.EXPECTED_TO_EXHAUST,
            expected_conflict_reason="replan_attempts_exhausted",
        )
        self.assertEqual(
            execution.replan_expectation,
            ReplanExpectation.EXPECTED_TO_EXHAUST,
        )


if __name__ == "__main__":
    unittest.main()
