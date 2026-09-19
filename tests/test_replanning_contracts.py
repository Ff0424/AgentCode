"""Deterministic contract and policy tests for V2-09.5."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.domain import ShoppingRequirement
from src.agentrec.replanning import (
    BoundedReplanPolicy,
    FailureDiagnosisService,
    FailureRecoverability,
    FailureType,
    ReplanAction,
    VerificationFailurePattern,
)
from src.agentrec.verification import (
    CandidateVerification,
    CandidateVerificationStatus,
    ConstraintVerification,
    ConstraintVerificationReason,
    ConstraintVerificationStatus,
    RequirementVerification,
)


def constraint(name: str, status: ConstraintVerificationStatus) -> ConstraintVerification:
    if status is ConstraintVerificationStatus.UNKNOWN:
        return ConstraintVerification(
            constraint=name, canonical_constraint=name.casefold(), status=status,
            reason=ConstraintVerificationReason.INSUFFICIENT_EVIDENCE,
        )
    if status is ConstraintVerificationStatus.CONTRADICTED:
        return ConstraintVerification(
            constraint=name, canonical_constraint=name.casefold(), status=status,
            reason=ConstraintVerificationReason.EXPLICIT_CONTRADICTION,
            contradicting_chunk_ids=(f"{name}-negative",),
        )
    return ConstraintVerification(
        constraint=name, canonical_constraint=name.casefold(), status=status,
        reason=ConstraintVerificationReason.EXPLICIT_SUPPORT,
        supporting_chunk_ids=(f"{name}-positive",),
    )


def verification(statuses: tuple[CandidateVerificationStatus, ...]) -> RequirementVerification:
    candidates = []
    for index, status in enumerate(statuses):
        constraint_status = (
            ConstraintVerificationStatus.UNKNOWN
            if status is CandidateVerificationStatus.UNVERIFIED
            else ConstraintVerificationStatus.CONTRADICTED
        )
        candidates.append(CandidateVerification(
            item_index=index, parent_asin=f"P-{index}", status=status,
            constraints=(constraint("HDMI", constraint_status),),
            verified_at_plan_version=0,
        ))
    return RequirementVerification(
        plan_id="plan", requirement_id="dock", verified_at_plan_version=0,
        candidates=tuple(candidates),
    )


class ReplanningContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diagnosis = FailureDiagnosisService()
        self.policy = BoundedReplanPolicy()
        self.requirement = ShoppingRequirement(
            requirement_id="dock", category="Hubs", max_budget=500,
            required_features=("HDMI", "USB-C"),
            soft_preferences=("laptop use",),
        )

    def diagnose(self, statuses, attempt=0):
        return self.diagnosis.diagnose_zero_eligible(
            verification=verification(statuses), plan_id="plan",
            requirement_id="dock", plan_version=0, attempt=attempt,
        )

    def test_diagnosis_patterns_are_deterministic(self):
        cases = (
            ((CandidateVerificationStatus.UNVERIFIED,) * 2,
             VerificationFailurePattern.ALL_CANDIDATES_UNVERIFIED),
            ((CandidateVerificationStatus.INELIGIBLE,) * 2,
             VerificationFailurePattern.ALL_CANDIDATES_INELIGIBLE),
            ((CandidateVerificationStatus.UNVERIFIED,
              CandidateVerificationStatus.INELIGIBLE),
             VerificationFailurePattern.MIXED_VERIFICATION_FAILURE),
        )
        for statuses, expected in cases:
            with self.subTest(expected=expected):
                first = self.diagnose(statuses)
                second = self.diagnose(statuses)
                self.assertEqual(first, second)
                self.assertEqual(first.verification_pattern, expected)
                self.assertEqual(first.failure_type, FailureType.NO_ELIGIBLE_CANDIDATES)
                self.assertEqual(first.recoverability, FailureRecoverability.RECOVERABLE)

    def test_attempt_one_is_exhausted_and_non_recoverable(self):
        result = self.diagnose((CandidateVerificationStatus.UNVERIFIED,), attempt=1)
        self.assertEqual(result.failure_type, FailureType.REPLAN_ATTEMPTS_EXHAUSTED)
        self.assertEqual(result.recoverability, FailureRecoverability.NON_RECOVERABLE)

    def test_policy_expands_only_five_to_ten_and_preserves_hard_constraints(self):
        diagnosis = self.diagnose((CandidateVerificationStatus.UNVERIFIED,))
        directive = self.policy.decide(
            diagnosis=diagnosis, requirement=self.requirement, current_top_k=5,
            current_attempt=0, max_replan_attempts=1,
        )
        self.assertEqual(directive.action, ReplanAction.EXPAND_CANDIDATE_POOL)
        self.assertEqual((directive.previous_top_k, directive.next_top_k), (5, 10))
        self.assertEqual(directive.preserved_category, self.requirement.category)
        self.assertEqual(directive.preserved_max_budget, self.requirement.max_budget)
        self.assertEqual(
            directive.preserved_required_features, self.requirement.required_features
        )

    def test_attempt_one_policy_stops(self):
        diagnosis = self.diagnose((CandidateVerificationStatus.INELIGIBLE,), attempt=1)
        directive = self.policy.decide(
            diagnosis=diagnosis, requirement=self.requirement, current_top_k=10,
            current_attempt=1, max_replan_attempts=1,
        )
        self.assertEqual(directive.action, ReplanAction.STOP_CONFLICT)
        self.assertIsNone(directive.next_top_k)

    def test_contracts_are_frozen_and_reject_extra_fields(self):
        diagnosis = self.diagnose((CandidateVerificationStatus.UNVERIFIED,))
        with self.assertRaises(ValidationError):
            diagnosis.attempt = 1
        payload = diagnosis.model_dump()
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            type(diagnosis).model_validate(payload)

    def test_stale_identity_and_eligible_input_are_rejected(self):
        with self.assertRaises(ValueError):
            self.diagnosis.diagnose_zero_eligible(
                verification=verification((CandidateVerificationStatus.UNVERIFIED,)),
                plan_id="wrong", requirement_id="dock", plan_version=0, attempt=0,
            )
        supported = CandidateVerification(
            item_index=1, parent_asin="P", status=CandidateVerificationStatus.ELIGIBLE,
            constraints=(constraint("HDMI", ConstraintVerificationStatus.SUPPORTED),),
            verified_at_plan_version=0,
        )
        with self.assertRaises(ValueError):
            self.diagnosis.diagnose_zero_eligible(
                verification=RequirementVerification(
                    plan_id="plan", requirement_id="dock", verified_at_plan_version=0,
                    candidates=(supported,),
                ),
                plan_id="plan", requirement_id="dock", plan_version=0, attempt=0,
            )


if __name__ == "__main__":
    unittest.main()
