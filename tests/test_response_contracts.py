"""Contract tests for V2-09.6 grounded response data models."""

from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from src.agentrec.domain import ItemSource
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.replanning import CandidateStatusSummary
from src.agentrec.response import (
    ConflictDecisionSummary,
    ConflictReason,
    ConflictResponseContext,
    EvidenceReference,
    FinalResponseResult,
    ProductDecisionSummary,
    ReadyResponseContext,
    ResponseKind,
    VerifiedConstraintClaim,
)
from src.agentrec.verification import ConstraintVerificationStatus


def evidence(chunk_id: str = "chunk-1") -> EvidenceReference:
    return EvidenceReference(
        chunk_id=chunk_id,
        chunk_type=KnowledgeChunkType.FEATURES,
        excerpt="Supports HDMI output.",
    )


def claim(**changes: object) -> VerifiedConstraintClaim:
    values = {
        "requirement_id": "dock",
        "parent_asin": "ASIN-1",
        "original_constraint": "HDMI",
        "canonical_constraint": "hdmi",
        "status": ConstraintVerificationStatus.SUPPORTED,
        "supporting_evidence": (evidence(),),
    }
    values.update(changes)
    return VerifiedConstraintClaim(**values)


def product(**changes: object) -> ProductDecisionSummary:
    values = {
        "requirement_id": "dock",
        "category": "USB Hubs",
        "parent_asin": "ASIN-1",
        "title": "USB-C Dock",
        "price": 99.99,
        "quantity": 1,
        "source": ItemSource.HYBRID,
        "selection_rationale": "selected_by_rank_policy",
        "verified_claims": (claim(),),
    }
    values.update(changes)
    return ProductDecisionSummary(**values)


def counts(*, eligible: int = 0) -> CandidateStatusSummary:
    return CandidateStatusSummary(
        total_candidates=2,
        eligible_count=eligible,
        unverified_count=2 - eligible,
        ineligible_count=0,
        supported_constraint_count=1,
        contradicted_constraint_count=0,
        unknown_constraint_count=1,
    )


def no_candidates(**changes: object) -> ConflictDecisionSummary:
    values = {
        "requirement_id": "dock",
        "category": "USB Hubs",
        "required_features": ("HDMI", "USB-C"),
        "reason": ConflictReason.NO_RECOMMENDATION_CANDIDATES,
        "replan_attempts_performed": 0,
        "candidate_pool_sizes": (5,),
    }
    values.update(changes)
    return ConflictDecisionSummary(**values)


def exhaustion(**changes: object) -> ConflictDecisionSummary:
    values = {
        "requirement_id": "dock",
        "category": "Single Board Computers",
        "required_features": ("HDMI", "USB-C"),
        "reason": ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
        "replan_attempts_performed": 1,
        "candidate_pool_sizes": (5, 10),
        "verification_status_counts": counts(),
        "failed_constraints": ("USB-C",),
        "unknown_constraints": ("USB-C",),
    }
    values.update(changes)
    return ConflictDecisionSummary(**values)


def allocation_exhausted(**changes: object) -> ConflictDecisionSummary:
    values = {
        "requirement_id": "mouse",
        "category": "Mouse",
        "required_features": (),
        "reason": ConflictReason.GOAL_ALLOCATION_EXHAUSTED,
        "replan_attempts_performed": 0,
        "candidate_pool_sizes": (),
    }
    values.update(changes)
    return ConflictDecisionSummary(**values)


class ResponseContractTests(unittest.TestCase):
    def test_models_are_frozen_and_forbid_extra_fields(self) -> None:
        value = evidence()
        with self.assertRaises(ValidationError):
            value.excerpt = "changed"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            EvidenceReference(
                chunk_id="x",
                chunk_type="features",
                excerpt="x",
                unexpected=True,
            )

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceReference(chunk_id=" ", chunk_type="features", excerpt="text")
        with self.assertRaises(ValidationError):
            FinalResponseResult(
                kind=ResponseKind.READY,
                text=" ",
                decision_summary=(product(),),
            )

    def test_invalid_money_and_non_finite_values_are_rejected(self) -> None:
        for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(price=value), self.assertRaises(ValidationError):
                product(price=value)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(total=value), self.assertRaises(ValidationError):
                ReadyResponseContext(
                    plan_id="plan",
                    plan_version=1,
                    currency="USD",
                    products=(product(),),
                    total_spent=value,
                    remaining_budget=1.0,
                )
            with self.subTest(remaining=value), self.assertRaises(ValidationError):
                ConflictResponseContext(
                    plan_id="plan",
                    plan_version=0,
                    currency="USD",
                    decision=no_candidates(),
                    total_spent=0.0,
                    remaining_budget=value,
                )

    def test_verified_claim_requires_supported_unique_nonempty_evidence(self) -> None:
        self.assertEqual(claim().status, ConstraintVerificationStatus.SUPPORTED)
        for status in (
            ConstraintVerificationStatus.UNKNOWN,
            ConstraintVerificationStatus.CONTRADICTED,
        ):
            with self.subTest(status=status), self.assertRaises(ValidationError):
                claim(status=status)
        with self.assertRaises(ValidationError):
            claim(supporting_evidence=())
        with self.assertRaises(ValidationError):
            claim(supporting_evidence=(evidence(), evidence()))

    def test_product_summary_has_only_response_safe_fields(self) -> None:
        value = product()
        self.assertEqual(value.price, 99.99)
        self.assertNotIn("item_index", type(value).model_fields)
        self.assertNotIn("score", type(value).model_fields)
        self.assertNotIn("similarity_score", type(value).model_fields)

    def test_ready_context_requires_nonempty_unique_products(self) -> None:
        with self.assertRaises(ValidationError):
            ReadyResponseContext(
                plan_id="plan",
                plan_version=1,
                currency="usd",
                products=(),
                total_spent=0.0,
                remaining_budget=500.0,
            )
        first = product()
        for duplicate in (
            product(parent_asin="ASIN-2"),
            product(requirement_id="mouse"),
        ):
            with self.subTest(duplicate=duplicate), self.assertRaises(ValidationError):
                ReadyResponseContext(
                    plan_id="plan",
                    plan_version=2,
                    currency="usd",
                    products=(first, duplicate),
                    total_spent=199.98,
                    remaining_budget=300.02,
                )
        valid = ReadyResponseContext(
            plan_id="plan",
            plan_version=1,
            currency="usd",
            products=(first,),
            total_spent=99.99,
            remaining_budget=400.01,
        )
        self.assertEqual(valid.currency, "USD")

    def test_no_candidate_conflict_shape(self) -> None:
        self.assertEqual(no_candidates().candidate_pool_sizes, (5,))
        self.assertEqual(
            no_candidates(
                replan_attempts_performed=1,
                candidate_pool_sizes=(5, 10),
            ).replan_attempts_performed,
            1,
        )
        with self.assertRaises(ValidationError):
            no_candidates(verification_status_counts=counts())
        with self.assertRaises(ValidationError):
            no_candidates(failed_constraints=("HDMI",))
        with self.assertRaises(ValidationError):
            no_candidates(replan_attempts_performed=1, candidate_pool_sizes=(5,))

    def test_exhaustion_conflict_shape(self) -> None:
        self.assertEqual(exhaustion().candidate_pool_sizes, (5, 10))
        with self.assertRaises(ValidationError):
            exhaustion(replan_attempts_performed=0)
        with self.assertRaises(ValidationError):
            exhaustion(candidate_pool_sizes=(10,))
        with self.assertRaises(ValidationError):
            exhaustion(verification_status_counts=None)
        with self.assertRaises(ValidationError):
            exhaustion(verification_status_counts=counts(eligible=1))
        with self.assertRaises(ValidationError):
            exhaustion(failed_constraints=())

    def test_goal_allocation_exhausted_conflict_shape(self) -> None:
        value = allocation_exhausted()
        self.assertEqual(value.candidate_pool_sizes, ())
        self.assertTrue(value.user_action_required)
        invalid = (
            {"candidate_pool_sizes": (5,)},
            {"replan_attempts_performed": 1},
            {"verification_status_counts": counts()},
            {"failed_constraints": ("HDMI",)},
            {"unknown_constraints": ("HDMI",)},
            {"contradicted_constraints": ("HDMI",)},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                allocation_exhausted(**changes)
        for factory in (no_candidates, exhaustion):
            with self.subTest(factory=factory), self.assertRaises(ValidationError):
                factory(candidate_pool_sizes=())

    def test_final_response_result_enforces_kind_summary_alignment(self) -> None:
        ready = FinalResponseResult(
            kind=ResponseKind.READY,
            text="Ready.",
            decision_summary=(product(),),
        )
        conflict = FinalResponseResult(
            kind=ResponseKind.CONFLICT,
            text="Conflict.",
            decision_summary=no_candidates(),
        )
        self.assertIsInstance(ready.decision_summary, tuple)
        self.assertIsInstance(conflict.decision_summary, ConflictDecisionSummary)
        with self.assertRaises(ValidationError):
            FinalResponseResult(
                kind=ResponseKind.READY,
                text="Wrong.",
                decision_summary=no_candidates(),
            )
        with self.assertRaises(ValidationError):
            FinalResponseResult(
                kind=ResponseKind.CONFLICT,
                text="Wrong.",
                decision_summary=(product(),),
            )
        with self.assertRaises(ValidationError):
            FinalResponseResult(
                kind=ResponseKind.READY,
                text="Wrong.",
                decision_summary=(),
            )


if __name__ == "__main__":
    unittest.main()
