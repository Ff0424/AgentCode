"""Deterministic regression tests for required-feature grounding."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.domain import ShoppingRequirement
from src.agentrec.evidence import (
    EvidenceCandidate,
    EvidenceSnippet,
    ProductEvidence,
    RequirementEvidence,
)
from src.agentrec.evidence.query_builder import EvidenceQueryBuilder
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.planning import (
    ConstraintGroundingStatus,
    ConstraintGroundingResult,
    DeterministicConstraintGrounder,
    GoalRequirementProposal,
    GoalToRequirementProjector,
    ShoppingGoalExtractionDecision,
    UnresolvedConstraintGroundingError,
)
from src.agentrec.tools import RecommendationToolArgs
from src.agentrec.verification import (
    ConstraintVerificationStatus,
    EvidenceConstraintVerifier,
)
from tests.fake_evidence import PROVENANCE


class ConstraintGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grounder = DeterministicConstraintGrounder()

    def test_hdmi_natural_language_wrappers_ground_canonically(self) -> None:
        for value in ("HDMI", "支持HDMI", "必须支持HDMI", "需要HDMI"):
            with self.subTest(value=value):
                result = self.grounder.ground(value)
                self.assertEqual(result.status, ConstraintGroundingStatus.GROUNDED)
                self.assertEqual(result.canonical_constraint, "HDMI")
                self.assertEqual(result.original_constraint, value)

    def test_usb_c_aliases_and_wrappers_ground_canonically(self) -> None:
        values = (
            "USB-C", "USB C", "Type-C", "Type C",
            "支持USB-C", "必须有Type-C",
        )
        for value in values:
            with self.subTest(value=value):
                result = self.grounder.ground(value)
                self.assertEqual(result.status, ConstraintGroundingStatus.GROUNDED)
                self.assertEqual(result.canonical_constraint, "USB-C")

    def test_unknown_constraints_remain_explicitly_unresolved(self) -> None:
        for value in ("支持量子传输", "一个没有注册的 feature"):
            with self.subTest(value=value):
                result = self.grounder.ground(value)
                self.assertEqual(result.status, ConstraintGroundingStatus.UNRESOLVED)
                self.assertIsNone(result.canonical_constraint)
                self.assertEqual(result.original_constraint, value)

    def test_grounding_contract_is_frozen_and_status_aligned(self) -> None:
        grounded = ConstraintGroundingResult(
            original_constraint=" HDMI ",
            canonical_constraint="HDMI",
            status=ConstraintGroundingStatus.GROUNDED,
        )
        self.assertEqual(grounded.original_constraint, "HDMI")
        with self.assertRaises(ValidationError):
            grounded.canonical_constraint = "USB-C"
        with self.assertRaises(ValidationError):
            ConstraintGroundingResult(
                original_constraint="HDMI",
                status=ConstraintGroundingStatus.GROUNDED,
            )
        with self.assertRaises(ValidationError):
            ConstraintGroundingResult(
                original_constraint="unknown",
                canonical_constraint="HDMI",
                status=ConstraintGroundingStatus.UNRESOLVED,
            )

    def test_projection_canonicalizes_before_downstream_use(self) -> None:
        for value in ("HDMI", "支持HDMI", "必须支持HDMI"):
            with self.subTest(value=value):
                projection = GoalToRequirementProjector().project(
                    ShoppingGoalExtractionDecision(
                        total_budget=200,
                        requirement_proposals=(GoalRequirementProposal(
                            category="Docking Stations",
                            required_features=(value,),
                        ),),
                    )
                )
                requirement = projection.requirements[0]
                self.assertEqual(requirement.required_features, ("HDMI",))
                recommendation_args = RecommendationToolArgs(
                    category=requirement.category,
                    required_features=requirement.required_features,
                )
                self.assertEqual(recommendation_args.required_features, ("HDMI",))
                self.assertEqual(
                    EvidenceQueryBuilder().build(requirement),
                    "Category: Docking Stations. Required product evidence: HDMI.",
                )

    def test_unknown_hard_constraint_fails_closed_before_domain_projection(self) -> None:
        decision = ShoppingGoalExtractionDecision(
            total_budget=200,
            requirement_proposals=(GoalRequirementProposal(
                category="Docking Stations",
                required_features=("支持量子传输",),
            ),),
        )
        with self.assertRaises(UnresolvedConstraintGroundingError) as caught:
            GoalToRequirementProjector().project(decision)
        self.assertEqual(
            caught.exception.results[0].status,
            ConstraintGroundingStatus.UNRESOLVED,
        )

    def test_agent_goal_entry_converts_unresolved_constraint_to_clarification(self) -> None:
        try:
            from tests.test_goal_execution_preparation import prepare, runner
        except ModuleNotFoundError as exc:
            if exc.name == "langgraph":
                self.skipTest("local environment does not provide langgraph")
            raise

        decision = ShoppingGoalExtractionDecision(
            total_budget=200,
            requirement_proposals=(GoalRequirementProposal(
                category="Docking Stations",
                required_features=("支持量子传输",),
            ),),
        )
        result = prepare(runner(decision))
        self.assertEqual(result.status.value, "clarification_required")
        self.assertIn("支持量子传输", result.clarification_question)
        self.assertIsNone(result.prepared_execution)

    def test_grounded_hdmi_is_supported_by_existing_verifier(self) -> None:
        requirement = ShoppingRequirement(
            requirement_id="dock",
            category="Docking Stations",
            required_features=self.grounder.ground_required_features(("必须支持HDMI",)),
        )
        snippet = EvidenceSnippet(
            rank=1,
            item_index=1,
            parent_asin="P-1",
            chunk_id="c-1",
            chunk_type=KnowledgeChunkType.FEATURES,
            part_index=0,
            text="supports HDMI output",
            similarity_score=0.8,
        )
        candidate = EvidenceCandidate(item_index=1, parent_asin="P-1")
        evidence = RequirementEvidence(
            plan_id="p",
            retrieved_at_plan_version=0,
            requirement_id="dock",
            query="feature evidence",
            candidates=(candidate,),
            products=(ProductEvidence(
                item_index=1,
                parent_asin="P-1",
                snippets=(snippet,),
            ),),
            provenance=PROVENANCE,
        )
        result = EvidenceConstraintVerifier().verify(
            requirement=requirement,
            evidence=evidence,
            current_plan_id="p",
            current_plan_version=0,
        )
        self.assertEqual(
            result.candidates[0].constraints[0].status,
            ConstraintVerificationStatus.SUPPORTED,
        )


if __name__ == "__main__":
    unittest.main()
