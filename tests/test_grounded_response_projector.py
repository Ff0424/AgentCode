"""Deterministic and fail-closed tests for GroundedResponseProjector."""

from __future__ import annotations

import unittest

from src.agentrec.domain import AgentState, ShoppingRequirement
from src.agentrec.evidence import EvidenceSnippet, ProductEvidence, RequirementEvidence
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.response import (
    ConflictReason,
    GroundedResponseProjectionError,
    GroundedResponseProjector,
    ProjectionErrorCode,
    ResponseKind,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.tools import RecommendationToolItem, RecommendationToolResult
from src.agentrec.verification import (
    CandidateVerificationStatus,
    ConstraintVerificationStatus,
    EvidenceConstraintVerifier,
)
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.fake_evidence import PROVENANCE, FakeEvidenceService
from tests.test_bounded_replan_workflow import (
    AttemptEvidenceService,
    PoolAwareTool,
    run as run_bounded,
)
from tests.test_shopping_workflow import FakeRecommendationTool
from tests.test_verified_workflow import MappedEvidenceService, graph as verified_graph, state as verified_state


PROJECTOR = GroundedResponseProjector()


def run_verified(*, features=("HDMI",), texts=None) -> ShoppingWorkflowState:
    mapped = texts or {
        "P-1": ("Supports HDMI output and includes USB-C port",),
        "P-2": ("Supports HDMI output and includes USB-C port",),
        "P-3": ("Supports HDMI output and includes USB-C port",),
    }
    return ShoppingWorkflowState.model_validate(
        verified_graph(
            MappedEvidenceService(mapped), EvidenceConstraintVerifier()
        ).invoke(verified_state(features=features))
    )


def run_two_requirements() -> ShoppingWorkflowState:
    service = ShoppingPlanService()
    plan = service.create_plan(
        plan_id="multi",
        user_id="user",
        currency="USD",
        total_budget=500,
        requirements=(
            ShoppingRequirement(requirement_id="dock", category="Dock", max_budget=200),
            ShoppingRequirement(requirement_id="mouse", category="Mouse", max_budget=100),
        ),
    )
    graph = build_shopping_workflow(
        FakeRecommendationTool({"Dock": 120, "Mouse": 30}),
        service,
        evidence_service=FakeEvidenceService(),
        verification_service=EvidenceConstraintVerifier(),
    )
    return ShoppingWorkflowState.model_validate(graph.invoke(ShoppingWorkflowState(
        agent_state=AgentState(user_id="user", session_id="session", shopping_plan=plan)
    )))


class FixedTextEvidenceService:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self._texts = texts

    def retrieve(self, *, plan_id, plan_version, requirement, candidates):
        products = []
        for position, candidate in enumerate(candidates):
            text = self._texts[position % len(self._texts)]
            products.append(ProductEvidence(
                item_index=candidate.item_index,
                parent_asin=candidate.parent_asin,
                snippets=(EvidenceSnippet(
                    rank=1,
                    item_index=candidate.item_index,
                    parent_asin=candidate.parent_asin,
                    chunk_id=f"{len(candidates)}-{candidate.parent_asin}",
                    chunk_type=KnowledgeChunkType.FEATURES,
                    part_index=0,
                    text=text,
                    similarity_score=0.8,
                ),),
            ))
        return RequirementEvidence(
            plan_id=plan_id,
            retrieved_at_plan_version=plan_version,
            requirement_id=requirement.requirement_id,
            query="HDMI support",
            candidates=candidates,
            products=tuple(products),
            provenance=PROVENANCE,
        )


def replace_selected_evidence(state, evidence):
    return state.model_copy(update={"selected_evidence": (evidence,)})


class ReadyProjectionTests(unittest.TestCase):
    def test_verified_product_and_multiple_features(self) -> None:
        final = run_verified(features=("HDMI", "USB-C"))
        context = PROJECTOR.project(final)
        self.assertEqual(context.kind, ResponseKind.READY)
        self.assertEqual(len(context.products[0].verified_claims), 2)
        self.assertEqual(
            tuple(x.original_constraint for x in context.products[0].verified_claims),
            ("HDMI", "USB-C"),
        )
        self.assertFalse(hasattr(context.products[0], "item_index"))
        reference = context.products[0].verified_claims[0].supporting_evidence[0]
        self.assertFalse(hasattr(reference, "similarity_score"))

    def test_no_required_features_has_no_claims(self) -> None:
        context = PROJECTOR.project(run_verified(features=()))
        self.assertEqual(context.products[0].verified_claims, ())

    def test_all_supporting_chunks_preserve_evidence_order(self) -> None:
        final = run_verified(texts={
            parent: ("Supports HDMI output", "Includes HDMI port")
            for parent in ("P-1", "P-2", "P-3")
        })
        claim = PROJECTOR.project(final).products[0].verified_claims[0]
        self.assertEqual(
            tuple(value.chunk_id for value in claim.supporting_evidence),
            ("P-1-c1", "P-1-c2"),
        )

    def test_multi_requirement_versions_and_stable_order(self) -> None:
        final = run_two_requirements()
        self.assertEqual(
            tuple(x.selected_at_plan_version for x in final.selected_evidence), (1, 2)
        )
        self.assertEqual(final.agent_state.shopping_plan.version, 2)
        context = PROJECTOR.project(final)
        self.assertEqual(
            tuple(x.requirement_id for x in context.products), ("dock", "mouse")
        )

    def test_product_facts_come_only_from_plan(self) -> None:
        final = run_verified()
        plan_item = final.agent_state.shopping_plan.selected_items[0]
        fake = final.last_tool_result.model_copy(update={
            "items": (final.last_tool_result.items[0].model_copy(update={
                "title": "TOOL OVERRIDE", "price": 1.0
            }),)
        })
        corrupted_runtime = final.model_copy(update={"last_tool_result": fake})
        product = PROJECTOR.project(corrupted_runtime).products[0]
        self.assertEqual((product.title, product.price, product.quantity), (
            plan_item.title, plan_item.price, plan_item.quantity
        ))

    def test_missing_or_ambiguous_selected_evidence_fails_closed(self) -> None:
        final = run_verified()
        for evidence in ((), (final.selected_evidence[0], final.selected_evidence[0])):
            with self.subTest(count=len(evidence)), self.assertRaises(
                GroundedResponseProjectionError
            ):
                PROJECTOR.project(final.model_copy(update={"selected_evidence": evidence}))

    def test_missing_verification_and_noneligible_candidate_fail_closed(self) -> None:
        final = run_verified()
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"selected_verifications": ()}))
        selected = final.selected_verifications[0]
        candidate = selected.candidate.model_copy(
            update={"status": CandidateVerificationStatus.UNVERIFIED}
        )
        corrupted = selected.model_copy(update={"candidate": candidate})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"selected_verifications": (corrupted,)}))

    def test_unsupported_constraint_and_missing_chunk_fail_closed(self) -> None:
        final = run_verified()
        selected = final.selected_verifications[0]
        constraint = selected.candidate.constraints[0]
        unsupported = constraint.model_copy(update={
            "status": ConstraintVerificationStatus.UNKNOWN,
            "supporting_chunk_ids": (),
        })
        candidate = selected.candidate.model_copy(update={"constraints": (unsupported,)})
        with self.assertRaises(GroundedResponseProjectionError) as captured:
            PROJECTOR.project(final.model_copy(update={
                "selected_verifications": (selected.model_copy(update={"candidate": candidate}),)
            }))
        self.assertEqual(captured.exception.code, ProjectionErrorCode.UNSUPPORTED_CLAIM)

        missing = constraint.model_copy(update={"supporting_chunk_ids": ("unknown",)})
        candidate = selected.candidate.model_copy(update={"constraints": (missing,)})
        with self.assertRaises(GroundedResponseProjectionError) as captured:
            PROJECTOR.project(final.model_copy(update={
                "selected_verifications": (selected.model_copy(update={"candidate": candidate}),)
            }))
        self.assertEqual(captured.exception.code, ProjectionErrorCode.MISSING_GROUNDING)

    def test_evidence_verification_identity_and_versions_fail_closed(self) -> None:
        final = run_verified()
        evidence = final.selected_evidence[0]
        bad_evidence = evidence.model_copy(update={"parent_asin": "WRONG"})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(replace_selected_evidence(final, bad_evidence))

        selected = final.selected_verifications[0]
        bad_candidate = selected.candidate.model_copy(update={"parent_asin": "WRONG"})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={
                "selected_verifications": (
                    selected.model_copy(update={"candidate": bad_candidate}),
                )
            }))
        bad_version = selected.model_copy(update={"selected_at_plan_version": 2})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"selected_verifications": (bad_version,)}))

    def test_plan_evaluation_mismatch_and_nonready_route_fail_closed(self) -> None:
        final = run_verified()
        bad_evaluation = final.evaluation.model_copy(update={"total_spent": 0.0})
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={"evaluation": bad_evaluation}))
        with self.assertRaises(GroundedResponseProjectionError) as captured:
            PROJECTOR.project(final.model_copy(update={"route": WorkflowRoute.ERROR}))
        self.assertEqual(
            captured.exception.code, ProjectionErrorCode.UNSUPPORTED_TERMINAL_STATE
        )

    def test_planner_rationale_and_unselected_tool_item_cannot_create_claim(self) -> None:
        final = run_verified()
        plan = final.agent_state.shopping_plan
        item = plan.selected_items[0].model_copy(update={
            "selected_reason": "Claims ImaginaryFeature and price $1"
        })
        changed_plan = plan.model_copy(update={"selected_items": (item,)})
        changed_agent = final.agent_state.model_copy(update={"shopping_plan": changed_plan})
        unselected = RecommendationToolItem(
            rank=2, item_index=999, parent_asin="UNSELECTED", title="Other",
            price=1.0, score=1.0, score_source="hybrid",
        )
        result = RecommendationToolResult(
            personalization_status="personalized", fallback_reason=None,
            returned_count=2, items=(final.last_tool_result.items[0], unselected),
        )
        context = PROJECTOR.project(final.model_copy(update={
            "agent_state": changed_agent, "last_tool_result": result
        }))
        self.assertEqual(context.products[0].parent_asin, item.parent_asin)
        self.assertNotEqual(context.products[0].parent_asin, "UNSELECTED")
        self.assertNotIn(
            "ImaginaryFeature",
            tuple(x.original_constraint for x in context.products[0].verified_claims),
        )

    def test_projection_does_not_mutate_state(self) -> None:
        final = run_verified()
        before = final.model_dump(mode="json")
        PROJECTOR.project(final)
        self.assertEqual(final.model_dump(mode="json"), before)


class ConflictProjectionTests(unittest.TestCase):
    def test_initial_empty_recommendation_has_no_fake_verification_or_retry(self) -> None:
        final = run_bounded(
            PoolAwareTool(empty=True),
            AttemptEvidenceService(success_on_expanded_pool=True),
        )
        context = PROJECTOR.project(final)
        self.assertEqual(context.decision.reason, ConflictReason.NO_RECOMMENDATION_CANDIDATES)
        self.assertEqual(context.decision.candidate_pool_sizes, (5,))
        self.assertEqual(context.decision.replan_attempts_performed, 0)
        self.assertIsNone(context.decision.verification_status_counts)
        self.assertEqual(context.decision.failed_constraints, ())

    def test_retry_exhaustion_all_unknown(self) -> None:
        final = run_bounded(
            PoolAwareTool(), AttemptEvidenceService(success_on_expanded_pool=False)
        )
        context = PROJECTOR.project(final)
        self.assertEqual(context.decision.reason, ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED)
        self.assertEqual(context.decision.candidate_pool_sizes, (5, 10))
        self.assertEqual(context.decision.unknown_constraints, ("HDMI",))
        self.assertEqual(context.decision.contradicted_constraints, ())

    def test_retry_exhaustion_all_contradicted(self) -> None:
        final = run_bounded(PoolAwareTool(), FixedTextEvidenceService(("No HDMI port",)))
        decision = PROJECTOR.project(final).decision
        self.assertEqual(decision.unknown_constraints, ())
        self.assertEqual(decision.contradicted_constraints, ("HDMI",))

    def test_retry_exhaustion_mixed_unknown_and_contradicted(self) -> None:
        final = run_bounded(PoolAwareTool(), FixedTextEvidenceService((
            "HDMI may be considered", "No HDMI port"
        )))
        decision = PROJECTOR.project(final).decision
        self.assertEqual(decision.unknown_constraints, ("HDMI",))
        self.assertEqual(decision.contradicted_constraints, ("HDMI",))

    def test_malformed_history_attempt_and_requirement_fail_closed(self) -> None:
        final = run_bounded(
            PoolAwareTool(), AttemptEvidenceService(success_on_expanded_pool=False)
        )
        corruptions = (
            final.model_copy(update={"failure_history": final.failure_history[:1]}),
            final.model_copy(update={"replan_attempt": 2}),
            final.model_copy(update={
                "agent_state": final.agent_state.model_copy(
                    update={"current_requirement_id": "wrong"}
                )
            }),
        )
        for value in corruptions:
            with self.subTest(value=value), self.assertRaises(
                GroundedResponseProjectionError
            ):
                PROJECTOR.project(value)

    def test_eligible_candidate_and_unsupported_conflict_fail_closed(self) -> None:
        final = run_bounded(
            PoolAwareTool(), AttemptEvidenceService(success_on_expanded_pool=False)
        )
        verification = final.current_verification
        first = verification.candidates[0].model_copy(
            update={"status": CandidateVerificationStatus.ELIGIBLE}
        )
        corrupted_verification = verification.model_copy(update={
            "candidates": (first, *verification.candidates[1:])
        })
        with self.assertRaises(GroundedResponseProjectionError):
            PROJECTOR.project(final.model_copy(update={
                "current_verification": corrupted_verification
            }))
        unsupported = final.model_copy(update={"agent_state": final.agent_state.model_copy(
            update={"error_state": "other_conflict"}
        )})
        with self.assertRaises(GroundedResponseProjectionError) as captured:
            PROJECTOR.project(unsupported)
        self.assertEqual(
            captured.exception.code, ProjectionErrorCode.UNSUPPORTED_TERMINAL_STATE
        )

    def test_conflict_projection_does_not_mutate_state(self) -> None:
        final = run_bounded(
            PoolAwareTool(), AttemptEvidenceService(success_on_expanded_pool=False)
        )
        before = final.model_dump(mode="json")
        PROJECTOR.project(final)
        self.assertEqual(final.model_dump(mode="json"), before)


if __name__ == "__main__":
    unittest.main()
