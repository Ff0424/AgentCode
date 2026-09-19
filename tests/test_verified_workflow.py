"""V2-09.4 workflow gate tests with deterministic in-memory dependencies."""

from __future__ import annotations

import unittest

from src.agentrec.domain import AgentState, ShoppingRequirement
from src.agentrec.evidence import EvidenceCandidate, EvidenceSnippet, ProductEvidence, RequirementEvidence
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.planning import FakePlanner, SelectCandidateDecision, SelectRequirementDecision
from src.agentrec.services import ShoppingPlanService
from src.agentrec.tools import RecommendationToolItem, RecommendationToolResult
from src.agentrec.verification import EvidenceConstraintVerifier
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.fake_evidence import PROVENANCE


class ThreeCandidateTool:
    def recommend(self, *, user_id, args, excluded_parent_asins=()):
        return RecommendationToolResult(personalization_status="personalized",
            fallback_reason=None, returned_count=3, items=tuple(
                RecommendationToolItem(rank=rank, item_index=rank,
                    parent_asin=f"P-{rank}", title=f"Product {rank}", price=50.0,
                    score=1.0 / rank, score_source="hybrid")
                for rank in range(1, 4)))


class MappedEvidenceService:
    def __init__(self, texts):
        self.texts = texts

    def retrieve(self, *, plan_id, plan_version, requirement, candidates):
        products = []
        for candidate in candidates:
            snippets = tuple(EvidenceSnippet(rank=index, item_index=candidate.item_index,
                parent_asin=candidate.parent_asin, chunk_id=f"{candidate.parent_asin}-c{index}",
                chunk_type=KnowledgeChunkType.FEATURES, part_index=index - 1,
                text=text, similarity_score=0.8)
                for index, text in enumerate(self.texts[candidate.parent_asin], 1))
            products.append(ProductEvidence(item_index=candidate.item_index,
                parent_asin=candidate.parent_asin, snippets=snippets))
        return RequirementEvidence(plan_id=plan_id, retrieved_at_plan_version=plan_version,
            requirement_id=requirement.requirement_id, query="feature evidence",
            candidates=candidates, products=tuple(products), provenance=PROVENANCE)


class CountingVerifier(EvidenceConstraintVerifier):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def verify(self, **kwargs):
        self.calls += 1
        return super().verify(**kwargs)


def state(*, features=("HDMI",)):
    service = ShoppingPlanService()
    plan = service.create_plan(plan_id="p", user_id="u", currency="USD", total_budget=500,
        requirements=(ShoppingRequirement(requirement_id="dock", category="Hubs",
            max_budget=100, required_features=features),))
    return ShoppingWorkflowState(agent_state=AgentState(user_id="u", session_id="s",
        shopping_plan=plan))


def graph(evidence, verifier, planner=None):
    return build_shopping_workflow(ThreeCandidateTool(), ShoppingPlanService(), planner=planner,
        evidence_service=evidence, verification_service=verifier)


class VerifiedWorkflowTests(unittest.TestCase):
    def test_only_eligible_candidates_enter_deterministic_selection(self):
        evidence = MappedEvidenceService({
            "P-1": ("Supports DisplayPort",),
            "P-2": ("Supports HDMI output",),
            "P-3": ("No HDMI port",),
        })
        final = ShoppingWorkflowState.model_validate(graph(evidence, EvidenceConstraintVerifier()).invoke(state()))
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual(final.agent_state.shopping_plan.selected_items[0].parent_asin, "P-2")
        self.assertEqual(len(final.selected_verifications), 1)
        selected = final.selected_verifications[0]
        self.assertEqual(selected.candidate.parent_asin, "P-2")
        self.assertEqual(selected.selected_at_plan_version, selected.candidate.verified_at_plan_version + 1)

    def test_zero_eligible_is_conflict_without_mutation(self):
        evidence = MappedEvidenceService({
            "P-1": ("Supports DisplayPort",), "P-2": ("No HDMI port",),
            "P-3": ("HDMI may be considered",),
        })
        final = ShoppingWorkflowState.model_validate(graph(evidence, EvidenceConstraintVerifier()).invoke(state()))
        self.assertEqual(final.route, WorkflowRoute.CONFLICT)
        self.assertEqual(final.agent_state.error_state, "constraint_verification:no_eligible_candidates")
        self.assertEqual(final.agent_state.shopping_plan.version, 0)

    def test_planner_cannot_select_unverified_or_ineligible(self):
        evidence = MappedEvidenceService({
            "P-1": ("Supports HDMI output",), "P-2": ("Supports DisplayPort",),
            "P-3": ("No HDMI port",),
        })
        for parent in ("P-2", "P-3"):
            planner = FakePlanner(decisions_by_key={
                "select_requirement:0": SelectRequirementDecision(plan_id="p", plan_version=0,
                    requirement_id="dock", reason="choose requirement"),
                "select_candidate:0:dock": SelectCandidateDecision(plan_id="p", plan_version=0,
                    requirement_id="dock", parent_asin=parent, reason="attempt bypass"),
            })
            final = ShoppingWorkflowState.model_validate(graph(evidence, EvidenceConstraintVerifier(), planner).invoke(state()))
            self.assertEqual(final.route, WorkflowRoute.ERROR)
            self.assertEqual(final.agent_state.error_state, "planner_validation:candidate_not_eligible")
            self.assertEqual(final.agent_state.shopping_plan.version, 0)

    def test_no_required_features_explicitly_skips_verifier(self):
        verifier = CountingVerifier()
        evidence = MappedEvidenceService({
            "P-1": ("general specifications",), "P-2": ("general specifications",),
            "P-3": ("general specifications",),
        })
        final = ShoppingWorkflowState.model_validate(graph(evidence, verifier).invoke(state(features=())))
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual(verifier.calls, 0)
        self.assertEqual(final.agent_state.shopping_plan.selected_items[0].parent_asin, "P-1")
        self.assertEqual(final.selected_verifications, ())


if __name__ == "__main__":
    unittest.main()
