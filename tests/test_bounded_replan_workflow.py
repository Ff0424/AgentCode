"""LangGraph tests for one bounded candidate-pool re-plan."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.agentrec.domain import AgentState, ShoppingRequirement
from src.agentrec.evidence import (
    EvidenceCandidate,
    EvidenceSnippet,
    ProductEvidence,
    RequirementEvidence,
)
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.services import ShoppingPlanService
from src.agentrec.tools import (
    RecommendationToolArgs,
    RecommendationToolItem,
    RecommendationToolResult,
)
from src.agentrec.verification import EvidenceConstraintVerifier
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.fake_evidence import PROVENANCE


class PoolAwareTool:
    def __init__(self, *, empty=False):
        self.empty = empty
        self.calls = []

    def recommend(self, *, user_id, args, excluded_parent_asins=()):
        self.calls.append(args)
        if self.empty:
            return RecommendationToolResult(
                personalization_status="personalized", fallback_reason=None,
                returned_count=0, items=(),
            )
        items = tuple(
            RecommendationToolItem(
                rank=rank, item_index=rank, parent_asin=f"P-{rank}",
                title=f"Product {rank}", price=50.0, score=1.0 / rank,
                score_source="hybrid",
            )
            for rank in range(1, args.top_k + 1)
        )
        return RecommendationToolResult(
            personalization_status="personalized", fallback_reason=None,
            returned_count=len(items), items=items,
        )


class AttemptEvidenceService:
    def __init__(self, *, success_on_expanded_pool: bool):
        self.success_on_expanded_pool = success_on_expanded_pool
        self.calls = []

    def retrieve(self, *, plan_id, plan_version, requirement, candidates):
        self.calls.append((plan_version, candidates))
        products = []
        for candidate in candidates:
            text = "HDMI may be considered"
            if (
                self.success_on_expanded_pool
                and len(candidates) == 10
                and candidate.parent_asin == "P-6"
            ):
                text = "Supports HDMI output"
            snippet = EvidenceSnippet(
                rank=1, item_index=candidate.item_index,
                parent_asin=candidate.parent_asin,
                chunk_id=f"{len(candidates)}-{candidate.parent_asin}",
                chunk_type=KnowledgeChunkType.FEATURES, part_index=0,
                text=text, similarity_score=0.8,
            )
            products.append(ProductEvidence(
                item_index=candidate.item_index, parent_asin=candidate.parent_asin,
                snippets=(snippet,),
            ))
        return RequirementEvidence(
            plan_id=plan_id, retrieved_at_plan_version=plan_version,
            requirement_id=requirement.requirement_id, query="HDMI support",
            candidates=candidates, products=tuple(products), provenance=PROVENANCE,
        )


class CorruptEvidenceService(AttemptEvidenceService):
    def retrieve(self, **kwargs):
        result = super().retrieve(**kwargs)
        return result.model_copy(update={"requirement_id": "wrong"})


def initial_state():
    plan_service = ShoppingPlanService()
    plan = plan_service.create_plan(
        plan_id="plan", user_id="user", currency="USD", total_budget=500,
        requirements=(ShoppingRequirement(
            requirement_id="dock", category="Hubs", max_budget=500,
            required_features=("HDMI",), soft_preferences=("laptop use",),
        ),),
    )
    return ShoppingWorkflowState(agent_state=AgentState(
        user_id="user", session_id="session", shopping_plan=plan,
    ))


def run(tool, evidence):
    graph = build_shopping_workflow(
        tool, ShoppingPlanService(), evidence_service=evidence,
        verification_service=EvidenceConstraintVerifier(),
    )
    return ShoppingWorkflowState.model_validate(
        graph.invoke(initial_state(), config={"recursion_limit": 80})
    )


class BoundedReplanWorkflowTests(unittest.TestCase):
    def test_zero_eligible_expands_once_then_succeeds_with_one_mutation(self):
        tool = PoolAwareTool()
        evidence = AttemptEvidenceService(success_on_expanded_pool=True)
        final = run(tool, evidence)
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual([value.top_k for value in tool.calls], [5, 10])
        self.assertEqual(final.replan_attempt, 1)
        self.assertEqual(final.agent_state.shopping_plan.version, 1)
        self.assertEqual(len(final.agent_state.shopping_plan.selected_items), 1)
        self.assertEqual(
            final.agent_state.shopping_plan.selected_items[0].parent_asin, "P-6"
        )
        self.assertEqual(len(final.failure_history), 1)
        self.assertEqual(len(final.replan_history), 1)
        self.assertIsNone(final.current_evidence)
        self.assertIsNone(final.current_verification)
        self.assertIsNone(final.current_evidence_attempt)
        self.assertIsNone(final.current_verification_attempt)
        self.assertIsNone(final.planner_decision)
        self.assertEqual([call[0] for call in evidence.calls], [0, 0])

    def test_retry_exhaustion_conflicts_without_attempt_two_or_mutation(self):
        tool = PoolAwareTool()
        final = run(tool, AttemptEvidenceService(success_on_expanded_pool=False))
        self.assertEqual(final.route, WorkflowRoute.CONFLICT)
        self.assertEqual([value.top_k for value in tool.calls], [5, 10])
        self.assertEqual(final.replan_attempt, 1)
        self.assertEqual(final.agent_state.shopping_plan.version, 0)
        self.assertEqual(final.agent_state.shopping_plan.selected_items, ())
        self.assertEqual(len(final.failure_history), 2)
        self.assertEqual(len(final.replan_history), 2)
        self.assertEqual(
            final.agent_state.error_state,
            "constraint_verification:replan_attempts_exhausted",
        )

    def test_no_recommendation_candidates_do_not_retry(self):
        tool = PoolAwareTool(empty=True)
        final = run(tool, AttemptEvidenceService(success_on_expanded_pool=True))
        self.assertEqual(final.route, WorkflowRoute.CONFLICT)
        self.assertEqual([value.top_k for value in tool.calls], [5])
        self.assertEqual(final.replan_attempt, 0)
        self.assertEqual(final.failure_history, ())
        self.assertEqual(final.agent_state.error_state, "no_recommendation_candidates")

    def test_hard_constraints_are_identical_across_attempts(self):
        tool = PoolAwareTool()
        run(tool, AttemptEvidenceService(success_on_expanded_pool=True))
        first, second = tool.calls
        self.assertEqual(first.category, second.category)
        self.assertEqual(first.max_price, second.max_price)
        self.assertEqual(first.required_features, second.required_features)

    def test_normal_success_still_calls_recommendation_once(self):
        tool = PoolAwareTool()
        evidence = AttemptEvidenceService(success_on_expanded_pool=True)
        # Make the initial pool use the same explicit support as the expanded pool.
        evidence.retrieve = lambda **kwargs: RequirementEvidence(
            plan_id=kwargs["plan_id"],
            retrieved_at_plan_version=kwargs["plan_version"],
            requirement_id=kwargs["requirement"].requirement_id,
            query="HDMI support", candidates=kwargs["candidates"],
            products=tuple(ProductEvidence(
                item_index=c.item_index, parent_asin=c.parent_asin,
                snippets=(EvidenceSnippet(
                    rank=1, item_index=c.item_index, parent_asin=c.parent_asin,
                    chunk_id=f"support-{c.parent_asin}",
                    chunk_type=KnowledgeChunkType.FEATURES, part_index=0,
                    text="Supports HDMI output", similarity_score=0.8,
                ),),
            ) for c in kwargs["candidates"]), provenance=PROVENANCE,
        )
        final = run(tool, evidence)
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual([value.top_k for value in tool.calls], [5])
        self.assertEqual(final.replan_attempt, 0)

    def test_attempt_binding_is_enforced(self):
        tool = PoolAwareTool()
        evidence_service = AttemptEvidenceService(success_on_expanded_pool=False)
        state = initial_state()
        # Produce a real evidence object and deliberately bind it to another attempt.
        result = tool.recommend(
            user_id="user",
            args=RecommendationToolArgs(
                top_k=5, category="Hubs", max_price=500,
                required_features=("HDMI",),
            ),
        )
        candidates = tuple(
            EvidenceCandidate(item_index=x.item_index, parent_asin=x.parent_asin)
            for x in result.items
        )
        evidence = evidence_service.retrieve(
            plan_id="plan", plan_version=0,
            requirement=state.agent_state.shopping_plan.requirements[0],
            candidates=candidates,
        )
        with self.assertRaises(ValidationError):
            ShoppingWorkflowState.model_validate(state.model_copy(update={
                "agent_state": state.agent_state.model_copy(update={
                    "current_requirement_id": "dock"
                }),
                "current_evidence": evidence,
                "current_evidence_attempt": 1,
            }).model_dump())

    def test_identity_error_does_not_enter_replan(self):
        tool = PoolAwareTool()
        final = run(tool, CorruptEvidenceService(success_on_expanded_pool=False))
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(final.failure_history, ())
        self.assertEqual(final.replan_history, ())


if __name__ == "__main__":
    unittest.main()
