"""Unit tests for the Web-safe Goal execution API projection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.agentrec.api.app import _project_chat_response, chat
from src.agentrec.api.dependencies import AgentRuntime
from src.agentrec.api.schemas import ChatRequest
from src.agentrec.domain import (
    AgentState,
    ItemSource,
    PlanStatus,
    RequirementStatus,
    ShoppingPlan,
    ShoppingPlanItem,
    ShoppingRequirement,
)
from src.agentrec.integration import GoalExecutionStatus
from src.agentrec.knowledge import KnowledgeChunkType
from src.agentrec.planning import GoalBudgetAllocation, RequirementBudgetAllocation
from src.agentrec.response import (
    ConflictDecisionSummary,
    ConflictReason,
    EvidenceReference,
    FinalResponseResult,
    ProductDecisionSummary,
    ResponseKind,
    VerifiedConstraintClaim,
)
from src.agentrec.verification import ConstraintVerificationStatus


def _workflow_state(*, ready: bool) -> SimpleNamespace:
    requirement = ShoppingRequirement(
        requirement_id="req-001",
        category="Docking Stations",
        quantity=2 if ready else 1,
        max_budget=100.0,
        required_features=("HDMI",),
        soft_preferences=("compact",),
        priority=3,
        status=(
            RequirementStatus.SATISFIED
            if ready
            else RequirementStatus.PENDING
        ),
    )
    items = (
        (
            ShoppingPlanItem(
                slot_id="slot-001",
                requirement_id="req-001",
                category="Docking Stations",
                parent_asin="P-DOCK",
                title="Grounded Dock",
                price=40.0,
                quantity=2,
                selected_reason="selected_by_rank_policy",
                constraints_satisfied=True,
                source=ItemSource.HYBRID,
                recommendation_score=0.9,
            ),
        )
        if ready
        else ()
    )
    plan = ShoppingPlan(
        plan_id="plan-api",
        user_id="user-api",
        currency="USD",
        total_budget=100.0,
        requirements=(requirement,),
        selected_items=items,
        status=PlanStatus.READY if ready else PlanStatus.DRAFT,
        version=1 if ready else 0,
    )
    return SimpleNamespace(
        agent_state=AgentState(
            user_id="user-api",
            session_id="session-api",
            shopping_plan=plan,
            error_state="private traceback /artifact/path" if not ready else None,
        ),
        goal_budget_allocation=GoalBudgetAllocation(
            total_budget=100.0,
            allocations=(
                RequirementBudgetAllocation(
                    requirement_id="req-001",
                    allocated_budget=100.0,
                ),
            ),
            unallocated_budget=0.0,
        ),
    )


def _ready_result() -> SimpleNamespace:
    claim = VerifiedConstraintClaim(
        requirement_id="req-001",
        parent_asin="P-DOCK",
        original_constraint="HDMI",
        canonical_constraint="hdmi",
        status=ConstraintVerificationStatus.SUPPORTED,
        supporting_evidence=(
            EvidenceReference(
                chunk_id="private-chunk-id",
                chunk_type=KnowledgeChunkType.FEATURES,
                excerpt="private raw evidence excerpt",
            ),
        ),
    )
    summary = ProductDecisionSummary(
        requirement_id="req-001",
        category="Docking Stations",
        parent_asin="P-DOCK",
        title="Grounded Dock",
        price=40.0,
        quantity=2,
        source=ItemSource.HYBRID,
        selection_rationale="private planner rationale",
        verified_claims=(claim,),
    )
    return SimpleNamespace(
        status=GoalExecutionStatus.READY,
        final_response=FinalResponseResult(
            kind=ResponseKind.READY,
            text=(
                "Grounded Dock P-DOCK quantity 2 USD 40.00 HDMI; "
                "total 80.00 remaining 20.00"
            ),
            decision_summary=(summary,),
        ),
        workflow_state=_workflow_state(ready=True),
        clarification_question=None,
    )


def _clarification_result() -> SimpleNamespace:
    return SimpleNamespace(
        status=GoalExecutionStatus.CLARIFICATION_REQUIRED,
        final_response=None,
        workflow_state=None,
        clarification_question="What is your total budget?",
    )


class _FakeRunner:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def run_goal(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class APIProjectionTests(unittest.TestCase):
    def test_ready_projects_plan_products_and_verified_requirements(self) -> None:
        response = _project_chat_response(_ready_result())

        self.assertEqual(response.status, "ready")
        self.assertIn("Grounded Dock", response.response)
        self.assertIn("P-DOCK", response.response)
        self.assertIn("USD 40.00", response.response)
        self.assertIsNotNone(response.conversation_summary)
        self.assertIn("1 项需求均已满足", response.conversation_summary)
        self.assertNotIn("Grounded Dock", response.conversation_summary)
        self.assertNotIn("P-DOCK", response.conversation_summary)
        self.assertNotIn("40.0", response.conversation_summary)
        self.assertNotIn("80.0", response.conversation_summary)
        self.assertNotIn("USD", response.conversation_summary)
        self.assertNotIn("HDMI", response.conversation_summary)
        self.assertIsNone(response.clarification)
        self.assertIsNone(response.conflict)
        self.assertEqual(response.plan.total_budget, 100.0)
        self.assertEqual(response.plan.total_spent, 80.0)
        self.assertEqual(response.plan.remaining_budget, 20.0)
        self.assertEqual(response.requirements[0].allocated_budget, 100.0)
        self.assertEqual(response.requirements[0].required_features, ["HDMI"])
        self.assertEqual(response.products[0].quantity, 2)
        self.assertEqual(response.products[0].subtotal, 80.0)
        self.assertEqual(
            response.products[0].verified_requirements[0].model_dump(),
            {"constraint": "HDMI", "status": "supported"},
        )
        serialized = response.model_dump_json()
        self.assertNotIn("private-chunk-id", serialized)
        self.assertNotIn("private raw evidence", serialized)
        self.assertNotIn("private planner rationale", serialized)
        self.assertNotIn("recommendation_score", serialized)

    def test_clarification_is_valid_without_fake_plan_or_products(self) -> None:
        response = _project_chat_response(_clarification_result())

        self.assertEqual(response.status, "clarification_required")
        self.assertIsNone(response.response)
        self.assertIsNone(response.conversation_summary)
        self.assertEqual(
            response.clarification.question,
            "What is your total budget?",
        )
        self.assertIsNone(response.plan)
        self.assertEqual(response.requirements, [])
        self.assertEqual(response.products, [])
        self.assertIsNone(response.conflict)

    def test_conflict_uses_safe_projected_summary(self) -> None:
        decision = ConflictDecisionSummary(
            requirement_id="req-001",
            category="Docking Stations",
            required_features=("HDMI",),
            reason=ConflictReason.NO_RECOMMENDATION_CANDIDATES,
            replan_attempts_performed=1,
            candidate_pool_sizes=(5, 10),
        )
        result = SimpleNamespace(
            status=GoalExecutionStatus.CONFLICT,
            final_response=FinalResponseResult(
                kind=ResponseKind.CONFLICT,
                text="Grounded conflict response",
                decision_summary=decision,
            ),
            workflow_state=_workflow_state(ready=False),
            clarification_question=None,
        )

        response = _project_chat_response(result)

        self.assertEqual(response.status, "conflict")
        self.assertEqual(response.response, "Grounded conflict response")
        self.assertIsNone(response.conversation_summary)
        self.assertEqual(response.conflict.reason, "no_recommendation_candidates")
        self.assertEqual(response.conflict.replan_attempts_performed, 1)
        self.assertTrue(response.conflict.user_action_required)
        self.assertEqual(response.conflict.required_features, ["HDMI"])
        serialized = response.model_dump_json()
        self.assertNotIn("private traceback", serialized)
        self.assertNotIn("source_error_code", serialized)
        self.assertNotIn("candidate_pool_sizes", serialized)

    def test_error_does_not_leak_internal_workflow_state(self) -> None:
        result = SimpleNamespace(
            status=GoalExecutionStatus.ERROR,
            final_response=None,
            workflow_state=_workflow_state(ready=False),
            clarification_question=None,
        )

        response = _project_chat_response(result)

        self.assertEqual(response.status, "error")
        self.assertIsNone(response.response)
        self.assertIsNone(response.conversation_summary)
        self.assertIsNone(response.plan)
        self.assertEqual(response.requirements, [])
        self.assertEqual(response.products, [])
        self.assertIsNone(response.conflict)
        self.assertNotIn("private traceback", response.model_dump_json())

    def test_optional_user_id_uses_default_and_explicit_user_is_preserved(self) -> None:
        runner = _FakeRunner(_clarification_result())
        runtime = AgentRuntime(runner=runner, default_user_id="real-default-user")

        chat(
            ChatRequest(session_id="session", query="shopping goal"),
            runtime,
        )
        chat(
            ChatRequest(
                user_id="explicit-user",
                session_id="session",
                query="shopping goal",
            ),
            runtime,
        )

        self.assertEqual(runner.calls[0]["user_id"], "real-default-user")
        self.assertEqual(runner.calls[1]["user_id"], "explicit-user")


if __name__ == "__main__":
    unittest.main()
