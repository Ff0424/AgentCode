"""Deterministic V2-08.4c tests; no real provider request is performed."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.agentrec.planning import (
    OpenAICompatiblePlannerProvider,
    PlannerProviderError,
    PlannerSchemaError,
    PlannerTimeoutError,
    SelectRequirementDecision,
    StructuredLLMPlanner,
)
from src.agentrec.services import ShoppingPlanService
from src.agentrec.workflows import ShoppingWorkflowState, WorkflowRoute, build_shopping_workflow
from tests.test_shopping_workflow import FakeRecommendationTool, initial_state
from tests.fake_evidence import FakeEvidenceService


class FakeTextProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, messages, timeout_seconds):
        self.calls.append((messages, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def decision(action, version, **fields):
    return json.dumps({
        "action": action,
        "plan_id": "workflow-demo",
        "plan_version": version,
        "reason": "deterministic provider decision",
        **fields,
    })


class LLMPlannerAdapterTests(unittest.TestCase):
    def test_valid_json_becomes_planner_decision(self) -> None:
        provider = FakeTextProvider([
            decision("select_requirement", 0, requirement_id="dock")
        ])
        planner = StructuredLLMPlanner(provider)
        output = planner.decide(context={
            "decision_key": "select_requirement:0",
            "plan_id": "workflow-demo",
            "plan_version": 0,
            "requirements": [],
            "embedding": [999],
            "gpu": "cuda:0",
        })
        self.assertIsInstance(output, SelectRequirementDecision)
        self.assertEqual(output.requirement_id, "dock")
        messages = provider.calls[0][0]
        self.assertNotIn("embedding", messages[1]["content"])
        self.assertNotIn("cuda", messages[1]["content"])

    def test_invalid_json_missing_and_extra_fields_raise_schema_error(self) -> None:
        invalid_values = (
            "not-json",
            json.dumps({
                "action": "select_requirement", "plan_id": "p",
                "plan_version": 0, "reason": "missing requirement",
            }),
            json.dumps({
                "action": "select_requirement", "plan_id": "p",
                "plan_version": 0, "requirement_id": "dock", "reason": "x",
                "price": 1,
            }),
        )
        for raw in invalid_values:
            with self.subTest(raw=raw):
                provider = FakeTextProvider([raw])
                planner = StructuredLLMPlanner(provider, schema_retries=0)
                with self.assertRaises(PlannerSchemaError):
                    planner.decide(context={
                        "decision_key": "select_requirement:0",
                        "plan_id": "p", "plan_version": 0, "requirements": [],
                    })

    def test_schema_retry_occurs_at_most_once(self) -> None:
        provider = FakeTextProvider([
            "invalid",
            decision("select_requirement", 0, requirement_id="dock"),
        ])
        output = StructuredLLMPlanner(provider, schema_retries=1).decide(context={
            "decision_key": "select_requirement:0",
            "plan_id": "workflow-demo", "plan_version": 0, "requirements": [],
        })
        self.assertEqual(output.requirement_id, "dock")
        self.assertEqual(len(provider.calls), 2)

    def test_timeout_and_provider_failures_are_explicit(self) -> None:
        with self.assertRaises(PlannerTimeoutError):
            StructuredLLMPlanner(FakeTextProvider([TimeoutError()])).decide(context={
                "decision_key": "select_requirement:0",
                "plan_id": "p", "plan_version": 0, "requirements": [],
            })
        with self.assertRaises(PlannerProviderError):
            StructuredLLMPlanner(FakeTextProvider([RuntimeError("network")])).decide(context={
                "decision_key": "select_requirement:0",
                "plan_id": "p", "plan_version": 0, "requirements": [],
            })

    def test_openai_compatible_provider_uses_json_mode(self) -> None:
        class Completions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"action":"x"}')
                )])

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = OpenAICompatiblePlannerProvider(model="deepseek-test", client=client)
        raw = provider.complete(
            messages=({"role": "system", "content": "test"},),
            timeout_seconds=12.0,
        )
        self.assertEqual(raw, '{"action":"x"}')
        self.assertEqual(completions.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(completions.kwargs["temperature"], 0.0)
        self.assertEqual(completions.kwargs["timeout"], 12.0)

    def test_real_adapter_contract_runs_through_existing_workflow(self) -> None:
        responses = []
        for version, requirement in enumerate(("dock", "mouse", "headphones")):
            responses.extend((
                decision("select_requirement", version, requirement_id=requirement),
                decision(
                    "select_candidate", version, requirement_id=requirement,
                    parent_asin=f"P-{requirement.upper()}",
                ),
            ))
        planner = StructuredLLMPlanner(FakeTextProvider(responses))
        graph = build_shopping_workflow(
            FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180}),
            ShoppingPlanService(),
            planner=planner,
            evidence_service=FakeEvidenceService(),
        )
        final = ShoppingWorkflowState.model_validate(
            graph.invoke(initial_state(), config={"recursion_limit": 50})
        )
        self.assertEqual(final.route, WorkflowRoute.READY)
        self.assertEqual(final.agent_state.shopping_plan.total_spent, 330)

    def test_workflow_still_rejects_fabricated_candidate(self) -> None:
        provider = FakeTextProvider([
            decision("select_requirement", 0, requirement_id="dock"),
            decision(
                "select_candidate", 0, requirement_id="dock",
                parent_asin="P-FABRICATED",
            ),
        ])
        graph = build_shopping_workflow(
            FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180}),
            ShoppingPlanService(), planner=StructuredLLMPlanner(provider),
            evidence_service=FakeEvidenceService(),
        )
        final = ShoppingWorkflowState.model_validate(graph.invoke(initial_state()))
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(
            final.agent_state.error_state,
            "planner_validation:candidate_not_in_tool_result",
        )

    def test_workflow_still_rejects_stale_version(self) -> None:
        provider = FakeTextProvider([
            decision("select_requirement", 99, requirement_id="dock")
        ])
        graph = build_shopping_workflow(
            FakeRecommendationTool({"Dock": 120, "Mouse": 30, "Headphones": 180}),
            ShoppingPlanService(), planner=StructuredLLMPlanner(provider),
            evidence_service=FakeEvidenceService(),
        )
        final = ShoppingWorkflowState.model_validate(graph.invoke(initial_state()))
        self.assertEqual(final.route, WorkflowRoute.ERROR)
        self.assertEqual(
            final.agent_state.error_state,
            "planner_validation:stale_plan_version",
        )


if __name__ == "__main__":
    unittest.main()
