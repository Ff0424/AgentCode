"""Unit tests for the shared shopping WorkflowRunner execution boundary."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agentrec.domain import AgentState, ShoppingPlan, ShoppingRequirement
from src.agentrec.integration import WorkflowRunner
from src.agentrec.workflows import ShoppingWorkflowState


def _dependencies() -> dict[str, object]:
    return {
        "planner": SimpleNamespace(decide=lambda **kwargs: None),
        "recommendation_tool": SimpleNamespace(recommend=lambda **kwargs: None),
        "evidence_service": SimpleNamespace(retrieve=lambda **kwargs: None),
        "verification_service": SimpleNamespace(verify=lambda **kwargs: None),
        "shopping_plan_service": SimpleNamespace(create_plan=lambda **kwargs: None),
    }


def _state() -> ShoppingWorkflowState:
    plan = ShoppingPlan(
        plan_id="plan",
        user_id="user",
        currency="USD",
        total_budget=100,
        requirements=(ShoppingRequirement(
            requirement_id="mouse",
            category="Mouse",
            max_budget=50,
        ),),
    )
    return ShoppingWorkflowState(agent_state=AgentState(
        user_id="user",
        session_id="session",
        shopping_plan=plan,
    ))


class _RecordingGraph:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[object, object]] = []

    def invoke(self, state, config=None):
        self.calls.append((state, config))
        return self.output


class WorkflowRunnerTests(unittest.TestCase):
    def test_valid_dependencies_are_accepted(self) -> None:
        runner = WorkflowRunner(**_dependencies())
        self.assertTrue(callable(runner.execute))

    def test_missing_dependency_methods_are_rejected(self) -> None:
        required = {
            "planner": "decide",
            "recommendation_tool": "recommend",
            "evidence_service": "retrieve",
            "verification_service": "verify",
            "shopping_plan_service": "create_plan",
        }
        for name, method in required.items():
            values = _dependencies()
            values[name] = SimpleNamespace()
            with self.subTest(name=name, method=method), self.assertRaises(TypeError):
                WorkflowRunner(**values)

    def test_execute_rejects_invalid_state_and_recursion_limit(self) -> None:
        runner = WorkflowRunner(**_dependencies())
        with self.assertRaises(TypeError):
            runner.execute(None)  # type: ignore[arg-type]
        initial = _state()
        for value in (True, False, 0, "50"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.execute(initial, recursion_limit=value)  # type: ignore[arg-type]

    def test_execute_builds_invokes_and_validates_workflow(self) -> None:
        dependencies = _dependencies()
        runner = WorkflowRunner(**dependencies)
        initial = _state()
        graph = _RecordingGraph(initial.model_dump(mode="json"))
        with patch(
            "src.agentrec.integration.workflow_runner.build_shopping_workflow",
            return_value=graph,
        ) as build:
            result = runner.execute(initial, recursion_limit=17)

        build.assert_called_once_with(
            dependencies["recommendation_tool"],
            dependencies["shopping_plan_service"],
            planner=dependencies["planner"],
            evidence_service=dependencies["evidence_service"],
            verification_service=dependencies["verification_service"],
        )
        self.assertEqual(graph.calls, [(initial, {"recursion_limit": 17})])
        self.assertIsInstance(result, ShoppingWorkflowState)
        self.assertEqual(result, initial)

    def test_execution_is_deterministic_for_the_same_input(self) -> None:
        runner = WorkflowRunner(**_dependencies())
        initial = _state()
        graph = _RecordingGraph(initial.model_dump(mode="json"))
        with patch(
            "src.agentrec.integration.workflow_runner.build_shopping_workflow",
            return_value=graph,
        ):
            first = runner.execute(initial)
            second = runner.execute(initial)
        self.assertEqual(first, second)
        self.assertEqual(
            graph.calls,
            [
                (initial, {"recursion_limit": 50}),
                (initial, {"recursion_limit": 50}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
