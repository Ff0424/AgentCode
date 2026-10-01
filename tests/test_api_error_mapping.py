"""HTTP boundary tests for safe AgentRec API error mapping."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agentrec.api.dependencies import AgentRuntime
from src.agentrec.integration import GoalExecutionStatus
from src.agentrec.planning import (
    PlannerProviderError,
    PlannerSchemaError,
    PlannerTimeoutError,
)
from src.agentrec.response import (
    ConflictDecisionSummary,
    ConflictReason,
    FinalResponseResult,
    ResponseKind,
)
from tests.test_api_projection import (
    _clarification_result,
    _ready_result,
    _workflow_state,
)


api_app_module = importlib.import_module("src.agentrec.api.app")

_REQUEST = {
    "session_id": "session-api-boundary",
    "query": "Build a shopping plan",
}
_FORBIDDEN_PUBLIC_TEXT = (
    "private exception detail",
    "traceback",
    "/private/",
    "sk-",
    "raw provider response",
    "PlannerTimeoutError",
    "PlannerSchemaError",
    "PlannerProviderError",
    "RuntimeError",
)


class _Runner:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def run_goal(self, **_kwargs):
        if self._error is not None:
            raise self._error
        return self._result


@contextmanager
def _client_for(runner: _Runner):
    runtime = AgentRuntime(runner=runner, default_user_id="known-user")
    with patch.object(api_app_module, "build_runtime", return_value=runtime):
        with TestClient(
            api_app_module.app,
            raise_server_exceptions=False,
        ) as client:
            yield client


def _sensitive_message() -> str:
    return (
        "private exception detail traceback /private/artifacts/model.npy "
        "sk-secret raw provider response"
    )


class APIErrorMappingTests(unittest.TestCase):
    def assert_safe_error(
        self,
        response,
        *,
        status_code: int,
        error_code: str,
    ) -> None:
        self.assertEqual(response.status_code, status_code)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["error_code"], error_code)
        self.assertIsInstance(body["message"], str)
        self.assertTrue(body["message"])
        serialized = json.dumps(body)
        for forbidden in _FORBIDDEN_PUBLIC_TEXT:
            self.assertNotIn(forbidden, serialized)

    def test_planner_timeout_maps_to_safe_504(self) -> None:
        runner = _Runner(error=PlannerTimeoutError(_sensitive_message()))
        with _client_for(runner) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assert_safe_error(
            response,
            status_code=504,
            error_code="provider_timeout",
        )

    def test_planner_schema_error_maps_to_safe_502(self) -> None:
        runner = _Runner(error=PlannerSchemaError(_sensitive_message()))
        with _client_for(runner) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assert_safe_error(
            response,
            status_code=502,
            error_code="provider_invalid_response",
        )

    def test_planner_provider_error_maps_to_safe_502(self) -> None:
        runner = _Runner(error=PlannerProviderError(_sensitive_message()))
        with _client_for(runner) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assert_safe_error(
            response,
            status_code=502,
            error_code="provider_unavailable",
        )

    def test_unexpected_runtime_error_maps_to_safe_500(self) -> None:
        runner = _Runner(error=RuntimeError(_sensitive_message()))
        with _client_for(runner) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assert_safe_error(
            response,
            status_code=500,
            error_code="internal_error",
        )

    def test_goal_error_result_maps_to_safe_500(self) -> None:
        result = SimpleNamespace(
            status=GoalExecutionStatus.ERROR,
            final_response=None,
            workflow_state=_workflow_state(ready=False),
            clarification_question=None,
        )
        with _client_for(_Runner(result=result)) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assert_safe_error(
            response,
            status_code=500,
            error_code="agent_execution_failed",
        )
        self.assertNotIn("error_state", response.text)

    def test_ready_remains_http_200_chat_response(self) -> None:
        with _client_for(_Runner(result=_ready_result())) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_clarification_remains_http_200_chat_response(self) -> None:
        with _client_for(_Runner(result=_clarification_result())) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "clarification_required")

    def test_conflict_remains_http_200_chat_response(self) -> None:
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
        with _client_for(_Runner(result=result)) as client:
            response = client.post("/api/v1/chat", json=_REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "conflict")

    def test_request_validation_remains_fastapi_422(self) -> None:
        with _client_for(_Runner(result=_clarification_result())) as client:
            response = client.post(
                "/api/v1/chat",
                json={"session_id": "session-api-boundary", "query": "   "},
            )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
