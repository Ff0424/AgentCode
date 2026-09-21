"""Deterministic V2-09.8.5 decision-layer E2E smoke.

Run from the repository root::

    python scripts/28_smoke_v2_09_8_5_decision_e2e.py

The smoke uses only production decision contracts/policy/executor and the
AgentTaskRunner with deterministic fake business dependencies. It performs no
network, LLM, model, database, or artifact access.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentrec.decision import (  # noqa: E402
    AgentIntent,
    DecisionContext,
    DecisionExecutor,
    DeterministicDecisionPolicy,
)
from tests.test_decision_e2e import (  # noqa: E402
    FORBIDDEN_RESPONSE_TERMS,
    _FailingPolicy,
    build_runner,
    confirmed_memory_store,
    resolve_context,
    run,
)
from src.agentrec.memory import RequirementMemoryMerger  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def main() -> int:
    print("================ DECISION E2E =================")
    policy = DeterministicDecisionPolicy()
    executor = DecisionExecutor()

    section("RECOMMENDATION WITHOUT MEMORY")
    without_memory = run(build_runner(
        decision_policy=policy,
        decision_executor=executor,
    ))
    print(without_memory.decision_plan.model_dump(mode="json"))

    section("RECOMMENDATION WITH MEMORY")
    with_memory_runner = build_runner(
        decision_policy=policy,
        decision_executor=executor,
        memory_store=confirmed_memory_store(),
        memory_merger=RequirementMemoryMerger(),
    )
    with_memory = run(with_memory_runner)
    print(with_memory.decision_plan.model_dump(mode="json"))

    section("INFORMATION")
    information = resolve_context(DecisionContext(
        intent=AgentIntent.INFORMATION,
        memory_available=False,
        requirement_complete=True,
        candidate_available=False,
        verification_required=False,
        response_required=True,
    ))
    print(information.model_dump(mode="json"))

    section("CLARIFICATION")
    clarification = resolve_context(DecisionContext(
        intent=AgentIntent.RECOMMENDATION,
        memory_available=False,
        requirement_complete=False,
        candidate_available=False,
        verification_required=False,
        response_required=True,
    ))
    print(clarification.model_dump(mode="json"))

    section("FAILURE ISOLATION")
    failed = run(build_runner(
        decision_policy=_FailingPolicy(),
        decision_executor=executor,
    ))
    print(f"decision_plan={failed.decision_plan!r}")
    print(f"workflow_state_present={failed.workflow_state is not None}")
    print(f"final_response_present={failed.final_response is not None}")

    section("INTEGRITY SUMMARY")
    second = run(build_runner(decision_policy=policy, decision_executor=executor))
    response_text = without_memory.final_response.text.casefold()
    checks = {
        "recommendation_without_memory": (
            not without_memory.decision_plan.use_memory
            and without_memory.decision_plan.retrieve_products
            and without_memory.decision_plan.verify_evidence
            and without_memory.decision_plan.generate_response
        ),
        "recommendation_with_memory": with_memory.decision_plan.use_memory,
        "information_route": information.generate_response and information.stop,
        "clarification_route": clarification.ask_clarification and clarification.stop,
        "deterministic": without_memory.decision_plan == second.decision_plan,
        "failure_isolated": (
            failed.decision_plan is None
            and failed.workflow_state is not None
            and failed.final_response is not None
        ),
        "response_security": all(
            term not in response_text for term in FORBIDDEN_RESPONSE_TERMS
        ),
    }
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"\nV2-09.8.5 DECISION E2E SMOKE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
