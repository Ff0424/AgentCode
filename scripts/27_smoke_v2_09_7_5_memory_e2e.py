"""Deterministic V2-09.7.5 multi-turn memory E2E smoke.

Run from the repository root::

    python scripts/27_smoke_v2_09_7_5_memory_e2e.py

The smoke uses production MemoryStore, RequirementMemoryMerger, and
AgentTaskRunner with deterministic fake recommendation/evidence dependencies.
It performs no network, model, database, or artifact access.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentrec.memory import (  # noqa: E402
    MemoryStore,
    PreferenceStatus,
    PreferenceType,
    RequirementMemoryMerger,
)
from tests.test_memory_multiturn_e2e import (  # noqa: E402
    FailingStore,
    ObservingMerger,
    build_runner,
    execute,
    final_requirement,
    memory_item,
)


def section(title: str) -> None:
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def main() -> int:
    print("================ MEMORY E2E =================")

    section("MEMORY INITIALIZATION")
    store = MemoryStore()
    merger = ObservingMerger()
    print("memory_store=MemoryStore")
    print("memory_merger=RequirementMemoryMerger")

    section("TURN 1 PREFERENCE CREATION")
    item = memory_item(
        PreferenceType.FEATURE,
        "USB-C",
        PreferenceStatus.CONFIRMED,
    )
    store.add(item)
    print(f"preference_id={item.id}")
    print(f"scope={item.scope.value}")
    print(f"type={item.type.value}")
    print(f"status={item.status.value}")

    section("TURN 2 RETRIEVAL")
    confirmed = store.get_confirmed_preferences("user-1")
    print(f"confirmed_count={len(confirmed)}")
    print(f"confirmed_values={tuple(value.value for value in confirmed)!r}")

    section("MERGED REQUIREMENT")
    result = execute(build_runner(memory_store=store, memory_merger=merger))
    requirement = final_requirement(result)
    print(f"merger_call_count={len(merger.calls)}")
    print(f"required_features={requirement.required_features!r}")
    print(f"max_budget={requirement.max_budget!r}")

    section("WORKFLOW RESULT")
    print(f"status={result.status.value}")
    print(f"route={result.workflow_state.route.value}")
    print(f"final_response_kind={result.final_response.kind.value}")
    print(f"memory_error={result.memory_error!r}")

    section("MEMORY FAILURE ISOLATION RESULT")
    failed = execute(build_runner(
        memory_store=FailingStore(),
        memory_merger=RequirementMemoryMerger(),
    ))
    print(f"status={failed.status.value}")
    print(f"workflow_state_present={failed.workflow_state is not None}")
    print(f"final_response_present={failed.final_response is not None}")
    print(f"memory_error={failed.memory_error!r}")

    section("INTEGRITY SUMMARY")
    checks = {
        "confirmed_memory_effective": "USB-C" in requirement.required_features,
        "production_merger_called": len(merger.calls) == 1,
        "workflow_state_has_no_memory_fields": {"memory", "preferences", "memory_error"}.isdisjoint(
            type(result.workflow_state).model_fields
        ),
        "normal_workflow_ready": result.status.value == "ready",
        "failure_isolated": (
            failed.memory_error == "memory_store_failed"
            and failed.status.value == "ready"
            and failed.workflow_state is not None
            and failed.final_response is not None
        ),
    }
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"\nV2-09.7.5 MEMORY E2E SMOKE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
