"""V2-09.5 real-artifact bounded re-plan smoke for the Ubuntu GPU server.

The script is an observation-only validation layer.  It first performs a
bounded, read-only discovery over real recommendation/evidence/verification
components, then executes one fresh production LangGraph workflow.  It never
changes an artifact, candidate status, verifier alias, or production rule.

Run from the repository root on the validated Ubuntu server::

    python scripts/26_smoke_v2_09_5_real_artifact_e2e.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentrec.domain import AgentState, ShoppingRequirement  # noqa: E402
from src.agentrec.evidence import (  # noqa: E402
    EvidenceCandidate,
    GroundedEvidenceService,
    RequirementEvidence,
)
from src.agentrec.planning import (  # noqa: E402
    SelectCandidateDecision,
    SelectRequirementDecision,
)
from src.agentrec.recommendation import RecommendationService  # noqa: E402
from src.agentrec.replanning import (  # noqa: E402
    BoundedReplanPolicy,
    FailureDiagnosisService,
    FailureRecoverability,
    FailureType,
    ReplanAction,
)
from src.agentrec.retrieval import (  # noqa: E402
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from src.agentrec.services import ShoppingPlanService  # noqa: E402
from src.agentrec.tools import (  # noqa: E402
    RecommendationToolAdapter,
    RecommendationToolArgs,
)
from src.agentrec.verification import (  # noqa: E402
    CandidateVerificationStatus,
    ConstraintVerificationStatus,
    EvidenceConstraintVerifier,
    RequirementVerification,
)
from src.agentrec.workflows import (  # noqa: E402
    ShoppingWorkflowState,
    WorkflowRoute,
    build_shopping_workflow,
)


USER_ID = "AEFKF6R2GUSK2AWPSWRR4ZO36JVQ"
REQUIREMENT_ID = "dock-hub"
MAX_PRICE = 500.0
FEATURE_SETS = (("HDMI", "USB-C"), ("HDMI",), ("USB-C",))
MAX_CATEGORY_ATTEMPTS = 24
MAX_PHASE_B_CASES = 6
TOP_K_INITIAL = 5
TOP_K_EXPANDED = 10
DISCOVERY_PLAN_ID = "v2-09-5-read-only-discovery"
CATEGORY_PATTERN = re.compile(r"(?<![a-z0-9])(?:dock|hubs?)(?![a-z0-9])", re.I)


def section(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def now() -> float:
    return time.perf_counter()


def milliseconds(started: float, ended: float | None = None) -> float:
    return ((now() if ended is None else ended) - started) * 1000.0


def git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def git_clean(root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short"], cwd=root, check=True,
            capture_output=True, text=True, timeout=10,
        )
        return not result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False


def contains_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(contains_key(item, key) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(contains_key(item, key) for item in value)
    return False


def identity(item: Any) -> tuple[int, str]:
    return item.item_index, item.parent_asin


def candidate_identities(result: Any) -> tuple[tuple[int, str], ...]:
    return tuple(identity(item) for item in result.items)


def eligible_identities(verification: RequirementVerification) -> tuple[tuple[int, str], ...]:
    return tuple(
        identity(item) for item in verification.candidates
        if item.status is CandidateVerificationStatus.ELIGIBLE
    )


class AuditedDeterministicPlanner:
    """Choose only from workflow-provided identities and retain audit context."""

    def __init__(self) -> None:
        self.contexts: list[Mapping[str, Any]] = []
        self.item_index_exposed = False
        self.selectable_parent_asins: tuple[str, ...] = ()

    def decide(self, *, context: Mapping[str, Any]):
        self.contexts.append(context)
        self.item_index_exposed |= contains_key(context, "item_index")
        key = context.get("decision_key")
        if not isinstance(key, str):
            raise ValueError("Planner context has no decision_key.")
        if key.startswith("select_requirement:"):
            requirements = context.get("requirements")
            if not isinstance(requirements, tuple) or not requirements:
                raise ValueError("Requirement planner received no requirements.")
            return SelectRequirementDecision(
                plan_id=context["plan_id"],
                plan_version=context["plan_version"],
                requirement_id=requirements[0]["requirement_id"],
                reason="v2_09_5_real_smoke_requirement",
            )
        if key.startswith("select_candidate:"):
            candidates = context.get("candidates")
            if not isinstance(candidates, tuple) or not candidates:
                raise ValueError("Candidate planner received no eligible candidates.")
            self.selectable_parent_asins = tuple(
                value["parent_asin"] for value in candidates
            )
            return SelectCandidateDecision(
                plan_id=context["plan_id"],
                plan_version=context["plan_version"],
                requirement_id=context["requirement_id"],
                parent_asin=self.selectable_parent_asins[0],
                reason="v2_09_5_first_eligible_candidate",
            )
        raise ValueError(f"Unsupported planner decision_key={key!r}.")


class RecordingRecommendationTool:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def recommend(self, **kwargs: Any):
        args = kwargs.get("args")
        started = now()
        try:
            result = self.delegate.recommend(**kwargs)
        except Exception:
            self.records.append({"call_index": len(self.records), "started": started,
                                 "ended": now(), "args": args, "error": True})
            raise
        ended = now()
        self.records.append({
            "call_index": len(self.records), "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended), "args": args,
            "excluded_parent_asins": kwargs.get("excluded_parent_asins", ()),
            "result": result, "identities": candidate_identities(result),
        })
        return result


class RecordingEvidenceService:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any):
        started = now()
        try:
            result = self.delegate.retrieve(**kwargs)
        except Exception:
            self.records.append({"call_index": len(self.records), "started": started,
                                 "ended": now(), "kwargs": kwargs, "error": True})
            raise
        ended = now()
        self.records.append({
            "call_index": len(self.records), "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended), "result": result,
            "plan_id": kwargs["plan_id"], "plan_version": kwargs["plan_version"],
            "requirement_id": kwargs["requirement"].requirement_id,
            "identities": tuple(identity(value) for value in kwargs["candidates"]),
            "chunk_ids": {
                identity(product): tuple(snippet.chunk_id for snippet in product.snippets)
                for product in result.products
            },
        })
        return result


class RecordingVerificationService:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def verify(self, **kwargs: Any):
        started = now()
        try:
            result = self.delegate.verify(**kwargs)
        except Exception:
            self.records.append({"call_index": len(self.records), "started": started,
                                 "ended": now(), "kwargs": kwargs, "error": True})
            raise
        ended = now()
        self.records.append({
            "call_index": len(self.records), "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended), "result": result,
            "plan_id": kwargs["current_plan_id"],
            "plan_version": kwargs["current_plan_version"],
            "identities": tuple(identity(value) for value in result.candidates),
        })
        return result


class RecordingDiagnosisService:
    def __init__(self, delegate: FailureDiagnosisService) -> None:
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def diagnose_zero_eligible(self, **kwargs: Any):
        started = now()
        try:
            result = self.delegate.diagnose_zero_eligible(**kwargs)
        except Exception:
            self.records.append({"call_index": len(self.records), "started": started,
                                 "ended": now(), "kwargs": kwargs, "error": True})
            raise
        ended = now()
        self.records.append({
            "call_index": len(self.records), "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended), "attempt": kwargs["attempt"],
            "result": result,
        })
        return result


class RecordingReplanPolicy:
    def __init__(self, delegate: BoundedReplanPolicy) -> None:
        self.delegate = delegate
        self.records: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any):
        started = now()
        try:
            result = self.delegate.decide(**kwargs)
        except Exception:
            self.records.append({"call_index": len(self.records), "started": started,
                                 "ended": now(), "kwargs": kwargs, "error": True})
            raise
        ended = now()
        self.records.append({
            "call_index": len(self.records), "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended),
            "attempt": kwargs["current_attempt"], "top_k": kwargs["current_top_k"],
            "diagnosis": kwargs["diagnosis"], "result": result,
        })
        return result


class RecordingShoppingPlanService:
    """Delegate domain operations; record but never simulate a mutation."""

    def __init__(self, delegate: ShoppingPlanService) -> None:
        self.delegate = delegate
        self.select_records: list[dict[str, Any]] = []
        self.evaluate_records: list[dict[str, Any]] = []

    def select_item(self, plan: Any, **kwargs: Any):
        started = now()
        result = self.delegate.select_item(plan, **kwargs)
        ended = now()
        self.select_records.append({
            "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended),
            "pre_version": plan.version, "post_version": result.version,
            "selected_parent_asin": kwargs["selected_parent_asin"],
        })
        return result

    def evaluate_plan(self, plan: Any):
        started = now()
        result = self.delegate.evaluate_plan(plan)
        ended = now()
        self.evaluate_records.append({
            "started": started, "ended": ended,
            "duration_ms": milliseconds(started, ended), "result": result,
        })
        return result


def discover_categories(catalog_path: Path) -> tuple[str, ...]:
    """Read real category members; do not reproduce catalog matching semantics."""

    # Collapse whitespace/case variants deterministically while preserving one
    # exact catalog spelling for the real RecommendationService call.
    values: dict[str, str] = {}
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid catalog JSON at line {line_number}.") from exc
            categories = record.get("categories")
            if not isinstance(categories, list):
                raise ValueError(f"Catalog categories at line {line_number} are invalid.")
            for raw in categories:
                if isinstance(raw, str):
                    category = " ".join(raw.split())
                    if category and CATEGORY_PATTERN.search(category):
                        normalized = category.casefold()
                        previous = values.get(normalized)
                        if previous is None or category < previous:
                            values[normalized] = category
    return tuple(sorted(values.values(), key=lambda value: (value.casefold(), value)))


def requirement(category: str, features: tuple[str, ...]) -> ShoppingRequirement:
    return ShoppingRequirement(
        requirement_id=REQUIREMENT_ID, category=category, quantity=1,
        max_budget=MAX_PRICE, required_features=features,
        soft_preferences=("suitable for laptop use",),
    )


def tool_call(tool: Any, category: str, features: tuple[str, ...], top_k: int):
    result = tool.recommend(
        user_id=USER_ID,
        args=RecommendationToolArgs(
            top_k=top_k, category=category, max_price=MAX_PRICE,
            required_features=features,
        ),
        excluded_parent_asins=(),
    )
    if result.personalization_status != "personalized":
        raise RuntimeError(
            f"Known user resolved as {result.personalization_status!r}; refusing fallback."
        )
    if any(item.score_source != "hybrid" for item in result.items):
        raise RuntimeError("Known-user discovery returned a non-hybrid candidate.")
    return result


def phase_b_probe(
    evidence_service: Any,
    verifier: Any,
    category: str,
    features: tuple[str, ...],
    top10: Any,
) -> tuple[RequirementEvidence, RequirementVerification]:
    current_requirement = requirement(category, features)
    evidence = evidence_service.retrieve(
        plan_id=DISCOVERY_PLAN_ID, plan_version=0,
        requirement=current_requirement,
        candidates=tuple(EvidenceCandidate(
            item_index=item.item_index, parent_asin=item.parent_asin,
        ) for item in top10.items),
    )
    verification = verifier.verify(
        requirement=current_requirement, evidence=evidence,
        current_plan_id=DISCOVERY_PLAN_ID, current_plan_version=0,
    )
    return evidence, verification


def perform_discovery(
    *, catalog_path: Path, recommendation_tool: Any,
    evidence_service: Any, verifier: Any,
) -> dict[str, Any]:
    started = now()
    category_started = now()
    categories = discover_categories(catalog_path)[:MAX_CATEGORY_ATTEMPTS]
    category_ms = milliseconds(category_started)
    phase_a_started = now()
    shortlist: list[dict[str, Any]] = []
    phase_a_cases = 0
    phase_a_calls = 0
    for category in categories:
        for features in FEATURE_SETS:
            phase_a_cases += 1
            top5 = tool_call(recommendation_tool, category, features, TOP_K_INITIAL)
            top10 = tool_call(recommendation_tool, category, features, TOP_K_EXPANDED)
            phase_a_calls += 2
            ids5, ids10 = candidate_identities(top5), candidate_identities(top10)
            unique = len(set(ids5)) == len(ids5) and len(set(ids10)) == len(ids10)
            prefix = ids5 == ids10[:len(ids5)]
            accepted = (
                top5.returned_count > 0
                and top10.returned_count > top5.returned_count
                and prefix and unique and bool(set(ids10) - set(ids5))
            )
            print(
                f"phase_a_case={phase_a_cases} category={category!r} "
                f"features={features!r} top5={top5.returned_count} "
                f"top10={top10.returned_count} prefix={prefix} unique={unique} "
                f"shortlisted={accepted}"
            )
            if accepted:
                shortlist.append({
                    "category": category, "features": features,
                    "top5": top5, "top10": top10,
                })
            if len(shortlist) >= MAX_PHASE_B_CASES:
                break
        if len(shortlist) >= MAX_PHASE_B_CASES:
            break
    phase_a_ms = milliseconds(phase_a_started)

    phase_b_started = now()
    success = None
    exhaustion = None
    phase_b_cases = 0
    candidate_retrieval_calls = 0
    evidence_ms = 0.0
    verification_ms = 0.0
    for case in shortlist[:MAX_PHASE_B_CASES]:
        phase_b_cases += 1
        ev_started = now()
        evidence, verification = phase_b_probe(
            evidence_service, verifier, case["category"],
            case["features"], case["top10"],
        )
        # GroundedEvidenceService performs one restricted retrieval per candidate.
        candidate_retrieval_calls += case["top10"].returned_count
        combined_ms = milliseconds(ev_started)
        # Wrapper records give the exact split; direct services may not expose it.
        if getattr(evidence_service, "records", None):
            evidence_ms += evidence_service.records[-1]["duration_ms"]
        if getattr(verifier, "records", None):
            verification_ms += verifier.records[-1]["duration_ms"]
        if not getattr(evidence_service, "records", None):
            evidence_ms += combined_ms
        first_count = case["top5"].returned_count
        first = verification.candidates[:first_count]
        all_candidates = verification.candidates
        first_eligible = tuple(
            identity(value) for value in first
            if value.status is CandidateVerificationStatus.ELIGIBLE
        )
        all_eligible = eligible_identities(verification)
        new_ids = set(candidate_identities(case["top10"])) - set(
            candidate_identities(case["top5"])
        )
        new_eligible = tuple(value for value in all_eligible if value in new_ids)
        case.update({
            "evidence": evidence, "verification": verification,
            "first_eligible": first_eligible, "all_eligible": all_eligible,
            "new_ids": new_ids, "new_eligible": new_eligible,
        })
        print(
            f"phase_b_case={phase_b_cases} category={case['category']!r} "
            f"features={case['features']!r} top5_eligible={first_eligible} "
            f"top10_eligible={all_eligible} new_eligible={new_eligible}"
        )
        if not first_eligible and new_eligible:
            success = case
            break
        if not first_eligible and not all_eligible and exhaustion is None:
            exhaustion = case
    phase_b_ms = milliseconds(phase_b_started)
    return {
        "categories": categories, "phase_a_cases": phase_a_cases,
        "phase_a_calls": phase_a_calls, "shortlist_count": len(shortlist),
        "phase_b_cases": phase_b_cases,
        "candidate_retrieval_calls": candidate_retrieval_calls,
        "category_ms": category_ms, "phase_a_ms": phase_a_ms,
        "phase_b_ms": phase_b_ms, "evidence_ms": evidence_ms,
        "verification_ms": verification_ms, "total_ms": milliseconds(started),
        "mode": "success" if success is not None else "exhaustion" if exhaustion is not None else "inconclusive",
        "case": success if success is not None else exhaustion,
    }


def new_state(
    plan_service: ShoppingPlanService, category: str, features: tuple[str, ...]
) -> ShoppingWorkflowState:
    plan = plan_service.create_plan(
        plan_id="v2-09-5-real-artifact-formal", user_id=USER_ID,
        currency="USD", total_budget=MAX_PRICE,
        requirements=(requirement(category, features),),
    )
    return ShoppingWorkflowState(agent_state=AgentState(
        user_id=USER_ID, session_id="v2-09-5-real-artifact-formal-session",
        shopping_plan=plan,
    ))


def referenced_chunk_ids(candidate: Any) -> set[str]:
    return {
        chunk_id
        for constraint in candidate.constraints
        for chunk_id in (
            *constraint.supporting_chunk_ids,
            *constraint.contradicting_chunk_ids,
        )
    }


def attempt_integrity(rec: dict[str, Any], ev: dict[str, Any], ver: dict[str, Any]) -> bool:
    recommendation_ids = rec["identities"]
    evidence: RequirementEvidence = ev["result"]
    verification: RequirementVerification = ver["result"]
    evidence_ids = tuple(identity(value) for value in evidence.products)
    verification_ids = tuple(identity(value) for value in verification.candidates)
    if recommendation_ids != ev["identities"] or evidence_ids != recommendation_ids:
        return False
    if verification_ids != recommendation_ids:
        return False
    chunks_by_identity = {
        identity(product): {snippet.chunk_id for snippet in product.snippets}
        for product in evidence.products
    }
    return all(
        referenced_chunk_ids(candidate) <= chunks_by_identity.get(identity(candidate), set())
        for candidate in verification.candidates
    )


def print_attempt(index: int, rec: dict[str, Any], ev: dict[str, Any], ver: dict[str, Any]) -> None:
    section(f"FORMAL ATTEMPT {index}")
    print(f"top_k={rec['args'].top_k}")
    print(f"candidate_identities={rec['identities']}")
    print(f"evidence_chunk_ids={ev['chunk_ids']}")
    for candidate in ver["result"].candidates:
        constraints = tuple(
            (value.constraint, value.status.value,
             value.supporting_chunk_ids, value.contradicting_chunk_ids)
            for value in candidate.constraints
        )
        print(
            f"candidate={identity(candidate)} status={candidate.status.value} "
            f"constraints={constraints}"
        )


def run_formal(
    *, case: dict[str, Any], base_tool: Any, base_evidence: Any,
    base_verifier: Any, base_plan_service: ShoppingPlanService,
) -> dict[str, Any]:
    recommendation = RecordingRecommendationTool(base_tool)
    evidence = RecordingEvidenceService(base_evidence)
    verification = RecordingVerificationService(base_verifier)
    diagnosis = RecordingDiagnosisService(FailureDiagnosisService())
    policy = RecordingReplanPolicy(BoundedReplanPolicy())
    plan_service = RecordingShoppingPlanService(base_plan_service)
    planner = AuditedDeterministicPlanner()
    graph_started = now()
    graph = build_shopping_workflow(
        recommendation, plan_service, planner=planner,
        evidence_service=evidence, verification_service=verification,
        diagnosis_service=diagnosis, replan_policy=policy,
    )
    graph_ms = milliseconds(graph_started)
    initial = new_state(base_plan_service, case["category"], case["features"])
    started = now()
    final = ShoppingWorkflowState.model_validate(
        graph.invoke(initial, config={"recursion_limit": 50})
    )
    return {
        "initial": initial, "final": final, "planner": planner,
        "recommendation": recommendation, "evidence": evidence,
        "verification": verification, "diagnosis": diagnosis,
        "policy": policy, "plan_service": plan_service,
        "graph_ms": graph_ms, "workflow_ms": milliseconds(started),
    }


def check_formal(run: dict[str, Any], mode: str) -> dict[str, bool]:
    recs = run["recommendation"].records
    evs = run["evidence"].records
    vers = run["verification"].records
    diagnoses = run["diagnosis"].records
    policies = run["policy"].records
    mutations = run["plan_service"].select_records
    initial: ShoppingWorkflowState = run["initial"]
    final: ShoppingWorkflowState = run["final"]
    planner: AuditedDeterministicPlanner = run["planner"]
    checks: dict[str, bool] = {
        "two_recommendation_calls": len(recs) == 2,
        "two_evidence_calls": len(evs) == 2,
        "two_verification_calls": len(vers) == 2,
    }
    if not all(checks.values()):
        return checks
    args0, args1 = recs[0]["args"], recs[1]["args"]
    ids0, ids1 = recs[0]["identities"], recs[1]["identities"]
    new_ids = set(ids1) - set(ids0)
    eligible0 = set(eligible_identities(vers[0]["result"]))
    eligible1 = set(eligible_identities(vers[1]["result"]))
    plan_json = json.dumps(
        final.agent_state.shopping_plan.to_domain_dict(), ensure_ascii=False
    ).casefold()
    checks.update({
        "top_k_5_to_10": args0.top_k == 5 and args1.top_k == 10,
        "constraints_preserved": (
            args0.category == args1.category
            and args0.max_price == args1.max_price
            and args0.required_features == args1.required_features
        ),
        "attempt0_nonempty_zero_eligible": bool(ids0) and not eligible0,
        "expanded_candidate_set": bool(new_ids),
        "attempt0_integrity": attempt_integrity(recs[0], evs[0], vers[0]),
        "attempt1_integrity": attempt_integrity(recs[1], evs[1], vers[1]),
        "fresh_evidence_calls": evs[0]["result"] is not evs[1]["result"],
        "fresh_verification_calls": vers[0]["result"] is not vers[1]["result"],
        "planner_item_index_hidden": not planner.item_index_exposed,
        "plan_domain_no_evidence": all(
            value not in plan_json for value in ("chunk", "evidence", "verification")
        ),
        "no_attempt_2": len(recs) == 2 and final.replan_attempt == 1,
        "initial_contract": (
            initial.agent_state.shopping_plan.version == 0
            and initial.recommendation_top_k == 5
            and initial.replan_attempt == 0
        ),
    })
    diagnosis0 = diagnoses[0]["result"] if diagnoses else None
    directive0 = policies[0]["result"] if policies else None
    checks["attempt0_diagnosis"] = bool(
        diagnosis0
        and diagnosis0.failure_type is FailureType.NO_ELIGIBLE_CANDIDATES
        and diagnosis0.recoverability is FailureRecoverability.RECOVERABLE
    )
    checks["attempt0_expand_directive"] = bool(
        directive0
        and directive0.action is ReplanAction.EXPAND_CANDIDATE_POOL
        and directive0.previous_top_k == 5
        and directive0.next_top_k == 10
    )
    checks["pre_mutation_version_stable"] = bool(
        all(value["plan_version"] == 0 for value in evs)
        and all(value["plan_version"] == 0 for value in vers)
        and diagnosis0
        and diagnosis0.diagnosed_at_plan_version == 0
        and directive0
        and directive0.source_plan_version == 0
    )
    if mode == "success":
        selected = final.agent_state.shopping_plan.selected_items
        selected_identity = None
        if selected:
            selected_parent = selected[0].parent_asin
            selected_identity = next(
                (value for value in ids1 if value[1] == selected_parent), None
            )
        checks.update({
            "new_candidate_is_eligible": bool(new_ids & eligible1),
            "selected_from_attempt1_eligible": selected_identity in eligible1,
            "planner_allowlist_exact": (
                set(planner.selectable_parent_asins)
                == {value[1] for value in eligible1}
            ),
            "exactly_one_mutation": len(mutations) == 1,
            "version_0_to_1": bool(
                len(mutations) == 1
                and mutations[0]["pre_version"] == 0
                and mutations[0]["post_version"] == 1
                and final.agent_state.shopping_plan.version == 1
            ),
            "final_ready": final.route is WorkflowRoute.READY,
            "success_transients_cleared": (
                final.current_evidence is None
                and final.current_verification is None
                and final.current_evidence_attempt is None
                and final.current_verification_attempt is None
                and final.current_failure_diagnosis is None
                and final.current_replan_directive is None
            ),
            "attempt0_history_retained": (
                len(final.failure_history) == 1 and len(final.replan_history) == 1
            ),
            "selected_only_provenance": (
                len(final.selected_evidence) == 1
                and len(final.selected_verifications) == 1
                and selected_identity is not None
                and identity(final.selected_evidence[0]) == selected_identity
                and identity(final.selected_verifications[0].candidate) == selected_identity
            ),
            "diagnosis_policy_counts": len(diagnoses) == 1 and len(policies) == 1,
        })
    else:
        diagnosis1 = diagnoses[1]["result"] if len(diagnoses) == 2 else None
        directive1 = policies[1]["result"] if len(policies) == 2 else None
        checks.update({
            "attempt1_zero_eligible": not eligible1,
            "attempt1_exhausted": bool(
                diagnosis1
                and diagnosis1.failure_type is FailureType.REPLAN_ATTEMPTS_EXHAUSTED
                and diagnosis1.recoverability is FailureRecoverability.NON_RECOVERABLE
            ),
            "attempt1_stop_directive": bool(
                directive1
                and directive1.action is ReplanAction.STOP_CONFLICT
                and directive1.next_top_k is None
            ),
            "diagnosis_policy_counts": len(diagnoses) == 2 and len(policies) == 2,
            "no_mutation": not mutations,
            "version_unchanged": final.agent_state.shopping_plan.version == 0,
            "selected_items_empty": not final.agent_state.shopping_plan.selected_items,
            "final_conflict": final.route is WorkflowRoute.CONFLICT,
            "exhaustion_error": (
                final.agent_state.error_state
                == "constraint_verification:replan_attempts_exhausted"
            ),
            "attempt1_runtime_bound": (
                final.current_evidence_attempt == 1
                and final.current_verification_attempt == 1
            ),
        })
    return checks


def print_latency(run: dict[str, Any]) -> None:
    section("POST-DISCOVERY FORMAL LATENCY")
    recs = run["recommendation"].records
    evs = run["evidence"].records
    vers = run["verification"].records
    diagnoses = run["diagnosis"].records
    policies = run["policy"].records
    for index in range(min(2, len(recs), len(evs), len(vers))):
        print(f"attempt{index}_recommendation_ms={recs[index]['duration_ms']:.3f}")
        print(f"attempt{index}_evidence_ms={evs[index]['duration_ms']:.3f}")
        print(f"attempt{index}_verification_ms={vers[index]['duration_ms']:.3f}")
    if len(recs) == 2 and vers:
        print(
            "diagnosis_replan_reset_interval_ms="
            f"{milliseconds(vers[0]['ended'], recs[1]['started']):.3f}"
        )
    print(f"diagnosis_total_ms={sum(x['duration_ms'] for x in diagnoses):.3f}")
    print(f"policy_total_ms={sum(x['duration_ms'] for x in policies):.3f}")
    plan_service = run["plan_service"]
    print(f"selection_update_ms={sum(x['duration_ms'] for x in plan_service.select_records):.3f}")
    print(f"evaluation_ms={sum(x['duration_ms'] for x in plan_service.evaluate_records):.3f}")
    print(f"graph_construction_ms={run['graph_ms']:.3f}")
    print(f"formal_workflow_total_ms={run['workflow_ms']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    serving_dir = root / "artifacts/recommendation/serving_v2"
    catalog_path = root / "data/processed/recommendation/product_catalog.jsonl"
    knowledge_dir = root / "artifacts/recommendation/knowledge"
    retrieval_dir = root / "artifacts/recommendation/retrieval"
    model_path = root / "models/bge-m3"

    section("ENVIRONMENT")
    import torch

    print(f"git_head={git_head(root)}")
    print(f"git_working_tree_clean={git_clean(root)}")
    print(f"langgraph_version={importlib.metadata.version('langgraph')}")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("This real-artifact smoke requires CUDA.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device.")
    device_index = 0 if device.index is None else device.index
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {device_index} is unavailable.")
    torch.cuda.set_device(device_index)
    print(f"gpu={torch.cuda.get_device_name(device_index)}")
    for name, path in (
        ("serving_path", serving_dir), ("catalog_path", catalog_path),
        ("knowledge_path", knowledge_dir), ("retrieval_path", retrieval_dir),
        ("bge_path", model_path),
    ):
        print(f"{name}={path}")

    section("STRICT INITIALIZATION")
    started = now()
    recommendation_service = RecommendationService(
        bundle_dir=serving_dir, catalog_path=catalog_path, device=args.device,
    )
    recommendation_init_ms = milliseconds(started)
    recommendation_tool = RecommendationToolAdapter(recommendation_service)
    started = now()
    artifacts = ChunkRetrievalArtifacts(
        retrieval_dir=retrieval_dir, knowledge_dir=knowledge_dir,
        validation_mode=ValidationMode.STRICT, expected_dimension=1024,
    )
    artifact_load_ms = milliseconds(started)
    started = now()
    encoder = BGEM3QueryEncoder(
        model_path=model_path, device=args.device, use_fp16=True,
        batch_size=8, max_length=2048, embedding_dimension=1024,
    )
    model_load_ms = milliseconds(started)
    started = now()
    retriever = ChunkRetriever(
        encoder=encoder, backend=NumPyExactBackend(artifacts.embeddings),
        artifacts=artifacts,
    )
    evidence_service = GroundedEvidenceService(retriever=retriever)
    verifier = EvidenceConstraintVerifier()
    plan_service = ShoppingPlanService()
    service_construction_ms = milliseconds(started)
    print(f"recommendation_service_init_ms={recommendation_init_ms:.3f}")
    print(f"retrieval_artifact_load_ms={artifact_load_ms:.3f}")
    print(f"bge_m3_load_ms={model_load_ms:.3f}")
    print(f"service_construction_ms={service_construction_ms:.3f}")
    print(f"chunk_count={artifacts.row_count}")
    print(f"embedding_dimension={artifacts.dimension}")
    print("validation_mode=STRICT")

    section("BOUNDED DISCOVERY")
    discovery_evidence = RecordingEvidenceService(evidence_service)
    discovery_verifier = RecordingVerificationService(verifier)
    discovery = perform_discovery(
        catalog_path=catalog_path, recommendation_tool=recommendation_tool,
        evidence_service=discovery_evidence, verifier=discovery_verifier,
    )
    print(f"categories_considered={len(discovery['categories'])}")
    print(f"phase_a_cases_tested={discovery['phase_a_cases']}")
    print(f"phase_a_recommendation_calls={discovery['phase_a_calls']}")
    print(f"phase_b_cases_tested={discovery['phase_b_cases']}")
    print(f"phase_b_candidate_retrieval_calls={discovery['candidate_retrieval_calls']}")
    print(f"category_discovery_ms={discovery['category_ms']:.3f}")
    print(f"phase_a_ms={discovery['phase_a_ms']:.3f}")
    print(f"phase_b_evidence_ms={discovery['evidence_ms']:.3f}")
    print(f"phase_b_verification_ms={discovery['verification_ms']:.3f}")
    print(f"phase_b_total_ms={discovery['phase_b_ms']:.3f}")
    print(f"discovery_total_ms={discovery['total_ms']:.3f}")
    if discovery["mode"] == "inconclusive":
        print("\nREAL REPLAN SUCCESS CASE NOT FOUND WITHIN BOUNDED DISCOVERY")
        print("DISCOVERY INCONCLUSIVE")
        return 2
    if discovery["mode"] == "exhaustion":
        print("\nREAL REPLAN SUCCESS CASE NOT FOUND WITHIN BOUNDED DISCOVERY")
        print("Using the first real retry-exhaustion fallback case.")
    else:
        print("\nREAL REPLAN SUCCESS CASE DISCOVERED")
    case = discovery["case"]
    assert case is not None
    print(f"selected_category={case['category']!r}")
    print(f"selected_required_features={case['features']!r}")

    # Fresh plan, state, planner, and recording counters isolate formal execution.
    formal = run_formal(
        case=case, base_tool=recommendation_tool,
        base_evidence=evidence_service, base_verifier=verifier,
        base_plan_service=plan_service,
    )
    recs = formal["recommendation"].records
    evs = formal["evidence"].records
    vers = formal["verification"].records
    for index in range(min(2, len(recs), len(evs), len(vers))):
        print_attempt(index, recs[index], evs[index], vers[index])

    section("DIAGNOSIS / DIRECTIVES")
    for record in formal["diagnosis"].records:
        value = record["result"]
        print(
            f"attempt={record['attempt']} failure_type={value.failure_type.value} "
            f"recoverability={value.recoverability.value} "
            f"verification_pattern={value.verification_pattern.value}"
        )
    for record in formal["policy"].records:
        value = record["result"]
        print(
            f"attempt={record['attempt']} action={value.action.value} "
            f"previous_top_k={value.previous_top_k} next_top_k={value.next_top_k}"
        )

    checks = check_formal(formal, discovery["mode"])
    print_latency(formal)
    section("INTEGRITY SUMMARY")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    if passed and discovery["mode"] == "success":
        print("\nV2-09.5 REAL ARTIFACT SUCCESS-AFTER-REPLAN SMOKE: PASS")
        return 0
    if passed:
        print("\nV2-09.5 REAL ARTIFACT RETRY-EXHAUSTION SMOKE: PASS")
        return 0
    print("\nV2-09.5 REAL ARTIFACT E2E SMOKE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
