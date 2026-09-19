"""V2-09.4 real-artifact end-to-end smoke for the Ubuntu GPU server.

This script is intentionally read-only.  It loads the published serving,
knowledge, and retrieval artifacts, then executes the production LangGraph
shopping workflow with real recommendation, retrieval, and verification
services.  It writes no artifact and contains no replacement business logic.

Run from the repository root on the validated Ubuntu server::

    python scripts/25_smoke_v2_09_4_real_artifact_e2e.py
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    GroundedEvidenceService,
    RequirementEvidence,
)
from src.agentrec.planning import (  # noqa: E402
    SelectCandidateDecision,
    SelectRequirementDecision,
)
from src.agentrec.recommendation import RecommendationService  # noqa: E402
from src.agentrec.retrieval import (  # noqa: E402
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from src.agentrec.services import ShoppingPlanService  # noqa: E402
from src.agentrec.tools import RecommendationToolAdapter  # noqa: E402
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
REQUIREMENT_ID = "hub"
CATEGORY = "Hubs"
MAX_PRICE = 500.0
REQUIRED_FEATURES = ("HDMI", "USB-C")
SOFT_PREFERENCES = ("suitable for laptop use",)
RUN_LABELS = ("cold-ish", "warm-1", "warm-2")


def section(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def git_head(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def contains_key(value: object, forbidden_key: str) -> bool:
    """Recursively inspect planner context without mutating it."""

    if isinstance(value, Mapping):
        return forbidden_key in value or any(
            contains_key(item, forbidden_key) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(contains_key(item, forbidden_key) for item in value)
    return False


class AuditedDeterministicPlanner:
    """Select the only requirement, then the first eligible context candidate.

    This smoke planner proposes identity only.  The production workflow still
    validates its decision and owns every mutation.  It also asserts that the
    system-only ``item_index`` is never exposed in planner context.
    """

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
            requirement_id = requirements[0].get("requirement_id")
            return SelectRequirementDecision(
                plan_id=context["plan_id"],
                plan_version=context["plan_version"],
                requirement_id=requirement_id,
                reason="real_artifact_smoke_deterministic_requirement",
            )
        if key.startswith("select_candidate:"):
            candidates = context.get("candidates")
            if not isinstance(candidates, tuple) or not candidates:
                raise ValueError("Candidate planner received no eligible candidates.")
            self.selectable_parent_asins = tuple(
                candidate["parent_asin"] for candidate in candidates
            )
            return SelectCandidateDecision(
                plan_id=context["plan_id"],
                plan_version=context["plan_version"],
                requirement_id=context["requirement_id"],
                parent_asin=self.selectable_parent_asins[0],
                reason="real_artifact_smoke_first_eligible_candidate",
            )
        raise ValueError(f"Unsupported planner decision_key={key!r}.")


class TimedRecommendationTool:
    def __init__(self, delegate: RecommendationToolAdapter) -> None:
        self.delegate = delegate
        self.calls = 0
        self.elapsed_ms = 0.0
        self.result = None

    def recommend(self, **kwargs: Any):
        started = time.perf_counter()
        try:
            result = self.delegate.recommend(**kwargs)
        finally:
            self.elapsed_ms += elapsed_ms(started)
            self.calls += 1
        self.result = result
        return result


class TimedEvidenceService:
    def __init__(self, delegate: GroundedEvidenceService) -> None:
        self.delegate = delegate
        self.calls = 0
        self.elapsed_ms = 0.0
        self.result: RequirementEvidence | None = None

    def retrieve(self, **kwargs: Any):
        started = time.perf_counter()
        try:
            result = self.delegate.retrieve(**kwargs)
        finally:
            self.elapsed_ms += elapsed_ms(started)
            self.calls += 1
        self.result = result
        return result


class TimedVerificationService:
    def __init__(self, delegate: EvidenceConstraintVerifier) -> None:
        self.delegate = delegate
        self.calls = 0
        self.elapsed_ms = 0.0
        self.result: RequirementVerification | None = None

    def verify(self, **kwargs: Any):
        started = time.perf_counter()
        try:
            result = self.delegate.verify(**kwargs)
        finally:
            self.elapsed_ms += elapsed_ms(started)
            self.calls += 1
        self.result = result
        return result


def new_initial_state(plan_service: ShoppingPlanService, run_number: int) -> ShoppingWorkflowState:
    requirement = ShoppingRequirement(
        requirement_id=REQUIREMENT_ID,
        category=CATEGORY,
        quantity=1,
        max_budget=MAX_PRICE,
        required_features=REQUIRED_FEATURES,
        soft_preferences=SOFT_PREFERENCES,
    )
    plan = plan_service.create_plan(
        plan_id=f"v2-09-4-real-smoke-{run_number}",
        user_id=USER_ID,
        currency="USD",
        total_budget=MAX_PRICE,
        requirements=(requirement,),
    )
    return ShoppingWorkflowState(
        agent_state=AgentState(
            user_id=USER_ID,
            session_id=f"v2-09-4-real-smoke-session-{run_number}",
            shopping_plan=plan,
        )
    )


def run_workflow(
    *,
    run_number: int,
    recommendation_tool: RecommendationToolAdapter,
    evidence_service: GroundedEvidenceService,
    verifier: EvidenceConstraintVerifier,
    plan_service: ShoppingPlanService,
) -> dict[str, Any]:
    timed_tool = TimedRecommendationTool(recommendation_tool)
    timed_evidence = TimedEvidenceService(evidence_service)
    timed_verifier = TimedVerificationService(verifier)
    planner = AuditedDeterministicPlanner()
    graph = build_shopping_workflow(
        timed_tool,
        plan_service,
        planner=planner,
        evidence_service=timed_evidence,
        verification_service=timed_verifier,
    )
    initial = new_initial_state(plan_service, run_number)
    started = time.perf_counter()
    final = ShoppingWorkflowState.model_validate(
        graph.invoke(initial, config={"recursion_limit": 50})
    )
    return {
        "initial": initial,
        "final": final,
        "recommendation": timed_tool.result,
        "evidence": timed_evidence.result,
        "verification": timed_verifier.result,
        "planner": planner,
        "calls": {
            "recommendation": timed_tool.calls,
            "evidence": timed_evidence.calls,
            "verification": timed_verifier.calls,
        },
        "latency": {
            "recommendation_ms": timed_tool.elapsed_ms,
            "evidence_ms": timed_evidence.elapsed_ms,
            "verification_ms": timed_verifier.elapsed_ms,
            "workflow_ms": elapsed_ms(started),
        },
    }


def print_requirement(state: ShoppingWorkflowState) -> None:
    requirement = state.agent_state.shopping_plan.requirements[0]
    print(json.dumps(requirement.model_dump(mode="json"), ensure_ascii=False, indent=2))


def print_candidates(result: Any) -> None:
    if result is None:
        print("No RecommendationToolResult was produced.")
        return
    for item in result.items:
        print(
            f"rank={item.rank} item_index={item.item_index} "
            f"parent_asin={item.parent_asin} title={item.title!r} price={item.price}"
        )


def print_evidence(evidence: RequirementEvidence | None) -> set[str]:
    if evidence is None:
        print("No RequirementEvidence was produced.")
        return set()
    ids: set[str] = set()
    for product in evidence.products:
        for snippet in product.snippets:
            ids.add(snippet.chunk_id)
            preview = " ".join(snippet.text.split())[:300]
            print(
                f"item_index={product.item_index} parent_asin={product.parent_asin} "
                f"chunk_id={snippet.chunk_id} score={snippet.similarity_score:.6f} "
                f"chunk_type={snippet.chunk_type.value} text={preview!r}"
            )
    return ids


def print_verification(verification: RequirementVerification | None) -> None:
    if verification is None:
        print("No RequirementVerification was produced.")
        return
    for candidate in verification.candidates:
        print(
            f"candidate item_index={candidate.item_index} "
            f"parent_asin={candidate.parent_asin} status={candidate.status.value}"
        )
        for constraint in candidate.constraints:
            print(
                f"  {constraint.constraint}: status={constraint.status.value} "
                f"reason={constraint.reason.value} "
                f"supporting_chunk_ids={list(constraint.supporting_chunk_ids)} "
                f"contradicting_chunk_ids={list(constraint.contradicting_chunk_ids)}"
            )


def candidate_groups(verification: RequirementVerification | None) -> dict[str, list[str]]:
    groups = {"ELIGIBLE": [], "UNVERIFIED": [], "INELIGIBLE": []}
    if verification is not None:
        for candidate in verification.candidates:
            groups[candidate.status.value.upper()].append(candidate.parent_asin)
    return groups


def inspect_run(run: dict[str, Any]) -> tuple[dict[str, bool], bool]:
    initial: ShoppingWorkflowState = run["initial"]
    final: ShoppingWorkflowState = run["final"]
    recommendation = run["recommendation"]
    evidence: RequirementEvidence | None = run["evidence"]
    verification: RequirementVerification | None = run["verification"]
    planner: AuditedDeterministicPlanner = run["planner"]
    initial_version = initial.agent_state.shopping_plan.version
    final_plan = final.agent_state.shopping_plan

    recommendation_pairs = set()
    recommendation_parents = set()
    unique_candidates = False
    if recommendation is not None:
        recommendation_pairs = {
            (item.item_index, item.parent_asin) for item in recommendation.items
        }
        recommendation_parents = {item.parent_asin for item in recommendation.items}
        unique_candidates = (
            len(recommendation_pairs) == recommendation.returned_count
            and len(recommendation_parents) == recommendation.returned_count
        )
    evidence_pairs = set()
    evidence_chunk_ids: set[str] = set()
    evidence_chunk_ids_by_pair: dict[tuple[int, str], set[str]] = {}
    if evidence is not None:
        evidence_pairs = {
            (product.item_index, product.parent_asin) for product in evidence.products
        }
        evidence_chunk_ids = {
            snippet.chunk_id
            for product in evidence.products
            for snippet in product.snippets
        }
        evidence_chunk_ids_by_pair = {
            (product.item_index, product.parent_asin): {
                snippet.chunk_id for snippet in product.snippets
            }
            for product in evidence.products
        }
    outside = sorted(evidence_pairs - recommendation_pairs)
    referenced_ids = set()
    references_match_candidate = True
    verification_pairs = set()
    unknown_promoted = False
    if verification is not None:
        verification_pairs = {
            (candidate.item_index, candidate.parent_asin)
            for candidate in verification.candidates
        }
        for candidate in verification.candidates:
            statuses = {constraint.status for constraint in candidate.constraints}
            unknown_promoted |= (
                ConstraintVerificationStatus.UNKNOWN in statuses
                and candidate.status is CandidateVerificationStatus.ELIGIBLE
            )
            for constraint in candidate.constraints:
                referenced_ids.update(constraint.supporting_chunk_ids)
                referenced_ids.update(constraint.contradicting_chunk_ids)
                references_match_candidate &= set(
                    constraint.supporting_chunk_ids
                    + constraint.contradicting_chunk_ids
                ) <= evidence_chunk_ids_by_pair.get(
                    (candidate.item_index, candidate.parent_asin), set()
                )

    zero_eligible = (
        verification is not None and not verification.eligible_parent_asins
    )
    selected = final_plan.selected_items[0] if final_plan.selected_items else None
    selected_verification = (
        final.selected_verifications[0] if final.selected_verifications else None
    )
    selected_evidence = final.selected_evidence[0] if final.selected_evidence else None
    selected_pair = None
    if selected_verification is not None:
        selected_pair = (
            selected_verification.candidate.item_index,
            selected_verification.candidate.parent_asin,
        )
    plan_domain = final_plan.to_domain_dict()
    serialized_plan = json.dumps(plan_domain, ensure_ascii=False).casefold()

    if zero_eligible:
        plan_transition = (
            final.route is WorkflowRoute.CONFLICT
            and final.agent_state.error_state
            == "constraint_verification:no_eligible_candidates"
            and not final_plan.selected_items
            and final_plan.version == initial_version
        )
        planner_gate = not planner.selectable_parent_asins
        identity_alignment = (
            unique_candidates
            and evidence_pairs == recommendation_pairs
            and verification_pairs == evidence_pairs
        )
    else:
        plan_transition = (
            selected_verification is not None
            and selected_evidence is not None
            and evidence is not None
            and verification is not None
            and evidence.retrieved_at_plan_version == initial_version
            and verification.verified_at_plan_version == initial_version
            and selected_verification.selected_at_plan_version == final_plan.version
            and selected_evidence.selected_at_plan_version == final_plan.version
            and final_plan.version == initial_version + 1
        )
        planner_gate = (
            selected is not None
            and selected.parent_asin in planner.selectable_parent_asins
            and verification is not None
            and selected.parent_asin in verification.eligible_parent_asins
            and not planner.item_index_exposed
        )
        identity_alignment = (
            unique_candidates
            and evidence_pairs == recommendation_pairs
            and verification_pairs == evidence_pairs
            and selected is not None
            and selected_pair in recommendation_pairs
            and selected.parent_asin == selected_pair[1]
            and selected_evidence is not None
            and (selected_evidence.item_index, selected_evidence.parent_asin)
            == selected_pair
        )

    checks = {
        "candidate_restriction": not outside and bool(recommendation_pairs),
        "verification_chunk_reference_integrity": (
            referenced_ids <= evidence_chunk_ids and references_match_candidate
        ),
        "identity_alignment": identity_alignment,
        "plan_version_transition": plan_transition,
        "planner_allowlist_integrity": planner_gate and not planner.item_index_exposed,
        "shopping_plan_evidence_leakage": (
            "evidence" not in serialized_plan and "chunk_id" not in serialized_plan
        ),
        "shopping_plan_verification_leakage": "verification" not in serialized_plan,
        "unknown_not_promoted_to_eligible": not unknown_promoted,
    }
    print(f"outside_candidate_set = {outside}")
    return checks, zero_eligible


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
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("This real-artifact smoke requires CUDA.")
    parsed_device = torch.device(args.device)
    if parsed_device.type != "cuda":
        raise ValueError("--device must be a CUDA device.")
    device_index = 0 if parsed_device.index is None else parsed_device.index
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {device_index} is unavailable.")
    torch.cuda.set_device(device_index)
    print(f"gpu={torch.cuda.get_device_name(device_index)}")
    print(f"serving_dir={serving_dir}")
    print(f"catalog_path={catalog_path}")
    print(f"knowledge_dir={knowledge_dir}")
    print(f"retrieval_dir={retrieval_dir}")
    print(f"bge_m3_path={model_path}")

    section("INITIALIZATION")
    started = time.perf_counter()
    recommendation_service = RecommendationService(
        bundle_dir=serving_dir,
        catalog_path=catalog_path,
        device=args.device,
    )
    recommendation_init = elapsed_ms(started)
    recommendation_tool = RecommendationToolAdapter(recommendation_service)

    started = time.perf_counter()
    retrieval_artifacts = ChunkRetrievalArtifacts(
        retrieval_dir=retrieval_dir,
        knowledge_dir=knowledge_dir,
        validation_mode=ValidationMode.STRICT,
        expected_dimension=1024,
    )
    retrieval_load = elapsed_ms(started)

    started = time.perf_counter()
    encoder = BGEM3QueryEncoder(
        model_path=model_path,
        device=args.device,
        use_fp16=True,
        batch_size=8,
        max_length=2048,
        embedding_dimension=1024,
    )
    model_load = elapsed_ms(started)
    backend = NumPyExactBackend(retrieval_artifacts.embeddings)
    retriever = ChunkRetriever(
        encoder=encoder, backend=backend, artifacts=retrieval_artifacts
    )
    evidence_service = GroundedEvidenceService(retriever=retriever)
    verifier = EvidenceConstraintVerifier()
    plan_service = ShoppingPlanService()
    print(f"recommendation_service_init_ms={recommendation_init:.3f}")
    print(f"retrieval_artifact_load_ms={retrieval_load:.3f}")
    print(f"bge_m3_load_ms={model_load:.3f}")
    print(f"chunk_count={retrieval_artifacts.row_count}")
    print(f"embedding_dimension={retrieval_artifacts.dimension}")

    section("REQUIREMENT")
    preview_state = new_initial_state(plan_service, 0)
    print_requirement(preview_state)

    runs = []
    for run_number, label in enumerate(RUN_LABELS, start=1):
        run = run_workflow(
            run_number=run_number,
            recommendation_tool=recommendation_tool,
            evidence_service=evidence_service,
            verifier=verifier,
            plan_service=plan_service,
        )
        run["label"] = label
        runs.append(run)
    audited = runs[0]

    section("REAL RECOMMENDATION CANDIDATES")
    print_candidates(audited["recommendation"])
    print(f"recommendation_calls={audited['calls']['recommendation']}")

    section("REAL EVIDENCE")
    print_evidence(audited["evidence"])

    section("CONSTRAINT VERIFICATION")
    print_verification(audited["verification"])

    section("CANDIDATE SUMMARY")
    groups = candidate_groups(audited["verification"])
    for name in ("ELIGIBLE", "UNVERIFIED", "INELIGIBLE"):
        print(f"{name}={groups[name]}")

    section("PLANNER GATE")
    planner: AuditedDeterministicPlanner = audited["planner"]
    verification: RequirementVerification | None = audited["verification"]
    eligible = () if verification is None else verification.eligible_parent_asins
    selected = audited["final"].selected_parent_asin
    if selected is None and audited["final"].agent_state.shopping_plan.selected_items:
        selected = audited["final"].agent_state.shopping_plan.selected_items[0].parent_asin
    print(f"eligible_parent_asins={list(eligible)}")
    print(f"planner_selectable_parent_asins={list(planner.selectable_parent_asins)}")
    print(f"selected_parent_asin={selected}")
    print(f"planner_context_contains_item_index={planner.item_index_exposed}")

    section("PLAN VERSION")
    evidence = audited["evidence"]
    final = audited["final"]
    selected_verification = (
        final.selected_verifications[0] if final.selected_verifications else None
    )
    print(f"initial_plan_version={audited['initial'].agent_state.shopping_plan.version}")
    print(
        "retrieved_at_plan_version="
        f"{None if evidence is None else evidence.retrieved_at_plan_version}"
    )
    print(
        "verified_at_plan_version="
        f"{None if verification is None else verification.verified_at_plan_version}"
    )
    print(
        "selected_at_plan_version="
        f"{None if selected_verification is None else selected_verification.selected_at_plan_version}"
    )
    print(f"final_plan_version={final.agent_state.shopping_plan.version}")

    section("IDENTITY / DOMAIN LEAKAGE")
    checks, zero_eligible = inspect_run(audited)
    print(
        "shopping_plan_json="
        + json.dumps(
            final.agent_state.shopping_plan.to_domain_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    section("LATENCY")
    for run in runs:
        values = " ".join(
            f"{name}={value:.3f}"
            for name, value in run["latency"].items()
        )
        print(f"{run['label']}: {values}")
    for metric in ("recommendation_ms", "evidence_ms", "verification_ms", "workflow_ms"):
        median_value = statistics.median(run["latency"][metric] for run in runs[1:])
        print(f"warm_median_{metric}={median_value:.3f}")

    section("INTEGRITY SUMMARY")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    all_integrity_passed = all(checks.values())
    recommendation_exists = (
        audited["recommendation"] is not None
        and audited["recommendation"].returned_count > 0
    )
    exactly_once = all(
        run["calls"]["recommendation"] == 1 for run in runs
    )
    print(f"one_recommendation_call_per_workflow: {'PASS' if exactly_once else 'FAIL'}")
    all_integrity_passed &= exactly_once

    if zero_eligible and all_integrity_passed:
        print("\nSYSTEM INTEGRITY: PASS")
        print("REAL DATA VERIFICATION: UNRESOLVED")
        return 0
    successful_mutation = (
        recommendation_exists
        and final.route is WorkflowRoute.READY
        and len(final.agent_state.shopping_plan.selected_items) == 1
    )
    if all_integrity_passed and successful_mutation:
        print("\nV2-09.4 REAL ARTIFACT E2E SMOKE: PASS")
        return 0
    print("\nV2-09.4 REAL ARTIFACT E2E SMOKE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
