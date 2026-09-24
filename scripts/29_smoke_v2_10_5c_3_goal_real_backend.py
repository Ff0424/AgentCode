"""Read-only V2-10.5c.3 real Recommendation feasibility probe.

This server-only smoke loads the published serving-v2 bundle and Product
Catalog through the production ``RecommendationService``.  It checks whether
one deterministic model-known user has recommendation candidates for the
three frozen Golden Goal requirements.  It does not initialize retrieval,
BGE-M3, verification, LangGraph, or Agent integration and writes no files.

Run from the repository root on the Ubuntu GPU server::

    /home/server/anaconda3/envs/agentrec/bin/python \
      scripts/29_smoke_v2_10_5c_3_goal_real_backend.py \
      --project-root /home/server/AgentCode \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from src.agentrec.domain import PlanStatus, ShoppingRequirement  # noqa: E402
from src.agentrec.evidence import (  # noqa: E402
    EvidenceCandidate,
    EvidenceEmptyError,
    GroundedEvidenceService,
    RequirementEvidence,
)
from src.agentrec.recommendation import RecommendationService  # noqa: E402
from src.agentrec.retrieval import (  # noqa: E402
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from src.agentrec.tools import (  # noqa: E402
    RecommendationToolAdapter,
    RecommendationToolArgs,
    RecommendationToolResult,
)
from src.agentrec.integration import AgentTaskRunner, GoalExecutionStatus  # noqa: E402
from src.agentrec.planning import (  # noqa: E402
    AllocationPreferenceType,
    FakeGoalExtractor,
    FakeRequirementExtractor,
    GoalAllocationPreference,
    GoalRequirementProposal,
    RequirementExtractionDecision,
    SelectCandidateDecision,
    SelectRequirementDecision,
    ShoppingGoalExtractionDecision,
)
from src.agentrec.response import ResponseKind  # noqa: E402
from src.agentrec.services import ShoppingPlanService  # noqa: E402
from src.agentrec.verification import (  # noqa: E402
    CandidateVerificationStatus,
    ConstraintVerificationStatus,
    EvidenceConstraintVerifier,
    RequirementVerification,
)
from src.agentrec.workflows import WorkflowRoute  # noqa: E402


@dataclass(frozen=True)
class ProbeSpec:
    """One frozen Golden Goal recommendation constraint set."""

    category: str
    max_price: float
    required_features: tuple[str, ...]


PROBES = (
    ProbeSpec("Docking Stations", 166.67, ("HDMI",)),
    ProbeSpec("Mice", 83.33, ()),
    ProbeSpec("Headphones", 250.00, ()),
)

DOCK_CATEGORY = "Docking Stations"
EVIDENCE_PLAN_ID = "v2-10-5c-3-stage-2"
EVIDENCE_PLAN_VERSION = 0
EVIDENCE_REQUIREMENT_ID = "req-001"
GOLDEN_TOTAL_BUDGET = 500.0
GOLDEN_REQUEST = (
    "我准备出差办公，预算 500 美元，需要扩展坞、鼠标和耳机。"
    "扩展坞必须支持 HDMI，鼠标尽量便宜，耳机可以多分一点预算。"
    "根据我以前的购买习惯帮我配一套。"
)
GOLDEN_REQUIREMENTS = (
    ("req-001", "Docking Stations", 166.67),
    ("req-002", "Mice", 83.33),
    ("req-003", "Headphones", 250.00),
)


@dataclass(frozen=True)
class EvidenceRuntime:
    """One initialized read-only evidence runtime and startup timings."""

    service: GroundedEvidenceService
    verifier: EvidenceConstraintVerifier
    retrieval_artifact_load_seconds: float
    bge_model_load_seconds: float


@dataclass(frozen=True)
class EvidenceProbeSummary:
    """Compact Stage-2 outcome without retaining model or artifact internals."""

    evidence: RequirementEvidence
    verification: RequirementVerification
    evidence_retrieval_seconds: float
    verification_seconds: float
    candidates_with_evidence: int
    supported_candidates: int
    eligible_candidates: int


class ArtifactInitializationError(RuntimeError):
    """Retrieval artifacts failed strict read-only initialization."""


class ModelInitializationError(RuntimeError):
    """The local BGE-M3 query encoder failed initialization."""


class GoldenValidationError(RuntimeError):
    """One sanitized Stage-3 acceptance failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GoldenDeterministicPlanner:
    """Select the first pending requirement and first eligible candidate."""

    _SELECTABLE_REQUIREMENT_STATUSES = {"pending", "candidate_selected"}

    def decide(self, *, context: Mapping[str, Any]):
        if not isinstance(context, Mapping):
            raise TypeError("Planner context must be a mapping.")
        decision_key = context.get("decision_key")
        if not isinstance(decision_key, str):
            raise ValueError("Planner context has no decision_key.")
        if decision_key.startswith("select_requirement:"):
            requirements = context.get("requirements")
            if not isinstance(requirements, tuple):
                raise ValueError("Requirement planner context is invalid.")
            selectable = tuple(
                requirement
                for requirement in requirements
                if requirement.get("status")
                in self._SELECTABLE_REQUIREMENT_STATUSES
            )
            if not selectable:
                raise ValueError("Requirement planner received no pending requirement.")
            return SelectRequirementDecision(
                plan_id=context["plan_id"],
                plan_version=context["plan_version"],
                requirement_id=selectable[0]["requirement_id"],
                reason="golden_real_backend_first_pending_requirement",
            )
        if decision_key.startswith("select_candidate:"):
            candidates = context.get("candidates")
            if not isinstance(candidates, tuple) or not candidates:
                raise ValueError("Candidate planner received no eligible candidate.")
            return SelectCandidateDecision(
                plan_id=context["plan_id"],
                plan_version=context["plan_version"],
                requirement_id=context["requirement_id"],
                parent_asin=candidates[0]["parent_asin"],
                reason="golden_real_backend_first_eligible_candidate",
            )
        raise ValueError(f"Unsupported planner decision_key={decision_key!r}.")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="AgentRec repository root.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device passed unchanged to RecommendationService.",
    )
    return parser.parse_args()


def _run_probe(
    *,
    adapter: RecommendationToolAdapter,
    user_id: str,
    spec: ProbeSpec,
) -> RecommendationToolResult:
    result = adapter.recommend(
        user_id=user_id,
        args=RecommendationToolArgs(
            top_k=5,
            category=spec.category,
            max_price=spec.max_price,
            required_features=spec.required_features,
        ),
        excluded_parent_asins=(),
    )

    # The production Catalog filter owns price eligibility.  This assertion
    # verifies its externally visible result without duplicating its logic.
    for item in result.items:
        if item.price is None or item.price > spec.max_price:
            raise AssertionError(
                f"{spec.category} returned an item outside max_price={spec.max_price:.2f}."
            )
    return result


def _print_probe(spec: ProbeSpec, result: RecommendationToolResult) -> None:
    _section(f"Probe: {spec.category}")
    print(f"max_price={spec.max_price:.2f}")
    print(f"required_features={spec.required_features}")
    print(f"personalization_status={result.personalization_status}")
    print(f"returned_count={result.returned_count}")
    if result.items:
        top = result.items[0]
        print(f"top_parent_asin={top.parent_asin}")
        print(f"top_title={top.title}")
        print(f"top_price={top.price:.2f}")
        print(f"top_score_source={top.score_source}")


def _dock_result(
    results: list[tuple[ProbeSpec, RecommendationToolResult]],
) -> RecommendationToolResult:
    matches = tuple(
        result for spec, result in results if spec.category == DOCK_CATEGORY
    )
    if len(matches) != 1:
        raise AssertionError("Stage 1 did not produce exactly one Dock result.")
    return matches[0]


def _to_evidence_candidates(
    result: RecommendationToolResult,
) -> tuple[EvidenceCandidate, ...]:
    """Preserve the complete trusted Tool-result identity order."""

    return tuple(
        EvidenceCandidate(
            item_index=item.item_index,
            parent_asin=item.parent_asin,
        )
        for item in result.items
    )


def _initialize_evidence_runtime(
    *,
    project_root: Path,
    device: str,
) -> EvidenceRuntime:
    if device != "cuda:0":
        raise ValueError("Stage 2 requires --device cuda:0.")

    knowledge_dir = project_root / "artifacts/recommendation/knowledge"
    retrieval_dir = project_root / "artifacts/recommendation/retrieval"
    model_path = project_root / "models/bge-m3"

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    started = time.perf_counter()
    try:
        artifacts = ChunkRetrievalArtifacts(
            retrieval_dir=retrieval_dir,
            knowledge_dir=knowledge_dir,
            validation_mode=ValidationMode.STRICT,
            expected_dimension=1024,
        )
    except Exception as exc:
        raise ArtifactInitializationError("ARTIFACT_ERROR") from exc
    retrieval_load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    try:
        encoder = BGEM3QueryEncoder(
            model_path=model_path,
            device=device,
            use_fp16=True,
            batch_size=8,
            max_length=2048,
            embedding_dimension=1024,
        )
    except Exception as exc:
        raise ModelInitializationError("MODEL_LOAD_ERROR") from exc
    model_load_seconds = time.perf_counter() - started

    backend = NumPyExactBackend(artifacts.embeddings)
    retriever = ChunkRetriever(
        encoder=encoder,
        backend=backend,
        artifacts=artifacts,
    )
    return EvidenceRuntime(
        service=GroundedEvidenceService(retriever=retriever),
        verifier=EvidenceConstraintVerifier(),
        retrieval_artifact_load_seconds=retrieval_load_seconds,
        bge_model_load_seconds=model_load_seconds,
    )


def _run_dock_evidence_probe(
    *,
    dock_result: RecommendationToolResult,
    runtime: EvidenceRuntime,
) -> EvidenceProbeSummary:
    candidates = _to_evidence_candidates(dock_result)
    if not candidates:
        raise ValueError("ZERO_RECOMMENDATION_CANDIDATES")

    requirement = ShoppingRequirement(
        requirement_id=EVIDENCE_REQUIREMENT_ID,
        category=DOCK_CATEGORY,
        quantity=1,
        max_budget=None,
        required_features=("HDMI",),
        soft_preferences=(),
        priority=3,
    )

    started = time.perf_counter()
    evidence = runtime.service.retrieve(
        plan_id=EVIDENCE_PLAN_ID,
        plan_version=EVIDENCE_PLAN_VERSION,
        requirement=requirement,
        candidates=candidates,
    )
    evidence_seconds = time.perf_counter() - started

    if (
        evidence.plan_id != EVIDENCE_PLAN_ID
        or evidence.retrieved_at_plan_version != EVIDENCE_PLAN_VERSION
        or evidence.requirement_id != EVIDENCE_REQUIREMENT_ID
    ):
        raise AssertionError("Evidence provenance context is misaligned.")
    expected_identities = tuple(
        (item.item_index, item.parent_asin) for item in candidates
    )
    candidate_identities = tuple(
        (item.item_index, item.parent_asin) for item in evidence.candidates
    )
    product_identities = tuple(
        (item.item_index, item.parent_asin) for item in evidence.products
    )
    if candidate_identities != expected_identities or product_identities != expected_identities:
        raise AssertionError("Recommendation and evidence identities are misaligned.")
    for product in evidence.products:
        if not product.snippets or any(
            snippet.item_index != product.item_index
            or snippet.parent_asin != product.parent_asin
            for snippet in product.snippets
        ):
            raise AssertionError("Evidence snippets do not fully cover candidate identity.")

    started = time.perf_counter()
    verification = runtime.verifier.verify(
        requirement=requirement,
        evidence=evidence,
        current_plan_id=EVIDENCE_PLAN_ID,
        current_plan_version=EVIDENCE_PLAN_VERSION,
    )
    verification_seconds = time.perf_counter() - started
    if (
        verification.plan_id != EVIDENCE_PLAN_ID
        or verification.verified_at_plan_version != EVIDENCE_PLAN_VERSION
        or verification.requirement_id != EVIDENCE_REQUIREMENT_ID
    ):
        raise AssertionError("Verification provenance context is misaligned.")
    verification_identities = tuple(
        (item.item_index, item.parent_asin) for item in verification.candidates
    )
    if verification_identities != expected_identities:
        raise AssertionError("Evidence and verification identities are misaligned.")

    supported = sum(
        any(
            constraint.status is ConstraintVerificationStatus.SUPPORTED
            for constraint in candidate.constraints
        )
        for candidate in verification.candidates
    )
    eligible = sum(
        candidate.status is CandidateVerificationStatus.ELIGIBLE
        for candidate in verification.candidates
    )
    return EvidenceProbeSummary(
        evidence=evidence,
        verification=verification,
        evidence_retrieval_seconds=evidence_seconds,
        verification_seconds=verification_seconds,
        candidates_with_evidence=len(evidence.products),
        supported_candidates=supported,
        eligible_candidates=eligible,
    )


def _print_evidence_environment(
    *,
    project_root: Path,
    runtime: EvidenceRuntime,
) -> None:
    _section("Evidence Environment")
    print(f"knowledge_root={project_root / 'artifacts/recommendation/knowledge'}")
    print(f"retrieval_root={project_root / 'artifacts/recommendation/retrieval'}")
    print(f"model_path={project_root / 'models/bge-m3'}")
    print(f"artifact_validation_mode={ValidationMode.STRICT.value}")
    print(
        "retrieval_artifact_load_seconds="
        f"{runtime.retrieval_artifact_load_seconds:.6f}"
    )
    print(f"bge_model_load_seconds={runtime.bge_model_load_seconds:.6f}")


def _print_evidence_summary(
    *,
    dock_result: RecommendationToolResult,
    summary: EvidenceProbeSummary,
) -> bool:
    evidence_by_identity = {
        (product.item_index, product.parent_asin): product
        for product in summary.evidence.products
    }
    verification_by_identity = {
        (candidate.item_index, candidate.parent_asin): candidate
        for candidate in summary.verification.candidates
    }

    _section("Dock Evidence")
    for item in dock_result.items:
        identity = (item.item_index, item.parent_asin)
        product = evidence_by_identity[identity]
        candidate = verification_by_identity[identity]
        statuses = tuple(value.status.value for value in candidate.constraints)
        verification_status = statuses[0] if len(statuses) == 1 else statuses
        print(f"candidate_rank={item.rank}")
        print(f"item_index={item.item_index}")
        print(f"parent_asin={item.parent_asin}")
        print(f"title={item.title}")
        print(f"price={item.price:.2f}")
        print(f"evidence_count={len(product.snippets)}")
        print(f"verification_status={verification_status}")
        print(f"candidate_status={candidate.status.value}")

    passed = (
        summary.candidates_with_evidence == dock_result.returned_count
        and summary.supported_candidates > 0
        and summary.eligible_candidates > 0
        and bool(summary.verification.eligible_parent_asins)
    )
    _section("Evidence Feasibility Summary")
    print(f"recommendation_candidates={dock_result.returned_count}")
    print(f"candidates_with_evidence={summary.candidates_with_evidence}")
    print(f"supported_candidates={summary.supported_candidates}")
    print(f"eligible_candidates={summary.eligible_candidates}")
    print(f"evidence_retrieval_seconds={summary.evidence_retrieval_seconds:.6f}")
    print(f"verification_seconds={summary.verification_seconds:.6f}")
    print(f"REAL_EVIDENCE_FEASIBILITY_PASS={passed}")
    return passed


def _verification_failure_reason(summary: EvidenceProbeSummary) -> str:
    """Classify a valid but non-eligible verification result without relaxing it."""

    statuses = tuple(
        constraint.status
        for candidate in summary.verification.candidates
        for constraint in candidate.constraints
    )
    if summary.eligible_candidates == 0:
        if ConstraintVerificationStatus.CONTRADICTED in statuses:
            return "HDMI_CONTRADICTED"
        if ConstraintVerificationStatus.UNKNOWN in statuses:
            return "HDMI_UNKNOWN"
        return "NO_ELIGIBLE_CANDIDATE"
    return "NO_ELIGIBLE_CANDIDATE"


def _money(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _build_goal_decision() -> ShoppingGoalExtractionDecision:
    return ShoppingGoalExtractionDecision(
        total_budget=GOLDEN_TOTAL_BUDGET,
        requirement_proposals=(
            GoalRequirementProposal(
                category="Docking Stations",
                quantity=1,
                max_budget=None,
                required_features=("HDMI",),
                soft_preferences=(),
                priority=3,
            ),
            GoalRequirementProposal(
                category="Mice",
                quantity=1,
                max_budget=None,
                required_features=(),
                soft_preferences=(),
                priority=3,
            ),
            GoalRequirementProposal(
                category="Headphones",
                quantity=1,
                max_budget=None,
                required_features=(),
                soft_preferences=(),
                priority=3,
            ),
        ),
        allocation_preferences=(
            GoalAllocationPreference(
                target_index=1,
                preference=AllocationPreferenceType.SAVE_MORE,
            ),
            GoalAllocationPreference(
                target_index=2,
                preference=AllocationPreferenceType.ALLOCATE_MORE,
            ),
        ),
        clarification_needed=False,
        clarification_question=None,
    )


def _build_legacy_requirement_extractor() -> FakeRequirementExtractor:
    """Satisfy the legacy constructor dependency unused by run_goal()."""

    return FakeRequirementExtractor(
        RequirementExtractionDecision(
            category="constructor-placeholder",
            quantity=1,
            max_budget=1.0,
            required_features=(),
            soft_preferences=(),
            priority=3,
            clarification_needed=False,
            reason="legacy constructor dependency unused by run_goal",
        )
    )


def _print_terminal_failure(result: Any) -> None:
    state = getattr(result, "workflow_state", None)
    if state is None:
        return
    print(f"error_state={state.agent_state.error_state}")
    diagnosis = state.current_failure_diagnosis
    print(
        "failure_diagnosis="
        f"{None if diagnosis is None else type(diagnosis).__name__}"
    )
    print(f"failure_history_count={len(state.failure_history)}")
    print(f"replan_history_count={len(state.replan_history)}")


def _validate_and_print_golden_result(result: Any, *, user_id: str) -> bool:
    if result.status is GoalExecutionStatus.CLARIFICATION_REQUIRED:
        raise GoldenValidationError("GOAL_CLARIFICATION_UNEXPECTED")
    if result.status is GoalExecutionStatus.CONFLICT:
        _print_terminal_failure(result)
        raise GoldenValidationError("WORKFLOW_CONFLICT")
    if result.status is GoalExecutionStatus.ERROR:
        _print_terminal_failure(result)
        raise GoldenValidationError("WORKFLOW_ERROR")
    if result.status is not GoalExecutionStatus.READY:
        raise GoldenValidationError("UNEXPECTED_TERMINAL_STATUS")
    if result.workflow_state is None or result.prepared_execution is None:
        raise GoldenValidationError("UNEXPECTED_TERMINAL_STATUS")
    if (
        result.workflow_state.route is not WorkflowRoute.READY
        or result.final_response is None
        or result.response_error is not None
        or result.memory_error is not None
    ):
        raise GoldenValidationError("RESPONSE_ERROR")

    prepared = result.prepared_execution
    allocation = prepared.budget_allocation
    projection = prepared.projection
    expected_ids = tuple(value[0] for value in GOLDEN_REQUIREMENTS)
    expected_categories = tuple(value[1] for value in GOLDEN_REQUIREMENTS)
    expected_allocations = tuple(_money(value[2]) for value in GOLDEN_REQUIREMENTS)
    if (
        tuple(value.requirement_id for value in projection.requirements)
        != expected_ids
        or tuple(value.category for value in projection.requirements)
        != expected_categories
        or any(value.max_budget is not None for value in projection.requirements)
        or tuple(value.requirement_id for value in allocation.allocations)
        != expected_ids
        or tuple(_money(value.allocated_budget) for value in allocation.allocations)
        != expected_allocations
        or _money(allocation.total_budget) != _money(GOLDEN_TOTAL_BUDGET)
        or _money(allocation.unallocated_budget) != Decimal("0.00")
    ):
        raise GoldenValidationError("GOAL_ALLOCATION_MISMATCH")

    state = result.workflow_state
    plan = state.agent_state.shopping_plan
    if (
        state.route is not WorkflowRoute.READY
        or plan.status is not PlanStatus.READY
        or len(plan.requirements) != 3
        or len(plan.selected_items) != 3
        or plan.version != 3
        or plan.is_over_budget
        or not plan.is_valid
        or plan.total_spent > GOLDEN_TOTAL_BUDGET
        or _money(plan.remaining_budget)
        != _money(GOLDEN_TOTAL_BUDGET - plan.total_spent)
    ):
        raise GoldenValidationError("NO_SELECTED_ITEM")
    if (
        tuple(value.requirement_id for value in plan.requirements) != expected_ids
        or tuple(value.category for value in plan.requirements) != expected_categories
        or tuple(value.requirement_id for value in plan.selected_items) != expected_ids
    ):
        raise GoldenValidationError("IDENTITY_MISMATCH")

    provenance = state.selected_recommendation_budget_provenance
    if len(provenance) != 3:
        raise GoldenValidationError("PROVENANCE_MISMATCH")
    for index, value in enumerate(provenance):
        expected_id, _category, expected_budget = GOLDEN_REQUIREMENTS[index]
        if (
            value.requirement_id != expected_id
            or _money(value.allocated_budget) != _money(expected_budget)
            or _money(value.effective_subtotal_budget) != _money(expected_budget)
            or _money(value.derived_unit_max_price) != _money(expected_budget)
            or value.explicit_max_budget is not None
            or _money(value.retained_subtotal_before_call) != Decimal("0.00")
            or value.remaining_quantity_before_call != 1
            or value.plan_version_before_call != index
        ):
            raise GoldenValidationError("PROVENANCE_MISMATCH")

    evidence_by_id = {
        value.requirement_id: value for value in state.selected_evidence
    }
    verification_by_id = {
        value.requirement_id: value for value in state.selected_verifications
    }
    if set(evidence_by_id) != set(expected_ids) or set(verification_by_id) != {
        "req-001"
    }:
        raise GoldenValidationError("IDENTITY_MISMATCH")

    plan_items = {value.requirement_id: value for value in plan.selected_items}
    for requirement_id, category, expected_cap in GOLDEN_REQUIREMENTS:
        item = plan_items.get(requirement_id)
        evidence = evidence_by_id.get(requirement_id)
        if (
            item is None
            or evidence is None
            or item.requirement_id != requirement_id
            or item.category != category
            or item.parent_asin != evidence.parent_asin
            or not item.parent_asin
            or not item.title
            or item.quantity != 1
            or item.price <= 0
            or _money(item.price) > _money(expected_cap)
        ):
            raise GoldenValidationError("IDENTITY_MISMATCH")

    dock_evidence = evidence_by_id["req-001"]
    dock_verification = verification_by_id["req-001"]
    dock_candidate = dock_verification.candidate
    if (
        dock_candidate.item_index != dock_evidence.item_index
        or dock_candidate.parent_asin != dock_evidence.parent_asin
        or dock_candidate.status is not CandidateVerificationStatus.ELIGIBLE
        or not any(
            constraint.status is ConstraintVerificationStatus.SUPPORTED
            and bool(constraint.supporting_chunk_ids)
            for constraint in dock_candidate.constraints
        )
    ):
        raise GoldenValidationError("IDENTITY_MISMATCH")

    response = result.final_response
    if (
        response.kind is not ResponseKind.READY
        or not isinstance(response.decision_summary, tuple)
        or len(response.decision_summary) != 3
    ):
        raise GoldenValidationError("RESPONSE_ERROR")
    for index, summary in enumerate(response.decision_summary):
        requirement_id = expected_ids[index]
        item = plan_items[requirement_id]
        if (
            summary.requirement_id != requirement_id
            or summary.parent_asin != item.parent_asin
        ):
            raise GoldenValidationError("IDENTITY_MISMATCH")
        if requirement_id == "req-001":
            if not any(
                claim.status is ConstraintVerificationStatus.SUPPORTED
                and claim.original_constraint == "HDMI"
                and bool(claim.supporting_evidence)
                for claim in summary.verified_claims
            ):
                raise GoldenValidationError("RESPONSE_ERROR")
        elif summary.verified_claims:
            raise GoldenValidationError("RESPONSE_ERROR")

    _section("Golden Goal")
    print(f"user_id={user_id}")
    print(f"total_budget={projection.total_budget:.2f}")
    for requirement in projection.requirements:
        print(
            f"requirement={requirement.requirement_id} "
            f"category={requirement.category} quantity={requirement.quantity} "
            f"required_features={requirement.required_features}"
        )
    print("allocation_preferences=Mice:save_more,Headphones:allocate_more")

    _section("Goal Allocation")
    category_by_id = {
        requirement.requirement_id: requirement.category
        for requirement in projection.requirements
    }
    for value in allocation.allocations:
        print(
            f"requirement_id={value.requirement_id} "
            f"category={category_by_id[value.requirement_id]} "
            f"allocated_budget={value.allocated_budget:.2f}"
        )

    _section("Selected Bundle")
    provenance_by_id = {value.requirement_id: value for value in provenance}
    for requirement_id in expected_ids:
        item = plan_items[requirement_id]
        evidence = evidence_by_id[requirement_id]
        selected_verification = verification_by_id.get(requirement_id)
        print(f"requirement_id={requirement_id}")
        print(f"category={item.category}")
        print(f"item_index={evidence.item_index}")
        print(f"parent_asin={item.parent_asin}")
        print(f"title={item.title}")
        print(f"unit_price={item.price:.2f}")
        print(f"quantity={item.quantity}")
        print(f"subtotal={item.price * item.quantity:.2f}")
        print(
            "derived_unit_cap="
            f"{provenance_by_id[requirement_id].derived_unit_max_price:.2f}"
        )
        print(
            "verification_status="
            + (
                "not_required"
                if selected_verification is None
                else selected_verification.candidate.status.value
            )
        )

    _section("Bundle Summary")
    print(f"total_spent={plan.total_spent:.2f}")
    print(f"remaining_budget={plan.remaining_budget:.2f}")
    print(f"plan_status={plan.status.value}")
    print(f"workflow_route={state.route.value}")
    print(f"goal_status={result.status.value}")
    print(f"plan_version={plan.version}")

    _section("Grounded Final Response")
    print(response.text)
    return True


def main() -> int:
    process_started = time.perf_counter()
    args = _parse_args()
    project_root = args.project_root.resolve()
    serving_dir = project_root / "artifacts/recommendation/serving_v2"
    catalog_path = (
        project_root / "data/processed/recommendation/product_catalog.jsonl"
    )

    started = time.perf_counter()
    service = RecommendationService(
        bundle_dir=serving_dir,
        catalog_path=catalog_path,
        device=args.device,
    )
    service_init_seconds = time.perf_counter() - started
    adapter = RecommendationToolAdapter(service)

    artifacts = service.artifacts
    if not artifacts.model_user_ids:
        raise RuntimeError("Serving bundle contains no model-known users.")
    user_id = artifacts.model_user_ids[0]
    canonical_user_index = artifacts.resolve_canonical_user(user_id)
    if canonical_user_index is None:
        raise AssertionError("First model user is absent from canonical user lookup.")
    model_user_row = artifacts.resolve_model_user_row(canonical_user_index)
    if model_user_row != 0:
        raise AssertionError(
            "model_user_ids[0] did not resolve to model_user_row=0."
        )

    _section("Environment")
    print(f"device={args.device}")
    print(f"service_init_seconds={service_init_seconds:.6f}")

    _section("User")
    print("selection_rule=model_user_ids[0]")
    print(f"user_id={user_id}")
    print(f"canonical_user_index={canonical_user_index}")
    print(f"model_user_row={model_user_row}")

    probe_started = time.perf_counter()
    results: list[tuple[ProbeSpec, RecommendationToolResult]] = []
    for spec in PROBES:
        result = _run_probe(adapter=adapter, user_id=user_id, spec=spec)
        results.append((spec, result))
        _print_probe(spec, result)
    probe_total_seconds = time.perf_counter() - probe_started

    statuses = {
        spec.category: (
            "PASS"
            if result.returned_count > 0
            and result.personalization_status == "personalized"
            else "ZERO_CANDIDATES"
            if result.returned_count == 0
            and result.personalization_status == "personalized"
            else "UNEXPECTED_PERSONALIZATION_STATUS"
        )
        for spec, result in results
    }
    all_pass = all(status == "PASS" for status in statuses.values())

    _section("Feasibility Summary")
    for spec in PROBES:
        print(f"{spec.category}: {statuses[spec.category]}")
    print(f"probe_total_seconds={probe_total_seconds:.6f}")
    print(f"ALL_RECOMMENDATION_PROBES_PASS={all_pass}")

    dock_result = _dock_result(results)
    if dock_result.returned_count == 0:
        _section("Evidence Feasibility Summary")
        print("failure_reason=ZERO_RECOMMENDATION_CANDIDATES")
        print("REAL_EVIDENCE_FEASIBILITY_PASS=False")
        return 1
    if dock_result.personalization_status != "personalized":
        _section("Evidence Feasibility Summary")
        print("failure_reason=UNEXPECTED_PERSONALIZATION_STATUS")
        print("REAL_EVIDENCE_FEASIBILITY_PASS=False")
        return 1

    try:
        evidence_runtime = _initialize_evidence_runtime(
            project_root=project_root,
            device=args.device,
        )
    except ArtifactInitializationError:
        _section("Evidence Feasibility Summary")
        print("failure_reason=ARTIFACT_ERROR")
        print("REAL_EVIDENCE_FEASIBILITY_PASS=False")
        return 1
    except ModelInitializationError:
        _section("Evidence Feasibility Summary")
        print("failure_reason=MODEL_LOAD_ERROR")
        print("REAL_EVIDENCE_FEASIBILITY_PASS=False")
        return 1

    _print_evidence_environment(
        project_root=project_root,
        runtime=evidence_runtime,
    )
    try:
        evidence_summary = _run_dock_evidence_probe(
            dock_result=dock_result,
            runtime=evidence_runtime,
        )
    except EvidenceEmptyError:
        _section("Evidence Feasibility Summary")
        print(f"recommendation_candidates={dock_result.returned_count}")
        print("failure_reason=EVIDENCE_EMPTY")
        print("REAL_EVIDENCE_FEASIBILITY_PASS=False")
        return 1

    evidence_pass = _print_evidence_summary(
        dock_result=dock_result,
        summary=evidence_summary,
    )
    if not evidence_pass:
        print(f"failure_reason={_verification_failure_reason(evidence_summary)}")
        return 1

    plan_service = ShoppingPlanService()
    runner = AgentTaskRunner(
        requirement_extractor=_build_legacy_requirement_extractor(),
        planner=GoldenDeterministicPlanner(),
        recommendation_tool=adapter,
        evidence_service=evidence_runtime.service,
        verification_service=evidence_runtime.verifier,
        shopping_plan_service=plan_service,
        memory_store=None,
        memory_merger=None,
        goal_extractor=FakeGoalExtractor(_build_goal_decision()),
    )
    goal_started = time.perf_counter()
    try:
        result = runner.run_goal(
            user_id=user_id,
            session_id="v2-10-5c-3-golden-session",
            plan_id="v2-10-5c-3-golden-plan",
            user_request=GOLDEN_REQUEST,
            currency="USD",
            recursion_limit=50,
        )
    except Exception:
        goal_execution_seconds = time.perf_counter() - goal_started
        _section("Golden E2E Summary")
        print("failure_reason=WORKFLOW_ERROR")
        print(f"goal_execution_seconds={goal_execution_seconds:.6f}")
        print(f"total_process_seconds={time.perf_counter() - process_started:.6f}")
        print("GOLDEN_REAL_BACKEND_E2E_PASS=False")
        return 1
    goal_execution_seconds = time.perf_counter() - goal_started

    try:
        golden_pass = _validate_and_print_golden_result(result, user_id=user_id)
    except GoldenValidationError as exc:
        _section("Golden E2E Summary")
        print(f"failure_reason={exc.code}")
        print(f"goal_execution_seconds={goal_execution_seconds:.6f}")
        print(f"total_process_seconds={time.perf_counter() - process_started:.6f}")
        print("GOLDEN_REAL_BACKEND_E2E_PASS=False")
        return 1

    state = result.workflow_state
    assert state is not None
    plan = state.agent_state.shopping_plan
    dock_verification = state.selected_verifications[0]
    dock_hdmi_supported = any(
        constraint.status is ConstraintVerificationStatus.SUPPORTED
        for constraint in dock_verification.candidate.constraints
    )
    _section("Golden E2E Summary")
    print(f"selected_requirements={len(plan.requirements)}")
    print(f"selected_items={len(plan.selected_items)}")
    print(f"within_budget={not plan.is_over_budget}")
    print(f"dock_hdmi_supported={dock_hdmi_supported}")
    print(f"response_grounded={result.final_response is not None}")
    print(f"goal_execution_seconds={goal_execution_seconds:.6f}")
    print(f"total_process_seconds={time.perf_counter() - process_started:.6f}")
    process_pass = all_pass and evidence_pass and golden_pass
    print(f"GOLDEN_REAL_BACKEND_E2E_PASS={process_pass}")
    return 0 if process_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
