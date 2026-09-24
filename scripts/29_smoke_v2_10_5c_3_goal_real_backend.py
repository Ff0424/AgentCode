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
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from src.agentrec.domain import ShoppingRequirement  # noqa: E402
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
from src.agentrec.verification import (  # noqa: E402
    CandidateVerificationStatus,
    ConstraintVerificationStatus,
    EvidenceConstraintVerifier,
    RequirementVerification,
)


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


def main() -> int:
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
    return 0 if all_pass and evidence_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
