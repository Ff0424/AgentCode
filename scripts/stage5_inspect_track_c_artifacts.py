"""Generate read-only Track C artifact inspection reports for human review.

The script starts from frozen human ``ExpectedGoal`` annotations. It calls the
real Recommendation and Evidence interfaces but never calls DeepSeek, the full
Agent workflow, or the Constraint Verifier. It does not mutate benchmark data.

Server example::

    python scripts/stage5_inspect_track_c_artifacts.py \
      --project-root /path/to/AgentCode \
      --device cuda:0 \
      --fallback-user-id RAW_CANONICAL_MODEL_UNSEEN_USER
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_FLOOR
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[1]
for import_root in (DEFAULT_PROJECT_ROOT, DEFAULT_PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from agentrec.domain import ShoppingRequirement  # noqa: E402
from agentrec.evaluation import (  # noqa: E402
    CandidateEvidenceInspection,
    CandidateInspection,
    CatalogFact,
    EvaluationCase,
    EvidenceSnippetInspection,
    IdentityInspection,
    InspectionRequirement,
    RecommendationInspection,
    TrackCArtifactInspectionReport,
    TrackCArtifactInspector,
    select_track_c_cases,
)
from agentrec.evidence import (  # noqa: E402
    EvidenceCandidate,
    EvidenceEmptyError,
    GroundedEvidenceService,
)
from agentrec.planning import (  # noqa: E402
    AllocationPreferenceType,
    DeterministicBudgetAllocator,
    GoalAllocationPreference,
    GoalRequirementProposal,
    GoalToRequirementProjector,
    ShoppingGoalExtractionDecision,
)
from agentrec.recommendation import RecommendationRequest, RecommendationService  # noqa: E402
from agentrec.retrieval import (  # noqa: E402
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)
from agentrec.tools import RecommendationToolAdapter, RecommendationToolArgs  # noqa: E402


_CENT = Decimal("0.01")
_MAX_REPORT_TEXT = 4000
_SECRET_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]+|AKIA[A-Z0-9]+)\b")
_WINDOWS_PATH_PATTERN = re.compile(r"(?i)\b[a-z]:[\\/][^\s]+")
_LINUX_PATH_PATTERN = re.compile(r"(?<!\w)/(?:home|root|opt|srv|mnt)/[^\s]+")


def _safe_text(value: object, *, max_length: int = _MAX_REPORT_TEXT) -> str | None:
    """Project bounded readable text while redacting secrets and machine paths."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        text = " ".join(str(item) for item in value if item is not None)
    else:
        text = str(value)
    text = " ".join(text.split())
    text = _SECRET_PATTERN.sub("[REDACTED_SECRET]", text)
    text = _WINDOWS_PATH_PATTERN.sub("[REDACTED_PATH]", text)
    text = _LINUX_PATH_PATTERN.sub("[REDACTED_PATH]", text)
    return text[:max_length] or None


class RealIdentityProbe:
    def __init__(self, service: RecommendationService) -> None:
        self._artifacts = service.artifacts

    def inspect(self, *, user_id: str, user_reference: str) -> IdentityInspection:
        canonical = self._artifacts.resolve_canonical_user(user_id)
        model_row = (
            None
            if canonical is None
            else self._artifacts.resolve_model_user_row(canonical)
        )
        if canonical is None:
            status = "unknown_user"
            reason = "raw_user_not_in_canonical_mapping"
        elif model_row is None:
            status = "canonical_known_model_unseen"
            reason = "canonical_user_not_in_model"
        else:
            status = "personalized"
            reason = None
        return IdentityInspection(
            user_reference=user_reference,
            canonical_known=canonical is not None,
            model_known=model_row is not None,
            personalization_status=status,
            fallback_reason=reason,
        )


class RealRecommendationProbe:
    def __init__(self, service: RecommendationService) -> None:
        self._service = service
        self._adapter = RecommendationToolAdapter(service)

    def recommend(
        self,
        *,
        user_id: str,
        requirement: InspectionRequirement,
        top_k: int,
    ) -> RecommendationInspection:
        args = RecommendationToolArgs(
            top_k=top_k,
            category=requirement.category,
            max_price=requirement.unit_max_price,
            required_features=requirement.required_features,
        )
        result = self._adapter.recommend(
            user_id=user_id,
            args=args,
            excluded_parent_asins=(),
        )
        request = RecommendationRequest(
            user_id=user_id,
            top_k=top_k,
            category=requirement.category,
            max_price=requirement.unit_max_price,
            required_features=requirement.required_features,
            excluded_parent_asins=(),
        )
        eligible = self._service.catalog.build_eligible_mask(request)
        candidates: list[CandidateInspection] = []
        for item in result.items:
            row = self._service.catalog.parent_asin_to_serving_row.get(item.parent_asin)
            if row is None:
                raise ValueError("Recommendation identity is missing from Product Catalog.")
            product = self._service.catalog.project_product(row)
            if product.item_index != item.item_index:
                raise ValueError("Recommendation and Catalog item identities differ.")
            candidates.append(
                CandidateInspection(
                    rank=item.rank,
                    parent_asin=item.parent_asin,
                    item_index=item.item_index,
                    title=_safe_text(item.title),
                    price=item.price,
                    categories=tuple(
                        value
                        for raw in product.categories
                        if (value := _safe_text(raw, max_length=500)) is not None
                    ),
                    source=item.score_source,
                    price_constraint_passed=(
                        item.price is not None
                        and item.price <= requirement.unit_max_price
                    ),
                    feature_filter_constraint_passed=bool(eligible[row]),
                )
            )
        return RecommendationInspection(
            top_k=top_k,
            personalization_status=result.personalization_status,
            fallback_reason=result.fallback_reason,
            candidates=tuple(candidates),
        )


class JsonlCatalogFactProvider:
    """Read exact Catalog records through a compact byte-offset identity index."""

    def __init__(self, catalog_path: Path) -> None:
        self._path = catalog_path
        self._offsets: dict[tuple[int, str], int] = {}
        with catalog_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                identity = (record.get("item_index"), record.get("parent_asin"))
                if (
                    isinstance(identity[0], bool)
                    or not isinstance(identity[0], int)
                    or not isinstance(identity[1], str)
                    or not identity[1]
                ):
                    raise ValueError("Product Catalog contains an invalid identity.")
                if identity in self._offsets:
                    raise ValueError("Product Catalog contains a duplicate identity.")
                self._offsets[identity] = offset

    @staticmethod
    def _strings(value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(
            projected
            for raw in value
            if isinstance(raw, str)
            and (projected := _safe_text(raw, max_length=1000)) is not None
        )

    def get_facts(
        self, identities: tuple[tuple[int, str], ...]
    ) -> tuple[CatalogFact, ...]:
        facts: list[CatalogFact] = []
        with self._path.open("rb") as handle:
            for identity in identities:
                offset = self._offsets.get(identity)
                if offset is None:
                    raise ValueError("Candidate identity is missing from raw Product Catalog.")
                handle.seek(offset)
                record = json.loads(handle.readline())
                price = record.get("price")
                safe_price = (
                    float(price)
                    if isinstance(price, (int, float))
                    and not isinstance(price, bool)
                    and math.isfinite(float(price))
                    and float(price) > 0
                    else None
                )
                facts.append(
                    CatalogFact(
                        item_index=identity[0],
                        parent_asin=identity[1],
                        title=_safe_text(record.get("title")),
                        categories=self._strings(record.get("categories")),
                        price=safe_price,
                        features=self._strings(record.get("features")),
                        description=_safe_text(record.get("description")),
                    )
                )
        return tuple(facts)


class RealEvidenceProbe:
    """Retrieve candidate-restricted snippets without invoking verification."""

    def __init__(self, service: GroundedEvidenceService) -> None:
        self._service = service

    def retrieve(
        self,
        *,
        case_id: str,
        requirement: InspectionRequirement,
        candidates: tuple[CandidateInspection, ...],
    ) -> tuple[CandidateEvidenceInspection, ...]:
        domain_requirement = ShoppingRequirement(
            requirement_id=requirement.requirement_id,
            category=requirement.category,
            quantity=requirement.quantity,
            max_budget=requirement.effective_budget,
            required_features=requirement.required_features,
            soft_preferences=(),
            priority=3,
        )
        results: list[CandidateEvidenceInspection] = []
        for candidate in candidates:
            try:
                evidence = self._service.retrieve(
                    plan_id=f"inspection-{case_id}",
                    plan_version=0,
                    requirement=domain_requirement,
                    candidates=(
                        EvidenceCandidate(
                            item_index=candidate.item_index,
                            parent_asin=candidate.parent_asin,
                        ),
                    ),
                )
            except EvidenceEmptyError:
                results.append(
                    CandidateEvidenceInspection(
                        parent_asin=candidate.parent_asin,
                        item_index=candidate.item_index,
                        error_code="no_evidence_chunks",
                    )
                )
                continue
            product = evidence.products[0]
            results.append(
                CandidateEvidenceInspection(
                    parent_asin=candidate.parent_asin,
                    item_index=candidate.item_index,
                    query=_safe_text(evidence.query),
                    snippets=tuple(
                        EvidenceSnippetInspection(
                            rank=value.rank,
                            evidence_reference=value.chunk_id,
                            source_field=value.chunk_type.value,
                            text=_safe_text(value.text) or "[EMPTY_AFTER_REDACTION]",
                            retrieval_score=value.similarity_score,
                        )
                        for value in product.snippets
                    ),
                )
            )
        return tuple(results)


def _load_cases(project_root: Path) -> tuple[EvaluationCase, ...]:
    path = project_root / "data/evaluation/curated_v1_pilot/cases.jsonl"
    return tuple(
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _to_requirements(case: EvaluationCase) -> tuple[InspectionRequirement, ...]:
    """Project frozen annotation truth through deterministic planning only."""

    goal = case.expected_goal
    decision = ShoppingGoalExtractionDecision(
        total_budget=goal.total_budget,
        requirement_proposals=tuple(
            GoalRequirementProposal(
                category=value.category,
                quantity=value.quantity,
                max_budget=value.max_budget,
                required_features=value.required_features,
                soft_preferences=value.soft_preferences,
                priority=value.priority,
            )
            for value in goal.ordered_requirement_proposals
        ),
        allocation_preferences=tuple(
            GoalAllocationPreference(
                target_index=value.target_index,
                preference=AllocationPreferenceType(value.preference.value),
            )
            for value in goal.allocation_preferences
        ),
        clarification_needed=False,
        clarification_question=None,
    )
    projection = GoalToRequirementProjector().project(decision)
    allocation = DeterministicBudgetAllocator().allocate(projection)
    allocated_by_id = {
        value.requirement_id: value.allocated_budget
        for value in allocation.allocations
    }
    results: list[InspectionRequirement] = []
    for index, requirement in enumerate(projection.requirements):
        allocated = Decimal(str(allocated_by_id[requirement.requirement_id]))
        explicit = (
            None
            if requirement.max_budget is None
            else Decimal(str(requirement.max_budget))
        )
        effective = allocated if explicit is None else min(allocated, explicit)
        unit = (effective / requirement.quantity).quantize(_CENT, rounding=ROUND_FLOOR)
        if unit <= 0:
            raise ValueError("Inspection requirement has no positive unit budget.")
        results.append(
            InspectionRequirement(
                requirement_index=index,
                requirement_id=requirement.requirement_id,
                category=requirement.category,
                quantity=requirement.quantity,
                allocated_budget=float(allocated),
                effective_budget=float(effective),
                unit_max_price=float(unit),
                required_features=requirement.required_features,
            )
        )
    return tuple(results)


def _render_text(report: TrackCArtifactInspectionReport) -> str:
    lines = [
        "=== AgentRec Track C Artifact Inspection ===",
        f"report_version={report.report_version}",
        "purpose=human adjudication only",
        "verifier_called=false",
        "ground_truth_mutated=false",
    ]
    for case in report.cases:
        lines.extend(
            (
                "",
                f"=== CASE {case.case_id} ===",
                f"user_policy={case.user_policy}",
                f"identity_status={case.identity.personalization_status}",
                f"canonical_known={case.identity.canonical_known}",
                f"model_known={case.identity.model_known}",
                f"fallback_reason={case.identity.fallback_reason}",
            )
        )
        for requirement in case.requirements:
            value = requirement.input_truth
            lines.extend(
                (
                    f"-- requirement[{value.requirement_index}] {value.category}",
                    f"quantity={value.quantity}",
                    f"allocated_budget={value.allocated_budget:.2f}",
                    f"effective_budget={value.effective_budget:.2f}",
                    f"unit_max_price={value.unit_max_price:.2f}",
                    f"required_features={value.required_features}",
                    f"top5_count={len(requirement.top5.candidates)}",
                    f"top10_count={len(requirement.top10.candidates)}",
                    f"new_candidates={requirement.new_candidate_identities}",
                )
            )
            for label, probe in (("top5", requirement.top5), ("top10", requirement.top10)):
                for candidate in probe.candidates:
                    lines.append(
                        f"{label} rank={candidate.rank} item_index={candidate.item_index} "
                        f"parent_asin={candidate.parent_asin} price={candidate.price} "
                        f"source={candidate.source} title={candidate.title}"
                    )
            for fact in requirement.catalog_facts:
                lines.append(
                    f"catalog item_index={fact.item_index} parent_asin={fact.parent_asin} "
                    f"categories={fact.categories} price={fact.price} "
                    f"features={fact.features} description={fact.description}"
                )
            for evidence in requirement.evidence:
                lines.append(
                    f"evidence item_index={evidence.item_index} "
                    f"parent_asin={evidence.parent_asin} query={evidence.query} "
                    f"error={evidence.error_code}"
                )
                for snippet in evidence.snippets:
                    lines.append(
                        f"  rank={snippet.rank} reference={snippet.evidence_reference} "
                        f"source={snippet.source_field} score={snippet.retrieval_score:.6f} "
                        f"text={snippet.text}"
                    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fallback-user-id",
        required=True,
        help="Raw canonical-known/model-unseen user chosen independently for inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cases = select_track_c_cases(_load_cases(project_root))
    service = RecommendationService(
        bundle_dir=project_root / "artifacts/recommendation/serving_v2",
        catalog_path=project_root / "data/processed/recommendation/product_catalog.jsonl",
        device=args.device,
    )
    retrieval_artifacts = ChunkRetrievalArtifacts(
        retrieval_dir=project_root / "artifacts/recommendation/retrieval",
        knowledge_dir=project_root / "artifacts/recommendation/knowledge",
        validation_mode=ValidationMode.STRICT,
        expected_dimension=1024,
    )
    encoder = BGEM3QueryEncoder(
        model_path=project_root / "models/bge-m3",
        device=args.device,
        use_fp16=True,
        batch_size=8,
        max_length=2048,
        embedding_dimension=1024,
    )
    retriever = ChunkRetriever(
        encoder=encoder,
        backend=NumPyExactBackend(retrieval_artifacts.embeddings),
        artifacts=retrieval_artifacts,
    )
    inspector = TrackCArtifactInspector(
        recommendation=RealRecommendationProbe(service),
        catalog=JsonlCatalogFactProvider(
            project_root / "data/processed/recommendation/product_catalog.jsonl"
        ),
        evidence=RealEvidenceProbe(GroundedEvidenceService(retriever=retriever)),
        identity=RealIdentityProbe(service),
    )

    if not service.artifacts.model_user_ids:
        raise RuntimeError("Serving bundle contains no model-known users.")
    known_user = service.artifacts.model_user_ids[0]
    fallback_user = args.fallback_user_id.strip()
    fallback_identity = RealIdentityProbe(service).inspect(
        user_id=fallback_user,
        user_reference="fallback-evaluation-user",
    )
    if not fallback_identity.canonical_known or fallback_identity.model_known:
        raise ValueError(
            "--fallback-user-id must be canonical-known and model-unseen."
        )

    case_reports = []
    for case in cases:
        fallback_case = case.case_id == "e2e_fallback_001"
        case_reports.append(
            inspector.inspect_case(
                case=case,
                user_id=fallback_user if fallback_case else known_user,
                user_reference=(
                    "fallback-evaluation-user"
                    if fallback_case
                    else "model_user_ids[0]"
                ),
                requirements=_to_requirements(case),
            )
        )
    report = TrackCArtifactInspectionReport(cases=tuple(case_reports))

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage5_track_c_artifact_inspection.json"
    text_path = output_dir / "stage5_track_c_artifact_inspection.txt"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    text_path.write_text(_render_text(report), encoding="utf-8")
    print("TRACK C ARTIFACT INSPECTION: COMPLETE")
    print("cases=6")
    print("verifier_called=false")
    print("ground_truth_mutated=false")
    print("json_report=output/stage5_track_c_artifact_inspection.json")
    print("text_report=output/stage5_track_c_artifact_inspection.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
