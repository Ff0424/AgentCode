"""Lightweight contracts and orchestration for Track C artifact inspection.

This module never assigns benchmark truth or invokes verification. Real
Recommendation, Catalog, and Evidence dependencies are injected by the server
script; unit tests use in-memory fakes.
"""

from __future__ import annotations

from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .contracts import EvaluationCase


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

TRACK_C_CASE_IDS = (
    "e2e_ready_001",
    "e2e_zero_candidate_001",
    "e2e_unknown_replan_001",
    "e2e_contradicted_001",
    "e2e_replan_recover_001",
    "e2e_fallback_001",
)


class InspectionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_index: Annotated[int, Field(strict=True, ge=0)]
    requirement_id: NonEmptyText
    category: NonEmptyText
    quantity: Annotated[int, Field(strict=True, ge=1)]
    allocated_budget: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    effective_budget: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    unit_max_price: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    required_features: tuple[NonEmptyText, ...] = ()


class CandidateInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Annotated[int, Field(strict=True, ge=1)]
    parent_asin: NonEmptyText
    item_index: Annotated[int, Field(strict=True, ge=0)]
    title: NonEmptyText | None = None
    price: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    categories: tuple[NonEmptyText, ...] = ()
    source: NonEmptyText
    price_constraint_passed: bool
    feature_filter_constraint_passed: bool


class RecommendationInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: Annotated[int, Field(strict=True, ge=1)]
    personalization_status: NonEmptyText
    fallback_reason: NonEmptyText | None = None
    candidates: tuple[CandidateInspection, ...] = ()

    @model_validator(mode="after")
    def validate_candidates(self) -> "RecommendationInspection":
        if tuple(value.rank for value in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("Candidate ranks must be exactly 1..N.")
        identities = tuple(
            (value.item_index, value.parent_asin) for value in self.candidates
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Candidate identities must be unique.")
        return self


class CatalogFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: NonEmptyText
    item_index: Annotated[int, Field(strict=True, ge=0)]
    title: NonEmptyText | None = None
    categories: tuple[NonEmptyText, ...] = ()
    price: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    features: tuple[NonEmptyText, ...] = ()
    description: NonEmptyText | None = None


class EvidenceSnippetInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Annotated[int, Field(strict=True, ge=1)]
    evidence_reference: NonEmptyText
    source_field: NonEmptyText
    text: NonEmptyText
    retrieval_score: Annotated[float, Field(ge=-1.00001, le=1.00001)]


class CandidateEvidenceInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: NonEmptyText
    item_index: Annotated[int, Field(strict=True, ge=0)]
    query: NonEmptyText | None = None
    snippets: tuple[EvidenceSnippetInspection, ...] = ()
    error_code: NonEmptyText | None = None


class IdentityInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_reference: NonEmptyText
    canonical_known: bool
    model_known: bool
    personalization_status: NonEmptyText
    fallback_reason: NonEmptyText | None = None


class RequirementArtifactInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_truth: InspectionRequirement
    top5: RecommendationInspection
    top10: RecommendationInspection
    new_candidate_identities: tuple[tuple[int, str], ...]
    catalog_facts: tuple[CatalogFact, ...]
    evidence: tuple[CandidateEvidenceInspection, ...] = ()


class CaseArtifactInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    user_policy: NonEmptyText
    identity: IdentityInspection
    requirements: tuple[RequirementArtifactInspection, ...]


class TrackCArtifactInspectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: NonEmptyText = "1.0"
    purpose: NonEmptyText = "human_adjudication_only"
    verifier_called: bool = False
    ground_truth_mutated: bool = False
    cases: tuple[CaseArtifactInspection, ...]

    @model_validator(mode="after")
    def validate_safety_flags(self) -> "TrackCArtifactInspectionReport":
        if self.verifier_called or self.ground_truth_mutated:
            raise ValueError("Inspection report must not verify or mutate ground truth.")
        if tuple(value.case_id for value in self.cases) != TRACK_C_CASE_IDS:
            raise ValueError("Inspection report must contain the six Track C cases in order.")
        return self


class RecommendationProbe(Protocol):
    def recommend(
        self,
        *,
        user_id: str,
        requirement: InspectionRequirement,
        top_k: int,
    ) -> RecommendationInspection: ...


class CatalogFactProvider(Protocol):
    def get_facts(
        self, identities: tuple[tuple[int, str], ...]
    ) -> tuple[CatalogFact, ...]: ...


class EvidenceProbe(Protocol):
    def retrieve(
        self,
        *,
        case_id: str,
        requirement: InspectionRequirement,
        candidates: tuple[CandidateInspection, ...],
    ) -> tuple[CandidateEvidenceInspection, ...]: ...


class IdentityProbe(Protocol):
    def inspect(self, *, user_id: str, user_reference: str) -> IdentityInspection: ...


def select_track_c_cases(cases: tuple[EvaluationCase, ...]) -> tuple[EvaluationCase, ...]:
    """Select exactly the frozen pending Track C cases without mutating them."""

    by_id = {value.case_id: value for value in cases}
    if len(by_id) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    missing = tuple(value for value in TRACK_C_CASE_IDS if value not in by_id)
    if missing:
        raise ValueError(f"Track C cases are missing: {missing!r}.")
    selected = tuple(by_id[value] for value in TRACK_C_CASE_IDS)
    if any("track_c" not in value.tags for value in selected):
        raise ValueError("Every selected case must carry the track_c tag.")
    return selected


class TrackCArtifactInspector:
    """Inspect injected read-only backends, never infer verification truth."""

    def __init__(
        self,
        *,
        recommendation: RecommendationProbe,
        catalog: CatalogFactProvider,
        evidence: EvidenceProbe,
        identity: IdentityProbe,
    ) -> None:
        self._recommendation = recommendation
        self._catalog = catalog
        self._evidence = evidence
        self._identity = identity

    def inspect_case(
        self,
        *,
        case: EvaluationCase,
        user_id: str,
        user_reference: str,
        requirements: tuple[InspectionRequirement, ...],
    ) -> CaseArtifactInspection:
        identity = self._identity.inspect(
            user_id=user_id,
            user_reference=user_reference,
        )
        requirement_reports: list[RequirementArtifactInspection] = []
        for requirement in requirements:
            top5 = self._recommendation.recommend(
                user_id=user_id, requirement=requirement, top_k=5
            )
            top10 = self._recommendation.recommend(
                user_id=user_id, requirement=requirement, top_k=10
            )
            top5_ids = tuple(
                (value.item_index, value.parent_asin) for value in top5.candidates
            )
            top10_ids = tuple(
                (value.item_index, value.parent_asin) for value in top10.candidates
            )
            top5_set = set(top5_ids)
            new_ids = tuple(value for value in top10_ids if value not in top5_set)
            all_ids = tuple(dict.fromkeys((*top5_ids, *top10_ids)))
            facts = self._catalog.get_facts(all_ids)
            fact_ids = tuple((value.item_index, value.parent_asin) for value in facts)
            if fact_ids != all_ids:
                raise ValueError("Catalog facts must exactly cover candidates in order.")
            evidence = (
                self._evidence.retrieve(
                    case_id=case.case_id,
                    requirement=requirement,
                    candidates=top10.candidates,
                )
                if requirement.required_features and top10.candidates
                else ()
            )
            requirement_reports.append(
                RequirementArtifactInspection(
                    input_truth=requirement,
                    top5=top5,
                    top10=top10,
                    new_candidate_identities=new_ids,
                    catalog_facts=facts,
                    evidence=evidence,
                )
            )
        return CaseArtifactInspection(
            case_id=case.case_id,
            user_policy=case.runtime_profile.user_identity_policy,
            identity=identity,
            requirements=tuple(requirement_reports),
        )
