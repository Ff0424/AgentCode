"""Immutable, runtime-independent contracts for AgentRec evaluation data.

The models in this module describe human annotation truth, reproducibility
profiles, and evaluation outputs.  They deliberately do not execute or import
the production Agent runtime.  Enum values that meet production boundaries
mirror the serialized values of those contracts without coupling benchmark
loading to runtime dependencies.
"""

from __future__ import annotations

import math
import numbers
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _finite_non_negative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field_name} must be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    return number


def _finite_positive(value: object, field_name: str) -> float:
    number = _finite_non_negative(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return number


def _stable_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    """Trim, deduplicate case-insensitively, and preserve first occurrence."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if key not in seen:
            result.append(normalized)
            seen.add(key)
    return tuple(result)


class EvaluationSplit(str, Enum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"


class AllocationPreference(str, Enum):
    """Annotation equivalent of the production allocation preference values."""

    SAVE_MORE = "save_more"
    ALLOCATE_MORE = "allocate_more"


class ExpectedGroundingStatus(str, Enum):
    GROUNDED = "grounded"
    UNRESOLVED = "unresolved"


class ExpectedTerminalStatus(str, Enum):
    """Terminal values supported by the current Goal execution contract."""

    READY = "ready"
    CONFLICT = "conflict"
    CLARIFICATION_REQUIRED = "clarification_required"
    ERROR = "error"


class VerificationTruthStatus(str, Enum):
    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


class ReplanExpectation(str, Enum):
    NOT_EXPECTED = "not_expected"
    EXPECTED_TO_RECOVER = "expected_to_recover"
    EXPECTED_TO_EXHAUST = "expected_to_exhaust"


class AdjudicationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DISPUTED = "disputed"


class EvaluationLayer(str, Enum):
    L0_CONTRACT_VALIDITY = "l0_contract_validity"
    L1_GOAL_UNDERSTANDING_GROUNDING = "l1_goal_understanding_grounding"
    L2_PROJECTION_ALLOCATION_TOOL_ARGUMENTS = (
        "l2_projection_allocation_tool_arguments"
    )
    L3_RECOMMENDATION_EVIDENCE_VERIFICATION = (
        "l3_recommendation_evidence_verification"
    )
    L4_WORKFLOW_REPLAN_PLAN = "l4_workflow_replan_plan"
    L5_E2E_API_SAFETY = "l5_e2e_api_safety"


class FailureCategory(str, Enum):
    MODEL_FAILURE = "model_failure"
    BUSINESS_CLARIFICATION = "business_clarification"
    BACKEND_FEASIBILITY = "backend_feasibility"
    DETERMINISTIC_REGRESSION = "deterministic_regression"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class FailureCode(str, Enum):
    SCHEMA_INVALID = "schema_invalid"
    REQUIREMENT_MISSING = "requirement_missing"
    WRONG_CATEGORY = "wrong_category"
    HARD_SOFT_CONFUSION = "hard_soft_confusion"
    WRONG_BUDGET = "wrong_budget"
    GROUNDING_UNRESOLVED = "grounding_unresolved"
    ZERO_CANDIDATE = "zero_candidate"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    VERIFICATION_UNKNOWN = "verification_unknown"
    VERIFICATION_CONTRADICTED = "verification_contradicted"
    REPLAN_EXHAUSTED = "replan_exhausted"
    BUDGET_VIOLATION = "budget_violation"
    PROVIDER_TIMEOUT = "provider_timeout"
    INTERNAL_ERROR = "internal_error"


class ExpectedRequirementProposal(BaseModel):
    """Human annotation for one ordered, identity-free requirement proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: NonEmptyText
    quantity: Annotated[int, Field(strict=True, ge=1, le=100)] = 1
    max_budget: float | None = None
    required_features: Annotated[
        tuple[NonEmptyText, ...], Field(max_length=20)
    ] = ()
    soft_preferences: Annotated[
        tuple[NonEmptyText, ...], Field(max_length=20)
    ] = ()
    priority: Annotated[int, Field(strict=True, ge=1, le=5)] = 3

    @field_validator("max_budget", mode="before")
    @classmethod
    def validate_max_budget(cls, value: object) -> object:
        return None if value is None else _finite_positive(value, "max_budget")

    @field_validator("required_features", "soft_preferences", mode="after")
    @classmethod
    def normalize_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _stable_unique(values)


class ExpectedAllocationPreference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_index: Annotated[int, Field(strict=True, ge=0)]
    preference: AllocationPreference


class ClarificationExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    needed: Annotated[bool, Field(strict=True)]
    question: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_question(self) -> "ClarificationExpectation":
        if self.needed != (self.question is not None):
            raise ValueError("question must be present exactly when clarification is needed.")
        return self


class ExpectedGoal(BaseModel):
    """Independent annotation truth for structured Goal extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_budget: float | None = None
    ordered_requirement_proposals: Annotated[
        tuple[ExpectedRequirementProposal, ...], Field(max_length=10)
    ] = ()
    allocation_preferences: tuple[ExpectedAllocationPreference, ...] = ()
    clarification: ClarificationExpectation

    @field_validator("total_budget", mode="before")
    @classmethod
    def validate_total_budget(cls, value: object) -> object:
        return None if value is None else _finite_positive(value, "total_budget")

    @model_validator(mode="after")
    def validate_goal(self) -> "ExpectedGoal":
        count = len(self.ordered_requirement_proposals)
        targets = tuple(value.target_index for value in self.allocation_preferences)
        if any(target >= count for target in targets):
            raise ValueError("Allocation target must reference an ordered proposal.")
        if len(set(targets)) != len(targets):
            raise ValueError("Each proposal may have at most one allocation preference.")
        if not self.clarification.needed:
            if self.total_budget is None:
                raise ValueError("A complete expected goal requires total_budget.")
            if not self.ordered_requirement_proposals:
                raise ValueError("A complete expected goal requires proposals.")
        return self


class GroundingExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_constraint: NonEmptyText
    expected_canonical_constraint: NonEmptyText | None = None
    expected_status: ExpectedGroundingStatus

    @model_validator(mode="after")
    def validate_grounding(self) -> "GroundingExpectation":
        if self.expected_status is ExpectedGroundingStatus.GROUNDED:
            if self.expected_canonical_constraint is None:
                raise ValueError("GROUNDED requires expected_canonical_constraint.")
        elif self.expected_canonical_constraint is not None:
            raise ValueError("UNRESOLVED cannot define a canonical constraint.")
        return self


class ExpectedToolArguments(BaseModel):
    """Expected safe arguments for one recommendation invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_index: Annotated[int, Field(strict=True, ge=0)]
    top_k: Annotated[int, Field(strict=True, ge=1, le=50)]
    category: NonEmptyText
    max_price: float | None = None
    required_features: tuple[NonEmptyText, ...] = ()

    @field_validator("max_price", mode="before")
    @classmethod
    def validate_max_price(cls, value: object) -> object:
        return None if value is None else _finite_positive(value, "max_price")


class ExpectedExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_terminal_statuses: Annotated[
        tuple[ExpectedTerminalStatus, ...], Field(min_length=1)
    ]
    expected_tool_arguments: tuple[ExpectedToolArguments, ...] = ()
    replan_expectation: ReplanExpectation = ReplanExpectation.NOT_EXPECTED
    expected_conflict_reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "ExpectedExecution":
        if len(set(self.allowed_terminal_statuses)) != len(
            self.allowed_terminal_statuses
        ):
            raise ValueError("allowed_terminal_statuses must be unique.")
        if (
            self.expected_conflict_reason is not None
            and ExpectedTerminalStatus.CONFLICT not in self.allowed_terminal_statuses
        ):
            raise ValueError("A conflict reason requires CONFLICT to be allowed.")
        return self


class ProductConstraintTruth(BaseModel):
    """Independent fact annotation for one product and hard constraint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: NonEmptyText
    item_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    constraint: NonEmptyText
    status: VerificationTruthStatus
    provenance: NonEmptyText


class EvidenceAnnotation(BaseModel):
    """Independent annotation of evidence, never copied from verifier output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_reference: NonEmptyText
    parent_asin: NonEmptyText
    constraint: NonEmptyText
    status: VerificationTruthStatus
    provenance: NonEmptyText


class ResponseRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_facts: tuple[NonEmptyText, ...] = ()
    forbidden_facts: tuple[NonEmptyText, ...] = ()
    forbidden_leakage: tuple[NonEmptyText, ...] = ()


class DecodingConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    max_tokens: Annotated[int, Field(strict=True, ge=1)] | None = None
    seed: int | None = None


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_retries: Annotated[int, Field(strict=True, ge=0)] = 0
    execution_retries: Annotated[int, Field(strict=True, ge=0)] = 0


class ArtifactVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_name: NonEmptyText
    version: NonEmptyText


class RuntimeProfile(BaseModel):
    """Secret-free reproducibility metadata for comparable evaluation runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_identifier: NonEmptyText
    provider: NonEmptyText
    adapter: NonEmptyText
    decoding: DecodingConfiguration = DecodingConfiguration()
    retry_policy: RetryPolicy = RetryPolicy()
    artifact_versions: tuple[ArtifactVersion, ...] = ()
    user_identity_policy: NonEmptyText

    @model_validator(mode="after")
    def validate_artifacts(self) -> "RuntimeProfile":
        names = tuple(value.artifact_name.casefold() for value in self.artifact_versions)
        if len(set(names)) != len(names):
            raise ValueError("Artifact names must be unique.")
        return self


class AnnotationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotator: NonEmptyText
    annotation_version: NonEmptyText
    adjudication_status: AdjudicationStatus
    provenance: NonEmptyText


class EvaluationCase(BaseModel):
    """Versioned benchmark case with annotation truth separated from runtime output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    schema_version: NonEmptyText
    split: EvaluationSplit
    locale: NonEmptyText
    tags: tuple[NonEmptyText, ...] = ()
    user_input: NonEmptyText
    runtime_profile: RuntimeProfile
    expected_goal: ExpectedGoal
    expected_grounding: tuple[GroundingExpectation, ...] = ()
    expected_execution: ExpectedExecution
    independent_product_truth: tuple[ProductConstraintTruth, ...] = ()
    evidence_annotations: tuple[EvidenceAnnotation, ...] = ()
    response_requirements: ResponseRequirements = ResponseRequirements()
    annotation_metadata: AnnotationMetadata

    @field_validator("tags", mode="after")
    @classmethod
    def deduplicate_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _stable_unique(values)


class MetricResult(BaseModel):
    """One computed metric result; calculation remains outside the contract layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: NonEmptyText
    value: float
    numerator: float
    denominator: float
    evaluation_scope: NonEmptyText

    @model_validator(mode="before")
    @classmethod
    def validate_numbers(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        copied["value"] = _finite_non_negative(copied.get("value"), "value")
        copied["numerator"] = _finite_non_negative(
            copied.get("numerator"), "numerator"
        )
        denominator = _finite_non_negative(copied.get("denominator"), "denominator")
        if denominator <= 0:
            raise ValueError("denominator must be greater than zero.")
        if copied["numerator"] > denominator:
            raise ValueError("numerator cannot exceed denominator.")
        copied["denominator"] = denominator
        return copied


class ComponentOutputReference(BaseModel):
    """Safe reference to a component output retained outside this contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: NonEmptyText
    reference: NonEmptyText


class EvaluationResult(BaseModel):
    """Evaluation outcome; it contains predictions/references, never annotation truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    execution_status: ExpectedTerminalStatus
    earliest_failure_layer: EvaluationLayer | None = None
    failure_category: FailureCategory | None = None
    failure_code: FailureCode | None = None
    component_outputs: tuple[ComponentOutputReference, ...] = ()
    metric_results: tuple[MetricResult, ...] = ()

    @model_validator(mode="after")
    def validate_failure_fields(self) -> "EvaluationResult":
        values = (
            self.earliest_failure_layer,
            self.failure_category,
            self.failure_code,
        )
        if any(value is None for value in values) and any(
            value is not None for value in values
        ):
            raise ValueError("Failure layer, category, and code must be set together.")
        return self
