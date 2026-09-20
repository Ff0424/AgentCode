"""Deterministic projection from terminal workflow state to grounded context.

The projector is the response security boundary: it copies public product
facts only from ShoppingPlan, constructs feature claims only through selected
SUPPORTED verification provenance, and recognizes only the two frozen P0
conflict shapes. It performs no rendering, I/O, service calls, or mutation.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import NoReturn

from ..domain import PlanStatus, RequirementStatus
from ..replanning import (
    CandidateStatusSummary,
    FailureReason,
    FailureRecoverability,
    FailureType,
    ReplanAction,
    ReplanReason,
    VerificationFailurePattern,
)
from ..verification import (
    CandidateVerificationStatus,
    ConstraintVerificationReason,
    ConstraintVerificationStatus,
)
from ..workflows import ShoppingWorkflowState, WorkflowRoute
from .contracts import (
    ConflictDecisionSummary,
    ConflictReason,
    ConflictResponseContext,
    EvidenceReference,
    GroundedResponseContext,
    ProductDecisionSummary,
    ReadyResponseContext,
    VerifiedConstraintClaim,
)


class ProjectionErrorCode(str, Enum):
    UNSUPPORTED_TERMINAL_STATE = "unsupported_terminal_state"
    STATE_INCONSISTENCY = "state_inconsistency"
    MISSING_GROUNDING = "missing_grounding"
    IDENTITY_OR_VERSION_MISMATCH = "identity_or_version_mismatch"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class GroundedResponseProjectionError(RuntimeError):
    """Detailed internal projection error; integration will sanitize it later."""

    def __init__(self, code: ProjectionErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


def _fail(code: ProjectionErrorCode, message: str) -> NoReturn:
    raise GroundedResponseProjectionError(code, message)


def _same_money(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def _requirement(state: ShoppingWorkflowState):
    requirement_id = state.agent_state.current_requirement_id
    matches = tuple(
        value
        for value in state.agent_state.shopping_plan.requirements
        if value.requirement_id == requirement_id
    )
    if len(matches) != 1:
        _fail(
            ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
            "Terminal conflict has no unique current requirement.",
        )
    return matches[0]


class GroundedResponseProjector:
    """Project a validated READY or CONFLICT workflow state without side effects."""

    def project(self, workflow_state: ShoppingWorkflowState) -> GroundedResponseContext:
        if not isinstance(workflow_state, ShoppingWorkflowState):
            raise TypeError("workflow_state must be a ShoppingWorkflowState.")
        try:
            if workflow_state.route is WorkflowRoute.READY:
                return self._project_ready(workflow_state)
            if workflow_state.route is WorkflowRoute.CONFLICT:
                return self._project_conflict(workflow_state)
            _fail(
                ProjectionErrorCode.UNSUPPORTED_TERMINAL_STATE,
                f"Unsupported terminal workflow route={workflow_state.route!r}.",
            )
        except GroundedResponseProjectionError:
            raise
        except Exception as exc:
            raise GroundedResponseProjectionError(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "Terminal workflow state could not be projected safely.",
            ) from exc

    @staticmethod
    def _project_ready(state: ShoppingWorkflowState) -> ReadyResponseContext:
        plan = state.agent_state.shopping_plan
        evaluation = state.evaluation
        if plan.status is not PlanStatus.READY or not plan.is_valid:
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "READY route requires a valid READY ShoppingPlan.",
            )
        if evaluation is None:
            _fail(ProjectionErrorCode.MISSING_GROUNDING, "READY evaluation is missing.")
        if (
            evaluation.plan_id != plan.plan_id
            or evaluation.plan_version != plan.version
            or evaluation.status is not plan.status
        ):
            _fail(
                ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                "Plan evaluation identity, version, or status is stale.",
            )
        if (
            not evaluation.is_ready
            or evaluation.has_hard_constraint_conflict
            or evaluation.is_over_budget
            or plan.has_hard_constraint_conflict
            or plan.is_over_budget
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "READY evaluation contains a budget or hard-constraint conflict.",
            )
        if not _same_money(evaluation.total_spent, plan.total_spent) or not _same_money(
            evaluation.remaining_budget, plan.remaining_budget
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "Evaluation money facts differ from ShoppingPlan domain facts.",
            )
        if (
            evaluation.requirement_count != len(plan.requirements)
            or evaluation.satisfied_requirement_count != len(plan.requirements)
            or evaluation.pending_requirement_count != 0
            or evaluation.conflict_requirement_count != 0
            or evaluation.issues
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "READY evaluation counts or issues differ from plan facts.",
            )
        if state.current_evidence is not None or state.current_verification is not None:
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "READY state cannot retain attempt-local grounding.",
            )

        requirements = {value.requirement_id: value for value in plan.requirements}
        if len(requirements) != len(plan.requirements):
            _fail(ProjectionErrorCode.STATE_INCONSISTENCY, "Requirement IDs are ambiguous.")
        if any(value.status is not RequirementStatus.SATISFIED for value in plan.requirements):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "READY response requires every requirement to be satisfied.",
            )
        items_by_requirement: dict[str, list[object]] = {
            value.requirement_id: [] for value in plan.requirements
        }
        for item in plan.selected_items:
            if item.requirement_id not in requirements:
                _fail(
                    ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                    "Selected item references an unknown requirement.",
                )
            items_by_requirement[item.requirement_id].append(item)
        if any(len(values) != 1 for values in items_by_requirement.values()):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "P0 response requires exactly one selected product per requirement.",
            )

        evidence_by_requirement = {
            value.requirement_id: value for value in state.selected_evidence
        }
        if len(evidence_by_requirement) != len(state.selected_evidence):
            _fail(ProjectionErrorCode.MISSING_GROUNDING, "Selected evidence is ambiguous.")
        if set(evidence_by_requirement) != set(requirements):
            _fail(
                ProjectionErrorCode.MISSING_GROUNDING,
                "Selected evidence must exactly cover READY requirements.",
            )
        verification_by_requirement = {
            value.requirement_id: value for value in state.selected_verifications
        }
        if len(verification_by_requirement) != len(state.selected_verifications):
            _fail(ProjectionErrorCode.MISSING_GROUNDING, "Selected verification is ambiguous.")
        expected_verified = {
            value.requirement_id for value in plan.requirements if value.required_features
        }
        if set(verification_by_requirement) != expected_verified:
            _fail(
                ProjectionErrorCode.MISSING_GROUNDING,
                "Selected verification does not exactly cover hard-feature requirements.",
            )

        products: list[ProductDecisionSummary] = []
        for requirement in plan.requirements:
            item = items_by_requirement[requirement.requirement_id][0]
            item_category = " ".join(item.category.split()).casefold()
            requirement_category = " ".join(requirement.category.split()).casefold()
            if item_category != requirement_category or item.quantity != requirement.quantity:
                _fail(
                    ProjectionErrorCode.STATE_INCONSISTENCY,
                    "Selected item category or quantity differs from its requirement.",
                )
            evidence = evidence_by_requirement[requirement.requirement_id]
            if (
                evidence.plan_id != plan.plan_id
                or evidence.requirement_id != requirement.requirement_id
                or evidence.parent_asin != item.parent_asin
                or evidence.selected_at_plan_version < 1
                or evidence.selected_at_plan_version > plan.version
            ):
                _fail(
                    ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                    "Selected evidence identity or version differs from the plan item.",
                )
            snippet_ids = tuple(value.chunk_id for value in evidence.snippets)
            if len(set(snippet_ids)) != len(snippet_ids):
                _fail(ProjectionErrorCode.MISSING_GROUNDING, "Evidence chunk IDs are ambiguous.")
            if any(
                value.item_index != evidence.item_index
                or value.parent_asin != evidence.parent_asin
                for value in evidence.snippets
            ):
                _fail(
                    ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                    "Evidence snippet identity differs from selected evidence.",
                )

            claims: tuple[VerifiedConstraintClaim, ...] = ()
            if requirement.required_features:
                selected_verification = verification_by_requirement[requirement.requirement_id]
                candidate = selected_verification.candidate
                if (
                    selected_verification.plan_id != plan.plan_id
                    or selected_verification.requirement_id != requirement.requirement_id
                    or selected_verification.selected_at_plan_version
                    != evidence.selected_at_plan_version
                    or candidate.parent_asin != item.parent_asin
                    or candidate.parent_asin != evidence.parent_asin
                    or candidate.item_index != evidence.item_index
                    or candidate.status is not CandidateVerificationStatus.ELIGIBLE
                    or candidate.verified_at_plan_version + 1
                    != selected_verification.selected_at_plan_version
                ):
                    _fail(
                        ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                        "Selected verification identity or version differs from evidence.",
                    )
                constraints = candidate.constraints
                if tuple(value.constraint for value in constraints) != requirement.required_features:
                    _fail(
                        ProjectionErrorCode.UNSUPPORTED_CLAIM,
                        "Verification constraints do not exactly match required features.",
                    )
                snippets_by_id = {value.chunk_id: value for value in evidence.snippets}
                built: list[VerifiedConstraintClaim] = []
                for constraint in constraints:
                    if (
                        constraint.status is not ConstraintVerificationStatus.SUPPORTED
                        or constraint.reason is not ConstraintVerificationReason.EXPLICIT_SUPPORT
                        or not constraint.supporting_chunk_ids
                        or constraint.contradicting_chunk_ids
                    ):
                        _fail(
                            ProjectionErrorCode.UNSUPPORTED_CLAIM,
                            "READY claims require explicit support without contradiction.",
                        )
                    if len(set(constraint.supporting_chunk_ids)) != len(
                        constraint.supporting_chunk_ids
                    ):
                        _fail(
                            ProjectionErrorCode.MISSING_GROUNDING,
                            "Supporting chunk IDs are ambiguous.",
                        )
                    supporting = set(constraint.supporting_chunk_ids)
                    if not supporting <= set(snippets_by_id):
                        _fail(
                            ProjectionErrorCode.MISSING_GROUNDING,
                            "A supporting chunk is absent from selected evidence.",
                        )
                    references = tuple(
                        EvidenceReference(
                            chunk_id=snippet.chunk_id,
                            chunk_type=snippet.chunk_type,
                            excerpt=snippet.text,
                        )
                        for snippet in evidence.snippets
                        if snippet.chunk_id in supporting
                    )
                    built.append(VerifiedConstraintClaim(
                        requirement_id=requirement.requirement_id,
                        parent_asin=item.parent_asin,
                        original_constraint=constraint.constraint,
                        canonical_constraint=constraint.canonical_constraint,
                        status=constraint.status,
                        supporting_evidence=references,
                    ))
                claims = tuple(built)

            products.append(ProductDecisionSummary(
                requirement_id=requirement.requirement_id,
                category=requirement.category,
                parent_asin=item.parent_asin,
                title=item.title,
                price=item.price,
                quantity=item.quantity,
                source=item.source,
                selection_rationale=item.selected_reason,
                verified_claims=claims,
            ))

        return ReadyResponseContext(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            currency=plan.currency,
            products=tuple(products),
            total_spent=plan.total_spent,
            remaining_budget=plan.remaining_budget,
        )

    def _project_conflict(self, state: ShoppingWorkflowState) -> ConflictResponseContext:
        result = state.last_tool_result
        if (
            result is not None
            and result.returned_count == 0
            and state.agent_state.error_state == "no_recommendation_candidates"
        ):
            return self._project_no_candidates(state)
        if state.agent_state.error_state == "constraint_verification:replan_attempts_exhausted":
            return self._project_exhaustion(state)
        _fail(
            ProjectionErrorCode.UNSUPPORTED_TERMINAL_STATE,
            "CONFLICT state does not match a supported P0 terminal pattern.",
        )

    @staticmethod
    def _project_no_candidates(state: ShoppingWorkflowState) -> ConflictResponseContext:
        plan = state.agent_state.shopping_plan
        requirement = _requirement(state)
        result = state.last_tool_result
        if result is None or result.returned_count != 0 or result.items:
            _fail(ProjectionErrorCode.STATE_INCONSISTENCY, "Tool result is not empty.")
        if state.recommendation_args is None or (
            state.recommendation_args.top_k != state.recommendation_top_k
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "Empty recommendation arguments do not match the active pool.",
            )
        GroundedResponseProjector._validate_recommendation_constraints(
            state, requirement
        )
        if any(
            value is not None
            for value in (
                state.current_evidence,
                state.current_verification,
                state.current_evidence_attempt,
                state.current_verification_attempt,
            )
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "No-candidate conflict cannot contain attempt-local grounding.",
            )
        if any(
            item.requirement_id == requirement.requirement_id
            for item in plan.selected_items
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "No-candidate requirement already has a selected item.",
            )

        if state.recommendation_top_k == 5:
            if (
                state.replan_attempt != 0
                or state.failure_history
                or state.replan_history
                or state.current_failure_diagnosis is not None
                or state.current_replan_directive is not None
            ):
                _fail(
                    ProjectionErrorCode.STATE_INCONSISTENCY,
                    "Initial empty recommendation contains false retry history.",
                )
            pools, attempts = (5,), 0
        elif state.recommendation_top_k == 10:
            GroundedResponseProjector._validate_single_expansion(state, requirement)
            pools, attempts = (5, 10), 1
        else:
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "No-candidate pool size is outside the frozen 5-to-10 policy.",
            )

        return ConflictResponseContext(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            currency=plan.currency,
            decision=ConflictDecisionSummary(
                requirement_id=requirement.requirement_id,
                category=requirement.category,
                required_features=requirement.required_features,
                reason=ConflictReason.NO_RECOMMENDATION_CANDIDATES,
                replan_attempts_performed=attempts,
                candidate_pool_sizes=pools,
            ),
            total_spent=plan.total_spent,
            remaining_budget=plan.remaining_budget,
        )

    @staticmethod
    def _validate_single_expansion(state: ShoppingWorkflowState, requirement) -> None:
        plan = state.agent_state.shopping_plan
        if (
            state.replan_attempt != 1
            or len(state.failure_history) != 1
            or len(state.replan_history) != 1
        ):
            _fail(ProjectionErrorCode.STATE_INCONSISTENCY, "Expansion history is malformed.")
        diagnosis = state.failure_history[0]
        directive = state.replan_history[0]
        if (
            state.current_failure_diagnosis != diagnosis
            or state.current_replan_directive != directive
            or diagnosis.plan_id != plan.plan_id
            or diagnosis.requirement_id != requirement.requirement_id
            or diagnosis.diagnosed_at_plan_version != plan.version
            or diagnosis.attempt != 0
            or diagnosis.failure_type is not FailureType.NO_ELIGIBLE_CANDIDATES
            or diagnosis.recoverability is not FailureRecoverability.RECOVERABLE
            or diagnosis.reason is not FailureReason.VERIFICATION_RETURNED_ZERO_ELIGIBLE
            or directive.plan_id != plan.plan_id
            or directive.requirement_id != requirement.requirement_id
            or directive.source_plan_version != plan.version
            or directive.attempt != 1
            or directive.action is not ReplanAction.EXPAND_CANDIDATE_POOL
            or directive.previous_top_k != 5
            or directive.next_top_k != 10
            or directive.reason is not ReplanReason.EXPAND_AFTER_ZERO_ELIGIBLE
            or directive.preserved_category != requirement.category
            or directive.preserved_max_budget != requirement.max_budget
            or directive.preserved_required_features != requirement.required_features
        ):
            _fail(
                ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                "Expansion diagnosis/directive does not match the frozen policy.",
            )

    @staticmethod
    def _validate_recommendation_constraints(state: ShoppingWorkflowState, requirement) -> None:
        args = state.recommendation_args
        if args is None or (
            args.category != requirement.category
            or args.max_price != requirement.max_budget
            or args.required_features != requirement.required_features
        ):
            _fail(
                ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                "Recommendation arguments differ from the current requirement.",
            )

    @staticmethod
    def _project_exhaustion(state: ShoppingWorkflowState) -> ConflictResponseContext:
        plan = state.agent_state.shopping_plan
        requirement = _requirement(state)
        if (
            state.replan_attempt != 1
            or state.recommendation_top_k != 10
            or state.current_evidence_attempt != 1
            or state.current_verification_attempt != 1
            or len(state.failure_history) != 2
            or len(state.replan_history) != 2
        ):
            _fail(ProjectionErrorCode.STATE_INCONSISTENCY, "Exhaustion history is malformed.")
        GroundedResponseProjector._validate_recommendation_constraints(
            state, requirement
        )
        if any(
            item.requirement_id == requirement.requirement_id
            for item in plan.selected_items
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "Exhausted requirement must not have mutated ShoppingPlan.",
            )
        first_diagnosis, final_diagnosis = state.failure_history
        expand, stop = state.replan_history
        if (
            state.current_failure_diagnosis != final_diagnosis
            or state.current_replan_directive != stop
            or first_diagnosis.attempt != 0
            or first_diagnosis.failure_type is not FailureType.NO_ELIGIBLE_CANDIDATES
            or first_diagnosis.recoverability is not FailureRecoverability.RECOVERABLE
            or first_diagnosis.reason is not FailureReason.VERIFICATION_RETURNED_ZERO_ELIGIBLE
            or final_diagnosis.attempt != 1
            or final_diagnosis.failure_type is not FailureType.REPLAN_ATTEMPTS_EXHAUSTED
            or final_diagnosis.recoverability is not FailureRecoverability.NON_RECOVERABLE
            or final_diagnosis.reason is not FailureReason.BOUNDED_REPLAN_ATTEMPT_EXHAUSTED
        ):
            _fail(ProjectionErrorCode.STATE_INCONSISTENCY, "Diagnosis sequence is invalid.")
        for diagnosis in (first_diagnosis, final_diagnosis):
            if (
                diagnosis.plan_id != plan.plan_id
                or diagnosis.requirement_id != requirement.requirement_id
                or diagnosis.diagnosed_at_plan_version != plan.version
            ):
                _fail(
                    ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                    "Failure diagnosis identity/version differs from the plan.",
                )
        if (
            expand.action is not ReplanAction.EXPAND_CANDIDATE_POOL
            or expand.attempt != 1
            or expand.previous_top_k != 5
            or expand.next_top_k != 10
            or expand.reason is not ReplanReason.EXPAND_AFTER_ZERO_ELIGIBLE
            or stop.action is not ReplanAction.STOP_CONFLICT
            or stop.attempt != 1
            or stop.previous_top_k != 10
            or stop.next_top_k is not None
            or stop.reason is not ReplanReason.ATTEMPTS_EXHAUSTED
        ):
            _fail(ProjectionErrorCode.STATE_INCONSISTENCY, "Replan action sequence is invalid.")
        for directive in (expand, stop):
            if (
                directive.plan_id != plan.plan_id
                or directive.requirement_id != requirement.requirement_id
                or directive.source_plan_version != plan.version
                or directive.preserved_category != requirement.category
                or directive.preserved_max_budget != requirement.max_budget
                or directive.preserved_required_features != requirement.required_features
            ):
                _fail(
                    ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                    "Replan directive changed identity or hard constraints.",
                )

        evidence = state.current_evidence
        verification = state.current_verification
        result = state.last_tool_result
        if evidence is None or verification is None or result is None or result.returned_count == 0:
            _fail(ProjectionErrorCode.MISSING_GROUNDING, "Final attempt grounding is missing.")
        expected_pairs = tuple((x.item_index, x.parent_asin) for x in result.items)
        evidence_candidate_pairs = tuple(
            (x.item_index, x.parent_asin) for x in evidence.candidates
        )
        evidence_pairs = tuple((x.item_index, x.parent_asin) for x in evidence.products)
        verification_pairs = tuple(
            (x.item_index, x.parent_asin) for x in verification.candidates
        )
        if (
            evidence.plan_id != plan.plan_id
            or evidence.requirement_id != requirement.requirement_id
            or evidence.retrieved_at_plan_version != plan.version
            or evidence_candidate_pairs != expected_pairs
            or evidence_pairs != expected_pairs
            or verification.plan_id != plan.plan_id
            or verification.requirement_id != requirement.requirement_id
            or verification.verified_at_plan_version != plan.version
            or verification_pairs != expected_pairs
        ):
            _fail(
                ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                "Final evidence/verification identity or version is stale.",
            )
        if verification.eligible_parent_asins:
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "Exhaustion cannot contain an eligible candidate.",
            )
        if any(
            tuple(value.constraint for value in candidate.constraints)
            != requirement.required_features
            for candidate in verification.candidates
        ):
            _fail(
                ProjectionErrorCode.IDENTITY_OR_VERSION_MISMATCH,
                "Final verification constraints differ from the requirement.",
            )
        for candidate in verification.candidates:
            constraint_statuses = tuple(value.status for value in candidate.constraints)
            expected_status = (
                CandidateVerificationStatus.INELIGIBLE
                if ConstraintVerificationStatus.CONTRADICTED in constraint_statuses
                else CandidateVerificationStatus.UNVERIFIED
                if ConstraintVerificationStatus.UNKNOWN in constraint_statuses
                else CandidateVerificationStatus.ELIGIBLE
            )
            if candidate.status is not expected_status:
                _fail(
                    ProjectionErrorCode.STATE_INCONSISTENCY,
                    "Candidate status differs from its constraint aggregation.",
                )

        statuses = tuple(value.status for value in verification.candidates)
        constraints = tuple(
            value
            for candidate in verification.candidates
            for value in candidate.constraints
        )
        summary = CandidateStatusSummary(
            total_candidates=len(statuses),
            eligible_count=statuses.count(CandidateVerificationStatus.ELIGIBLE),
            unverified_count=statuses.count(CandidateVerificationStatus.UNVERIFIED),
            ineligible_count=statuses.count(CandidateVerificationStatus.INELIGIBLE),
            supported_constraint_count=sum(
                value.status is ConstraintVerificationStatus.SUPPORTED
                for value in constraints
            ),
            contradicted_constraint_count=sum(
                value.status is ConstraintVerificationStatus.CONTRADICTED
                for value in constraints
            ),
            unknown_constraint_count=sum(
                value.status is ConstraintVerificationStatus.UNKNOWN
                for value in constraints
            ),
        )
        failed = tuple(dict.fromkeys(
            value.constraint
            for value in constraints
            if value.status is not ConstraintVerificationStatus.SUPPORTED
        ))
        unverified_count = statuses.count(CandidateVerificationStatus.UNVERIFIED)
        ineligible_count = statuses.count(CandidateVerificationStatus.INELIGIBLE)
        expected_pattern = (
            VerificationFailurePattern.ALL_CANDIDATES_UNVERIFIED
            if unverified_count == len(statuses)
            else VerificationFailurePattern.ALL_CANDIDATES_INELIGIBLE
            if ineligible_count == len(statuses)
            else VerificationFailurePattern.MIXED_VERIFICATION_FAILURE
        )
        if (
            final_diagnosis.candidate_status_summary != summary
            or final_diagnosis.failed_constraints != failed
            or final_diagnosis.verification_pattern is not expected_pattern
        ):
            _fail(
                ProjectionErrorCode.STATE_INCONSISTENCY,
                "Final diagnosis does not match final verification facts.",
            )
        unknown = tuple(
            feature
            for feature in requirement.required_features
            if any(
                value.constraint == feature
                and value.status is ConstraintVerificationStatus.UNKNOWN
                for value in constraints
            )
        )
        contradicted = tuple(
            feature
            for feature in requirement.required_features
            if any(
                value.constraint == feature
                and value.status is ConstraintVerificationStatus.CONTRADICTED
                for value in constraints
            )
        )

        return ConflictResponseContext(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            currency=plan.currency,
            decision=ConflictDecisionSummary(
                requirement_id=requirement.requirement_id,
                category=requirement.category,
                required_features=requirement.required_features,
                reason=ConflictReason.REPLAN_ATTEMPTS_EXHAUSTED,
                replan_attempts_performed=1,
                candidate_pool_sizes=(5, 10),
                verification_status_counts=summary,
                failed_constraints=failed,
                unknown_constraints=unknown,
                contradicted_constraints=contradicted,
            ),
            total_spent=plan.total_spent,
            remaining_budget=plan.remaining_budget,
        )
