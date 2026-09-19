"""Deterministic, evidence-only hard-feature verification."""

from __future__ import annotations

import re

from ..domain import ShoppingRequirement
from ..evidence import ProductEvidence, RequirementEvidence
from .aliases import FeatureAliasRegistry, normalize_feature_text
from .contracts import (
    CandidateVerification,
    CandidateVerificationStatus,
    ConstraintVerification,
    ConstraintVerificationReason,
    ConstraintVerificationStatus,
    RequirementVerification,
)


class ConstraintVerificationError(RuntimeError):
    """Fail-closed integrity or stale-state error."""


class VerificationIdentityError(ConstraintVerificationError):
    pass


class VerificationStaleError(ConstraintVerificationError):
    pass


_NEGATIVE_PREFIXES = (
    r"does\s+not\s+support\s+",
    r"doesn['’]t\s+support\s+",
    r"not\s+compatible\s+with\s+",
    r"without\s+",
    r"no\s+",
    r"lacks?\s+",
    r"lack\s+of\s+",
)
_POSITIVE_PREFIXES = (
    r"supports?\s+",
    r"compatible\s+with\s+",
    r"includes?\s+",
    r"has\s+",
    r"with\s+",
    r"equipped\s+with\s+",
)
_POSITIVE_SUFFIX = re.compile(
    r"^\s*(?:output|input|port|ports|connector|connectors|interface|interfaces|hub|support|supported|compatible)\b"
)


def _alias_pattern(alias: str) -> str:
    parts = [re.escape(value) for value in re.split(r"[\s-]+", alias) if value]
    return r"[\s-]+".join(parts)


def _signals(text: str, aliases: tuple[str, ...]) -> tuple[bool, bool]:
    normalized = normalize_feature_text(text)
    support = False
    contradiction = False
    for alias in aliases:
        pattern = _alias_pattern(alias)
        matcher = re.compile(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])")
        for match in matcher.finditer(normalized):
            prefix = normalized[max(0, match.start() - 45):match.start()]
            suffix = normalized[match.end():match.end() + 24]
            if any(re.search(value + r"$", prefix) for value in _NEGATIVE_PREFIXES):
                contradiction = True
                continue
            exact = normalized == match.group(0)
            positive_prefix = any(
                re.search(value + r"$", prefix) for value in _POSITIVE_PREFIXES
            )
            positive_suffix = _POSITIVE_SUFFIX.search(suffix) is not None
            if exact or positive_prefix or positive_suffix:
                support = True
    return support, contradiction


class EvidenceConstraintVerifier:
    """Verify required features using only an existing evidence snapshot."""

    def __init__(self, alias_registry: FeatureAliasRegistry | None = None) -> None:
        self._aliases = alias_registry or FeatureAliasRegistry()

    def _verify_constraint(
        self, constraint: str, product: ProductEvidence
    ) -> ConstraintVerification:
        canonical, aliases = self._aliases.resolve(constraint)
        supporting: list[str] = []
        contradicting: list[str] = []
        known_ids = {value.chunk_id for value in product.snippets}
        if len(known_ids) != len(product.snippets):
            raise VerificationIdentityError("Product evidence has duplicate chunk IDs.")
        for snippet in product.snippets:
            if snippet.item_index != product.item_index or snippet.parent_asin != product.parent_asin:
                raise VerificationIdentityError("Snippet identity differs from its candidate.")
            positive, negative = _signals(snippet.text, aliases)
            if positive:
                supporting.append(snippet.chunk_id)
            if negative:
                contradicting.append(snippet.chunk_id)
        supporting_ids = tuple(dict.fromkeys(supporting))
        contradicting_ids = tuple(dict.fromkeys(contradicting))
        if not set(supporting_ids + contradicting_ids) <= known_ids:
            raise VerificationIdentityError("Verification referenced an unknown chunk ID.")
        if supporting_ids and not contradicting_ids:
            status, reason = ConstraintVerificationStatus.SUPPORTED, ConstraintVerificationReason.EXPLICIT_SUPPORT
        elif contradicting_ids and not supporting_ids:
            status, reason = ConstraintVerificationStatus.CONTRADICTED, ConstraintVerificationReason.EXPLICIT_CONTRADICTION
        elif supporting_ids and contradicting_ids:
            status, reason = ConstraintVerificationStatus.UNKNOWN, ConstraintVerificationReason.CONFLICTING_EVIDENCE
        else:
            status, reason = ConstraintVerificationStatus.UNKNOWN, ConstraintVerificationReason.INSUFFICIENT_EVIDENCE
        return ConstraintVerification(
            constraint=constraint,
            canonical_constraint=canonical,
            status=status,
            reason=reason,
            supporting_chunk_ids=supporting_ids,
            contradicting_chunk_ids=contradicting_ids,
        )

    def verify(
        self,
        *,
        requirement: ShoppingRequirement,
        evidence: RequirementEvidence,
        current_plan_id: str,
        current_plan_version: int,
    ) -> RequirementVerification:
        if not isinstance(requirement, ShoppingRequirement):
            raise TypeError("requirement must be a ShoppingRequirement.")
        if not isinstance(evidence, RequirementEvidence):
            raise TypeError("evidence must be RequirementEvidence.")
        if not requirement.required_features:
            raise ValueError("Requirements without hard features must skip verification.")
        if evidence.plan_id != current_plan_id or evidence.requirement_id != requirement.requirement_id:
            raise VerificationIdentityError("Requirement evidence identity mismatch.")
        if evidence.retrieved_at_plan_version != current_plan_version:
            raise VerificationStaleError("Requirement evidence is stale.")
        evidence_pairs = tuple((value.item_index, value.parent_asin) for value in evidence.products)
        candidate_pairs = tuple((value.item_index, value.parent_asin) for value in evidence.candidates)
        if evidence_pairs != candidate_pairs or len(set(evidence_pairs)) != len(evidence_pairs):
            raise VerificationIdentityError("Candidate evidence identity is ambiguous.")
        candidates: list[CandidateVerification] = []
        for product in evidence.products:
            constraints = tuple(
                self._verify_constraint(value, product)
                for value in requirement.required_features
            )
            statuses = tuple(value.status for value in constraints)
            status = (
                CandidateVerificationStatus.INELIGIBLE
                if ConstraintVerificationStatus.CONTRADICTED in statuses
                else CandidateVerificationStatus.UNVERIFIED
                if ConstraintVerificationStatus.UNKNOWN in statuses
                else CandidateVerificationStatus.ELIGIBLE
            )
            candidates.append(CandidateVerification(
                item_index=product.item_index,
                parent_asin=product.parent_asin,
                status=status,
                constraints=constraints,
                verified_at_plan_version=current_plan_version,
            ))
        return RequirementVerification(
            plan_id=current_plan_id,
            requirement_id=requirement.requirement_id,
            verified_at_plan_version=current_plan_version,
            candidates=tuple(candidates),
        )
