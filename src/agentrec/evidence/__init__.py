"""Grounded evidence contracts, deterministic query construction, and service."""

from .contracts import (
    EvidenceCandidate,
    EvidenceProvenance,
    EvidenceSnippet,
    EvidenceStatus,
    ProductEvidence,
    RequirementEvidence,
    SelectedProductEvidence,
)
from .query_builder import EvidenceQueryBuilder
from .service import (
    EvidenceCandidateError,
    EvidenceEmptyError,
    EvidenceIdentityError,
    GroundedEvidenceError,
    GroundedEvidenceService,
)

__all__ = [
    "EvidenceCandidate",
    "EvidenceCandidateError",
    "EvidenceEmptyError",
    "EvidenceIdentityError",
    "EvidenceProvenance",
    "EvidenceQueryBuilder",
    "EvidenceSnippet",
    "EvidenceStatus",
    "GroundedEvidenceError",
    "GroundedEvidenceService",
    "ProductEvidence",
    "RequirementEvidence",
    "SelectedProductEvidence",
]
