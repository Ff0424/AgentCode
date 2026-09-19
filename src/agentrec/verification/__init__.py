"""Public deterministic evidence-verification contracts and service."""

from .aliases import FeatureAliasRegistry, normalize_feature_text
from .contracts import (
    CandidateVerification,
    CandidateVerificationStatus,
    ConstraintVerification,
    ConstraintVerificationReason,
    ConstraintVerificationStatus,
    RequirementVerification,
    SelectedCandidateVerification,
)
from .service import (
    ConstraintVerificationError,
    EvidenceConstraintVerifier,
    VerificationIdentityError,
    VerificationStaleError,
)

__all__ = [
    "CandidateVerification", "CandidateVerificationStatus",
    "ConstraintVerification", "ConstraintVerificationError",
    "ConstraintVerificationReason", "ConstraintVerificationStatus",
    "EvidenceConstraintVerifier", "FeatureAliasRegistry",
    "RequirementVerification", "SelectedCandidateVerification",
    "VerificationIdentityError", "VerificationStaleError",
    "normalize_feature_text",
]
