"""Public contracts and deterministic projection for grounded responses."""

from typing import TYPE_CHECKING, Any

from .contracts import (
    ConflictDecisionSummary,
    ConflictReason,
    ConflictResponseContext,
    EvidenceReference,
    FinalResponseResult,
    GroundedResponseContext,
    ProductDecisionSummary,
    ReadyResponseContext,
    ResponseErrorCode,
    ResponseKind,
    VerifiedConstraintClaim,
)
from .renderer import DeterministicFinalResponseRenderer

if TYPE_CHECKING:
    from .projector import (
        GroundedResponseProjectionError,
        GroundedResponseProjector,
        ProjectionErrorCode,
    )


_PROJECTOR_EXPORTS = {
    "GroundedResponseProjectionError",
    "GroundedResponseProjector",
    "ProjectionErrorCode",
}


def __getattr__(name: str) -> Any:
    """Keep pure response contracts importable without loading LangGraph."""

    if name in _PROJECTOR_EXPORTS:
        from . import projector

        return getattr(projector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConflictDecisionSummary",
    "ConflictReason",
    "ConflictResponseContext",
    "DeterministicFinalResponseRenderer",
    "EvidenceReference",
    "FinalResponseResult",
    "GroundedResponseContext",
    "GroundedResponseProjectionError",
    "GroundedResponseProjector",
    "ProductDecisionSummary",
    "ProjectionErrorCode",
    "ReadyResponseContext",
    "ResponseErrorCode",
    "ResponseKind",
    "VerifiedConstraintClaim",
]
