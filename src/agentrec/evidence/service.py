"""Identity-safe evidence retrieval over trusted recommendation candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain import ShoppingRequirement
from ..retrieval import RetrievalResult
from .contracts import (
    EvidenceCandidate,
    EvidenceProvenance,
    EvidenceSnippet,
    ProductEvidence,
    RequirementEvidence,
)
from .query_builder import EvidenceQueryBuilder


class GroundedEvidenceError(RuntimeError):
    """Base class for fail-closed evidence integration errors."""


class EvidenceCandidateError(GroundedEvidenceError):
    pass


class EvidenceIdentityError(GroundedEvidenceError):
    pass


class EvidenceEmptyError(GroundedEvidenceError):
    pass


TOP_K_PER_CANDIDATE = 3


def _derive_provenance(retriever: Any) -> EvidenceProvenance:
    artifacts = getattr(retriever, "artifacts", None)
    manifest = getattr(artifacts, "manifest", None)
    knowledge_dir = getattr(artifacts, "knowledge_dir", None)
    backend = getattr(retriever, "backend", None)
    if not isinstance(manifest, dict) or knowledge_dir is None or backend is None:
        raise ValueError("Retriever does not expose validated artifact provenance.")
    knowledge_path = Path(knowledge_dir) / "knowledge_manifest.json"
    try:
        knowledge_manifest = json.loads(knowledge_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read knowledge artifact provenance.") from exc
    chunks_sha = manifest.get("source", {}).get("sha256")
    if chunks_sha != knowledge_manifest.get("outputs", {}).get("chunks.jsonl", {}).get("sha256"):
        raise ValueError("Knowledge and retrieval provenance fingerprints differ.")
    model = manifest.get("model", {})
    model_name = model.get("name") or model.get("path")
    return EvidenceProvenance(
        knowledge_artifact_version=knowledge_manifest.get("artifact_version"),
        knowledge_chunks_sha256=chunks_sha,
        retrieval_artifact_version=manifest.get("artifact_version"),
        embedding_model=model_name,
        retrieval_backend=type(backend).__name__,
    )


class GroundedEvidenceService:
    """Retrieve evidence per candidate without recommendation-layer coupling."""

    def __init__(
        self,
        *,
        retriever: Any,
        query_builder: EvidenceQueryBuilder | None = None,
        provenance: EvidenceProvenance | None = None,
    ) -> None:
        if retriever is None or not callable(getattr(retriever, "search", None)):
            raise TypeError("retriever must provide search().")
        self._retriever = retriever
        self._query_builder = query_builder or EvidenceQueryBuilder()
        self._provenance = provenance or _derive_provenance(retriever)

    def retrieve(
        self,
        *,
        plan_id: str,
        plan_version: int,
        requirement: ShoppingRequirement,
        candidates: tuple[EvidenceCandidate, ...],
    ) -> RequirementEvidence:
        if not isinstance(candidates, tuple):
            raise TypeError("candidates must be a tuple of EvidenceCandidate values.")
        if not candidates:
            raise EvidenceCandidateError("Evidence retrieval requires non-empty candidates.")
        if any(not isinstance(value, EvidenceCandidate) for value in candidates):
            raise TypeError("Every candidate must be an EvidenceCandidate.")
        pairs = tuple((value.item_index, value.parent_asin) for value in candidates)
        if len(set(pairs)) != len(pairs):
            raise EvidenceCandidateError("Evidence candidate identities must be unique.")
        query = self._query_builder.build(requirement)
        products: list[ProductEvidence] = []
        for candidate in candidates:
            # A one-item restriction is deliberate: each recommendation candidate gets coverage.
            result = self._retriever.search(
                query,
                top_k=TOP_K_PER_CANDIDATE,
                candidate_item_indices=(candidate.item_index,),
            )
            if not isinstance(result, RetrievalResult):
                raise TypeError("ChunkRetriever returned an invalid result type.")
            if not result.candidate_restricted:
                raise EvidenceIdentityError("Retriever removed the candidate hard boundary.")
            if result.returned_count == 0:
                raise EvidenceEmptyError(
                    f"No chunks exist for candidate item_index={candidate.item_index}."
                )
            snippets: list[EvidenceSnippet] = []
            for chunk in result.chunks:
                if (
                    chunk.item_index != candidate.item_index
                    or chunk.parent_asin != candidate.parent_asin
                ):
                    raise EvidenceIdentityError(
                        "Retrieved chunk is outside the canonical candidate identity."
                    )
                snippets.append(EvidenceSnippet(
                    rank=chunk.rank,
                    item_index=chunk.item_index,
                    parent_asin=chunk.parent_asin,
                    chunk_id=chunk.chunk_id,
                    chunk_type=chunk.chunk_type,
                    part_index=chunk.part_index,
                    text=chunk.text,
                    similarity_score=chunk.similarity_score,
                ))
            products.append(ProductEvidence(
                item_index=candidate.item_index,
                parent_asin=candidate.parent_asin,
                snippets=tuple(snippets),
            ))
        return RequirementEvidence(
            plan_id=plan_id,
            retrieved_at_plan_version=plan_version,
            requirement_id=requirement.requirement_id,
            query=query,
            candidates=candidates,
            products=tuple(products),
            provenance=self._provenance,
        )
