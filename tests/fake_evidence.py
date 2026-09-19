"""Deterministic evidence-service fake shared by workflow tests."""

from src.agentrec.evidence import (
    EvidenceProvenance,
    EvidenceSnippet,
    ProductEvidence,
    RequirementEvidence,
)
from src.agentrec.knowledge import KnowledgeChunkType


PROVENANCE = EvidenceProvenance(
    knowledge_artifact_version="knowledge-v1",
    knowledge_chunks_sha256="a" * 64,
    retrieval_artifact_version="retrieval-v1",
    embedding_model="fake-bge-m3",
    retrieval_backend="fake-exact",
)


class FakeEvidenceService:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def retrieve(self, *, plan_id, plan_version, requirement, candidates):
        self.calls.append((plan_id, plan_version, requirement, candidates))
        if self.fail:
            raise RuntimeError("synthetic retrieval failure")
        feature_texts = requirement.required_features or ("product specifications",)
        products = tuple(
            ProductEvidence(
                item_index=candidate.item_index,
                parent_asin=candidate.parent_asin,
                snippets=tuple(EvidenceSnippet(
                    rank=rank,
                    item_index=candidate.item_index,
                    parent_asin=candidate.parent_asin,
                    chunk_id=f"chunk-{candidate.item_index}-{rank}",
                    chunk_type=KnowledgeChunkType.FEATURES,
                    part_index=rank - 1,
                    text=f"Supports {feature} port.",
                    similarity_score=0.8,
                ) for rank, feature in enumerate(feature_texts, 1)),
            ) for candidate in candidates
        )
        return RequirementEvidence(
            plan_id=plan_id,
            retrieved_at_plan_version=plan_version,
            requirement_id=requirement.requirement_id,
            query=f"Category: {requirement.category}",
            candidates=candidates,
            products=products,
            provenance=PROVENANCE,
        )
