"""Canonical product knowledge schemas, builder, and validation."""

from .artifacts import build_knowledge_artifacts, validate_knowledge_artifacts
from .schemas import KnowledgeChunk, KnowledgeChunkType, ProductDocument

__all__ = [
    "KnowledgeChunk",
    "KnowledgeChunkType",
    "ProductDocument",
    "build_knowledge_artifacts",
    "validate_knowledge_artifacts",
]
