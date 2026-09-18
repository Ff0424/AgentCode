"""Read-only semantic retrieval artifact construction and validation."""

from .artifact_loader import ChunkRetrievalArtifacts, ValidationMode
from .backends import BackendSearchResult, ChunkRetrievalBackend, NumPyExactBackend
from .chunk_retriever import ChunkRetriever
from .contracts import RetrievedChunk, RetrievalResult
from .embedding_artifacts import (
    BGEM3ChunkEncoder,
    ChunkEmbeddingConfig,
    build_chunk_embedding_artifacts,
    validate_chunk_embedding_artifacts,
)
from .query_encoder import BGEM3QueryEncoder, QueryEncoder, normalize_query_embeddings

__all__ = [
    "BGEM3ChunkEncoder",
    "BGEM3QueryEncoder",
    "BackendSearchResult",
    "ChunkRetrievalArtifacts",
    "ChunkRetrievalBackend",
    "ChunkEmbeddingConfig",
    "ChunkRetriever",
    "NumPyExactBackend",
    "QueryEncoder",
    "RetrievedChunk",
    "RetrievalResult",
    "ValidationMode",
    "build_chunk_embedding_artifacts",
    "normalize_query_embeddings",
    "validate_chunk_embedding_artifacts",
]
