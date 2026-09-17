"""Fingerprint and validate cached semantic item-embedding artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.recommendation.datasets.canonical_items import CanonicalItemUniverse
from experiments.recommendation.datasets.semantic_config import SemanticConfig


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expected_provenance(config: SemanticConfig, universe: CanonicalItemUniverse) -> dict:
    return {
        "canonical_item_indices_sha256": universe.item_indices_sha256,
        "canonical_identity_sha256": universe.identity_sha256,
        "item_mapping_sha256": sha256_file(config.item_mapping_path),
        "parent_asins_sha256": sha256_file(config.parent_asins_path),
        "product_catalog_sha256": sha256_file(config.product_catalog_path),
    }


def validate_embedding_artifacts(
    config: SemanticConfig,
    universe: CanonicalItemUniverse,
) -> tuple[np.ndarray, dict]:
    """Open the NPY without copying and validate its complete metadata contract."""

    if not config.item_embedding_path.is_file():
        raise FileNotFoundError(f"Item embeddings not found: {config.item_embedding_path}")
    if not config.item_embedding_metadata_path.is_file():
        raise FileNotFoundError(
            f"Item embedding metadata not found: {config.item_embedding_metadata_path}"
        )
    with config.item_embedding_metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    # Copy-on-write mmap remains disk read-only while allowing zero-copy tensor wrapping.
    embeddings = np.load(config.item_embedding_path, mmap_mode="c", allow_pickle=False)
    expected_shape = (universe.item_indices.size, config.embedding_dim)
    if embeddings.shape != expected_shape or embeddings.dtype != np.dtype("float32"):
        raise ValueError(
            f"Embedding array is {embeddings.shape}/{embeddings.dtype}; expected "
            f"{expected_shape}/float32."
        )
    expected = {
        "num_items": universe.item_indices.size,
        "embedding_dim": config.embedding_dim,
        "model": str(config.model_name_or_path),
        "text_fields": list(config.text_fields),
        "normalize_embeddings": config.normalize_embeddings,
        "row_mapping": "row i -> sorted canonical_item_indices[i]",
        "provenance": expected_provenance(config, universe),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Embedding metadata mismatch for {key!r}.")
    # Chunked validation avoids copying the complete matrix into memory.
    for start in range(0, embeddings.shape[0], 4096):
        block = np.asarray(embeddings[start : start + 4096])
        if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) == 0):
            raise ValueError(f"Invalid embedding values in rows {start}:{start+len(block)}.")
    return embeddings, metadata
