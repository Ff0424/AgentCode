"""
Build the formal FAISS index from validated Product Embedding artifacts.

This script never loads BGE-M3, generates embeddings, changes the source
artifacts, or performs retrieval evaluation. Rows are added in source order so
that FAISS row ``i`` remains aligned with embedding row ``i`` and
``product_ids[i]``.

Inputs:
    data/processed/rag/product_embeddings.npy
    data/processed/rag/product_ids.json
    data/processed/rag/product_embeddings.meta.json

Outputs:
    data/processed/rag/product_embeddings.faiss
    data/processed/rag/faiss_index.meta.json
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# 1. Formal artifact and index configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "processed" / "rag"

EMBEDDING_PATH = RAG_DIR / "product_embeddings.npy"
PRODUCT_IDS_PATH = RAG_DIR / "product_ids.json"
EMBEDDING_META_PATH = RAG_DIR / "product_embeddings.meta.json"

INDEX_PATH = RAG_DIR / "product_embeddings.faiss"
INDEX_META_PATH = RAG_DIR / "faiss_index.meta.json"

EXPECTED_ROWS = 125_762
EXPECTED_DIMENSION = 1024
EXPECTED_SHAPE = (EXPECTED_ROWS, EXPECTED_DIMENSION)
EXPECTED_DTYPE = np.dtype(np.float32)
ADD_CHUNK_SIZE = 4096
SANITY_CHECK_ROWS = (0, 100, 1000, EXPECTED_ROWS - 1)
RECONSTRUCTION_RTOL = 0.0
RECONSTRUCTION_ATOL = float(np.finfo(np.float32).eps)


# ============================================================
# 2. Small runtime, JSON, and path helpers
# ============================================================

def utc_now() -> str:
    """Return an unambiguous timestamp for the published metadata."""

    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_faiss():
    """Import FAISS only when the builder is executed."""

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed in the active environment. Install the "
            "project's intended FAISS package before building the index."
        ) from exc
    return faiss


def normalized_path(value) -> Path | None:
    """Resolve a metadata path for comparison without requiring it to exist."""

    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# ============================================================
# 3. Strict pre-build source validation
# ============================================================

def validate_required_inputs() -> None:
    missing = [
        path
        for path in (EMBEDDING_PATH, PRODUCT_IDS_PATH, EMBEDDING_META_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required formal artifacts are missing: "
            + ", ".join(str(path) for path in missing)
        )


def refuse_existing_outputs() -> None:
    """Never overwrite either member of the formal index artifact pair."""

    existing = [path for path in (INDEX_PATH, INDEX_META_PATH) if path.exists()]
    if existing:
        raise FileExistsError(
            "Formal FAISS output already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def validate_input_fingerprint(fingerprint) -> None:
    """Require the complete SHA-256 fingerprint produced by script 13."""

    if not isinstance(fingerprint, dict):
        raise ValueError("Embedding metadata input_fingerprint must be an object.")
    if fingerprint.get("algorithm") != "sha256":
        raise ValueError("Embedding metadata fingerprint algorithm is not sha256.")
    sha256 = fingerprint.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("Embedding metadata contains an invalid SHA-256 value.")
    if fingerprint.get("row_count") != EXPECTED_ROWS:
        raise ValueError("Embedding metadata fingerprint row_count is incorrect.")


def validate_sources():
    """Load sources read-only and enforce the formal script 13 contract."""

    validate_required_inputs()

    try:
        product_ids = read_json(PRODUCT_IDS_PATH)
        metadata = read_json(EMBEDDING_META_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not load source JSON artifact: {exc}") from exc

    if not isinstance(product_ids, list):
        raise ValueError("product_ids.json must contain a top-level list.")
    if len(product_ids) != EXPECTED_ROWS:
        raise ValueError(
            f"Product ID count is {len(product_ids):,}; expected {EXPECTED_ROWS:,}."
        )
    if not isinstance(metadata, dict):
        raise ValueError("Embedding metadata must contain a top-level object.")

    try:
        # Memory mapping avoids copying the 125762 x 1024 source matrix.
        embeddings = np.load(EMBEDDING_PATH, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load the embedding artifact: {exc}") from exc

    if embeddings.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Embedding shape is {embeddings.shape}; expected {EXPECTED_SHAPE}."
        )
    if embeddings.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"Embedding dtype is {embeddings.dtype}; expected {EXPECTED_DTYPE}."
        )

    run_config = metadata.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("Embedding metadata run_config must be an object.")

    checks = {
        "shape": metadata.get("shape") == list(embeddings.shape),
        "dtype": metadata.get("dtype") == embeddings.dtype.name,
        "embedding_path": (
            normalized_path(metadata.get("embedding_path"))
            == EMBEDDING_PATH.resolve()
        ),
        "product_ids_path": (
            normalized_path(metadata.get("product_ids_path"))
            == PRODUCT_IDS_PATH.resolve()
        ),
        "run_config.target_rows": run_config.get("target_rows") == EXPECTED_ROWS,
        "run_config.embedding_dimension": (
            run_config.get("embedding_dimension") == EXPECTED_DIMENSION
        ),
        "run_config.artifact_dtype": (
            run_config.get("artifact_dtype") == EXPECTED_DTYPE.name
        ),
        "run_config.additional_l2_normalization_applied": (
            run_config.get("additional_l2_normalization_applied") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "Embedding metadata is inconsistent with the formal artifacts: "
            + ", ".join(failures)
        )

    validate_input_fingerprint(metadata.get("input_fingerprint"))
    return embeddings, product_ids, metadata


# ============================================================
# 4. Ordered IndexFlatIP construction and sanity checks
# ============================================================

def build_index(faiss, embeddings):
    """Add every source row exactly once and in ascending row order."""

    index = faiss.IndexFlatIP(EXPECTED_DIMENSION)
    for start_row in range(0, EXPECTED_ROWS, ADD_CHUNK_SIZE):
        end_row = min(start_row + ADD_CHUNK_SIZE, EXPECTED_ROWS)
        # FAISS expects a contiguous float32 matrix; no normalization is applied.
        block = np.ascontiguousarray(embeddings[start_row:end_row])
        index.add(block)
        print(f"Added rows [0, {end_row:,}) / {EXPECTED_ROWS:,}")

    validate_index_structure(index)
    return index


def validate_index_structure(index) -> None:
    if index.ntotal != EXPECTED_ROWS:
        raise ValueError(
            f"FAISS ntotal is {index.ntotal:,}; expected {EXPECTED_ROWS:,}."
        )
    if index.d != EXPECTED_DIMENSION:
        raise ValueError(
            f"FAISS dimension is {index.d}; expected {EXPECTED_DIMENSION}."
        )


def run_reconstruction_checks(index, embeddings, stage: str) -> list:
    """Strictly verify that representative rows preserve their float32 values."""

    results = []
    for row in SANITY_CHECK_ROWS:
        source = np.asarray(embeddings[row], dtype=EXPECTED_DTYPE)
        reconstructed = np.asarray(index.reconstruct(row))

        if reconstructed.shape != source.shape:
            raise ValueError(
                f"{stage} reconstruction row {row} has shape "
                f"{reconstructed.shape}; expected {source.shape}."
            )
        if reconstructed.dtype != EXPECTED_DTYPE:
            raise ValueError(
                f"{stage} reconstruction row {row} has dtype "
                f"{reconstructed.dtype}; expected {EXPECTED_DTYPE}."
            )

        max_absolute_error = float(np.max(np.abs(reconstructed - source)))
        matches = bool(
            np.allclose(
                reconstructed,
                source,
                rtol=RECONSTRUCTION_RTOL,
                atol=RECONSTRUCTION_ATOL,
                equal_nan=False,
            )
        )
        if not matches:
            raise ValueError(
                f"{stage} reconstruction mismatch for row {row}: maximum "
                f"absolute error={max_absolute_error:.9g}, allowed "
                f"atol={RECONSTRUCTION_ATOL:.9g}."
            )

        results.append(
            {
                "row": row,
                "max_absolute_error": max_absolute_error,
                "rtol": RECONSTRUCTION_RTOL,
                "atol": RECONSTRUCTION_ATOL,
            }
        )
        print(
            f"{stage} reconstruction row {row:,}: "
            f"max abs error={max_absolute_error:.9g}"
        )
    return results


def run_self_search_diagnostics(index, embeddings) -> list:
    """Record Top-1 behavior without treating a non-self result as failure."""

    results = []
    for row in SANITY_CHECK_ROWS:
        query = np.ascontiguousarray(embeddings[row : row + 1])
        scores, indices = index.search(query, 1)
        returned_row = int(indices[0, 0])
        score = float(scores[0, 0])
        results.append(
            {
                "query_row": row,
                "top1_row": returned_row,
                "top1_is_self": returned_row == row,
                "score": score,
            }
        )
        print(
            f"Self-search diagnostic row {row:,}: Top-1 row={returned_row:,}, "
            f"is_self={returned_row == row}, score={score:.8f}"
        )
    return results


# ============================================================
# 5. Safe serialization and atomic publication
# ============================================================

def build_index_metadata(
    source_metadata: dict,
    in_memory_reconstruction_checks: list,
    self_search_diagnostics: list,
) -> dict:
    return {
        "version": 1,
        "created_at": utc_now(),
        "index_type": "faiss.IndexFlatIP",
        "metric": "inner_product",
        "dimension": EXPECTED_DIMENSION,
        "ntotal": EXPECTED_ROWS,
        "embedding_dtype": EXPECTED_DTYPE.name,
        "source_embedding_file": str(EMBEDDING_PATH),
        "source_product_ids_file": str(PRODUCT_IDS_PATH),
        "source_embedding_metadata_file": str(EMBEDDING_META_PATH),
        "source_embedding_fingerprint": source_metadata["input_fingerprint"],
        "additional_l2_normalization_applied": False,
        "row_alignment": (
            "FAISS row i == product_embeddings row i == product_ids[i]"
        ),
        "in_memory_reconstruction_checks": in_memory_reconstruction_checks,
        "self_search_diagnostics": self_search_diagnostics,
    }


def write_json_file(path: Path, value) -> None:
    """Fully flush a new JSON file before it becomes a formal artifact."""

    with path.open("x", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def publish_artifacts(faiss, index, embeddings, metadata: dict) -> None:
    """Validate temporary files, then atomically replace each formal path."""

    index_temp = INDEX_PATH.with_name(f".{INDEX_PATH.name}.{os.getpid()}.tmp")
    meta_temp = INDEX_META_PATH.with_name(
        f".{INDEX_META_PATH.name}.{os.getpid()}.tmp"
    )

    if index_temp.exists() or meta_temp.exists():
        raise FileExistsError("Process-specific temporary output already exists.")

    try:
        faiss.write_index(index, str(index_temp))
        # Force the completed FAISS bytes to storage before publication.
        with index_temp.open("rb") as index_file:
            os.fsync(index_file.fileno())

        reloaded = faiss.read_index(str(index_temp))
        validate_index_structure(reloaded)
        metadata["reloaded_reconstruction_checks"] = run_reconstruction_checks(
            reloaded,
            embeddings,
            stage="Reloaded index",
        )
        write_json_file(meta_temp, metadata)

        # Recheck immediately before publishing; normal runs never overwrite.
        refuse_existing_outputs()
        os.replace(index_temp, INDEX_PATH)
        os.replace(meta_temp, INDEX_META_PATH)
    finally:
        # Clean up only process-owned temporary files after a handled failure.
        for temporary_path in (index_temp, meta_temp):
            if temporary_path.exists():
                temporary_path.unlink()


# ============================================================
# 6. Program orchestration and entry
# ============================================================

def build_faiss_artifacts() -> None:
    refuse_existing_outputs()
    embeddings, product_ids, source_metadata = validate_sources()
    faiss = load_faiss()

    print("Building faiss.IndexFlatIP(1024) without additional normalization.")
    index = build_index(faiss, embeddings)
    in_memory_checks = run_reconstruction_checks(
        index,
        embeddings,
        stage="In-memory index",
    )
    self_search_diagnostics = run_self_search_diagnostics(index, embeddings)
    metadata = build_index_metadata(
        source_metadata,
        in_memory_checks,
        self_search_diagnostics,
    )
    publish_artifacts(faiss, index, embeddings, metadata)

    print(f"FAISS index written: {INDEX_PATH}")
    print(f"Index metadata written: {INDEX_META_PATH}")
    print("FINAL STATUS: PASS")


def main() -> int:
    build_faiss_artifacts()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("FINAL STATUS: FAIL", file=sys.stderr)
        raise SystemExit(1) from exc
