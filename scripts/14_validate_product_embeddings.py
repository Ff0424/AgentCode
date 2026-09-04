"""
Independently validate the formal Product Embedding artifacts.

This script is read-only. It does not load BGE-M3, generate embeddings, modify
artifacts, or build a FAISS index.

Validated files:
    data/processed/rag/product_documents_v2.jsonl
    data/processed/rag/product_embeddings.npy
    data/processed/rag/product_ids.json
    data/processed/rag/product_embeddings.meta.json
    data/processed/rag/product_embeddings.checkpoint.json
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# 1. Formal artifact configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "processed" / "rag"

DOCUMENT_PATH = RAG_DIR / "product_documents_v2.jsonl"
EMBEDDING_PATH = RAG_DIR / "product_embeddings.npy"
PRODUCT_IDS_PATH = RAG_DIR / "product_ids.json"
META_PATH = RAG_DIR / "product_embeddings.meta.json"
CHECKPOINT_PATH = RAG_DIR / "product_embeddings.checkpoint.json"

EXPECTED_ROWS = 125_762
EXPECTED_DIMENSION = 1024
EXPECTED_SHAPE = (EXPECTED_ROWS, EXPECTED_DIMENSION)
EXPECTED_DTYPE = np.dtype(np.float32)
SCAN_CHUNK_SIZE = 512

REQUIRED_PATHS = (
    ("Product Documents V2", DOCUMENT_PATH),
    ("Embedding matrix", EMBEDDING_PATH),
    ("Product IDs", PRODUCT_IDS_PATH),
    ("Embedding metadata", META_PATH),
    ("Embedding checkpoint", CHECKPOINT_PATH),
)


# ============================================================
# 2. Validation report
# ============================================================

class ValidationReport:
    """Collect readable checks and determine the final process exit status."""

    def __init__(self):
        self.errors = []

    def passed(self, name: str, detail: str) -> None:
        print(f"[PASS] {name}: {detail}")

    def failed(self, name: str, detail: str) -> None:
        message = f"{name}: {detail}"
        self.errors.append(message)
        print(f"[FAIL] {message}")

    def check(self, name: str, condition: bool, success: str, failure: str) -> None:
        if condition:
            self.passed(name, success)
        else:
            self.failed(name, failure)

    def finish(self) -> int:
        print("\n===== FINAL VALIDATION SUMMARY =====")
        if not self.errors:
            print("Checks failed: 0")
            print("FINAL STATUS: PASS")
            return 0

        print(f"Checks failed: {len(self.errors)}")
        for index, error in enumerate(self.errors, start=1):
            print(f"  {index}. {error}")
        print("FINAL STATUS: FAIL")
        return 1


# ============================================================
# 3. JSON and file helpers
# ============================================================

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def validate_required_files(report: ValidationReport) -> bool:
    """Report every required path before attempting dependent checks."""

    print("===== Required Files =====")
    all_present = True
    for label, path in REQUIRED_PATHS:
        exists = path.is_file()
        report.check(
            label,
            exists,
            str(path),
            f"missing file {path}",
        )
        all_present = all_present and exists
    return all_present


# ============================================================
# 4. Product IDs and Product Documents alignment
# ============================================================

def validate_product_ids(product_ids, report: ValidationReport) -> None:
    """Validate the standalone authoritative row-to-product mapping."""

    report.check(
        "Product ID JSON type",
        isinstance(product_ids, list),
        "top-level value is a list",
        f"expected list, found {type(product_ids).__name__}",
    )
    if not isinstance(product_ids, list):
        return

    empty_count = sum(
        1
        for product_id in product_ids
        if not isinstance(product_id, str) or not product_id.strip()
    )
    report.check(
        "Product ID count",
        len(product_ids) == EXPECTED_ROWS,
        f"{len(product_ids):,}",
        f"found {len(product_ids):,}, expected {EXPECTED_ROWS:,}",
    )
    report.check(
        "Product IDs non-empty",
        empty_count == 0,
        "empty IDs=0",
        f"empty IDs={empty_count:,}",
    )


def scan_documents(product_ids, report: ValidationReport) -> dict:
    """Recompute fingerprint while validating document IDs row by row."""

    digest = hashlib.sha256()
    document_count = 0
    empty_id_count = 0
    duplicate_id_count = 0
    alignment_mismatch_count = 0
    seen_ids = set()

    with DOCUMENT_PATH.open("rb") as input_file:
        for source_line, raw_line in enumerate(input_file, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue

            try:
                document = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"Invalid Product Document JSON at line {source_line}: {exc}"
                ) from exc

            product_id = document.get("id")
            if not isinstance(product_id, str) or not product_id.strip():
                empty_id_count += 1
            elif product_id in seen_ids:
                duplicate_id_count += 1
            else:
                seen_ids.add(product_id)

            if (
                document_count >= len(product_ids)
                or document.get("id") != product_ids[document_count]
            ):
                alignment_mismatch_count += 1

            document_count += 1

    fingerprint = {
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "size_bytes": DOCUMENT_PATH.stat().st_size,
        "row_count": document_count,
    }

    print("\n===== Product Document Alignment =====")
    report.check(
        "Document count",
        document_count == EXPECTED_ROWS,
        f"{document_count:,}",
        f"found {document_count:,}, expected {EXPECTED_ROWS:,}",
    )
    report.check(
        "Document IDs non-empty",
        empty_id_count == 0,
        "empty IDs=0",
        f"empty IDs={empty_id_count:,}",
    )
    report.check(
        "Document IDs unique",
        duplicate_id_count == 0,
        "duplicate IDs=0",
        f"duplicate IDs={duplicate_id_count:,}",
    )
    report.check(
        "Document-to-product ID row alignment",
        alignment_mismatch_count == 0,
        "mismatched rows=0",
        f"mismatched rows={alignment_mismatch_count:,}",
    )

    print(f"Current input SHA-256: {fingerprint['sha256']}")
    return fingerprint


# ============================================================
# 5. Memory-mapped embedding validation
# ============================================================

def scan_embeddings(embeddings, report: ValidationReport) -> dict:
    """Scan matrix blocks without copying the full embedding matrix into RAM."""

    nan_count = 0
    inf_count = 0
    zero_vector_count = 0
    norm_count = 0
    norm_sum = 0.0
    norm_min = None
    norm_max = None

    for start_row in range(0, embeddings.shape[0], SCAN_CHUNK_SIZE):
        end_row = min(start_row + SCAN_CHUNK_SIZE, embeddings.shape[0])
        block = embeddings[start_row:end_row]

        nan_count += int(np.isnan(block).sum())
        inf_count += int(np.isinf(block).sum())
        finite_rows = np.isfinite(block).all(axis=1)

        if finite_rows.any():
            # float64 accumulation makes the reported global mean stable.
            norms = np.linalg.norm(block[finite_rows], axis=1)
            zero_vector_count += int(np.count_nonzero(norms == 0.0))
            norm_count += int(norms.size)
            norm_sum += float(norms.sum(dtype=np.float64))
            block_min = float(norms.min())
            block_max = float(norms.max())
            norm_min = block_min if norm_min is None else min(norm_min, block_min)
            norm_max = block_max if norm_max is None else max(norm_max, block_max)

    norm_mean = norm_sum / norm_count if norm_count else None

    print("\n===== Embedding Matrix =====")
    report.check(
        "Embedding shape",
        embeddings.shape == EXPECTED_SHAPE,
        str(embeddings.shape),
        f"found {embeddings.shape}, expected {EXPECTED_SHAPE}",
    )
    report.check(
        "Embedding dtype",
        embeddings.dtype == EXPECTED_DTYPE,
        str(embeddings.dtype),
        f"found {embeddings.dtype}, expected {EXPECTED_DTYPE}",
    )
    report.check(
        "Embedding rows",
        embeddings.shape[0] == EXPECTED_ROWS,
        f"{embeddings.shape[0]:,}",
        f"found {embeddings.shape[0]:,}, expected {EXPECTED_ROWS:,}",
    )
    report.check(
        "NaN values",
        nan_count == 0,
        "NaN=0",
        f"NaN={nan_count:,}",
    )
    report.check(
        "Infinite values",
        inf_count == 0,
        "Inf=0",
        f"Inf={inf_count:,}",
    )
    report.check(
        "Zero vectors",
        zero_vector_count == 0,
        "zero-vector=0",
        f"zero-vector={zero_vector_count:,}",
    )

    norm_statistics = {
        "count": norm_count,
        "min": norm_min,
        "mean": norm_mean,
        "max": norm_max,
    }
    if norm_count:
        print(
            "L2 norm min/mean/max: "
            f"{norm_min:.6f}/{norm_mean:.6f}/{norm_max:.6f}"
        )
    else:
        report.failed("L2 norm statistics", "no finite vectors were available")

    return norm_statistics


# ============================================================
# 6. Metadata and checkpoint validation
# ============================================================

def expected_run_config() -> dict:
    """Return the fixed formal configuration defined by script 13."""

    return {
        "model_path": str(PROJECT_ROOT / "models" / "bge-m3"),
        "device": "cuda:0",
        "use_fp16": True,
        "max_length": 2048,
        "batch_size": 8,
        "chunk_size": 512,
        "embedding_dimension": EXPECTED_DIMENSION,
        "artifact_dtype": EXPECTED_DTYPE.name,
        "target_rows": EXPECTED_ROWS,
        "output_prefix": None,
        "additional_l2_normalization_applied": False,
    }


def validate_metadata(
    metadata: dict,
    embeddings,
    current_fingerprint: dict,
    report: ValidationReport,
) -> None:
    print("\n===== Embedding Metadata =====")
    actual_shape = list(embeddings.shape)
    actual_dtype = embeddings.dtype.name
    report.check(
        "Metadata shape",
        metadata.get("shape") == actual_shape == list(EXPECTED_SHAPE),
        f"{metadata.get('shape')}",
        f"metadata={metadata.get('shape')!r}, actual={actual_shape}",
    )
    report.check(
        "Metadata dtype",
        metadata.get("dtype") == actual_dtype == EXPECTED_DTYPE.name,
        actual_dtype,
        f"metadata={metadata.get('dtype')!r}, actual={actual_dtype!r}",
    )
    report.check(
        "Metadata run_config",
        metadata.get("run_config") == expected_run_config(),
        "matches the formal script 13 configuration",
        "does not match the formal script 13 configuration",
    )
    report.check(
        "Metadata input fingerprint",
        metadata.get("input_fingerprint") == current_fingerprint,
        "matches current Product Documents V2",
        "does not match current Product Documents V2",
    )


def validate_checkpoint(
    checkpoint: dict,
    metadata: dict,
    current_fingerprint: dict,
    report: ValidationReport,
) -> None:
    print("\n===== Completion Checkpoint =====")
    report.check(
        "Checkpoint status",
        checkpoint.get("status") == "completed",
        "completed",
        f"found {checkpoint.get('status')!r}, expected 'completed'",
    )
    report.check(
        "Checkpoint next_row",
        checkpoint.get("next_row") == EXPECTED_ROWS,
        f"{EXPECTED_ROWS:,}",
        f"found {checkpoint.get('next_row')!r}, expected {EXPECTED_ROWS:,}",
    )
    norm_count = checkpoint.get("norm_state", {}).get("count")
    report.check(
        "Checkpoint norm count",
        norm_count == EXPECTED_ROWS,
        f"{EXPECTED_ROWS:,}",
        f"found {norm_count!r}, expected {EXPECTED_ROWS:,}",
    )
    report.check(
        "Checkpoint run_config",
        checkpoint.get("run_config") == metadata.get("run_config"),
        "matches metadata run_config",
        "does not match metadata run_config",
    )
    report.check(
        "Checkpoint input fingerprint",
        checkpoint.get("input_fingerprint") == current_fingerprint,
        "matches current Product Documents V2",
        "does not match current Product Documents V2",
    )


# ============================================================
# 7. Validation orchestration
# ============================================================

def validate_artifacts() -> int:
    report = ValidationReport()
    if not validate_required_files(report):
        print("\nDependent validation checks were skipped because files are missing.")
        return report.finish()

    try:
        product_ids = read_json(PRODUCT_IDS_PATH)
        metadata = read_json(META_PATH)
        checkpoint = read_json(CHECKPOINT_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        report.failed("Artifact JSON loading", str(exc))
        return report.finish()

    validate_product_ids(product_ids, report)
    if not isinstance(product_ids, list):
        print("\nDependent row-alignment checks were skipped: product IDs are invalid.")
        return report.finish()
    report.check(
        "Metadata JSON type",
        isinstance(metadata, dict),
        "top-level value is an object",
        f"expected object, found {type(metadata).__name__}",
    )
    report.check(
        "Checkpoint JSON type",
        isinstance(checkpoint, dict),
        "top-level value is an object",
        f"expected object, found {type(checkpoint).__name__}",
    )
    if not isinstance(metadata, dict) or not isinstance(checkpoint, dict):
        print("\nDependent metadata/checkpoint checks were skipped: invalid JSON type.")
        return report.finish()

    try:
        current_fingerprint = scan_documents(product_ids, report)
    except (OSError, ValueError) as exc:
        report.failed("Product Document scan", str(exc))
        return report.finish()

    try:
        # mmap_mode keeps the 125762 x 1024 matrix out of process memory.
        embeddings = np.load(EMBEDDING_PATH, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        report.failed("Embedding matrix loading", str(exc))
        return report.finish()

    if embeddings.ndim != 2:
        report.failed(
            "Embedding rank",
            f"found ndim={embeddings.ndim}, expected a two-dimensional matrix",
        )
        return report.finish()

    scan_embeddings(embeddings, report)
    validate_metadata(metadata, embeddings, current_fingerprint, report)
    validate_checkpoint(checkpoint, metadata, current_fingerprint, report)

    print("\n===== Cross-Artifact Counts =====")
    report.check(
        "Documents / IDs / embedding rows",
        (
            current_fingerprint["row_count"]
            == len(product_ids)
            == embeddings.shape[0]
            == EXPECTED_ROWS
        ),
        f"all counts={EXPECTED_ROWS:,}",
        (
            f"documents={current_fingerprint['row_count']:,}, "
            f"IDs={len(product_ids):,}, embeddings={embeddings.shape[0]:,}, "
            f"expected={EXPECTED_ROWS:,}"
        ),
    )

    return report.finish()


# ============================================================
# 8. Program entry
# ============================================================

def main() -> int:
    print("Product Embedding Artifact Validation")
    print("Read-only validation; BGE-M3 is not loaded.\n")
    return validate_artifacts()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nUNEXPECTED VALIDATION ERROR: {exc}", file=sys.stderr)
        print("FINAL STATUS: FAIL", file=sys.stderr)
        raise SystemExit(1) from exc
