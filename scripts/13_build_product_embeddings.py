"""
Build the formal dense embedding artifact for Product Documents V2.

The script uses the local BGE-M3 model on a single CUDA GPU and writes dense
embeddings incrementally to a NumPy ``.npy`` memmap. Product IDs are emitted in
the exact same row order as the embedding matrix. No FAISS index is built.

Input:
    data/processed/rag/product_documents_v2.jsonl

Outputs:
    data/processed/rag/product_embeddings.npy
    data/processed/rag/product_ids.json
    data/processed/rag/product_embeddings.meta.json
    data/processed/rag/product_embeddings.checkpoint.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


# ============================================================
# 1. Fixed model and artifact configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "product_documents_v2.jsonl"
)
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "rag"

DEVICE = "cuda:0"
USE_FP16 = True
MAX_LENGTH = 2048
BATCH_SIZE = 8
EMBEDDING_DIMENSION = 1024
EMBEDDING_DTYPE = np.dtype(np.float32)
DEFAULT_CHUNK_SIZE = 512


# ============================================================
# 2. Small JSON and time helpers
# ============================================================

def utc_now() -> str:
    """Return a stable UTC timestamp for metadata and checkpoints."""

    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value) -> None:
    """Write JSON through a sibling temporary file and atomically replace it."""

    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())

    # os.replace is atomic when source and destination share a filesystem.
    os.replace(temporary_path, path)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


# ============================================================
# 3. Runtime and input validation
# ============================================================

def load_runtime_dependencies():
    """Import GPU dependencies only when the builder is actually executed."""

    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Install a CUDA-compatible PyTorch "
            "build and FlagEmbedding in the active environment."
        ) from exc

    return torch, BGEM3FlagModel


def validate_environment(torch) -> None:
    """Require the checked local model and exactly selected cuda:0 device."""

    if not DOCUMENT_PATH.is_file():
        raise FileNotFoundError(f"Input document file not found: {DOCUMENT_PATH}")

    required_model_files = (
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
    )
    missing_files = [
        filename
        for filename in required_model_files
        if not (MODEL_PATH / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Local BGE-M3 model is incomplete; missing: "
            + ", ".join(missing_files)
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("cuda:0 is unavailable; a CUDA-capable GPU is required.")

    torch.cuda.set_device(0)


def fingerprint_input(path: Path) -> dict:
    """Compute a content fingerprint and count non-empty JSONL records."""

    digest = hashlib.sha256()
    row_count = 0

    with path.open("rb") as input_file:
        for line in input_file:
            digest.update(line)
            if line.strip():
                row_count += 1

    return {
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
    }


def iter_documents(path: Path, limit: int):
    """Yield validated ``(row_index, product_id, text)`` tuples in file order."""

    row_index = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            if row_index >= limit:
                break

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at source line {line_number}: {exc}"
                ) from exc

            product_id = document.get("id")
            text = document.get("text")
            if not isinstance(product_id, str) or not product_id.strip():
                raise ValueError(
                    f"Missing or invalid product id at source line {line_number}."
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing or empty product text at source line {line_number}."
                )

            yield row_index, product_id, text
            row_index += 1


# ============================================================
# 4. Product ID preflight and output path isolation
# ============================================================

def preflight_product_ids(path: Path, target_rows: int) -> list:
    """Freeze and validate the authoritative row-to-product ID sequence."""

    product_ids = []
    seen_ids = set()

    print(f"Preflighting {target_rows:,} Product Documents...")
    for _, product_id, _ in iter_documents(path, target_rows):
        if product_id in seen_ids:
            raise ValueError(f"Duplicate product id found during preflight: {product_id}")
        seen_ids.add(product_id)
        product_ids.append(product_id)

    if len(product_ids) != target_rows:
        raise ValueError(
            f"Preflight found {len(product_ids):,} documents; "
            f"expected {target_rows:,}."
        )

    print(f"Preflight passed: {len(product_ids):,} unique, non-empty product IDs.")
    return product_ids


def build_output_paths(output_prefix) -> dict:
    """Resolve formal or prefix-isolated artifact paths in the RAG directory."""

    filename_prefix = f"{output_prefix}_" if output_prefix else ""
    return {
        "embedding": OUTPUT_DIR / f"{filename_prefix}product_embeddings.npy",
        "product_ids": OUTPUT_DIR / f"{filename_prefix}product_ids.json",
        "meta": OUTPUT_DIR / f"{filename_prefix}product_embeddings.meta.json",
        "checkpoint": (
            OUTPUT_DIR / f"{filename_prefix}product_embeddings.checkpoint.json"
        ),
    }


# ============================================================
# 5. Artifact and checkpoint state
# ============================================================

def build_run_config(
    target_rows: int,
    chunk_size: int,
    output_prefix,
) -> dict:
    """Return every setting that must remain identical across resume runs."""

    return {
        "model_path": str(MODEL_PATH),
        "device": DEVICE,
        "use_fp16": USE_FP16,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "chunk_size": chunk_size,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "artifact_dtype": EMBEDDING_DTYPE.name,
        "target_rows": target_rows,
        "output_prefix": output_prefix,
        "additional_l2_normalization_applied": False,
    }


def initial_norm_state() -> dict:
    return {
        "count": 0,
        "min": None,
        "sum": 0.0,
        "max": None,
    }


def update_norm_state(norm_state: dict, norms: np.ndarray) -> None:
    """Update resumable norm aggregates without storing per-row norms."""

    chunk_min = float(norms.min())
    chunk_max = float(norms.max())
    norm_state["count"] += int(norms.size)
    norm_state["sum"] += float(norms.sum(dtype=np.float64))
    norm_state["min"] = (
        chunk_min
        if norm_state["min"] is None
        else min(norm_state["min"], chunk_min)
    )
    norm_state["max"] = (
        chunk_max
        if norm_state["max"] is None
        else max(norm_state["max"], chunk_max)
    )


def make_checkpoint(
    input_fingerprint: dict,
    run_config: dict,
    next_row: int,
    last_product_id,
    norm_state: dict,
    started_at: str,
    status: str,
) -> dict:
    return {
        "version": 1,
        "status": status,
        "input_path": str(DOCUMENT_PATH),
        "input_fingerprint": input_fingerprint,
        "run_config": run_config,
        "next_row": next_row,
        "last_product_id": last_product_id,
        "norm_state": norm_state,
        "started_at": started_at,
        "updated_at": utc_now(),
    }


def validate_memmap_shape_dtype(array, target_rows: int) -> None:
    expected_shape = (target_rows, EMBEDDING_DIMENSION)
    if array.shape != expected_shape:
        raise ValueError(
            f"Embedding shape mismatch: found {array.shape}, "
            f"expected {expected_shape}."
        )
    if array.dtype != EMBEDDING_DTYPE:
        raise ValueError(
            f"Embedding dtype mismatch: found {array.dtype}, "
            f"expected {EMBEDDING_DTYPE}."
        )


def validate_finite_rows(array, row_count: int, chunk_size: int) -> None:
    """Check an existing or completed memmap without loading it all into RAM."""

    for start_row in range(0, row_count, chunk_size):
        end_row = min(start_row + chunk_size, row_count)
        if not np.isfinite(array[start_row:end_row]).all():
            raise ValueError(
                f"Non-finite embedding value found in rows "
                f"[{start_row}, {end_row})."
            )


def initialize_new_run(
    input_fingerprint: dict,
    run_config: dict,
    output_paths: dict,
):
    """Create an empty memmap and a committed row-zero checkpoint."""

    existing_paths = [path for path in output_paths.values() if path.exists()]
    if existing_paths:
        formatted = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Output artifacts already exist. Refusing to overwrite them; "
            f"use --resume for a compatible interrupted run. Found: {formatted}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_rows = run_config["target_rows"]
    embeddings = open_memmap(
        output_paths["embedding"],
        mode="w+",
        dtype=EMBEDDING_DTYPE,
        shape=(target_rows, EMBEDDING_DIMENSION),
    )
    embeddings.flush()

    started_at = utc_now()
    norm_state = initial_norm_state()
    checkpoint = make_checkpoint(
        input_fingerprint=input_fingerprint,
        run_config=run_config,
        next_row=0,
        last_product_id=None,
        norm_state=norm_state,
        started_at=started_at,
        status="running",
    )
    write_json_atomic(output_paths["checkpoint"], checkpoint)
    return embeddings, checkpoint


def load_resume_run(
    input_fingerprint: dict,
    run_config: dict,
    output_paths: dict,
):
    """Load and strictly validate the last atomically committed boundary."""

    if not output_paths["checkpoint"].is_file():
        raise FileNotFoundError(
            f"Cannot resume without checkpoint: {output_paths['checkpoint']}"
        )
    if not output_paths["embedding"].is_file():
        raise FileNotFoundError(
            f"Cannot resume without embedding memmap: {output_paths['embedding']}"
        )

    checkpoint = read_json(output_paths["checkpoint"])
    if checkpoint.get("input_fingerprint") != input_fingerprint:
        raise ValueError("Input fingerprint differs from the checkpoint.")
    if checkpoint.get("run_config") != run_config:
        raise ValueError(
            "Run configuration differs from the checkpoint. Resume with the "
            "same --limit and --chunk-size values."
        )

    next_row = checkpoint.get("next_row")
    target_rows = run_config["target_rows"]
    if not isinstance(next_row, int) or not 0 <= next_row <= target_rows:
        raise ValueError(f"Invalid checkpoint next_row: {next_row!r}")

    norm_state = checkpoint.get("norm_state", {})
    if norm_state.get("count") != next_row:
        raise ValueError(
            "Checkpoint norm count does not match its committed row boundary."
        )

    embeddings = open_memmap(output_paths["embedding"], mode="r+")
    validate_memmap_shape_dtype(embeddings, target_rows)
    validate_finite_rows(embeddings, next_row, run_config["chunk_size"])
    return embeddings, checkpoint


# ============================================================
# 6. BGE-M3 encoding and incremental commit
# ============================================================

def encode_dense(model, texts, torch) -> np.ndarray:
    """Encode one chunk and normalize only its storage type, not vector norms."""

    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vectors = output["dense_vecs"]

    if torch.is_tensor(dense_vectors):
        dense_vectors = dense_vectors.detach().cpu().numpy()

    # Persist float32 regardless of the model's FP16 internal computation.
    embeddings = np.asarray(dense_vectors, dtype=EMBEDDING_DTYPE)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Model returned shape {embeddings.shape}; expected {expected_shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Model returned NaN or infinite embedding values.")

    return embeddings


def commit_chunk(
    embeddings_memmap,
    chunk_embeddings: np.ndarray,
    chunk_ids: list,
    start_row: int,
    input_fingerprint: dict,
    run_config: dict,
    checkpoint: dict,
    output_paths: dict,
) -> dict:
    """Write, flush, then atomically advance the recoverable checkpoint."""

    end_row = start_row + len(chunk_ids)
    embeddings_memmap[start_row:end_row] = chunk_embeddings

    norms = np.linalg.norm(chunk_embeddings, axis=1)
    norm_state = dict(checkpoint["norm_state"])
    update_norm_state(norm_state, norms)

    # The checkpoint must never advance until all matrix bytes are flushed.
    embeddings_memmap.flush()
    updated_checkpoint = make_checkpoint(
        input_fingerprint=input_fingerprint,
        run_config=run_config,
        next_row=end_row,
        last_product_id=chunk_ids[-1],
        norm_state=norm_state,
        started_at=checkpoint["started_at"],
        status="running",
    )
    write_json_atomic(output_paths["checkpoint"], updated_checkpoint)
    return updated_checkpoint


# ============================================================
# 7. Final row-aligned artifacts and metadata
# ============================================================

def finalize_artifacts(
    embeddings_memmap,
    product_ids: list,
    input_fingerprint: dict,
    run_config: dict,
    checkpoint: dict,
    elapsed_seconds: float,
    output_paths: dict,
) -> None:
    """Validate the complete matrix and atomically publish IDs and metadata."""

    target_rows = run_config["target_rows"]
    if len(product_ids) != target_rows:
        raise ValueError(
            f"Product ID count {len(product_ids)} does not match {target_rows} rows."
        )
    if checkpoint["next_row"] != target_rows:
        raise ValueError("Checkpoint has not reached the target row count.")

    validate_memmap_shape_dtype(embeddings_memmap, target_rows)
    validate_finite_rows(
        embeddings_memmap,
        target_rows,
        run_config["chunk_size"],
    )
    embeddings_memmap.flush()

    norm_state = checkpoint["norm_state"]
    norm_mean = norm_state["sum"] / norm_state["count"]
    norm_statistics = {
        "min": norm_state["min"],
        "mean": norm_mean,
        "max": norm_state["max"],
    }

    # This array is the authoritative row-to-product mapping for the .npy file.
    write_json_atomic(output_paths["product_ids"], product_ids)

    metadata = {
        "version": 1,
        "input_path": str(DOCUMENT_PATH),
        "input_fingerprint": input_fingerprint,
        "embedding_path": str(output_paths["embedding"]),
        "product_ids_path": str(output_paths["product_ids"]),
        "shape": [target_rows, EMBEDDING_DIMENSION],
        "dtype": EMBEDDING_DTYPE.name,
        "row_alignment": "product_ids[i] corresponds to product_embeddings[i]",
        "run_config": run_config,
        "norm_statistics": norm_statistics,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "completed_at": utc_now(),
    }
    write_json_atomic(output_paths["meta"], metadata)

    completed_checkpoint = make_checkpoint(
        input_fingerprint=input_fingerprint,
        run_config=run_config,
        next_row=target_rows,
        last_product_id=product_ids[-1],
        norm_state=norm_state,
        started_at=checkpoint["started_at"],
        status="completed",
    )
    write_json_atomic(output_paths["checkpoint"], completed_checkpoint)

    print("\nEmbedding artifact completed.")
    print(f"Embeddings : {output_paths['embedding']}")
    print(f"Product IDs: {output_paths['product_ids']}")
    print(f"Metadata   : {output_paths['meta']}")
    print(f"Shape      : ({target_rows:,}, {EMBEDDING_DIMENSION})")
    print(f"Dtype      : {EMBEDDING_DTYPE.name}")
    print(
        "Norms      : "
        f"min={norm_statistics['min']:.6f}, "
        f"mean={norm_statistics['mean']:.6f}, "
        f"max={norm_statistics['max']:.6f}"
    )


# ============================================================
# 8. Build orchestration
# ============================================================

def build_embeddings(args, torch, model) -> None:
    input_fingerprint = fingerprint_input(DOCUMENT_PATH)
    source_rows = input_fingerprint["row_count"]
    target_rows = source_rows if args.limit is None else min(args.limit, source_rows)
    if target_rows <= 0:
        raise ValueError("The selected input contains no rows to embed.")

    # Preflight freezes the authoritative ID order before creating any memmap.
    product_ids = preflight_product_ids(DOCUMENT_PATH, target_rows)
    output_paths = build_output_paths(args.output_prefix)
    run_config = build_run_config(
        target_rows,
        args.chunk_size,
        args.output_prefix,
    )
    if args.resume:
        embeddings_memmap, checkpoint = load_resume_run(
            input_fingerprint,
            run_config,
            output_paths,
        )
    else:
        embeddings_memmap, checkpoint = initialize_new_run(
            input_fingerprint,
            run_config,
            output_paths,
        )

    start_row = checkpoint["next_row"]
    resume_boundary_product_id = checkpoint["last_product_id"]
    print(f"Input rows       : {source_rows:,}")
    print(f"Target rows      : {target_rows:,}")
    print(f"Committed rows   : {start_row:,}")
    print(f"Chunk size       : {args.chunk_size:,}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Device / FP16    : {DEVICE} / {USE_FP16}")
    print(f"Input SHA-256    : {input_fingerprint['sha256']}")
    print(f"Output prefix    : {args.output_prefix or '(formal artifacts)'}")

    pending_ids = []
    pending_texts = []
    started_timer = time.perf_counter()

    # Every encoding row must match the ID order frozen by the preflight scan.
    document_iterator = iter_documents(DOCUMENT_PATH, target_rows)
    for row_index, product_id, text in document_iterator:
        if product_id != product_ids[row_index]:
            raise ValueError(
                f"Product ID order changed at row {row_index}: "
                f"expected {product_ids[row_index]!r}, found {product_id!r}."
            )

        if row_index < start_row:
            continue

        if row_index == start_row and start_row > 0:
            if product_ids[start_row - 1] != resume_boundary_product_id:
                raise ValueError(
                    "Checkpoint last_product_id does not match the input row order."
                )

        # Use the preflight-frozen ID, not a newly derived mapping, for this row.
        pending_ids.append(product_ids[row_index])
        pending_texts.append(text)

        is_final_row = row_index + 1 == target_rows
        if len(pending_texts) < args.chunk_size and not is_final_row:
            continue

        chunk_start = checkpoint["next_row"]
        chunk_embeddings = encode_dense(model, pending_texts, torch)
        checkpoint = commit_chunk(
            embeddings_memmap=embeddings_memmap,
            chunk_embeddings=chunk_embeddings,
            chunk_ids=pending_ids,
            start_row=chunk_start,
            input_fingerprint=input_fingerprint,
            run_config=run_config,
            checkpoint=checkpoint,
            output_paths=output_paths,
        )

        elapsed = time.perf_counter() - started_timer
        docs_per_second = (
            (checkpoint["next_row"] - start_row) / elapsed
            if elapsed > 0
            else 0.0
        )
        print(
            f"Committed {checkpoint['next_row']:,}/{target_rows:,} rows | "
            f"this run {docs_per_second:.2f} docs/s"
        )

        pending_ids = []
        pending_texts = []
        del chunk_embeddings

    if start_row > 0 and product_ids[start_row - 1] != resume_boundary_product_id:
        raise ValueError(
            "Checkpoint last_product_id does not match the input row order."
        )

    finalize_artifacts(
        embeddings_memmap=embeddings_memmap,
        product_ids=product_ids,
        input_fingerprint=input_fingerprint,
        run_config=run_config,
        checkpoint=checkpoint,
        elapsed_seconds=time.perf_counter() - started_timer,
        output_paths=output_paths,
    )


# ============================================================
# 9. Command-line entry
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build row-aligned float32 BGE-M3 embeddings from Product "
            "Documents V2 using local cuda:0."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N documents (useful for validation runs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last flushed and atomically committed checkpoint.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Prefix artifact filenames to isolate smoke tests; omitted means "
            "the formal product_embeddings/product_ids filenames."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows committed per checkpoint (default: {DEFAULT_CHUNK_SIZE}).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer.")
    if args.output_prefix is not None:
        if not args.output_prefix.strip():
            parser.error("--output-prefix cannot be empty or whitespace.")
        if Path(args.output_prefix).name != args.output_prefix:
            parser.error("--output-prefix must be a filename prefix, not a path.")
        if args.output_prefix in {".", ".."}:
            parser.error("--output-prefix must not be '.' or '..'.")

    return args


def main() -> int:
    args = parse_args()
    torch, BGEM3FlagModel = load_runtime_dependencies()
    validate_environment(torch)

    print(f"Loading local BGE-M3 model: {MODEL_PATH}")
    model = BGEM3FlagModel(
        str(MODEL_PATH),
        use_fp16=USE_FP16,
        devices=[DEVICE],
    )
    build_embeddings(args, torch, model)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
