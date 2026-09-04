"""
Benchmark BGE-M3 dense embedding throughput for Product Documents V2.

This script is deliberately isolated from the data-processing pipeline. It reads
real Product Documents, runs in-memory inference, prints benchmark statistics,
and never writes embeddings, checkpoints, or vector indexes.

Input:
    data/processed/rag/product_documents_v2.jsonl
    models/bge-m3/

Benchmark matrix:
    Token groups: Short (256-512), Medium (800-1200), Long (1800-2048)
    Batch sizes: 8, 16, 32
    One warm-up followed by three measured runs for every combination
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# 1. Paths and benchmark configuration
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

BATCH_SIZES = (8, 16, 32)
MEASURED_RUNS = 3
MAX_LENGTH = 2048
DEFAULT_SAMPLES_PER_GROUP = 256
DEFAULT_SEED = 20260831


@dataclass(frozen=True)
class TokenGroup:
    """Inclusive token-length range used to select benchmark documents."""

    name: str
    min_tokens: int
    max_tokens: int


TOKEN_GROUPS = (
    TokenGroup("Short", 256, 512),
    TokenGroup("Medium", 800, 1200),
    TokenGroup("Long", 1800, MAX_LENGTH),
)


# ============================================================
# 2. Dependency and environment validation
# ============================================================

def load_runtime_dependencies():
    """Import GPU-only dependencies with an actionable error message."""

    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Benchmark dependencies are missing. Install a CUDA-compatible "
            "PyTorch build, FlagEmbedding, and transformers in the active "
            "environment before running this script."
        ) from exc

    return torch, BGEM3FlagModel, AutoTokenizer


def validate_inputs(torch) -> None:
    """Fail early when source data, local weights, or CUDA are unavailable."""

    if not DOCUMENT_PATH.is_file():
        raise FileNotFoundError(
            f"Product Documents V2 not found: {DOCUMENT_PATH}"
        )

    required_model_files = (
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
    )
    missing_files = [
        name for name in required_model_files
        if not (MODEL_PATH / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Local BGE-M3 model is incomplete; missing: "
            + ", ".join(missing_files)
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This benchmark requires one NVIDIA GPU."
        )


# ============================================================
# 3. Real-document sampling by tokenizer length
# ============================================================

def find_token_group(token_count: int):
    """Return the configured group containing token_count, if any."""

    for group in TOKEN_GROUPS:
        if group.min_tokens <= token_count <= group.max_tokens:
            return group
    return None


def sample_documents(tokenizer, samples_per_group: int, seed: int):
    """
    Select a reproducible reservoir sample from each real token-length group.

    Reservoir sampling avoids retaining all 125,762 document texts in memory and
    gives every eligible document in a group an equal chance of being selected.
    """

    rng = random.Random(seed)
    samples = {group.name: [] for group in TOKEN_GROUPS}
    eligible_counts = {group.name: 0 for group in TOKEN_GROUPS}
    document_count = 0

    print(f"Scanning documents: {DOCUMENT_PATH}")

    with DOCUMENT_PATH.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            text = document.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            # Match the V2 validation convention by including special tokens.
            token_count = len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            group = find_token_group(token_count)
            if group is None:
                continue

            document_count += 1
            eligible_counts[group.name] += 1
            seen_in_group = eligible_counts[group.name]
            candidate = {
                "id": document.get("id"),
                "text": text,
                "tokens": token_count,
            }

            reservoir = samples[group.name]
            if len(reservoir) < samples_per_group:
                reservoir.append(candidate)
            else:
                replacement_index = rng.randrange(seen_in_group)
                if replacement_index < samples_per_group:
                    reservoir[replacement_index] = candidate

            if line_number % 10_000 == 0:
                print(f"Tokenized {line_number:,} JSONL records...")

    print(f"Eligible non-empty documents: {document_count:,}")
    for group in TOKEN_GROUPS:
        group_samples = samples[group.name]
        if len(group_samples) < max(BATCH_SIZES):
            raise RuntimeError(
                f"{group.name} has only {len(group_samples)} sampled documents; "
                f"at least {max(BATCH_SIZES)} are required for batch size 32."
            )

        lengths = np.asarray(
            [sample["tokens"] for sample in group_samples],
            dtype=np.int32,
        )
        print(
            f"{group.name:<6}: eligible={eligible_counts[group.name]:>7,}, "
            f"sampled={len(group_samples):>4,}, "
            f"tokens={lengths.min()}..{lengths.max()}, "
            f"mean={lengths.mean():.1f}"
        )

    return samples


# ============================================================
# 4. One dense-embedding inference pass
# ============================================================

def encode_dense(model, texts, batch_size: int):
    """Run dense-only BGE-M3 inference and return a NumPy array."""

    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = output["dense_vecs"]

    # FlagEmbedding normally returns NumPy here; this keeps reporting robust.
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)

    return embeddings


def is_cuda_oom(torch, error: RuntimeError) -> bool:
    """Recognize CUDA OOM across supported PyTorch versions."""

    out_of_memory_error = getattr(torch.cuda, "OutOfMemoryError", None)
    if out_of_memory_error is not None and isinstance(error, out_of_memory_error):
        return True
    return "out of memory" in str(error).lower()


def recover_from_oom(torch) -> None:
    """Release transient references cached by CUDA after a failed run."""

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ============================================================
# 5. Warm-up and three-run benchmark
# ============================================================

def benchmark_case(torch, model, group: TokenGroup, documents, batch_size: int):
    """Benchmark one token group and batch size, safely handling CUDA OOM."""

    texts = [document["text"] for document in documents]
    warmup_texts = texts[:batch_size]

    print(
        f"\n--- Group={group.name} | batch_size={batch_size} | "
        f"documents={len(texts)} ---"
    )

    try:
        # Warm-up initializes CUDA kernels and allocator state outside timing.
        _ = encode_dense(model, warmup_texts, batch_size)
        torch.cuda.synchronize()
        del _
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if not is_cuda_oom(torch, exc):
            raise
        print(f"WARM-UP OOM: {exc}")
        recover_from_oom(torch)
        return []

    results = []

    for run_number in range(1, MEASURED_RUNS + 1):
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()  # Ensure no earlier GPU work enters timing.
            started_at = time.perf_counter()

            embeddings = encode_dense(model, texts, batch_size)

            torch.cuda.synchronize()  # Wait for all asynchronous kernels.
            elapsed = time.perf_counter() - started_at
            peak_vram_gib = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
            docs_per_second = len(texts) / elapsed

            norms = np.linalg.norm(
                embeddings.astype(np.float32, copy=False),
                axis=1,
            )
            result = {
                "elapsed": elapsed,
                "docs_per_second": docs_per_second,
                "peak_vram_gib": peak_vram_gib,
                "shape": embeddings.shape,
                "dtype": embeddings.dtype,
                "norm_min": float(norms.min()),
                "norm_mean": float(norms.mean()),
                "norm_max": float(norms.max()),
            }
            results.append(result)

            print(
                f"Run {run_number}: elapsed={elapsed:.3f}s | "
                f"docs/s={docs_per_second:.2f} | "
                f"peak VRAM={peak_vram_gib:.3f} GiB | "
                f"embedding shape={embeddings.shape} | "
                f"dtype={embeddings.dtype} | "
                f"norm min/mean/max="
                f"{norms.min():.6f}/{norms.mean():.6f}/{norms.max():.6f}"
            )

            # Explicit deletion prevents one run's output from affecting the next.
            del embeddings, norms
        except RuntimeError as exc:
            if not is_cuda_oom(torch, exc):
                raise
            print(f"Run {run_number}: OOM | {exc}")
            recover_from_oom(torch)
            break

    if results:
        print(
            "Summary: "
            f"elapsed median={np.median([r['elapsed'] for r in results]):.3f}s | "
            f"docs/s median="
            f"{np.median([r['docs_per_second'] for r in results]):.2f} | "
            f"peak VRAM max="
            f"{max(r['peak_vram_gib'] for r in results):.3f} GiB"
        )

    return results


# ============================================================
# 6. Program entry
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark local BGE-M3 FP16 batch sizes on real V2 documents."
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=DEFAULT_SAMPLES_PER_GROUP,
        help=(
            "Number of real documents sampled per token group "
            f"(default: {DEFAULT_SAMPLES_PER_GROUP})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Reservoir-sampling seed (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_group < max(BATCH_SIZES):
        raise ValueError(
            f"--samples-per-group must be at least {max(BATCH_SIZES)}."
        )

    torch, BGEM3FlagModel, AutoTokenizer = load_runtime_dependencies()
    validate_inputs(torch)

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    print(f"CUDA device: cuda:{device_index} ({device_name})")
    print(f"Local model: {MODEL_PATH}")
    print("Precision: FP16")

    # Both tokenizer and model are loaded only from the checked local directory.
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )
    samples = sample_documents(
        tokenizer,
        samples_per_group=args.samples_per_group,
        seed=args.seed,
    )

    model = BGEM3FlagModel(
        str(MODEL_PATH),
        use_fp16=True,
        devices=[f"cuda:{device_index}"],  # Explicitly constrain to one GPU.
    )

    for group in TOKEN_GROUPS:
        for batch_size in BATCH_SIZES:
            benchmark_case(
                torch=torch,
                model=model,
                group=group,
                documents=samples[group.name],
                batch_size=batch_size,
            )

    print("\nBenchmark completed. No embeddings or indexes were saved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
