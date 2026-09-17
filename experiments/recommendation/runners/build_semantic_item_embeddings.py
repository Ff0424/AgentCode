"""Build the canonical BGE-M3 item embedding cache for V2-06.4.

This GPU-side offline step reads the frozen Product Catalog, builds deterministic
pure-text inputs, and writes row-aligned float32 embeddings plus strict metadata.
It does not construct users, evaluate recommendation metrics, or modify inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.recommendation.datasets.canonical_items import load_canonical_item_universe
from experiments.recommendation.datasets.product_text import load_canonical_product_texts
from experiments.recommendation.datasets.semantic_artifacts import (
    expected_provenance,
    validate_embedding_artifacts,
)
from experiments.recommendation.datasets.semantic_config import (
    DEFAULT_CONFIG_PATH,
    load_semantic_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical BGE-M3 item embeddings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse a complete matching cache instead of rebuilding it.",
    )
    return parser.parse_args()


def _temporary_path(output: Path, suffix: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=suffix
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = _temporary_path(path, ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_gpu_dependencies():
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("PyTorch and FlagEmbedding are required on the GPU server.") from exc
    return torch, BGEM3FlagModel


def parse_cuda_device_index(device: str, default_index: int = 0) -> int:
    """Parse ``cuda``/``cuda:N`` without importing or initializing PyTorch.

    Bare ``cuda`` maps to ``default_index``. The runtime passes PyTorch's
    current CUDA device as that default; unit tests can use the documented
    fallback value of zero without requiring a CUDA installation.
    """

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty CUDA device string.")
    if isinstance(default_index, bool) or not isinstance(default_index, int):
        raise TypeError("default_index must be a non-negative integer.")
    if default_index < 0:
        raise ValueError("default_index must be a non-negative integer.")
    normalized = device.strip()
    if normalized == "cuda":
        return default_index
    match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", normalized)
    if match is None:
        raise ValueError(
            f"device must be 'cuda' or 'cuda:<non-negative index>', got {device!r}."
        )
    return int(match.group(1))


def _select_cuda_device(torch, device: str) -> int:
    """Validate and select one CUDA device before using CUDA utility APIs."""

    # Parse invalid device strings even on a host without CUDA for clearer errors.
    default_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    device_index = parse_cuda_device_index(device, default_index=default_index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this runtime.")
    device_count = int(torch.cuda.device_count())
    if device_index >= device_count:
        raise RuntimeError(
            f"CUDA device index {device_index} is out of range; "
            f"available device count={device_count}."
        )
    torch.cuda.set_device(device_index)
    return device_index


def run(config_path: Path, reuse_existing: bool) -> str:
    config = load_semantic_config(config_path.resolve())
    universe = load_canonical_item_universe(
        config.item_mapping_path, config.parent_asins_path
    )
    outputs_exist = (
        config.item_embedding_path.exists(),
        config.item_embedding_metadata_path.exists(),
    )
    if any(outputs_exist):
        if not all(outputs_exist):
            raise FileExistsError("Incomplete embedding cache exists; inspect it manually.")
        if not reuse_existing:
            raise FileExistsError(
                "Embedding cache already exists. Use --reuse-existing to validate it."
            )
        validate_embedding_artifacts(config, universe)
        return "reused"
    if not config.model_name_or_path.is_dir():
        raise FileNotFoundError(
            f"Local BGE-M3 model directory not found: {config.model_name_or_path}"
        )

    texts = load_canonical_product_texts(
        config.product_catalog_path, universe, config.text_fields
    )
    text_digest = hashlib.sha256()
    for item_index, text in zip(universe.item_indices, texts):
        text_digest.update(f"{int(item_index)}\t{text}\n".encode("utf-8"))

    torch, BGEM3FlagModel = _load_gpu_dependencies()
    device_index = _select_cuda_device(torch, config.device)
    print(
        f"Using CUDA device {device_index}: {torch.cuda.get_device_name(device_index)}",
        flush=True,
    )
    model = BGEM3FlagModel(
        str(config.model_name_or_path),
        use_fp16=config.use_fp16,
        devices=[config.device],
    )
    torch.cuda.reset_peak_memory_stats()
    temporary_embedding = _temporary_path(config.item_embedding_path, ".npy")
    memmap = np.lib.format.open_memmap(
        temporary_embedding,
        mode="w+",
        dtype=np.float32,
        shape=(universe.item_indices.size, config.embedding_dim),
    )
    norm_min, norm_max, norm_sum = float("inf"), 0.0, 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        for start in range(0, len(texts), config.embedding_batch_size):
            stop = min(start + config.embedding_batch_size, len(texts))
            output = model.encode(
                texts[start:stop],
                batch_size=config.embedding_batch_size,
                max_length=config.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = np.ascontiguousarray(output["dense_vecs"], dtype=np.float32)
            if vectors.shape != (stop - start, config.embedding_dim):
                raise ValueError(f"Unexpected BGE-M3 output shape at rows {start}:{stop}.")
            if not np.isfinite(vectors).all():
                raise ValueError(f"Non-finite BGE-M3 vectors at rows {start}:{stop}.")
            if config.normalize_embeddings:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError(f"Zero BGE-M3 vector at rows {start}:{stop}.")
                vectors /= norms
            norms = np.linalg.norm(vectors, axis=1)
            memmap[start:stop] = vectors
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
            norm_sum += float(norms.sum(dtype=np.float64))
            if start % (config.embedding_batch_size * 100) == 0:
                print(f"embedded={stop:,}/{len(texts):,}", flush=True)
        memmap.flush()
        del memmap
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metadata = {
            "status": "completed",
            "num_items": universe.item_indices.size,
            "embedding_dim": config.embedding_dim,
            "dtype": "float32",
            "model": str(config.model_name_or_path),
            "text_fields": list(config.text_fields),
            "empty_product_text_count": sum(not text for text in texts),
            "product_text_sha256": text_digest.hexdigest(),
            "normalize_embeddings": config.normalize_embeddings,
            "max_length": config.max_length,
            "embedding_batch_size": config.embedding_batch_size,
            "use_fp16": config.use_fp16,
            "row_mapping": "row i -> sorted canonical_item_indices[i]",
            "provenance": expected_provenance(config, universe),
            "norm_statistics": {
                "count": universe.item_indices.size,
                "min": norm_min,
                "mean": norm_sum / universe.item_indices.size,
                "max": norm_max,
            },
            "embedding_generation_seconds": elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Publish the large array first and metadata last; readers require both.
        os.replace(temporary_embedding, config.item_embedding_path)
        _write_json_atomic(config.item_embedding_metadata_path, metadata)
    except Exception:
        try:
            del memmap
        except UnboundLocalError:
            pass
        temporary_embedding.unlink(missing_ok=True)
        raise
    validate_embedding_artifacts(config, universe)
    return "built"


def main() -> int:
    args = parse_args()
    try:
        result = run(args.config, args.reuse_existing)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Semantic item embedding cache {result} successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
