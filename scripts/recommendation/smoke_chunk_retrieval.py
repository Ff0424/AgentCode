"""Run real V2-09.2b chunk retrieval smoke tests and latency measurements."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentrec.retrieval import (
    BGEM3QueryEncoder,
    ChunkRetrievalArtifacts,
    ChunkRetriever,
    NumPyExactBackend,
    ValidationMode,
)


DEFAULT_QUERIES = (
    "Does this docking station support HDMI and dual monitors?",
    "USB-C hub with ethernet and 4K HDMI output",
    "Noise cancelling headphones with long battery life",
    "Dash camera with GPS and good night recording",
    "Portable Bluetooth speaker suitable for outdoor use",
)


class TimedEncoder:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.last_seconds = 0.0

    @property
    def embedding_dimension(self) -> int:
        return self.wrapped.embedding_dimension

    def encode(self, queries):
        started = time.perf_counter()
        value = self.wrapped.encode(queries)
        self.last_seconds = time.perf_counter() - started
        return value


class TimedBackend:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.last_seconds = 0.0

    @property
    def row_count(self) -> int:
        return self.wrapped.row_count

    @property
    def dimension(self) -> int:
        return self.wrapped.dimension

    def search(self, query_embedding, *, top_k, candidate_rows=None):
        started = time.perf_counter()
        value = self.wrapped.search(
            query_embedding, top_k=top_k, candidate_rows=candidate_rows
        )
        self.last_seconds = time.perf_counter() - started
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-mode", required=True,
        choices=[mode.value for mode in ValidationMode],
        help="Use strict for first deployment acceptance; trusted_published for routine startup.",
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--retrieval-dir", type=Path,
        default=PROJECT_ROOT / "artifacts/recommendation/retrieval",
    )
    parser.add_argument(
        "--knowledge-dir", type=Path,
        default=PROJECT_ROOT / "artifacts/recommendation/knowledge",
    )
    parser.add_argument(
        "--model-path", type=Path, default=PROJECT_ROOT / "models/bge-m3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be non-negative and --repeat must be positive.")
    load_started = time.perf_counter()
    artifacts = ChunkRetrievalArtifacts(
        retrieval_dir=args.retrieval_dir, knowledge_dir=args.knowledge_dir,
        validation_mode=args.validation_mode,
    )
    artifact_load_seconds = time.perf_counter() - load_started

    model_started = time.perf_counter()
    encoder = TimedEncoder(BGEM3QueryEncoder(args.model_path))
    model_load_seconds = time.perf_counter() - model_started
    backend = TimedBackend(NumPyExactBackend(artifacts.embeddings))
    retriever = ChunkRetriever(encoder=encoder, backend=backend, artifacts=artifacts)

    print(f"Validation mode: {args.validation_mode}")
    print(f"Artifact load latency: {artifact_load_seconds:.6f} s")
    print(f"Model load latency: {model_load_seconds:.6f} s")
    for query in args.queries or DEFAULT_QUERIES:
        for _ in range(args.warmup):
            retriever.search(query, top_k=args.top_k)
        encode_times: list[float] = []
        retrieval_times: list[float] = []
        projection_times: list[float] = []
        total_times: list[float] = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            result = retriever.search(query, top_k=args.top_k)
            total_seconds = time.perf_counter() - started
            encode_times.append(encoder.last_seconds)
            retrieval_times.append(backend.last_seconds)
            projection_times.append(max(
                0.0, total_seconds - encoder.last_seconds - backend.last_seconds
            ))
            total_times.append(total_seconds)
        print(f"\nQuery: {query}")
        print(f"Warm-up runs: {args.warmup}; measured runs: {args.repeat}")
        print(f"Query encode median latency: {statistics.median(encode_times):.6f} s")
        print(f"Retrieval median latency: {statistics.median(retrieval_times):.6f} s")
        print(f"Projection median latency: {statistics.median(projection_times):.6f} s")
        print(
            f"Total latency median/min/max: {statistics.median(total_times):.6f} / "
            f"{min(total_times):.6f} / {max(total_times):.6f} s"
        )
        for chunk in result.chunks:
            preview = " ".join(chunk.text.split())[:240]
            print(
                f"{chunk.rank}. score={chunk.similarity_score:.8f} "
                f"row={chunk.embedding_row} chunk_id={chunk.chunk_id} "
                f"parent_asin={chunk.parent_asin} item_index={chunk.item_index} "
                f"type={chunk.chunk_type.value} text={preview}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
