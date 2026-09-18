"""Build V2-09.2a BGE-M3 chunk embedding artifacts.

Input:  artifacts/recommendation/knowledge/chunks.jsonl
Output: artifacts/recommendation/retrieval/{chunk_embeddings.npy,
        chunk_metadata.json,retrieval_manifest.json}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentrec.retrieval import (
    BGEM3ChunkEncoder,
    ChunkEmbeddingConfig,
    build_chunk_embedding_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path,
        default=PROJECT_ROOT / "artifacts/recommendation/knowledge/chunks.jsonl",
    )
    parser.add_argument(
        "--model-path", type=Path, default=PROJECT_ROOT / "models/bge-m3",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-rows", type=int, default=4096)
    parser.add_argument(
        "--resume-from", type=Path,
        help="Explicit unpublished temporary directory containing export_progress.json.",
    )
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Permit a bounded row-range export for native-crash localization.",
    )
    args = parser.parse_args()

    formal_output = PROJECT_ROOT / "artifacts/recommendation/retrieval"
    output_dir = args.output_dir or formal_output
    partial_requested = args.start_row != 0 or args.max_rows is not None
    if args.resume_from is not None and partial_requested:
        parser.error("--resume-from cannot be combined with --start-row/--max-rows.")
    if args.resume_from is not None and args.diagnostic:
        parser.error("Formal resume cannot be combined with --diagnostic.")
    if partial_requested and not args.diagnostic:
        parser.error("--start-row/--max-rows require --diagnostic.")
    if args.diagnostic and args.output_dir is None:
        parser.error("--diagnostic requires an explicit non-formal --output-dir.")
    if args.diagnostic and output_dir.resolve() == formal_output.resolve():
        parser.error("Diagnostic exports cannot target the formal retrieval directory.")

    # These values intentionally match the established AgentRec BGE-M3 pipeline.
    config = ChunkEmbeddingConfig()
    encoder = BGEM3ChunkEncoder(args.model_path, config)
    manifest = build_chunk_embedding_artifacts(
        chunks_path=args.chunks, output_dir=output_dir,
        encoder=encoder, config=config, start_row=args.start_row,
        max_rows=args.max_rows, progress_rows=args.progress_rows,
        diagnostic=args.diagnostic, resume_from=args.resume_from,
        model_path=args.model_path,
    )
    embedding = manifest["embedding"]
    print("V2-09.2a CHUNK EMBEDDING EXPORT: PASS")
    print(f"Chunks: {embedding['shape'][0]:,}")
    print(f"Embedding shape: {tuple(embedding['shape'])}")
    print(f"Output: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
