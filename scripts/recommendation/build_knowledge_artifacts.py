"""Build the V2-09.1 canonical read-only product knowledge artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentrec.knowledge import build_knowledge_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--item-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunk-chars", type=int, default=1600)
    args = parser.parse_args()
    manifest = build_knowledge_artifacts(
        catalog_path=args.catalog, item_mapping_path=args.item_mapping,
        output_dir=args.output_dir, max_chunk_chars=args.max_chunk_chars,
    )
    print(f"Products: {manifest['product_count']:,}")
    print(f"Chunks: {manifest['chunk_count']:,}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
