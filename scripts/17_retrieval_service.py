"""
Reusable runtime product retrieval backed by local BGE-M3 and FAISS.

The component loads only the formal FAISS index, its row-aligned product IDs,
the index metadata, and the local BGE-M3 model. It never rebuilds product
embeddings or the index, reads Product Documents, evaluates retrieval, or
writes artifacts.

Example:
    python scripts/17_retrieval_service.py \
        --query "USB-C hub with HDMI and ethernet" \
        --top-k 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ============================================================
# 1. Fixed runtime and artifact configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "processed" / "rag"

INDEX_PATH = RAG_DIR / "product_embeddings.faiss"
PRODUCT_IDS_PATH = RAG_DIR / "product_ids.json"
INDEX_META_PATH = RAG_DIR / "faiss_index.meta.json"
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"

EXPECTED_ROWS = 125_762
EXPECTED_DIMENSION = 1024
EXPECTED_INDEX_TYPE = "faiss.IndexFlatIP"
EXPECTED_METADATA_METRIC = "inner_product"

DEVICE = "cuda:0"
USE_FP16 = True
BATCH_SIZE = 8
MAX_LENGTH = 2048
SEARCH_CHUNK_SIZE = 512


# ============================================================
# 2. Product retrieval runtime component
# ============================================================

class ProductRetriever:
    """Encode user queries and map FAISS rows to formal product IDs."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        index_path: Path = INDEX_PATH,
        product_ids_path: Path = PRODUCT_IDS_PATH,
        index_meta_path: Path = INDEX_META_PATH,
    ):
        self.model_path = Path(model_path)
        self.index_path = Path(index_path)
        self.product_ids_path = Path(product_ids_path)
        self.index_meta_path = Path(index_meta_path)

        self._faiss, self._torch, model_class = self._load_dependencies()
        self._validate_cuda_and_local_model()
        self.product_ids, self.index_metadata, self.index = (
            self._load_and_validate_artifacts()
        )

        # The local path and offline flags prevent Hugging Face network access.
        self.model = model_class(
            str(self.model_path),
            use_fp16=USE_FP16,
            devices=[DEVICE],
        )

    @staticmethod
    def _load_dependencies():
        """Import heavyweight runtime packages only during initialization."""

        # These flags affect only this process and force transformers/hub calls
        # to use local files instead of reaching the Hugging Face network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            import faiss
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing runtime dependencies. FAISS, CUDA-compatible PyTorch, "
                "and FlagEmbedding are required."
            ) from exc
        return faiss, torch, BGEM3FlagModel

    def _validate_cuda_and_local_model(self) -> None:
        """Require cuda:0 and the same checked local model files as script 13."""

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local BGE-M3 model not found: {self.model_path}")

        required_model_files = (
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "tokenizer.json",
            "sentencepiece.bpe.model",
        )
        missing = [
            filename
            for filename in required_model_files
            if not (self.model_path / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Local BGE-M3 model is incomplete; missing: " + ", ".join(missing)
            )

        if not self._torch.cuda.is_available() or self._torch.cuda.device_count() < 1:
            raise RuntimeError("cuda:0 is unavailable; ProductRetriever requires CUDA.")
        self._torch.cuda.set_device(0)

    @staticmethod
    def _read_json(path: Path):
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    def _normalized_metadata_path(self, value) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def _load_and_validate_artifacts(self):
        """Load formal artifacts read-only and enforce their runtime contract."""

        missing = [
            path
            for path in (
                self.index_path,
                self.product_ids_path,
                self.index_meta_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Required retrieval artifact is missing: "
                + ", ".join(str(path) for path in missing)
            )

        try:
            product_ids = self._read_json(self.product_ids_path)
            metadata = self._read_json(self.index_meta_path)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not load retrieval artifact JSON: {exc}") from exc

        if not isinstance(product_ids, list):
            raise ValueError("product_ids.json must contain a top-level list.")
        if len(product_ids) != EXPECTED_ROWS:
            raise ValueError(
                f"Product ID count is {len(product_ids):,}; "
                f"expected {EXPECTED_ROWS:,}."
            )
        if any(
            not isinstance(product_id, str) or not product_id.strip()
            for product_id in product_ids
        ):
            raise ValueError("product_ids.json contains an empty or invalid ID.")
        if not isinstance(metadata, dict):
            raise ValueError("faiss_index.meta.json must contain an object.")

        metadata_checks = {
            "index_type": metadata.get("index_type") == EXPECTED_INDEX_TYPE,
            "metric": metadata.get("metric") == EXPECTED_METADATA_METRIC,
            "dimension": metadata.get("dimension") == EXPECTED_DIMENSION,
            "ntotal": metadata.get("ntotal") == EXPECTED_ROWS,
            "additional_l2_normalization_applied": (
                metadata.get("additional_l2_normalization_applied") is False
            ),
            "source_product_ids_file": (
                self._normalized_metadata_path(
                    metadata.get("source_product_ids_file")
                )
                == self.product_ids_path.resolve()
            ),
        }
        failed_checks = [
            name for name, passed in metadata_checks.items() if not passed
        ]
        if failed_checks:
            raise ValueError(
                "FAISS metadata validation failed: " + ", ".join(failed_checks)
            )

        try:
            index = self._faiss.read_index(str(self.index_path))
        except RuntimeError as exc:
            raise ValueError(f"Could not load the FAISS index: {exc}") from exc

        if type(index).__name__ != "IndexFlatIP":
            raise ValueError(
                f"Loaded FAISS index type is {type(index).__name__}; "
                "expected IndexFlatIP."
            )
        if index.ntotal != EXPECTED_ROWS:
            raise ValueError(
                f"FAISS ntotal is {index.ntotal:,}; expected {EXPECTED_ROWS:,}."
            )
        if index.d != EXPECTED_DIMENSION:
            raise ValueError(
                f"FAISS dimension is {index.d}; expected {EXPECTED_DIMENSION}."
            )
        if index.metric_type != self._faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS metric is not inner product.")

        return product_ids, metadata, index

    @staticmethod
    def _validate_query(query: str, position: int | None = None) -> None:
        """Reject non-string and whitespace-only queries with useful context."""

        label = "query" if position is None else f"queries[{position}]"
        if not isinstance(query, str):
            raise TypeError(f"{label} must be a string, found {type(query).__name__}.")
        if not query.strip():
            raise ValueError(f"{label} must be a non-empty string.")

    def _validate_top_k(self, top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive integer.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if top_k > self.index.ntotal:
            raise ValueError(
                f"top_k={top_k:,} exceeds index.ntotal={self.index.ntotal:,}."
            )

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries exactly like the existing dense embedding pipeline."""

        output = self.model.encode(
            queries,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = output.get("dense_vecs")
        if dense_vectors is None:
            raise ValueError("BGE-M3 output does not contain dense_vecs.")
        if self._torch.is_tensor(dense_vectors):
            dense_vectors = dense_vectors.detach().cpu().numpy()

        # FAISS consumes contiguous float32 queries; vector norms are untouched.
        embeddings = np.ascontiguousarray(dense_vectors, dtype=np.float32)
        expected_shape = (len(queries), EXPECTED_DIMENSION)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"BGE-M3 returned shape {embeddings.shape}; "
                f"expected {expected_shape}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("BGE-M3 returned NaN or infinite query embeddings.")
        return embeddings

    def _format_results(self, rows, scores) -> list[dict]:
        results = []
        for rank, (row, score) in enumerate(zip(rows, scores), start=1):
            row = int(row)
            if row < 0 or row >= len(self.product_ids):
                raise RuntimeError(f"FAISS returned invalid row index {row}.")
            results.append(
                {
                    "rank": rank,
                    "product_id": self.product_ids[row],
                    "score": float(score),
                    "row": row,
                }
            )
        return results

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve Top-K product rows for one non-empty query."""

        self._validate_query(query)
        self._validate_top_k(top_k)
        query_embedding = self._encode_queries([query])
        scores, rows = self.index.search(query_embedding, top_k)
        return self._format_results(rows[0], scores[0])

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[dict]]:
        """Retrieve multiple queries while preserving their exact input order."""

        if not isinstance(queries, list):
            raise TypeError("queries must be a list of strings.")
        if not queries:
            return []

        for position, query in enumerate(queries):
            self._validate_query(query, position)
        self._validate_top_k(top_k)

        embeddings = self._encode_queries(queries)
        all_results = []
        for start in range(0, len(queries), SEARCH_CHUNK_SIZE):
            end = min(start + SEARCH_CHUNK_SIZE, len(queries))
            scores, rows = self.index.search(embeddings[start:end], top_k)
            for result_rows, result_scores in zip(rows, scores):
                all_results.append(
                    self._format_results(result_rows, result_scores)
                )

        if len(all_results) != len(queries):
            raise RuntimeError("Internal batch result count does not match queries.")
        return all_results


# ============================================================
# 3. Minimal read-only CLI for one manual query
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Search the formal local BGE-M3 + FAISS product index."
    )
    parser.add_argument("--query", required=True, help="Non-empty product query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retriever = ProductRetriever()
    results = retriever.search(args.query, args.top_k)

    print(f"Query: {args.query}\n")
    for result in results:
        print(f"{result['rank']}. product_id={result['product_id']}")
        print(f"   row={result['row']}")
        print(f"   score={result['score']:.8f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
