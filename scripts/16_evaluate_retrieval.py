"""
Evaluate BGE-M3 dense product retrieval against the formal FAISS index.

The quantitative evaluation samples products with a fixed seed and creates a
title query plus a short deterministic feature-style query for each product.
The matching product ID is the ground truth. A small fixed set of shopping
intent queries is reported separately for human inspection and is never mixed
into the quantitative metrics.

Inputs:
    data/processed/rag/product_documents_v2.jsonl
    data/processed/rag/product_embeddings.faiss
    data/processed/rag/product_ids.json
    data/processed/rag/faiss_index.meta.json
    models/bge-m3

Outputs:
    data/processed/rag/retrieval_evaluation.json
    data/processed/rag/retrieval_evaluation_cases.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# 1. Fixed evaluation and model configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "processed" / "rag"

DOCUMENT_PATH = RAG_DIR / "product_documents_v2.jsonl"
INDEX_PATH = RAG_DIR / "product_embeddings.faiss"
PRODUCT_IDS_PATH = RAG_DIR / "product_ids.json"
INDEX_META_PATH = RAG_DIR / "faiss_index.meta.json"
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"

DEFAULT_OUTPUT_PREFIX = "retrieval_evaluation"

EXPECTED_ROWS = 125_762
EXPECTED_DIMENSION = 1024
EXPECTED_DTYPE = "float32"
RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 2000
DEVICE = "cuda:0"
USE_FP16 = True
MAX_LENGTH = 2048
BATCH_SIZE = 8
ENCODE_CHUNK_SIZE = 512
SEARCH_CHUNK_SIZE = 512
TOP_K = 20
RECALL_CUTOFFS = (1, 5, 10, 20)

QUALITATIVE_QUERIES = (
    "wireless computer speakers with bluetooth and gaming mode",
    "dash camera with GPS and good night recording",
    "noise cancelling headphones with long battery life",
    "USB-C hub for a laptop with HDMI and ethernet",
    "portable bluetooth speaker for outdoor use",
)


# ============================================================
# 2. Runtime, JSON, time, and path helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def normalized_path(value) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_runtime_dependencies():
    """Delay heavyweight imports until a real evaluation is requested."""

    try:
        import faiss
        import torch
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing evaluation dependencies. FAISS, CUDA-compatible PyTorch, "
            "and FlagEmbedding are required."
        ) from exc
    return faiss, torch, BGEM3FlagModel


def validate_runtime(torch) -> None:
    required_model_files = (
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
    )
    missing_model_files = [
        filename
        for filename in required_model_files
        if not (MODEL_PATH / filename).is_file()
    ]
    if missing_model_files:
        raise FileNotFoundError(
            "Local BGE-M3 model is incomplete; missing: "
            + ", ".join(missing_model_files)
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("cuda:0 is unavailable; evaluation requires one CUDA GPU.")
    torch.cuda.set_device(0)


# ============================================================
# 3. Formal artifact validation
# ============================================================

def build_output_paths(output_prefix: str) -> dict:
    """Derive an isolated report pair from one validated filename prefix."""

    return {
        "report": RAG_DIR / f"{output_prefix}.json",
        "cases": RAG_DIR / f"{output_prefix}_cases.jsonl",
    }


def refuse_existing_outputs(output_paths: dict) -> None:
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Evaluation output already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def validate_required_inputs() -> None:
    required = (DOCUMENT_PATH, INDEX_PATH, PRODUCT_IDS_PATH, INDEX_META_PATH)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required evaluation input is missing: "
            + ", ".join(str(path) for path in missing)
        )


def validate_index_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("FAISS metadata must contain a top-level object.")

    fingerprint = metadata.get("source_embedding_fingerprint")
    checks = {
        "index_type": metadata.get("index_type") == "faiss.IndexFlatIP",
        "metric": metadata.get("metric") == "inner_product",
        "dimension": metadata.get("dimension") == EXPECTED_DIMENSION,
        "ntotal": metadata.get("ntotal") == EXPECTED_ROWS,
        "embedding_dtype": metadata.get("embedding_dtype") == EXPECTED_DTYPE,
        "source_product_ids_file": (
            normalized_path(metadata.get("source_product_ids_file"))
            == PRODUCT_IDS_PATH.resolve()
        ),
        "additional_l2_normalization_applied": (
            metadata.get("additional_l2_normalization_applied") is False
        ),
        "source_embedding_fingerprint": (
            isinstance(fingerprint, dict)
            and fingerprint.get("algorithm") == "sha256"
            and fingerprint.get("row_count") == EXPECTED_ROWS
            and isinstance(fingerprint.get("sha256"), str)
            and len(fingerprint["sha256"]) == 64
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "FAISS metadata is inconsistent with the formal index: "
            + ", ".join(failures)
        )


def load_and_validate_artifacts(faiss):
    validate_required_inputs()
    try:
        product_ids = read_json(PRODUCT_IDS_PATH)
        index_metadata = read_json(INDEX_META_PATH)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not load artifact JSON: {exc}") from exc

    if not isinstance(product_ids, list):
        raise ValueError("product_ids.json must contain a top-level list.")
    if len(product_ids) != EXPECTED_ROWS:
        raise ValueError(
            f"Product ID count is {len(product_ids):,}; expected {EXPECTED_ROWS:,}."
        )
    if any(not isinstance(product_id, str) or not product_id.strip() for product_id in product_ids):
        raise ValueError("product_ids.json contains an empty or invalid product ID.")

    validate_index_metadata(index_metadata)
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except RuntimeError as exc:
        raise ValueError(f"Could not load the FAISS index: {exc}") from exc

    if index.ntotal != EXPECTED_ROWS:
        raise ValueError(
            f"FAISS ntotal is {index.ntotal:,}; expected {EXPECTED_ROWS:,}."
        )
    if index.d != EXPECTED_DIMENSION:
        raise ValueError(
            f"FAISS dimension is {index.d}; expected {EXPECTED_DIMENSION}."
        )
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            "FAISS metric is not inner product as required by IndexFlatIP."
        )
    return product_ids, index_metadata, index


# ============================================================
# 4. Document parsing and deterministic sampling
# ============================================================

def extract_section(text: str, label: str) -> str | None:
    """Extract one single-line section emitted by script 11."""

    prefix = f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value or None
    return None


def remove_identifier(text: str, identifier: str) -> str:
    if not identifier:
        return text
    return re.sub(re.escape(identifier), " ", text, flags=re.IGNORECASE)


def build_feature_query(features: str, product_id: str) -> str | None:
    """Create a concise feature intent without copying the full feature block."""

    cleaned = remove_identifier(features, product_id)
    parts = []
    for raw_part in cleaned.split("|"):
        part = " ".join(raw_part.split()).strip(" ,;:-")
        if part and part.casefold() not in {item.casefold() for item in parts}:
            parts.append(part)
        if len(parts) == 3:
            break

    if not parts:
        return None

    # Bound both semantic breadth and length; this is intentionally not a copy
    # of the complete Features section.
    feature_words = " ; ".join(parts).split()
    shortened = " ".join(feature_words[:36]).strip()
    if not shortened:
        return None
    return f"Looking for a product with these features: {shortened}"


def scan_documents_and_sample(product_ids: list, sample_size: int) -> tuple:
    """Validate all rows while reservoir-sampling eligible products uniformly."""

    rng = random.Random(RANDOM_SEED)
    reservoir = []
    titles_by_row = []
    document_count = 0
    eligible_count = 0

    with DOCUMENT_PATH.open("r", encoding="utf-8") as input_file:
        for source_line, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid Product Document JSON at line {source_line}: {exc}"
                ) from exc

            if document_count >= len(product_ids):
                raise ValueError("Document count exceeds the Product ID artifact.")

            product_id = document.get("id")
            metadata = document.get("metadata")
            parent_asin = metadata.get("parent_asin") if isinstance(metadata, dict) else None
            if not isinstance(product_id, str) or not product_id.strip():
                raise ValueError(f"Invalid document id at source line {source_line}.")
            if product_id != parent_asin:
                raise ValueError(
                    f"document id != metadata.parent_asin at row {document_count}."
                )
            if product_id != product_ids[document_count]:
                raise ValueError(
                    f"document id != product_ids[row] at row {document_count}."
                )

            text = document.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Document text is invalid at row {document_count}.")
            title = extract_section(text, "Title")
            features = extract_section(text, "Features")
            titles_by_row.append(title)

            if title and features:
                feature_query = build_feature_query(features, product_id)
                if feature_query:
                    candidate = {
                        "row": document_count,
                        "product_id": product_id,
                        "title_query": remove_identifier(title, product_id).strip(),
                        "feature_query": feature_query,
                    }
                    if candidate["title_query"]:
                        eligible_count += 1
                        if len(reservoir) < sample_size:
                            reservoir.append(candidate)
                        else:
                            replacement = rng.randrange(eligible_count)
                            if replacement < sample_size:
                                reservoir[replacement] = candidate

            document_count += 1

    if document_count != EXPECTED_ROWS:
        raise ValueError(
            f"Document count is {document_count:,}; expected {EXPECTED_ROWS:,}."
        )
    if len(titles_by_row) != EXPECTED_ROWS:
        raise ValueError("Internal title mapping is not row-aligned.")
    if eligible_count < sample_size:
        raise ValueError(
            f"Only {eligible_count:,} products have usable Title and Features; "
            f"cannot sample {sample_size:,}."
        )

    # Row sorting makes case output stable and easy to audit while preserving
    # the uniformly selected reservoir membership.
    reservoir.sort(key=lambda item: item["row"])
    return reservoir, titles_by_row, document_count, eligible_count


# ============================================================
# 5. Query encoding and FAISS search
# ============================================================

def encode_queries(model, queries: list, torch) -> np.ndarray:
    blocks = []
    for start in range(0, len(queries), ENCODE_CHUNK_SIZE):
        end = min(start + ENCODE_CHUNK_SIZE, len(queries))
        output = model.encode(
            queries[start:end],
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        if torch.is_tensor(dense_vectors):
            dense_vectors = dense_vectors.detach().cpu().numpy()
        block = np.asarray(dense_vectors, dtype=np.float32)
        if block.shape != (end - start, EXPECTED_DIMENSION):
            raise ValueError(
                f"BGE-M3 returned shape {block.shape}; expected "
                f"{(end - start, EXPECTED_DIMENSION)}."
            )
        if not np.isfinite(block).all():
            raise ValueError("BGE-M3 returned NaN or infinite query embeddings.")
        blocks.append(block)
        print(f"Encoded queries [0, {end:,}) / {len(queries):,}")

    # Conversion is storage-only; no additional L2 normalization is performed.
    return np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)


def search_queries(index, query_embeddings: np.ndarray) -> tuple:
    score_blocks = []
    row_blocks = []
    for start in range(0, len(query_embeddings), SEARCH_CHUNK_SIZE):
        end = min(start + SEARCH_CHUNK_SIZE, len(query_embeddings))
        scores, rows = index.search(query_embeddings[start:end], TOP_K)
        score_blocks.append(scores)
        row_blocks.append(rows)
    return np.vstack(score_blocks), np.vstack(row_blocks)


# ============================================================
# 6. Quantitative metrics and qualitative results
# ============================================================

def target_rank(target_product_id: str, result_rows, product_ids: list):
    for rank, row in enumerate(result_rows, start=1):
        if row >= 0 and product_ids[int(row)] == target_product_id:
            return rank
    return None


def build_cases(samples: list, scores, rows, product_ids: list) -> list:
    cases = []
    case_index = 0
    for sample in samples:
        for query_type in ("title", "feature"):
            result_rows = rows[case_index]
            result_ids = [product_ids[int(row)] for row in result_rows if row >= 0]
            result_scores = [
                float(score)
                for score, row in zip(scores[case_index], result_rows)
                if row >= 0
            ]
            cases.append(
                {
                    "query_type": query_type,
                    "query": sample[f"{query_type}_query"],
                    "target_product_id": sample["product_id"],
                    "target_rank": target_rank(
                        sample["product_id"], result_rows, product_ids
                    ),
                    "top20_product_ids": result_ids,
                    "top20_scores": result_scores,
                }
            )
            case_index += 1
    return cases


def calculate_metrics(cases: list, query_type: str) -> dict:
    selected = [case for case in cases if case["query_type"] == query_type]
    ranks = [case["target_rank"] for case in selected]
    metrics = {
        f"recall@{cutoff}": (
            sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)
        )
        for cutoff in RECALL_CUTOFFS
    }
    metrics["mrr@20"] = sum(
        0.0 if rank is None else 1.0 / rank for rank in ranks
    ) / len(ranks)
    metrics["query_count"] = len(selected)
    metrics["targets_not_in_top20"] = sum(rank is None for rank in ranks)
    return metrics


def build_qualitative_results(
    scores,
    rows,
    product_ids: list,
    titles_by_row: list,
) -> list:
    results = []
    for query_index, query in enumerate(QUALITATIVE_QUERIES):
        top5 = []
        for score, row in zip(scores[query_index, :5], rows[query_index, :5]):
            if row < 0:
                continue
            row = int(row)
            top5.append(
                {
                    "rank": len(top5) + 1,
                    "product_id": product_ids[row],
                    "title": titles_by_row[row],
                    "score": float(score),
                }
            )
        results.append({"query": query, "top5": top5})
    return results


# ============================================================
# 7. Safe output publication
# ============================================================

def write_json_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def write_cases_new(path: Path, cases: list) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        for case in cases:
            output_file.write(json.dumps(case, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def publish_outputs(report: dict, cases: list, output_paths: dict) -> None:
    report_path = output_paths["report"]
    cases_path = output_paths["cases"]
    report_temp = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    cases_temp = cases_path.with_name(f".{cases_path.name}.{os.getpid()}.tmp")
    if report_temp.exists() or cases_temp.exists():
        raise FileExistsError("Process-specific temporary evaluation output exists.")

    try:
        write_json_new(report_temp, report)
        write_cases_new(cases_temp, cases)
        refuse_existing_outputs(output_paths)
        os.replace(cases_temp, cases_path)
        os.replace(report_temp, report_path)
    finally:
        for temporary_path in (report_temp, cases_temp):
            if temporary_path.exists():
                temporary_path.unlink()


# ============================================================
# 8. Evaluation orchestration and CLI
# ============================================================

def evaluate(sample_size: int, output_prefix: str) -> None:
    output_paths = build_output_paths(output_prefix)
    refuse_existing_outputs(output_paths)
    faiss, torch, BGEM3FlagModel = load_runtime_dependencies()
    validate_runtime(torch)
    product_ids, index_metadata, index = load_and_validate_artifacts(faiss)

    samples, titles_by_row, document_count, eligible_count = (
        scan_documents_and_sample(product_ids, sample_size)
    )
    quantitative_queries = []
    for sample in samples:
        quantitative_queries.extend(
            (sample["title_query"], sample["feature_query"])
        )
    all_queries = quantitative_queries + list(QUALITATIVE_QUERIES)

    print(f"Loading local BGE-M3 model: {MODEL_PATH}")
    model = BGEM3FlagModel(
        str(MODEL_PATH),
        use_fp16=USE_FP16,
        devices=[DEVICE],
    )
    query_embeddings = encode_queries(model, all_queries, torch)
    scores, rows = search_queries(index, query_embeddings)

    quantitative_count = len(quantitative_queries)
    cases = build_cases(
        samples,
        scores[:quantitative_count],
        rows[:quantitative_count],
        product_ids,
    )
    qualitative_results = build_qualitative_results(
        scores[quantitative_count:],
        rows[quantitative_count:],
        product_ids,
        titles_by_row,
    )

    report = {
        "version": 1,
        "completed_at": utc_now(),
        "evaluation_config": {
            "random_seed": RANDOM_SEED,
            "sample_size": sample_size,
            "eligible_product_count": eligible_count,
            "query_types": ["title", "feature"],
            "search_top_k": TOP_K,
            "model_path": str(MODEL_PATH),
            "device": DEVICE,
            "use_fp16": USE_FP16,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "embedding_dimension": EXPECTED_DIMENSION,
            "additional_l2_normalization_applied": False,
        },
        "validated_artifacts": {
            "document_count": document_count,
            "product_id_count": len(product_ids),
            "faiss_ntotal": int(index.ntotal),
            "faiss_dimension": int(index.d),
            "faiss_index_metadata": str(INDEX_META_PATH),
            "source_embedding_fingerprint": index_metadata.get(
                "source_embedding_fingerprint"
            ),
        },
        "quantitative_metrics": {
            "title": calculate_metrics(cases, "title"),
            "feature": calculate_metrics(cases, "feature"),
        },
        "qualitative_queries": qualitative_results,
        "case_output": str(output_paths["cases"]),
    }
    publish_outputs(report, cases, output_paths)
    print(f"Evaluation report written: {output_paths['report']}")
    print(f"Case-level results written: {output_paths['cases']}")
    print("FINAL STATUS: PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BGE-M3 + IndexFlatIP product retrieval."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of products sampled with seed 42 (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Filename prefix for the report pair; defaults to "
            f"'{DEFAULT_OUTPUT_PREFIX}'."
        ),
    )
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be a positive integer.")
    if args.sample_size > EXPECTED_ROWS:
        parser.error(f"--sample-size cannot exceed {EXPECTED_ROWS}.")
    if not args.output_prefix.strip():
        parser.error("--output-prefix cannot be empty or whitespace.")
    if Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be a filename prefix, not a path.")
    if args.output_prefix in {".", ".."}:
        parser.error("--output-prefix must not be '.' or '..'.")
    return args


def main() -> int:
    args = parse_args()
    evaluate(args.sample_size, args.output_prefix)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("FINAL STATUS: FAIL", file=sys.stderr)
        raise SystemExit(1) from exc
