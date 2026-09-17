"""Strict dependency-free configuration loader for V2-06.4."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.recommendation.datasets.product_text import SUPPORTED_TEXT_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiments/recommendation/configs/semantic.yaml"
REQUIRED_KEYS = {
    "model_name_or_path", "dataset_path", "product_catalog_path", "item_mapping_path",
    "parent_asins_path", "item_embedding_path", "item_embedding_metadata_path",
    "report_path", "popularity_report_path", "lightgcn_report_path", "text_fields",
    "normalize_embeddings", "embedding_batch_size", "eval_user_batch_size",
    "max_length", "embedding_dim", "topk", "device", "use_fp16",
}


@dataclass(frozen=True)
class SemanticConfig:
    model_name_or_path: Path
    dataset_path: Path
    product_catalog_path: Path
    item_mapping_path: Path
    parent_asins_path: Path
    item_embedding_path: Path
    item_embedding_metadata_path: Path
    report_path: Path
    popularity_report_path: Path
    lightgcn_report_path: Path
    text_fields: tuple[str, ...]
    normalize_embeddings: bool
    embedding_batch_size: int
    eval_user_batch_size: int
    max_length: int
    embedding_dim: int
    topk: tuple[int, ...]
    device: str
    use_fp16: bool


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def load_semantic_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SemanticConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Semantic config not found: {path}")
    raw: dict[str, Any] = {}
    for line_number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number}: {original!r}")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in raw:
            raise ValueError(f"Duplicate config key: {key!r}.")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            raw[key] = lowered == "true"
        elif value.startswith("[") and value.endswith("]"):
            raw[key] = [part.strip() for part in value[1:-1].split(",") if part.strip()]
            if all(part.isdecimal() for part in raw[key]):
                raw[key] = [int(part) for part in raw[key]]
        else:
            try:
                raw[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                raw[key] = value
    if raw.keys() != REQUIRED_KEYS:
        raise ValueError(
            f"Config keys mismatch; missing={sorted(REQUIRED_KEYS-raw.keys())}, "
            f"extra={sorted(raw.keys()-REQUIRED_KEYS)}."
        )
    for flag in ("normalize_embeddings", "use_fp16"):
        if not isinstance(raw[flag], bool):
            raise TypeError(f"{flag} must be true or false.")
    fields = raw["text_fields"]
    if not isinstance(fields, list) or not fields or len(set(fields)) != len(fields):
        raise ValueError("text_fields must be a non-empty duplicate-free list.")
    if any(field not in SUPPORTED_TEXT_FIELDS for field in fields):
        raise ValueError(f"Unsupported text_fields: {fields!r}.")
    topk = raw["topk"]
    if not isinstance(topk, list) or tuple(topk) != (10, 20, 50):
        raise ValueError("V2-06.4 requires topk [10, 20, 50].")
    if not isinstance(raw["device"], str) or not raw["device"].strip():
        raise ValueError("device must be a non-empty string.")
    return SemanticConfig(
        model_name_or_path=_path(raw["model_name_or_path"], "model_name_or_path"),
        dataset_path=_path(raw["dataset_path"], "dataset_path"),
        product_catalog_path=_path(raw["product_catalog_path"], "product_catalog_path"),
        item_mapping_path=_path(raw["item_mapping_path"], "item_mapping_path"),
        parent_asins_path=_path(raw["parent_asins_path"], "parent_asins_path"),
        item_embedding_path=_path(raw["item_embedding_path"], "item_embedding_path"),
        item_embedding_metadata_path=_path(raw["item_embedding_metadata_path"], "item_embedding_metadata_path"),
        report_path=_path(raw["report_path"], "report_path"),
        popularity_report_path=_path(raw["popularity_report_path"], "popularity_report_path"),
        lightgcn_report_path=_path(raw["lightgcn_report_path"], "lightgcn_report_path"),
        text_fields=tuple(fields),
        normalize_embeddings=raw["normalize_embeddings"],
        embedding_batch_size=_positive_int(raw["embedding_batch_size"], "embedding_batch_size"),
        eval_user_batch_size=_positive_int(raw["eval_user_batch_size"], "eval_user_batch_size"),
        max_length=_positive_int(raw["max_length"], "max_length"),
        embedding_dim=_positive_int(raw["embedding_dim"], "embedding_dim"),
        topk=tuple(topk),
        device=raw["device"].strip(),
        use_fp16=raw["use_fp16"],
    )
