"""Strict configuration loader for the V2-06.5 hybrid experiment."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiments/recommendation/configs/hybrid.yaml"
REQUIRED = {
    "semantic_config_path", "lightgcn_checkpoint_path", "report_path",
    "popularity_report_path", "lightgcn_report_path", "semantic_report_path",
    "alpha_grid", "epsilon", "eval_user_batch_size", "topk",
    "selection_metric", "device",
}


@dataclass(frozen=True)
class HybridConfig:
    semantic_config_path: Path
    lightgcn_checkpoint_path: Path
    report_path: Path
    popularity_report_path: Path
    lightgcn_report_path: Path
    semantic_report_path: Path
    alpha_grid: tuple[float, ...]
    epsilon: float
    eval_user_batch_size: int
    topk: tuple[int, ...]
    selection_metric: str
    device: str


def _parse(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        return [ast.literal_eval(part.strip()) for part in value[1:-1].split(",")]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_hybrid_config(path: str | Path = DEFAULT_CONFIG_PATH) -> HybridConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Hybrid config not found: {path}")
    raw = {}
    for number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid hybrid config line {number}: {original!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in raw:
            raise ValueError(f"Duplicate hybrid config key: {key!r}.")
        raw[key] = _parse(value.strip())
    if raw.keys() != REQUIRED:
        raise ValueError(
            f"Hybrid config keys mismatch; missing={sorted(REQUIRED-raw.keys())}, "
            f"extra={sorted(raw.keys()-REQUIRED)}."
        )
    alpha_grid = tuple(float(value) for value in raw["alpha_grid"])
    if alpha_grid != tuple(round(i / 10, 1) for i in range(11)):
        raise ValueError("alpha_grid must be [0.0, 0.1, ..., 1.0].")
    epsilon = float(raw["epsilon"])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    batch = raw["eval_user_batch_size"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("eval_user_batch_size must be a positive integer.")
    topk = tuple(raw["topk"])
    if topk != (10, 20, 50):
        raise ValueError("V2-06.5 requires topk [10, 20, 50].")
    if raw["selection_metric"] != "NDCG@20":
        raise ValueError("V2-06.5 selection_metric must be NDCG@20.")
    if not isinstance(raw["device"], str):
        raise TypeError("device must be a CUDA device string.")
    return HybridConfig(
        semantic_config_path=_path(raw["semantic_config_path"], "semantic_config_path"),
        lightgcn_checkpoint_path=_path(raw["lightgcn_checkpoint_path"], "lightgcn_checkpoint_path"),
        report_path=_path(raw["report_path"], "report_path"),
        popularity_report_path=_path(raw["popularity_report_path"], "popularity_report_path"),
        lightgcn_report_path=_path(raw["lightgcn_report_path"], "lightgcn_report_path"),
        semantic_report_path=_path(raw["semantic_report_path"], "semantic_report_path"),
        alpha_grid=alpha_grid,
        epsilon=epsilon,
        eval_user_batch_size=batch,
        topk=topk,
        selection_metric=raw["selection_metric"],
        device=raw["device"].strip(),
    )
