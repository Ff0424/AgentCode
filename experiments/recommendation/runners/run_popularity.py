"""
Run the AgentRec V2 popularity recommendation baseline.

The runner reads the frozen train/valid/test NPZ files through the shared
Dataset Loader, fits ``PopularityRecommender`` using train only, produces
train-seen-filtered Top-K rankings for test users, evaluates them through the
shared ``evaluate_ranking()``, and writes a Markdown experiment report.

Example:
    python -m experiments.recommendation.runners.run_popularity \
        --dataset-path data/processed/recommendation/experiments \
        --output reports/recommendation/popularity_baseline.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.recommendation.datasets.load_dataset import load_dataset
from experiments.recommendation.evaluation.ranking_metrics import evaluate_ranking
from experiments.recommendation.models.popularity import PopularityRecommender


# ============================================================
# 1. Experiment defaults and CLI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "recommendation" / "experiments"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "recommendation" / "popularity_baseline.md"
)
DEFAULT_TOPK = (10, 20, 50)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("K values must be positive integers.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the AgentRec V2 popularity baseline on test users."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Directory containing frozen train.npz, valid.npz, and test.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_TOPK),
        help="Unique evaluation cutoffs, for example: --topk 10 20 50.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing report after a successful run.",
    )
    return parser.parse_args()


# ============================================================
# 2. Dataset and result validation
# ============================================================

def _validate_topk(topk: list[int]) -> tuple[int, ...]:
    values = tuple(topk)
    if len(set(values)) != len(values):
        raise ValueError(f"--topk contains duplicate values: {values}.")
    return values


def _validate_test_leave_one_out(test_user: np.ndarray) -> None:
    """The frozen LOO test split must contain exactly one row per user."""

    if test_user.size == 0:
        raise ValueError("The test split is empty.")
    if np.unique(test_user).size != test_user.size:
        raise ValueError(
            "The test split contains duplicate users; leave-one-out evaluation "
            "requires exactly one target row per test user."
        )


def _validate_rankings(rankings: np.ndarray, user_count: int, max_k: int) -> None:
    expected_shape = (user_count, max_k)
    if rankings.shape != expected_shape:
        raise ValueError(
            f"Ranking shape is {rankings.shape}; expected {expected_shape}."
        )
    if rankings.dtype != np.dtype("int32"):
        raise TypeError(f"Rankings must use int32; got {rankings.dtype}.")
    if rankings.size and np.any(rankings < 0):
        raise ValueError("Rankings contain negative canonical item indices.")


# ============================================================
# 3. Markdown report publication
# ============================================================

def _format_report(
    dataset_path: Path,
    topk: tuple[int, ...],
    metrics: dict[str, float],
    dataset,
    model: PopularityRecommender,
) -> str:
    train_users = int(np.unique(dataset.train_user).size)
    train_items = int(np.unique(dataset.train_item).size)
    test_users = int(np.unique(dataset.test_user).size)
    test_items = int(np.unique(dataset.test_item).size)

    lines = [
        "# AgentRec V2 Popularity Baseline",
        "",
        "## Experiment Configuration",
        "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Dataset path: `{dataset_path}`",
        "- Evaluation split: `test`",
        f"- Top-K cutoffs: `{list(topk)}`",
        "- Fit data: training interactions only",
        "- Seen-item filtering: enabled using each user's training history",
        "- Tie-break: interaction frequency descending, canonical `item_index` ascending",
        "",
        "## Dataset Scale",
        "",
        "| Split | Interactions | Unique Users | Unique Items |",
        "| --- | ---: | ---: | ---: |",
        f"| Train | {dataset.train_user.size:,} | {train_users:,} | {train_items:,} |",
        f"| Valid | {dataset.valid_user.size:,} | {np.unique(dataset.valid_user).size:,} | {np.unique(dataset.valid_item).size:,} |",
        f"| Test | {dataset.test_user.size:,} | {test_users:,} | {test_items:,} |",
        "",
        f"Train-observed popularity candidates: **{model.popular_items.size:,}**.",
        "",
        "## Method",
        "",
        "The baseline counts each canonical item interaction in the frozen training split. "
        "Every test user receives the same global popularity order after removing items "
        "already seen by that user in train. Validation and test interactions are never "
        "used to fit frequencies. This provides a deterministic non-personalized baseline "
        "for later LightGCN, Semantic, and Hybrid experiments.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for k in topk:
        lines.append(f"| Recall@{k} | {metrics[f'Recall@{k}']:.8f} |")
    for k in topk:
        lines.append(f"| NDCG@{k} | {metrics[f'NDCG@{k}']:.8f} |")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "This result freezes the first V2 recommendation baseline under the shared "
            "temporal split and ranking-metric contract. It should be used as the minimum "
            "reference point for subsequent personalized and hybrid models.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report_atomic(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output report already exists: {path}. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


# ============================================================
# 4. Experiment orchestration
# ============================================================

def run_experiment(
    dataset_path: Path,
    output_path: Path,
    topk: tuple[int, ...],
    overwrite: bool = False,
) -> dict[str, float]:
    """Fit, evaluate, report, and return the popularity baseline metrics."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output report already exists: {output_path}. Use --overwrite to replace it."
        )
    dataset = load_dataset(dataset_path)
    _validate_test_leave_one_out(dataset.test_user)

    model = PopularityRecommender().fit(dataset.train_user, dataset.train_item)
    max_k = max(topk)
    rankings = model.recommend(dataset.test_user, max_k)
    _validate_rankings(rankings, dataset.test_user.size, max_k)

    metrics = evaluate_ranking(
        ranked_items_by_user=rankings,
        relevant_items_by_user=dataset.test_item,
        topk=topk,
    )
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Evaluation produced non-finite metrics: {metrics}.")

    report = _format_report(dataset_path, topk, metrics, dataset, model)
    _write_report_atomic(output_path, report, overwrite=overwrite)
    return metrics


def main() -> int:
    args = parse_args()
    try:
        topk = _validate_topk(args.topk)
        metrics = run_experiment(
            dataset_path=args.dataset_path,
            output_path=args.output,
            topk=topk,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("========== AgentRec V2 Popularity Baseline ==========")
    for k in topk:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.8f}")
    for k in topk:
        print(f"NDCG@{k}: {metrics[f'NDCG@{k}']:.8f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
