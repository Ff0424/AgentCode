"""Evaluate the V2-06.4 BGE-M3 semantic recommendation baseline.

Task definition:
    train-history item text embeddings -> mean user semantic profile ->
    full canonical item ranking -> train-seen filtering -> valid/test held-out
    evaluation. No natural-language query, collaborative graph, or target item
    enters profile construction.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.recommendation.datasets.canonical_items import (
    load_canonical_item_universe,
    map_canonical_items,
)
from experiments.recommendation.datasets.load_dataset import load_dataset
from experiments.recommendation.datasets.semantic_artifacts import validate_embedding_artifacts
from experiments.recommendation.datasets.semantic_config import (
    DEFAULT_CONFIG_PATH,
    load_semantic_config,
)
from experiments.recommendation.evaluation.ranking_metrics import evaluate_ranking
from experiments.recommendation.models.semantic import (
    build_mean_user_profiles,
    validate_normalized_embeddings,
)
from experiments.recommendation.runners.build_semantic_item_embeddings import (
    _select_cuda_device,
)


@dataclass(frozen=True)
class EvaluationResult:
    overall: dict[str, float]
    warm: dict[str, float]
    cold: dict[str, float] | None
    warm_targets: int
    cold_targets: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-item semantic recommendation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _compact_users(dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    users = np.unique(
        np.concatenate((dataset.train_user, dataset.valid_user, dataset.test_user))
    ).astype(np.int32, copy=False)
    mapped: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        values = getattr(dataset, f"{split}_user")
        rows = np.searchsorted(users, values)
        if np.any(rows >= users.size) or not np.array_equal(users[rows], values):
            raise ValueError(f"Unable to map every {split} user.")
        mapped[split] = np.ascontiguousarray(rows, dtype=np.int64)
    return users, mapped


def _seen_history(train_users: np.ndarray, train_items: np.ndarray, num_users: int):
    order = np.lexsort((train_items, train_users))
    sorted_users, sorted_items = train_users[order], train_items[order]
    keys = sorted_users.astype(np.int64) * (int(train_items.max()) + 1) + sorted_items
    if np.unique(keys).size != keys.size:
        raise ValueError("Train split contains duplicate user-item pairs.")
    counts = np.bincount(sorted_users, minlength=num_users)
    offsets = np.empty(num_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, np.ascontiguousarray(sorted_items, dtype=np.int64)


def _evaluate(
    profiles: torch.Tensor,
    item_embeddings: torch.Tensor,
    eval_users: np.ndarray,
    target_items: np.ndarray,
    canonical_items: np.ndarray,
    train_observed_rows: np.ndarray,
    seen_offsets: np.ndarray,
    seen_items: np.ndarray,
    batch_size: int,
    topk: tuple[int, ...],
) -> EvaluationResult:
    torch.cuda.synchronize()
    started = time.perf_counter()
    max_k = max(topk)
    rankings = np.empty((eval_users.size, max_k), dtype=np.int32)
    for start in range(0, eval_users.size, batch_size):
        stop = min(start + batch_size, eval_users.size)
        batch_users = eval_users[start:stop]
        users_tensor = torch.as_tensor(batch_users, device=profiles.device)
        scores = profiles[users_tensor] @ item_embeddings.T
        mask_rows: list[int] = []
        mask_cols: list[int] = []
        for local_row, user in enumerate(batch_users):
            history = seen_items[seen_offsets[user] : seen_offsets[user + 1]]
            mask_rows.extend([local_row] * len(history))
            mask_cols.extend(history.tolist())
        if mask_rows:
            scores[
                torch.as_tensor(mask_rows, device=scores.device),
                torch.as_tensor(mask_cols, device=scores.device),
            ] = -torch.inf
        top_rows = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True).indices
        rankings[start:stop] = canonical_items[top_rows.cpu().numpy()]
        del scores, top_rows

    torch.cuda.synchronize()
    scoring_elapsed = time.perf_counter() - started
    overall = evaluate_ranking(rankings, target_items, topk)
    target_rows = map_canonical_items(target_items, canonical_items, "evaluation target")
    warm_mask = train_observed_rows[target_rows]
    warm = evaluate_ranking(rankings[warm_mask], target_items[warm_mask], topk)
    cold_count = int((~warm_mask).sum())
    cold = (
        evaluate_ranking(rankings[~warm_mask], target_items[~warm_mask], topk)
        if cold_count
        else None
    )
    return EvaluationResult(
        overall=overall,
        warm=warm,
        cold=cold,
        warm_targets=int(warm_mask.sum()),
        cold_targets=cold_count,
        elapsed_seconds=scoring_elapsed,
    )


def _read_baseline(path: Path, topk: tuple[int, ...]) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline report not found: {path}")
    text = path.read_text(encoding="utf-8")
    if "lightgcn" in path.name.lower():
        if "## Test Metrics Overall" not in text:
            raise ValueError(f"LightGCN test-overall section not found in {path}.")
        metric_text = text.split("## Test Metrics Overall", 1)[1].split("##", 1)[0]
    else:
        metric_text = text
    result = {}
    for prefix in ("Recall", "NDCG"):
        for k in topk:
            name = f"{prefix}@{k}"
            matches = re.findall(
                rf"\|\s*{re.escape(name)}\s*\|\s*([0-9.]+)\s*\|", metric_text
            )
            if not matches:
                raise ValueError(f"Metric {name} not found in {path}.")
            result[name] = float(matches[0])
    return result


def _metric_table(title: str, result: dict[str, float] | None, topk) -> list[str]:
    lines = [f"## {title}", "", "| Metric | Value |", "| --- | ---: |"]
    for prefix in ("Recall", "NDCG"):
        for k in topk:
            value = 0.0 if result is None else result[f"{prefix}@{k}"]
            lines.append(f"| {prefix}@{k} | {value:.8f} |")
    return lines + [""]


def _write_report(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(config_path: Path, overwrite: bool) -> Path:
    config = load_semantic_config(config_path.resolve())
    if config.report_path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {config.report_path}. Use --overwrite.")
    device_index = _select_cuda_device(torch, config.device)
    device = torch.device("cuda", device_index)
    if device.type != "cuda":
        raise RuntimeError(f"Semantic evaluation requires CUDA, got {device!s}.")
    dataset = load_dataset(config.dataset_path)
    universe = load_canonical_item_universe(config.item_mapping_path, config.parent_asins_path)
    for split in ("train", "valid", "test"):
        map_canonical_items(getattr(dataset, f"{split}_item"), universe.item_indices, f"{split} item")
    embeddings_array, metadata = validate_embedding_artifacts(config, universe)
    torch.cuda.reset_peak_memory_stats()
    item_embeddings = torch.as_tensor(embeddings_array, device=device).contiguous()
    if config.normalize_embeddings:
        validate_normalized_embeddings(item_embeddings)

    _, users = _compact_users(dataset)
    train_item_rows = map_canonical_items(dataset.train_item, universe.item_indices, "train item")
    profiles, _ = build_mean_user_profiles(
        torch.as_tensor(users["train"], device=device),
        torch.as_tensor(train_item_rows.astype(np.int64), device=device),
        item_embeddings,
        int(max(users["train"].max(), users["valid"].max(), users["test"].max())) + 1,
        config.normalize_embeddings,
    )
    if config.normalize_embeddings:
        validate_normalized_embeddings(profiles)
    elif not torch.isfinite(profiles).all():
        raise ValueError("User semantic profiles contain non-finite values.")
    seen_offsets, seen_items = _seen_history(
        users["train"], train_item_rows.astype(np.int64), profiles.shape[0]
    )
    observed = np.zeros(universe.item_indices.size, dtype=bool)
    observed[np.unique(train_item_rows)] = True
    validation = _evaluate(
        profiles, item_embeddings, users["valid"], dataset.valid_item,
        universe.item_indices, observed, seen_offsets, seen_items,
        config.eval_user_batch_size, config.topk,
    )
    test = _evaluate(
        profiles, item_embeddings, users["test"], dataset.test_item,
        universe.item_indices, observed, seen_offsets, seen_items,
        config.eval_user_batch_size, config.topk,
    )
    popularity = _read_baseline(config.popularity_report_path, config.topk)
    lightgcn = _read_baseline(config.lightgcn_report_path, config.topk)
    lines = [
        "# AgentRec V2 Semantic Recommendation Baseline", "",
        f"- Generated at UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- Status: completed on the configured GPU runtime", "",
        "## Dataset and Task Definition", "",
        f"- Canonical items: {universe.item_indices.size:,}",
        f"- Train interactions: {dataset.train_user.size:,}",
        f"- Validation targets: {dataset.valid_user.size:,}",
        f"- Test targets: {dataset.test_user.size:,}",
        "- Task: mean train-history item-text embeddings predict held-out future items.",
        "- Validation/test targets are never used to build user profiles.", "",
        "## Product Embeddings and User Profiles", "",
        f"- Model: `{config.model_name_or_path}`", f"- Text fields: `{list(config.text_fields)}`",
        f"- Embedding dimension: {config.embedding_dim}",
        f"- Normalization: {config.normalize_embeddings}; mean profiles are normalized again.",
        f"- Embedding generation time: {metadata['embedding_generation_seconds']:.2f} s", "",
        f"- Embedding generation peak GPU memory: {metadata['peak_gpu_memory_gib']:.3f} GiB", "",
        "## Full Ranking Protocol", "",
        f"- Candidate universe: all {universe.item_indices.size:,} canonical items",
        "- One full score pass per user batch; Top-50 is reused for every K.",
        "- Every user's complete train-seen history is masked.", "",
    ]
    lines += _metric_table("Validation Overall", validation.overall, config.topk)
    lines += _metric_table("Validation Warm Start", validation.warm, config.topk)
    lines += _metric_table("Validation Cold Start", validation.cold, config.topk)
    lines += _metric_table("Test Overall", test.overall, config.topk)
    lines += _metric_table("Test Warm Start", test.warm, config.topk)
    lines += _metric_table("Test Cold Start", test.cold, config.topk)
    lines += [
        "## Warm and Cold Targets", "",
        "| Split | Warm | Cold |", "| --- | ---: | ---: |",
        f"| Validation | {validation.warm_targets:,} | {validation.cold_targets:,} |",
        f"| Test | {test.warm_targets:,} | {test.cold_targets:,} |", "",
        "## Runtime", "",
        f"- Validation evaluation: {validation.elapsed_seconds:.2f} s",
        f"- Test evaluation: {test.elapsed_seconds:.2f} s",
        f"- Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB", "",
        "## Baseline Comparison", "",
        "| Method | Recall@20 | NDCG@20 |", "| --- | ---: | ---: |",
        f"| Popularity | {popularity['Recall@20']:.8f} | {popularity['NDCG@20']:.8f} |",
        f"| LightGCN | {lightgcn['Recall@20']:.8f} | {lightgcn['NDCG@20']:.8f} |",
        f"| Semantic | {test.overall['Recall@20']:.8f} | {test.overall['NDCG@20']:.8f} |", "",
        "## Conclusion", "",
        "This is a pure text-semantic recommendation baseline under the frozen temporal "
        "split, canonical identity, full-item ranking, and train-seen-filtering contract.", "",
    ]
    _write_report(config.report_path, "\n".join(lines), overwrite)
    return config.report_path


def main() -> int:
    args = parse_args()
    try:
        path = run(args.config, args.overwrite)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Semantic evaluation completed. Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
