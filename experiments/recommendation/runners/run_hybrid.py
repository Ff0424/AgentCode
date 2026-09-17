"""Run validation-selected Uniform and Cold-aware V2-06.5 hybrid baselines.

The runner rebuilds the frozen train graph only to load the existing LightGCN
checkpoint, validates and reuses the existing BGE-M3 item embedding cache, and
constructs semantic user profiles from train history. It never trains or embeds.
Both score sources are normalized per user over the full canonical catalog.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.recommendation.datasets.canonical_items import (
    load_canonical_item_universe,
    map_canonical_items,
)
from experiments.recommendation.datasets.hybrid_config import (
    DEFAULT_CONFIG_PATH,
    load_hybrid_config,
)
from experiments.recommendation.datasets.load_dataset import load_dataset
from experiments.recommendation.datasets.semantic_artifacts import (
    sha256_file,
    validate_embedding_artifacts,
)
from experiments.recommendation.datasets.semantic_config import load_semantic_config
from experiments.recommendation.evaluation.loo_metrics import (
    PartitionMetrics,
    evaluate_rank_partitions,
)
from experiments.recommendation.models.hybrid import (
    fuse_cold_aware,
    fuse_uniform,
    per_user_zscore,
)
from experiments.recommendation.models.lightgcn import LightGCN
from experiments.recommendation.models.semantic import (
    build_mean_user_profiles,
    validate_normalized_embeddings,
)
from experiments.recommendation.runners.build_semantic_item_embeddings import (
    _select_cuda_device,
)
from experiments.recommendation.runners.run_lightgcn import build_normalized_adjacency


VARIANTS = ("uniform", "cold_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-06.5 hybrid recommendation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _compact_users(dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    universe = np.unique(
        np.concatenate((dataset.train_user, dataset.valid_user, dataset.test_user))
    ).astype(np.int32, copy=False)
    mapped = {}
    for split in ("train", "valid", "test"):
        values = getattr(dataset, f"{split}_user")
        rows = np.searchsorted(universe, values)
        if np.any(rows >= universe.size) or not np.array_equal(universe[rows], values):
            raise ValueError(f"Unable to map every {split} user.")
        mapped[split] = np.ascontiguousarray(rows, dtype=np.int64)
    return universe, mapped


def _seen_history(train_users: np.ndarray, train_items: np.ndarray, num_users: int):
    order = np.lexsort((train_items, train_users))
    users, items = train_users[order], train_items[order]
    if np.any((users[1:] == users[:-1]) & (items[1:] == items[:-1])):
        raise ValueError("Train split contains duplicate compact user-item pairs.")
    counts = np.bincount(users, minlength=num_users)
    offsets = np.empty(num_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, np.ascontiguousarray(items, dtype=np.int64)


def _mask_coordinates(users, offsets, items):
    counts = offsets[users + 1] - offsets[users]
    rows = np.repeat(np.arange(users.size, dtype=np.int64), counts)
    # Explicit loop keeps the CSR slicing readable and only materializes one batch.
    columns = [items[offsets[user] : offsets[user + 1]] for user in users]
    return rows, np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)


def _target_ranks(top_rows: torch.Tensor, target_rows: torch.Tensor) -> np.ndarray:
    matches = top_rows.eq(target_rows.unsqueeze(1))
    found = matches.any(dim=1)
    ranks = torch.zeros(top_rows.shape[0], dtype=torch.int16, device=top_rows.device)
    # Compute positions as int64, then cast only the bounded 1..Top-K ranks for storage.
    hit_ranks = (
        matches[found]
        .to(torch.int64)
        .argmax(dim=1)
        .add(1)
        .to(dtype=ranks.dtype)
    )
    ranks[found] = hit_ranks
    return ranks.cpu().numpy()


def evaluate_alpha_grid(
    collaborative_users: torch.Tensor,
    collaborative_items: torch.Tensor,
    semantic_users: torch.Tensor,
    semantic_items: torch.Tensor,
    eval_users: np.ndarray,
    target_rows: np.ndarray,
    train_observed: np.ndarray,
    seen_offsets: np.ndarray,
    seen_items: np.ndarray,
    alphas: tuple[float, ...],
    epsilon: float,
    batch_size: int,
    topk: tuple[int, ...],
) -> tuple[dict[str, dict[float, PartitionMetrics]], float]:
    """Score one split in batches and retain only each target's Top-50 rank."""

    rank_arrays = {
        variant: {alpha: np.zeros(eval_users.size, dtype=np.int16) for alpha in alphas}
        for variant in VARIANTS
    }
    observed_tensor = torch.as_tensor(
        train_observed, dtype=torch.bool, device=collaborative_users.device
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, eval_users.size, batch_size):
            stop = min(start + batch_size, eval_users.size)
            users = eval_users[start:stop]
            user_tensor = torch.as_tensor(users, device=collaborative_users.device)
            target_tensor = torch.as_tensor(
                target_rows[start:stop], device=collaborative_users.device
            )
            collaborative = per_user_zscore(
                collaborative_users[user_tensor] @ collaborative_items.T, epsilon
            )
            semantic = per_user_zscore(
                semantic_users[user_tensor] @ semantic_items.T, epsilon
            )
            mask_rows, mask_items = _mask_coordinates(users, seen_offsets, seen_items)
            row_tensor = torch.as_tensor(mask_rows, device=collaborative.device)
            item_tensor = torch.as_tensor(mask_items, device=collaborative.device)
            for alpha in alphas:
                uniform = fuse_uniform(collaborative, semantic, alpha)
                uniform[row_tensor, item_tensor] = -torch.inf
                top_rows = torch.topk(uniform, max(topk), dim=1, sorted=True).indices
                rank_arrays["uniform"][alpha][start:stop] = _target_ranks(
                    top_rows, target_tensor
                )
                cold_aware = fuse_cold_aware(
                    collaborative, semantic, alpha, observed_tensor
                )
                cold_aware[row_tensor, item_tensor] = -torch.inf
                top_rows = torch.topk(cold_aware, max(topk), dim=1, sorted=True).indices
                rank_arrays["cold_aware"][alpha][start:stop] = _target_ranks(
                    top_rows, target_tensor
                )
                del uniform, cold_aware, top_rows
            del collaborative, semantic
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    warm_mask = train_observed[target_rows]
    results = {
        variant: {
            alpha: evaluate_rank_partitions(ranks, warm_mask, topk)
            for alpha, ranks in by_alpha.items()
        }
        for variant, by_alpha in rank_arrays.items()
    }
    return results, elapsed


def _select_alpha(results: dict[float, PartitionMetrics], metric: str) -> float:
    # max() preserves the first grid value on an exact metric tie.
    return max(results, key=lambda alpha: results[alpha].overall[metric])


def _read_test_metrics(path: Path, heading: str, topk) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline report not found: {path}")
    text = path.read_text(encoding="utf-8")
    if heading:
        marker = f"## {heading}"
        if marker not in text:
            raise ValueError(f"Section {marker!r} not found in {path}.")
        text = text.split(marker, 1)[1].split("##", 1)[0]
    metrics = {}
    for prefix in ("Recall", "NDCG"):
        for k in topk:
            name = f"{prefix}@{k}"
            match = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([0-9.]+)\s*\|", text)
            if match is None:
                raise ValueError(f"Missing {name} in {path}.")
            metrics[name] = float(match.group(1))
    return metrics


def _metric_rows(result: PartitionMetrics, topk) -> list[str]:
    rows = []
    for scope, metrics in (("Overall", result.overall), ("Warm", result.warm), ("Cold", result.cold)):
        for prefix in ("Recall", "NDCG"):
            for k in topk:
                value = 0.0 if metrics is None else metrics[f"{prefix}@{k}"]
                rows.append(f"| {scope} | {prefix}@{k} | {value:.8f} |")
    return rows


def _atomic_report(path: Path, text: str, overwrite: bool) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(config_path: Path, overwrite: bool) -> Path:
    config = load_hybrid_config(config_path.resolve())
    semantic_config = load_semantic_config(config.semantic_config_path)
    if config.device != semantic_config.device:
        raise ValueError("Hybrid and semantic configs must select the same CUDA device.")
    if not semantic_config.normalize_embeddings:
        raise ValueError("V2-06.5 requires the frozen normalized semantic embeddings.")
    if config.report_path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {config.report_path}. Use --overwrite.")
    device_index = _select_cuda_device(torch, config.device)
    device = torch.device("cuda", device_index)
    torch.cuda.reset_peak_memory_stats()

    dataset = load_dataset(semantic_config.dataset_path)
    universe = load_canonical_item_universe(
        semantic_config.item_mapping_path, semantic_config.parent_asins_path
    )
    canonical_users, user_rows = _compact_users(dataset)
    item_rows = {
        split: map_canonical_items(
            getattr(dataset, f"{split}_item"), universe.item_indices, f"{split} item"
        ).astype(np.int64)
        for split in ("train", "valid", "test")
    }
    train_observed = np.zeros(universe.item_indices.size, dtype=bool)
    train_observed[np.unique(item_rows["train"])] = True
    seen_offsets, seen_items = _seen_history(
        user_rows["train"], item_rows["train"], canonical_users.size
    )

    semantic_array, semantic_metadata = validate_embedding_artifacts(
        semantic_config, universe
    )
    semantic_items = torch.as_tensor(semantic_array, device=device).contiguous()
    validate_normalized_embeddings(semantic_items)
    semantic_users, _ = build_mean_user_profiles(
        torch.as_tensor(user_rows["train"], device=device),
        torch.as_tensor(item_rows["train"], device=device),
        semantic_items,
        canonical_users.size,
        normalize=True,
    )
    validate_normalized_embeddings(semantic_users)

    if not config.lightgcn_checkpoint_path.is_file():
        raise FileNotFoundError(f"LightGCN checkpoint not found: {config.lightgcn_checkpoint_path}")
    checkpoint = torch.load(
        config.lightgcn_checkpoint_path, map_location=device, weights_only=True
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("LightGCN checkpoint lacks model_config.")
    adjacency = build_normalized_adjacency(
        user_rows["train"].astype(np.int32), item_rows["train"].astype(np.int32),
        canonical_users.size, universe.item_indices.size, device,
    )
    model = LightGCN(
        canonical_users.size, universe.item_indices.size,
        int(model_config["embedding_dim"]), int(model_config["num_layers"]), adjacency,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        collaborative_users, collaborative_items = model.propagate()

    validation, validation_seconds = evaluate_alpha_grid(
        collaborative_users, collaborative_items, semantic_users, semantic_items,
        user_rows["valid"], item_rows["valid"], train_observed,
        seen_offsets, seen_items, config.alpha_grid, config.epsilon,
        config.eval_user_batch_size, config.topk,
    )
    selected = {
        variant: _select_alpha(validation[variant], config.selection_metric)
        for variant in VARIANTS
    }
    test_alphas = tuple(sorted(set(selected.values())))
    test_grid, test_seconds = evaluate_alpha_grid(
        collaborative_users, collaborative_items, semantic_users, semantic_items,
        user_rows["test"], item_rows["test"], train_observed,
        seen_offsets, seen_items, test_alphas, config.epsilon,
        config.eval_user_batch_size, config.topk,
    )
    test = {variant: test_grid[variant][selected[variant]] for variant in VARIANTS}

    popularity = _read_test_metrics(config.popularity_report_path, "", config.topk)
    lightgcn = _read_test_metrics(config.lightgcn_report_path, "Test Metrics Overall", config.topk)
    semantic = _read_test_metrics(config.semantic_report_path, "Test Overall", config.topk)
    lines = [
        "# AgentRec V2 Hybrid Recommendation Baseline", "",
        f"- Generated at UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Canonical items: {universe.item_indices.size:,}",
        f"- Train/Valid/Test: {dataset.train_user.size:,} / {dataset.valid_user.size:,} / {dataset.test_user.size:,}",
        f"- Train-observed / train-unseen items: {train_observed.sum():,} / {(~train_observed).sum():,}", "",
        "## Artifact Provenance", "",
        f"- LightGCN checkpoint: `{config.lightgcn_checkpoint_path}`",
        f"- LightGCN checkpoint SHA-256: `{sha256_file(config.lightgcn_checkpoint_path)}`",
        f"- Semantic embeddings: `{semantic_config.item_embedding_path}`",
        f"- Semantic metadata: `{semantic_config.item_embedding_metadata_path}`",
        f"- Semantic embedding SHA-256: `{sha256_file(semantic_config.item_embedding_path)}`",
        f"- Semantic metadata SHA-256: `{sha256_file(semantic_config.item_embedding_metadata_path)}`",
        f"- Semantic identity SHA-256: `{semantic_metadata['provenance']['canonical_identity_sha256']}`", "",
        "## Fusion and Normalization", "",
        "Each source is normalized per user across the full 125,762-item score vector using "
        "Z-score: (score - mean) / (population_std + epsilon). Uniform Hybrid uses "
        "(1-alpha)*LightGCN + alpha*Semantic. Cold-aware Hybrid uses the same formula for "
        "train-observed items and Semantic-only normalized scores for train-unseen items.",
        f"Epsilon: {config.epsilon}.", "",
        "## Validation Alpha Sweep", "",
        "| Variant | Alpha | Recall@20 | NDCG@20 |", "| --- | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        for alpha in config.alpha_grid:
            metrics = validation[variant][alpha].overall
            lines.append(f"| {variant} | {alpha:.1f} | {metrics['Recall@20']:.8f} | {metrics['NDCG@20']:.8f} |")
    for variant in VARIANTS:
        alpha = selected[variant]
        lines += ["", f"## {variant.replace('_', ' ').title()} Selected Result", "",
                  f"- Selected alpha by Validation NDCG@20: **{alpha:.1f}**", "",
                  "### Validation", "", "| Scope | Metric | Value |", "| --- | --- | ---: |"]
        lines += _metric_rows(validation[variant][alpha], config.topk)
        lines += ["", "### Test", "", "| Scope | Metric | Value |", "| --- | --- | ---: |"]
        lines += _metric_rows(test[variant], config.topk)
    lines += ["", "## Baseline Comparison", "",
              "| Method | Recall@20 | NDCG@20 |", "| --- | ---: | ---: |",
              f"| Popularity | {popularity['Recall@20']:.8f} | {popularity['NDCG@20']:.8f} |",
              f"| LightGCN | {lightgcn['Recall@20']:.8f} | {lightgcn['NDCG@20']:.8f} |",
              f"| Semantic | {semantic['Recall@20']:.8f} | {semantic['NDCG@20']:.8f} |",
              f"| Uniform Hybrid | {test['uniform'].overall['Recall@20']:.8f} | {test['uniform'].overall['NDCG@20']:.8f} |",
              f"| Cold-aware Hybrid | {test['cold_aware'].overall['Recall@20']:.8f} | {test['cold_aware'].overall['NDCG@20']:.8f} |", "",
              "## Runtime", "", f"- Validation alpha sweep: {validation_seconds:.2f} s",
              f"- Final test evaluation: {test_seconds:.2f} s",
              f"- Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB", "",
              "## Conclusion", "",
              "Alpha values were selected exclusively by Validation NDCG@20. Test was evaluated "
              "once after freezing each variant's alpha. The report preserves Overall, Warm, and "
              "Cold metrics under the shared full-item protocol.", ""]
    _atomic_report(config.report_path, "\n".join(lines), overwrite)
    return config.report_path


def main() -> int:
    args = parse_args()
    try:
        path = run(args.config, args.overwrite)
    except (FileNotFoundError, FloatingPointError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Hybrid experiment completed. Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
