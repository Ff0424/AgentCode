# AgentRec V2 Recommendation Experiments

This directory provides the model-independent experiment framework shared by
Popularity, LightGCN, Semantic, and Hybrid recommendation experiments. V2-06.1
defines dataset and evaluation contracts only; it does not implement or train
LightGCN.

## Directory Structure

```text
experiments/recommendation/
├── configs/
│   └── base.yaml
├── datasets/
│   └── load_dataset.py
├── evaluation/
│   └── ranking_metrics.py
├── models/
├── runners/
└── README.md
```

Responsibilities:

- `datasets/`: validates and loads frozen recommendation splits.
- `evaluation/`: model-independent top-K ranking metrics.
- `models/`: future model implementations with a shared ranking contract.
- `runners/`: future CLI orchestration for training and evaluation.
- `configs/`: experiment parameters shared across runners.

## Frozen Dataset Format

The default dataset directory is:

```text
data/processed/recommendation/experiments/
├── train.npz
├── valid.npz
└── test.npz
```

Every NPZ contains two aligned, one-dimensional `int32` arrays:

- `user_indices`: canonical Recommendation user indices.
- `item_indices`: canonical Recommendation item indices.

The loader exposes the unified interface:

```python
from experiments.recommendation.datasets.load_dataset import load_dataset

dataset = load_dataset("data/processed/recommendation/experiments")

dataset.train_user
dataset.train_item
dataset.valid_user
dataset.valid_item
dataset.test_user
dataset.test_item
```

Returned arrays are read-only. Canonical indices are intentionally preserved
and can be sparse. Models requiring compact embedding-table indices must build
an explicit model-local mapping and map predictions back to canonical
`item_index` before Product Catalog or Agent integration.

## Base Configuration

`configs/base.yaml` freezes the shared initial settings:

```yaml
dataset_path: data/processed/recommendation/experiments
topk: [10, 20, 50]
seed: 42
```

PyYAML is not a V2-06.1 dependency. A future runner may parse this file after
its configuration interface and direct dependency are explicitly introduced.

## Ranking Metrics

`evaluation/ranking_metrics.py` implements binary-relevance:

- Recall@10, Recall@20, Recall@50
- NDCG@10, NDCG@20, NDCG@50

Metrics are macro-averaged across evaluation users. Rankings must not contain
duplicate items, and every user must have at least one ground-truth item. The
current leave-one-out dataset may pass its ground-truth item as a scalar:

```python
from experiments.recommendation.evaluation.ranking_metrics import evaluate_ranking

metrics = evaluate_ranking(
    ranked_items_by_user=[[31, 42, 99], [7, 18, 24]],
    relevant_items_by_user=[42, 24],
    topk=[10, 20, 50],
)
```

## Experiment Flow

Future experiment runners should follow one shared flow:

1. Load `base.yaml` and model-specific settings.
2. Load the frozen temporal dataset through `load_dataset()`.
3. Train or construct the model from train interactions only.
4. Use valid interactions for hyperparameter/model selection only.
5. Produce duplicate-free ranked canonical item IDs for evaluation users.
6. Filter each user's train-seen items before scoring metrics.
7. Report Recall@K and NDCG@K through `evaluate_ranking()`.
8. Evaluate test once after configuration is frozen.

Cold items absent from train cannot be learned by a collaborative model. Future
runners must report their count and apply the experiment protocol consistently
when computing primary collaborative metrics.

## Adding a Model

Add one focused module under `models/` and one corresponding CLI under
`runners/`. A model implementation should:

- consume the unified `RecommendationDataset` interface;
- avoid changing frozen canonical arrays;
- use train data only for fitting and popularity statistics;
- return ranked canonical `item_index` values in descending preference order;
- exclude items already seen in the applicable user history;
- keep model-specific dependencies and configuration explicit;
- delegate Recall/NDCG calculation to the shared evaluation module.

This keeps Popularity, LightGCN, Semantic, and Hybrid results directly
comparable without duplicating dataset or metric logic.

## Validation Commands

Inspect the frozen dataset:

```bash
python -m experiments.recommendation.datasets.load_dataset \
  --dataset-path data/processed/recommendation/experiments
```

Run the deterministic metric example:

```bash
python -m experiments.recommendation.evaluation.ranking_metrics
```
