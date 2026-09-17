# AgentRec V2 LightGCN Baseline

## Experiment Configuration

- Dataset path: `/home/server/AgentCode/data/processed/recommendation/experiments`
- Checkpoint path: `/home/server/AgentCode/artifacts/recommendation/lightgcn_best.pt`
- Report path: `/home/server/AgentCode/reports/recommendation/lightgcn_baseline.md`
- Fit data: training interactions only
- Validation selects the checkpoint; test is evaluated once afterward

## Dataset Statistics

| Split | Interactions | Unique Users | Unique Items |
| --- | ---: | ---: | ---: |
| Train | 4,257,087 | 357,974 | 125,684 |
| Valid | 354,291 | 354,291 | 79,929 |
| Test | 354,291 | 354,291 | 73,732 |

- Canonical users represented in the positive split: 357,974
- Canonical item universe (V2-04 product layer): 125,762
- Positive-split observed item universe: 125,750
- Train-observed item universe: 125,684
- Train-unseen/cold items: 78
- Canonical items without a positive split interaction: 12

## Model Architecture

LightGCN uses 64-dimensional user/item ego embeddings, 3 rounds of normalized sparse graph propagation, and mean aggregation over the ego plus propagated layers. It contains no feature transformation, nonlinear activation, attention, semantic feature, or reranker.

## Hyperparameters

| Parameter | Value |
| --- | ---: |
| embedding_dim | 64 |
| num_layers | 3 |
| learning_rate | 0.001 |
| batch_size | 262,144 |
| maximum_epochs | 30 |
| reg_weight | 0.0001 |
| seed | 42 |
| topk | [10, 20, 50] |
| eval_user_batch_size | 512 |
| early_stopping_patience | 5 |
| early_stopping_metric | NDCG@20 |

## Graph Construction

The graph has 483,736 nodes and 8,514,174 directed edges after adding both directions for every train interaction. Each edge uses D^-1/2 A D^-1/2 normalization. Canonical IDs are mapped to compact model-local indices and mapped back before evaluation.

## BPR Loss and Negative Sampling

Each train interaction supplies one user-positive pair. One negative item is sampled uniformly from the full canonical item universe for every pair and every epoch. Vectorized rejection sampling checks encoded user-item keys against all train-seen pairs, so a positive or other seen item cannot be sampled. The objective is mean -log sigmoid(s(u,i+) - s(u,i-)) plus L2 regularization on the sampled ego embeddings.

## Training Configuration

- Device: `cuda:0` (NVIDIA GeForce RTX 4090 D)
- Optimizer: Adam
- Validation: full-item Top-50 after every epoch
- Seen-item filtering: all train interactions
- Best validation epoch: **6**
- Epochs executed: 11
- Best epoch total/ranking/regularization loss: 0.53891354 / 0.53886359 / 0.49953041

## Best Validation Metrics Overall

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.01307400 |
| Recall@20 | 0.02177588 |
| Recall@50 | 0.03596196 |
| NDCG@10 | 0.00696183 |
| NDCG@20 | 0.00919043 |
| NDCG@50 | 0.01198065 |

## Best Validation Metrics Warm Start

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.01307869 |
| Recall@20 | 0.02178369 |
| Recall@50 | 0.03597486 |
| NDCG@10 | 0.00696432 |
| NDCG@20 | 0.00919372 |
| NDCG@50 | 0.01198494 |

## Test Metrics Overall

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.00981397 |
| Recall@20 | 0.01698604 |
| Recall@50 | 0.02732782 |
| NDCG@10 | 0.00515527 |
| NDCG@20 | 0.00700258 |
| NDCG@50 | 0.00903287 |

## Test Metrics Warm Start

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.00982337 |
| Recall@20 | 0.01700231 |
| Recall@50 | 0.02735399 |
| NDCG@10 | 0.00516021 |
| NDCG@20 | 0.00700929 |
| NDCG@50 | 0.00904152 |

## Warm and Cold Targets

| Split | Warm Targets | Cold Targets | Cold Rate |
| --- | ---: | ---: | ---: |
| Valid | 354,164 | 127 | 0.035846% |
| Test | 353,952 | 339 | 0.095684% |

## Best Validation Metrics Cold Start

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.00000000 |
| Recall@20 | 0.00000000 |
| Recall@50 | 0.00000000 |
| NDCG@10 | 0.00000000 |
| NDCG@20 | 0.00000000 |
| NDCG@50 | 0.00000000 |

## Test Metrics Cold Start

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.00000000 |
| Recall@20 | 0.00000000 |
| Recall@50 | 0.00000000 |
| NDCG@10 | 0.00000000 |
| NDCG@20 | 0.00000000 |
| NDCG@50 | 0.00000000 |

## Runtime

- Generated at UTC: 2026-09-12T04:48:56+00:00
- Training compute time: 18.30 seconds
- Total validation evaluation time: 16.59 seconds
- Final test evaluation time: 1.41 seconds
- Peak allocated GPU memory: 2.035 GiB

## Comparison with Popularity

| Method | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.00981397 | 0.01670943 | 0.02708226 | 0.00512282 | 0.00690304 | 0.00894656 |
| LightGCN | 0.00981397 | 0.01698604 | 0.02732782 | 0.00515527 | 0.00700258 | 0.00903287 |
| Absolute delta | -0.00000000 | +0.00027661 | +0.00024556 | +0.00003245 | +0.00009954 | +0.00008631 |
| Relative delta | -0.00% | +1.66% | +0.91% | +0.63% | +1.44% | +0.96% |

## Conclusion

The comparison above uses the same frozen temporal split, full canonical item universe, train-seen filtering, one Top-50 ranking per user, and shared Recall/NDCG implementation. Validation selected the checkpoint; test metrics were computed only after selection.
