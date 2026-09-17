# AgentRec V2 Popularity Baseline

## Experiment Configuration

- Generated at (UTC): 2026-09-11T15:35:35+00:00
- Dataset path: `data\processed\recommendation\experiments`
- Evaluation split: `test`
- Top-K cutoffs: `[10, 20, 50]`
- Fit data: training interactions only
- Seen-item filtering: enabled using each user's training history
- Tie-break: interaction frequency descending, canonical `item_index` ascending

## Dataset Scale

| Split | Interactions | Unique Users | Unique Items |
| --- | ---: | ---: | ---: |
| Train | 4,257,087 | 357,974 | 125,684 |
| Valid | 354,291 | 354,291 | 79,929 |
| Test | 354,291 | 354,291 | 73,732 |

Train-observed popularity candidates: **125,684**.

## Method

The baseline counts each canonical item interaction in the frozen training split. Every test user receives the same global popularity order after removing items already seen by that user in train. Validation and test interactions are never used to fit frequencies. This provides a deterministic non-personalized baseline for later LightGCN, Semantic, and Hybrid experiments.

## Metrics

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.00981397 |
| Recall@20 | 0.01670943 |
| Recall@50 | 0.02708226 |
| NDCG@10 | 0.00512282 |
| NDCG@20 | 0.00690304 |
| NDCG@50 | 0.00894656 |

## Conclusion

This result freezes the first V2 recommendation baseline under the shared temporal split and ranking-metric contract. It should be used as the minimum reference point for subsequent personalized and hybrid models.
