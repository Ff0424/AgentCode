# Enriched Interaction Validation Report

## Validation Purpose

Validate that rating and timestamp values recovered from the original Amazon Reviews'23 Electronics stream are aligned exactly with the frozen 10-core canonical `(user_index, item_index)` row order.

## Inputs and Output

- Frozen interactions: `data/processed/recommendation/interactions_10core.npz`
- Canonical user mapping: `data/processed/recommendation/user_mapping.json`
- Canonical item mapping: `data/processed/recommendation/item_mapping.json`
- 10-core product identities: `data/processed/recommendation/parent_asins_10core.json`
- Original reviews: `data/raw/reviews_Electronics.jsonl.gz`
- Enriched artifact: `data/processed/recommendation/experiments/enriched_interactions.npz`

## Dataset Overview

| Measure | Value |
| --- | ---: |
| Source reviews scanned | 43,886,944 |
| Enriched interactions | 6,173,972 |
| Unique canonical users | 358,484 |
| Unique canonical items | 125,762 |

## Identity and Order Validation

| Check | Result |
| --- | --- |
| User/item identity matches | 6,173,972 / 6,173,972 (100.0000%) |
| User/item identity mismatches | 0 |
| Row order vs. `interactions_10core.npz` | PASS (exact full-array equality) |
| `parent_asins_10core.json` vs. `item_mapping.json` | PASS |

## Artifact Schema

| Array | Shape | dtype |
| --- | --- | --- |
| `user_indices` | `(6173972,)` | `int32` |
| `item_indices` | `(6173972,)` | `int32` |
| `ratings` | `(6173972,)` | `float32` |
| `timestamps` | `(6173972,)` | `int64` |

## Rating Validation

- Valid finite range: **PASS** (`1` to `5`)
- Mean rating: `4.311053`

| Rating | Count | Share |
| ---: | ---: | ---: |
| 1 | 422,694 | 6.8464% |
| 2 | 259,948 | 4.2104% |
| 3 | 428,082 | 6.9337% |
| 4 | 926,755 | 15.0107% |
| 5 | 4,136,493 | 66.9989% |

## Timestamp Validation

- Integer Unix milliseconds, non-negative and UTC-convertible: **PASS**
- Minimum: `943669237000` (1999-11-27T02:20:37+00:00)
- Maximum: `1694023450911` (2023-09-06T18:04:10.911000+00:00)

## Final Conclusion

**PASS** — `enriched_interactions.npz` contains exactly 6,173,972 validated interactions. Canonical user/item identities and complete row order match the frozen 10-core artifact exactly; every recovered rating and timestamp satisfies the V2-05.2 data contract. The dataset is ready for deterministic positive-feedback derivation and temporal splitting in the next Recommendation experiment stage.
