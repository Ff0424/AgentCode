# Product Catalog Quality Report

## Dataset Overview

本报告对 AgentRec V2-04 生成的 canonical Product Catalog 进行全量数据质量检查。Catalog 用于将 Recommendation System 的 `item_index` 统一映射为 RAG 与 Agent 使用的 `parent_asin` 和结构化商品信息。

| Metric | Value |
| --- | ---: |
| Catalog size | 125,762 |
| Unique `item_index` | 125,762 |
| Unique `parent_asin` | 125,762 |

数据来源：

- `data/processed/recommendation/product_catalog.jsonl`
- `data/processed/recommendation/item_mapping.json`

## Identity Quality

| Check | Result |
| --- | ---: |
| Duplicate `item_index` | 0 |
| Duplicate `parent_asin` | 0 |
| Mapping-consistent rows | 125,762 |
| Mapping-inconsistent rows | 0 |
| Mapping consistency | 100.0000% |

每条 catalog record 的 `item_index` 均可在 Recommendation `item_mapping.json` 中找到，并映射到同一条 record 的 `parent_asin`。因此 Recommendation 输出可以稳定转换为 RAG 和 Agent 使用的统一商品身份。

## Metadata Coverage

缺失定义为字段值为 JSON `null`、空字符串或空列表。Coverage 按 `(catalog size - missing count) / catalog size` 计算。

| Field | Present | Missing | Coverage |
| --- | ---: | ---: | ---: |
| `title` | 125,756 | 6 | 99.9952% |
| `categories` | 121,447 | 4,315 | 96.5689% |
| `price` | 68,655 | 57,107 | 54.5912% |
| `average_rating` | 125,762 | 0 | 100.0000% |
| `rating_number` | 125,762 | 0 | 100.0000% |
| `features` | 114,335 | 11,427 | 90.9138% |
| `description` | 71,744 | 54,018 | 57.0474% |

## Data Quality Analysis

### Identity Failure

未发现 identity failure：

- `item_index` 全量唯一；
- `parent_asin` 全量唯一；
- `item_index -> parent_asin` 与 Recommendation mapping 全量一致；
- 没有缺失的 `parent_asin`。

Identity layer 检查结果：**PASS**。

### Schema Failure

全量 125,762 条 JSONL records 均可解析，且符合 canonical catalog schema：

- `item_index` 为非负整数；
- `parent_asin` 为非空字符串；
- `categories` 和 `features` 为字符串列表；
- `title` 和 `description` 为字符串或合法 `null`；
- `price` 和 `average_rating` 为 finite number 或合法 `null`；
- `rating_number` 为非负整数或合法 `null`。

Schema/parse failure 数量：**0**。

Schema 检查结果：**PASS**。

### Upstream Metadata Missing

Catalog 中的内容缺失来自上游 Amazon Product Documents/metadata，不属于身份映射或 JSON schema 错误：

- `title` 和 `categories` 覆盖率较高，分别为 99.9952% 和 96.5689%；
- `features` 覆盖率为 90.9138%，适合支持大部分语义检索和 RAG 场景，但不能保证每个商品都有结构化 feature 内容；
- `average_rating` 和 `rating_number` 覆盖率均为 100%；
- `price` 覆盖率为 54.5912%，启用价格 hard constraint 时，无价格商品需要按现有业务规则过滤；
- `description` 覆盖率为 57.0474%，缺失 description 的商品仍可使用 title、categories、features 和其他 metadata；
- 不可用的 source price 保持为 `null`，没有猜测或生成虚假数值。

这些缺失会影响部分商品信息丰富度和约束可用性，但不会破坏 Recommendation、RAG 与 Agent 之间的 canonical identity alignment。

## Final Conclusion

**V2-04 Product Catalog Quality：PASS**

Product Catalog 满足 Recommendation-RAG-Agent 统一商品实体层的核心要求：

- 125,762 个 10-core 商品全部进入 catalog；
- Recommendation `item_index` 与 canonical `parent_asin` 一一对应；
- identity mapping consistency 为 100%；
- duplicate identity 和 schema failure 均为 0；
- Agent-facing 商品字段具有稳定、可解析的 schema；
- 上游 metadata 缺失已被透明保留和量化，没有使用 fake data 填充。

因此，V2-04 已建立可供后续 Personalized Recommendation、RAG Product Knowledge 和 Stateful Shopping Plan 使用的统一商品实体基础。后续模块仍需正确处理可选 metadata 缺失，尤其是 price 和 description。
