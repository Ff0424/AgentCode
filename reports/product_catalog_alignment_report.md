# Product Catalog Alignment Validation Report

## 验证目的

验证 Recommendation 输出的 `item_index` 能否通过 Product Catalog 稳定映射为 Agent 可使用的 `parent_asin` 和结构化商品信息。

## 数据规模

- Product Catalog：`data/processed/recommendation/product_catalog.jsonl`
- Item Mapping：`data/processed/recommendation/item_mapping.json`
- Catalog rows：125,762
- Sample size：100
- Random seed：42

## 检查项

- JSONL 每行是否为合法 JSON object；
- canonical JSON schema 和字段类型是否正确；
- `item_index` 与 `parent_asin` 是否全局唯一；
- sampled `item_index` 是否能通过 `item_mapping.json` 映射到同一 `parent_asin`；
- title、categories、price、average_rating、rating_number 的完整性；
- 数值字段是否为 finite JSON numbers 或合法 null。

## 验证结果

### 全量 Catalog 检查

| Check | Result |
| --- | ---: |
| Rows | 125,762 |
| Unique item indices | 125,762 |
| Unique parent ASINs | 125,762 |
| Duplicate item indices | 0 |
| Duplicate parent ASINs | 0 |
| Schema/parse errors | 0 |

### 随机样本对齐检查

| Check | Result |
| --- | ---: |
| Passed | 100 |
| Failed | 0 |
| Missing parent_asin | 0 |
| Missing title | 0 |
| Missing categories | 5 |
| Missing price | 52 |
| Missing average_rating | 0 |
| Missing rating_number | 0 |

### 全量内容缺失统计

合法的 null/空内容不属于 schema 解析错误，但在此单独记录。

| Field | Missing |
| --- | ---: |
| parent_asin | 0 |
| title | 6 |
| categories | 4,315 |
| price | 57,107 |
| average_rating | 0 |
| rating_number | 0 |

## 阶段结论

**V2-04.3 Product Catalog Alignment：PASS**

Recommendation `item_index` 可以通过 Product Catalog 稳定转换为 Agent-facing 商品实体。
