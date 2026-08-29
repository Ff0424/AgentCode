We first performed a streaming scan over the Amazon Electronics review dataset. The dataset contains 43,886,944 review records.
我们首先对 Amazon Electronics Review 数据进行了流式扫描，共包含 43,886,944 条 Review 记录。
    Amazon Electronics Metadata
          ↓
    10-Core Products
    125,762
          ↓
    Metadata Coverage
    100%
          ↓
    字段质量分析
    Title / Features / Details / Description
          ↓
    Product Document V1
          ↓
    BGE-M3 Tokenizer Length Analysis
          ↓
    发现：
    P99 = 2,274
    Max = 23,228
          ↓
    Product Document V2
    字段优先级 + 2048 Token Budget
          ↓
    Validation
          ↓
    125,762 Documents
    0 Empty
    0 Over-limit
          ↓
        PASS
基于 Amazon Reviews'23 Electronics 数据构建商品知识库，对 12.6 万商品元数据进行字段质量与 BGE-M3 Token 长度分析，设计字段优先级与 Token Budget 策略，将极端文档长度从 23K+ tokens 控制至 2,048 tokens，并保持完整商品覆盖。