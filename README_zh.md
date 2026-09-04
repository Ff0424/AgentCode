# AgentRec

**面向电子产品购物决策的 LLM Agent + RAG + 推荐系统。**

## 1. 项目概述

AgentRec 是一个基于 Amazon Reviews'23 Electronics 数据域构建的商品目录驱动型购物决策助手。系统将商品元数据构造成适合检索的 Product Documents V2，使用 BGE-M3 稠密向量和 FAISS 精确内积搜索召回候选商品，通过显式元数据约束与透明的启发式重排序筛选结果，再向 LLM 提供精简的商品证据，生成基于真实目录信息的回答。

完整系统包括：

- 125,762 条经过验证的 Product Documents V2；
- BGE-M3 稠密语义检索；
- `faiss.IndexFlatIP` 精确向量搜索；
- 价格、评分和评分数量过滤；
- relevance-first 多信号重排序；
- 确定性的 RAG 上下文构建；
- provider-neutral 的有界 Agent 工具循环；
- DeepSeek OpenAI-compatible adapter；
- FastAPI 后端和同源 Web Demo。

AgentRec 不是通用聊天机器人。它的回答基于检索和商品工具所提供的电子产品目录信息。

## 2. 核心能力

- **商品语料构建：** 根据字段优先级和 2,048-token budget，将 Amazon Electronics 元数据构造成紧凑的 Product Documents V2。
- **稠密语义检索：** 使用本地 BGE-M3 编码 query，并在 1,024 维 embedding 空间中检索商品。
- **精确向量搜索：** 使用 `faiss.IndexFlatIP`，不是近似最近邻索引。
- **结构化硬约束：** 当推荐工具收到相应参数时，支持 `max_price`、`min_rating` 和 `min_rating_count`。
- **Relevance-first 重排序：** 只在语义相关候选池内结合语义相关性、评分和热度进行重排。
- **RAG 上下文精简：** 在确定性的字段数量与字符预算内格式化排序后的商品事实。
- **Provider-neutral Agent 编排：** 将工具循环逻辑与具体 LLM provider 解耦。
- **DeepSeek Tool Calling：** 将内部工具调用协议转换为 DeepSeek 使用的 OpenAI-compatible 格式。
- **可复用 Runtime Services：** 提供检索、商品仓库、推荐、RAG 和 Agent 组件。
- **Web 应用：** 由 FastAPI 同源提供轻量 HTML/CSS/JavaScript 界面。

HDMI、Ethernet、续航或降噪等偏好目前通过语义检索和基于 RAG 证据的 LLM 推理处理，不是有严格保证的结构化硬过滤条件。

## 3. 系统架构

```text
                                  在线 Runtime

用户
  -> Web Demo
  -> FastAPI /api/chat
  -> Agent
  -> Agent Tools
  -> Recommendation Service V2
       -> BGE-M3 query embedding
       -> FAISS IndexFlatIP 精确检索
       -> Product Repository 查询
       -> 元数据硬约束过滤
       -> relevance-first 启发式重排序
  -> RAG Context Builder
  -> 通过 OpenAI-compatible adapter 调用 DeepSeek
  -> 基于商品证据的回答

                                  离线 Pipeline

Amazon Reviews'23 Electronics
  -> 交互数据预处理和 10-core 筛选
  -> Product Documents V2
  -> BGE-M3 商品 embeddings
  -> embedding 验证
  -> 构建 FAISS IndexFlatIP
  -> retrieval evaluation
  -> runtime artifacts
```

离线 pipeline 负责生成并验证不可变的检索 artifacts。在线 runtime 只加载这些 artifacts，不会在每次请求时重新生成商品 embeddings 或重建 FAISS 索引。

## 4. 离线 Pipeline

编号脚本主要覆盖以下阶段：

1. 流式扫描并分析 Amazon Electronics reviews 和 metadata。
2. 准备交互数据、比较 K-core 配置并构建最终 10-core 数据集。
3. 分析商品 metadata，按字段优先级和 2,048-token 上限构建 Product Documents V2。
4. 测试 BGE-M3 batch size，生成 float32 商品 embeddings，并验证 row alignment、fingerprint、shape、dtype、finite values 和 norm。
5. 构建保持行顺序的 `faiss.IndexFlatIP(1024)`，通过 reconstruction 验证向量内容和序列化结果。
6. 使用确定性抽样评估 title query 和 feature query，并生成 case-level 输出。

正式商品 embedding 配置使用本地 BGE-M3、`cuda:0`、FP16 inference、`max_length=2048` 和 `batch_size=8`。持久化的商品 embeddings 为 float32，pipeline 不执行额外 L2 normalization。

## 5. 在线 Runtime Pipeline

每个聊天请求执行以下流程：

```text
购物 query
  -> 语义候选召回
  -> 商品详情查询
  -> 可选的元数据硬约束过滤
  -> relevance pool 选择
  -> 确定性启发式重排序
  -> 紧凑 RAG context
  -> 有界 LLM tool loop
  -> 基于商品目录的最终回答
```

Retrieval Service 将每个 FAISS row 映射回严格对齐的 product ID。Product Repository 在初始化时一次性加载 Product Documents V2，后续查询不需要重复扫描 JSONL 文件。

## 6. Recommendation Service

`RecommendationServiceV2` 是当前正式的推荐 baseline，使用 relevance-first pipeline：

1. 语义检索 `candidate_k` 个候选商品。
2. 在保持原始 retrieval 顺序的前提下批量获取商品记录。
3. 应用可选硬约束：
   - `max_price`
   - `min_rating`
   - `min_rating_count`
4. 从过滤后 retrieval rank 最靠前的商品中选取最多 `rerank_pool_k` 个候选。
5. 在 relevance pool 内归一化原始语义分数。
6. 计算评分分数和热度分数。
7. 计算确定性加权分数并返回最终 Top-K。

重排序公式为：

```text
final_score =
    0.70 * normalized_semantic_score
  + 0.20 * rating_score
  + 0.10 * popularity_score
```

`rating_score` 为平均评分除以 5 后裁剪至 `[0, 1]`。`popularity_score` 为当前 relevance pool 内对 `log1p(rating_number)` 进行 min-max normalization 后的结果。确定性 tie-break 依次使用 semantic score、原始 retrieval score、retrieval rank 和 product ID。

这是一个透明的启发式重排序 baseline，不是 learned ranker。当某项硬约束启用时，缺少对应 metadata 的商品会被过滤；未启用约束时，缺失的 rating/popularity 信号使用代码定义的 fallback score。

## 7. RAG 与 Agent 设计

Agent 层由职责明确的组件组成：

- **RAG Context Builder（`20_rag_context_builder.py`）：** 将排序后的推荐结果转换为确定性的事实型上下文，限制总字符数、feature 数量、product details 和 description 长度，并保持排名前缀顺序。
- **Agent Tools（`21_agent_tools.py`）：** 暴露三个结构化工具：
  - `recommend_products`
  - `get_product_details`
  - `compare_products`
- **Agent Orchestration（`22_agent.py`）：** 实现 provider-neutral 的有界 tool loop，并验证 tool calls、处理显式失败。
- **LLM Adapter（`23_llm_adapter.py`）：** 将 provider-neutral messages 和 tool calls 转换为 DeepSeek 使用的 OpenAI-compatible 格式。

在一次 Agent run 内，完整工具结果只保留在内存中的 tool history 中，用于调试和后端处理；发送给 LLM 的是精简投影，避免原始商品文档和不必要的 retrieval 字段占用模型上下文。它不是持久化存储。HTTP response 只返回紧凑的 tool-call summary。

每个 `/api/chat` 请求都会启动一次全新的 Agent run。工具历史不会跨请求持久化，因此当前应用不支持多轮持久化对话记忆。

## 8. Retrieval Evaluation

经过验证的检索语料包含 **125,762 条 Product Documents V2**。商品和 query embeddings 均使用 BGE-M3 的 **1,024 维**稠密向量，并通过 `IndexFlatIP` 执行精确内积搜索。

受控 benchmark 使用固定随机种子 42，对每种 query 类型分别抽样 2,000 个商品：

- 从商品标题生成的 title queries；
- 从商品 features 确定性生成的短 feature queries。

| Query 类型 | Recall\@1 | Recall\@5 | Recall\@10 | Recall\@20 |   MRR\@20 |
| ---------- | --------: | --------: | ---------: | ---------: | --------: |
| Title      |    0.9255 |    0.9935 |     0.9960 |     0.9980 | 0.9561996 |
| Feature    |    0.4950 |    0.7060 |     0.7660 |     0.8135 | 0.5869288 |

这是一个使用 weak supervision/pseudo ground truth 的 controlled known-item retrieval benchmark：生成 query 的原商品被视为目标商品。目标商品没有被召回，不代表所有已召回的替代商品都不相关；这些指标不能解释为经过人工标注的推荐质量。

## 9. Reranking Analysis

V1 baseline 在整个候选集合上直接应用评分与热度信号。在固定 USB-C hub sanity analysis 中，这导致了较大的排名扰动：

- Retrieval Top-10 在 V1 Final Top-10 中保留：**4/10**
- V1 Final Top-10 的平均 retrieval rank：**25.0**

V2 首先将重排序范围限制在 retrieval relevance pool 中。对应分析使用大小为 20 的 pool：

- Retrieval Top-10 在 V2 Final Top-10 中保留：**10/10**
- V2 Final Top-10 的平均 retrieval rank：**5.5**

这一改善部分来自 V2 的结构设计：relevance pool 之外的商品无法进入最终排名。结果表明 V2 对该诊断 query 提供了更受控的相关性保护，但不能据此宣称它对所有 query 或业务目标都具有普遍优势。

## 10. 项目结构

```text
AgentCode/
├── scripts/
│   ├── 01-16                     # 离线数据、embedding、索引与评估
│   ├── 17_retrieval_service.py
│   ├── 18_product_repository.py
│   ├── 19_recommendation_service.py       # V1 baseline
│   ├── 19_recommendation_service_v2.py    # 当前 relevance-first baseline
│   ├── 19_analyze_reranking*.py
│   ├── 20_rag_context_builder.py
│   ├── 21_agent_tools.py
│   ├── 22_agent.py
│   ├── 23_llm_adapter.py
│   └── 24_fastapi_backend.py
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/                         # 本地数据和生成产物，不提交 Git
├── models/                       # 本地模型权重，不提交 Git
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── README_zh.md
```

编号文件目前保留原位，以维持已实现 pipeline。离线脚本、runtime services 和应用组件仅在职责上区分，尚未迁移到独立 Python package。

## 11. 依赖与已验证环境

AgentRec 使用双环境开发方式。

### 开发环境

- Windows
- VS Code / Codex
- 用于代码编辑、Git 和静态检查
- 不要求保存完整数据集、模型、embeddings 或 FAISS artifacts

### 已验证 Runtime 环境

- Ubuntu
- Python 3.12
- NVIDIA GeForce RTX 4090 D
- PyTorch 2.12.1+cu132
- torchvision 0.27.1+cu132
- FlagEmbedding 1.4.2
- faiss-cpu 1.15.0
- sentence-transformers 6.0.0
- transformers 5.16.1
- FastAPI 0.141.1
- Uvicorn 0.52.4
- Pydantic 2.13.5
- OpenAI SDK 3.7.0

BGE-M3、embedding、FAISS、Recommendation、RAG、Agent、FastAPI 和 Web Demo 的完整链路已经在 Ubuntu GPU 环境中验证。不声明轻量 Windows 开发工作区可以直接运行完整 GPU pipeline。

## 12. 安装

在目标 runtime 机器上创建并激活 Python 3.12 环境。

必须在项目 requirements 之前单独安装 PyTorch。请按照 PyTorch 官方安装指南，为当前机器的 CUDA 环境选择合适版本。已验证服务器使用 PyTorch `2.12.1+cu132`；项目有意不在 `requirements.txt` 中固定这一平台相关构建。

安装适合当前 CUDA 环境的 PyTorch 后，执行：

```bash
pip install -r requirements.txt
```

运行 GPU pipeline 前，应确认 PyTorch 能够访问目标 CUDA device。

## 13. 数据集与模型准备

AgentRec 使用：

- **数据集：** Amazon Reviews'23 Electronics
- **Embedding 模型：** BAAI/bge-m3

Git 仓库不包含：

- Amazon 原始 reviews 或 metadata；
- 处理后的交互数据集；
- Product Documents；
- BGE-M3 模型权重；
- 生成的商品 embeddings；
- FAISS index 或 evaluation artifacts。

运行离线 pipeline 前，需要将数据集和本地模型放置到脚本预期的位置。数据集与模型的 license 和使用条款请参考各自上游来源。

## 14. Artifact 布局

正式离线 pipeline 在 `data/processed/rag/` 下生成：

```text
product_documents_v2.jsonl
product_embeddings.npy
product_ids.json
product_embeddings.meta.json
product_embeddings.checkpoint.json
product_embeddings.faiss
faiss_index.meta.json
```

在线 retrieval/recommendation stack 直接加载：

- `product_documents_v2.jsonl`
- `product_embeddings.faiss`
- `product_ids.json`
- `faiss_index.meta.json`

Embedding matrix、embedding metadata 和 completed checkpoint 属于经过验证的构建与 provenance artifact，由离线验证和索引构建阶段使用。所有生成产物都不应提交 Git。

本地 BGE-M3 模型预期位于：

```text
models/bge-m3/
```

## 15. 启动系统

AgentRec 从 process environment 读取 DeepSeek API key：

```bash
export DEEPSEEK_API_KEY="..."
```

不要提交真实 key。`.env.example` 仅用于说明配置：应用没有使用 `python-dotenv`，不会自动加载该文件。

在仓库根目录启动 FastAPI：

```bash
uvicorn scripts.24_fastapi_backend:app --host 0.0.0.0 --port 8000
```

应用启动时会加载本地模型、FAISS index、product IDs 和 Product Documents。

启动成功后：

- Web Demo：`http://localhost:8000/`
- Swagger UI：`http://localhost:8000/docs`
- Health endpoint：`GET http://localhost:8000/health`

Web Demo 和 API 通过相同 origin 和端口提供服务。

## 16. API 示例

### Request

```http
POST /api/chat
Content-Type: application/json
```

```json
{
  "message": "Recommend a USB-C hub under 50 dollars with HDMI and ethernet."
}
```

### 紧凑 Response Contract

```json
{
  "answer": "Catalog-grounded recommendation text.",
  "steps": 2,
  "tool_call_count": 1,
  "tool_calls": [
    {
      "step": 1,
      "tool_name": "recommend_products",
      "ok": true
    }
  ]
}
```

公开 chat endpoint 不返回完整 recommendation objects、原始商品文档、retrieval scores 或内部 tool payloads。

## 17. Web Demo

浏览器 Demo 使用原生 HTML、CSS 和 JavaScript 实现，提供：

- 对 `/api/chat` 的同源请求；
- 前端代码中不包含 API key 或 provider credentials；
- 通过 `textContent` 安全地以纯文本形式渲染回答；
- Enter 发送、Shift+Enter 换行；
- loading 状态和友好错误提示；
- 确定性的示例购物问题；
- 由 FastAPI 提供的响应式单页面界面。

Assistant 回答支持一个有意受限的 Markdown 子集：段落和换行、粗体、无序列表、有序列表，以及一级至三级标题。Formatter 不使用 `innerHTML`；它只创建 allowlist 内的元素，并通过 text node 或 `textContent` 插入全部 LLM 内容。

## 18. 安全说明

- `DEEPSEEK_API_KEY` 仅由服务器进程读取。
- API key 不会发送到或保存在浏览器应用中。
- Git 忽略 `.env` 和 `.env.*`，`.env.example` 不包含真实 key。
- Git 忽略生成的数据、模型、embeddings 和 FAISS artifacts。
- 当前 Web Demo/API 没有 authentication 或 rate limiting。
- 后端会串行访问共享的同步 Agent/retrieval runtime，但它不是生产级并发或分布式 serving 架构。
- 在缺少认证、限流、传输安全、请求控制、监控和 secret management 时，不应将 Demo 直接暴露到公网。

## 19. 已知限制

- 每个 chat request 都是无状态、单轮请求；没有持久化 conversation memory。
- 只有价格、平均评分和评分数量是显式结构化硬约束。
- HDMI、Ethernet、codec、尺寸或接口类型等任意 feature 约束依赖语义检索和 RAG 推理，不是保证满足的 hard filter。
- Recommendation Service V2 是确定性 heuristic reranker，不是 learned ranker。
- 当前正式 artifacts 固定假设 125,762 个商品、1,024 维 BGE-M3 dense vectors 和 `IndexFlatIP` inner-product search。
- 项目 pipeline 不对商品 embeddings 执行额外 L2 normalization。
- Runtime 面向 GPU，BGE-M3 inference 当前固定使用 `cuda:0`。
- 应用没有生产级 authentication、rate limiting、水平扩展或分布式模型 serving。
- 共享模型状态通过串行 Agent execution 保护，因此请求并发能力有限。
- Web Demo 的 Markdown formatter 有意保持受限：不支持 links、images、attributes、raw HTML 或其他 Markdown 功能。
- Controlled known-item retrieval metrics 不能替代人工相关性或推荐质量评估。

## 20. Roadmap

- 提取并严格验证结构化商品 feature constraints。
- 基于人工或任务级评价研究 learned reranking。
- 增加可选的多轮 conversation memory。
- 在不启用 raw HTML 的前提下，评估其他可安全加入 allowlist 的 Markdown 功能。
- 增加生产级 authentication、rate limiting、observability 和部署控制。
- 将 runtime components 渐进迁移到常规可 import 的 Python package。
- 增加无需加载完整模型栈的自动化 contract、unit 和 API tests。

## 21. 致谢

AgentRec 基于以下项目和资源构建：

- Amazon Reviews'23
- BAAI BGE-M3
- FAISS
- FastAPI
- DeepSeek API

权威文档、引用格式、license 和使用条款请参考各上游项目或数据集。
