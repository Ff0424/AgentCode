# AgentRec

**LLM Agent + RAG + Recommendation system for electronics shopping decisions.**

## 1. Overview

AgentRec is a catalog-grounded shopping decision assistant built on the Amazon Reviews'23 Electronics domain. It turns product metadata into retrieval-ready Product Documents V2, retrieves candidates with BGE-M3 dense embeddings and exact FAISS inner-product search, applies explicit metadata constraints and transparent heuristic reranking, and gives an LLM compact product evidence for tool-grounded answers.

The end-to-end system combines:

- 125,762 validated Product Documents V2;
- BGE-M3 dense semantic retrieval;
- exact `faiss.IndexFlatIP` vector search;
- price, rating, and rating-count filtering;
- relevance-first multi-signal reranking;
- deterministic RAG context construction;
- a provider-neutral bounded Agent loop;
- a DeepSeek OpenAI-compatible adapter;
- a FastAPI backend and same-origin Web Demo.

AgentRec is not a general-purpose chatbot. Its answers are grounded in the electronics catalog exposed through its retrieval and product tools.

## 2. Key Features

- **Product corpus construction:** builds compact Product Documents V2 from Amazon Electronics metadata using field priorities and a 2,048-token budget.
- **Dense semantic retrieval:** encodes queries with the local BGE-M3 model and retrieves products from a 1,024-dimensional embedding space.
- **Exact vector search:** uses `faiss.IndexFlatIP`, not an approximate-nearest-neighbor index.
- **Structured hard constraints:** supports `max_price`, `min_rating`, and `min_rating_count` when these values are supplied to the recommendation tool.
- **Relevance-first reranking:** limits reranking to a semantic relevance pool, then combines semantic relevance, rating, and popularity.
- **RAG context minimization:** formats ranked product facts under deterministic field and character budgets.
- **Provider-neutral Agent orchestration:** separates tool-loop behavior from the concrete LLM provider.
- **DeepSeek tool calling:** translates the internal tool-call protocol to the OpenAI-compatible provider format.
- **Runtime services:** exposes reusable retrieval, repository, recommendation, RAG, and Agent components.
- **Web application:** serves a lightweight HTML/CSS/JavaScript interface from the FastAPI origin.

Preferences such as HDMI, Ethernet, battery life, or noise cancellation are currently handled as semantic retrieval preferences and LLM-grounded evidence. They are not guaranteed structured hard filters.

## 3. Architecture

```text
                                  ONLINE RUNTIME

User
  -> Web Demo
  -> FastAPI /api/chat
  -> Agent
  -> Agent Tools
  -> Recommendation Service V2
       -> BGE-M3 query embedding
       -> FAISS IndexFlatIP exact retrieval
       -> Product Repository lookup
       -> hard metadata filtering
       -> relevance-first heuristic reranking
  -> RAG Context Builder
  -> DeepSeek through the OpenAI-compatible adapter
  -> grounded answer

                                  OFFLINE PIPELINE

Amazon Reviews'23 Electronics
  -> interaction preprocessing and 10-core selection
  -> Product Documents V2
  -> BGE-M3 product embeddings
  -> embedding validation
  -> FAISS IndexFlatIP construction
  -> retrieval evaluation
  -> runtime artifacts
```

The offline pipeline creates and validates immutable retrieval artifacts. The online runtime loads those artifacts once and does not rebuild product embeddings or the FAISS index per request.

## 4. Offline Pipeline

The numbered scripts cover the following stages:

1. Stream and inspect Amazon Electronics reviews and metadata.
2. Prepare interactions, compare K-core configurations, and build the selected 10-core dataset.
3. Analyze product metadata and construct Product Documents V2 with prioritized fields and a 2,048-token limit.
4. Benchmark BGE-M3 batch sizes, generate float32 product embeddings, and validate row alignment, fingerprints, shape, dtype, finite values, and norms.
5. Build a row-preserving `faiss.IndexFlatIP(1024)` index and verify its serialized vectors by reconstruction.
6. Evaluate title-query and feature-query retrieval with deterministic sampling and case-level outputs.

The formal product embedding configuration uses local BGE-M3 inference on `cuda:0`, FP16 inference, `max_length=2048`, and `batch_size=8`. Stored product embeddings are float32, and the pipeline does not apply additional L2 normalization.

## 5. Runtime Pipeline

For each chat request, the runtime follows this flow:

```text
shopping query
  -> semantic candidate retrieval
  -> product lookup
  -> optional hard metadata constraints
  -> relevance-pool selection
  -> deterministic heuristic reranking
  -> compact RAG context
  -> bounded LLM tool loop
  -> final catalog-grounded answer
```

The retrieval service maps each FAISS row back to the row-aligned product ID. The product repository loads Product Documents V2 once at initialization and serves structured product facts without rescanning the JSONL file for every lookup.

## 6. Recommendation Service

`RecommendationServiceV2` is the current recommendation baseline. Its pipeline is deliberately relevance-first:

1. Retrieve `candidate_k` semantic candidates.
2. Fetch the corresponding product records while preserving retrieval order.
3. Apply optional hard constraints:
   - `max_price`
   - `min_rating`
   - `min_rating_count`
4. Select at most `rerank_pool_k` products from the highest-ranked remaining retrieval candidates.
5. Normalize raw semantic scores within that relevance pool.
6. Compute rating and popularity scores.
7. Produce a deterministic weighted score and return the final Top-K.

The score is:

```text
final_score =
    0.70 * normalized_semantic_score
  + 0.20 * rating_score
  + 0.10 * popularity_score
```

`rating_score` is the clipped average rating divided by five. `popularity_score` is a min-max normalized `log1p(rating_number)` value within the current relevance pool. Deterministic tie-breaking uses semantic score, raw retrieval score, retrieval rank, and product ID.

This is a transparent heuristic reranking baseline, not a learned ranker. Missing metadata is excluded when its corresponding hard constraint is active; otherwise missing rating/popularity signals receive the defined fallback scores.

## 7. RAG and Agent Design

The Agent layer is separated into focused components:

- **RAG Context Builder** (`20_rag_context_builder.py`): converts ranked recommendations into deterministic, fact-only context. It limits total context size, feature count, product details, and description length while preserving ranked-prefix order.
- **Agent Tools** (`21_agent_tools.py`): exposes three structured tools:
  - `recommend_products`
  - `get_product_details`
  - `compare_products`
- **Agent orchestration** (`22_agent.py`): implements a provider-neutral, bounded tool loop with validated tool calls and explicit failure handling.
- **LLM adapter** (`23_llm_adapter.py`): converts provider-neutral messages and tool calls into the OpenAI-compatible format used by DeepSeek.

Within one Agent run, complete tool results are retained in the in-memory tool history for debugging and backend processing. A compact projection is sent back to the LLM so that raw document text and unnecessary retrieval fields do not consume model context. The HTTP response exposes only a compact tool-call summary.

Each `/api/chat` request starts a fresh Agent run. Tool history is not persisted across requests, so the current application does not provide multi-turn conversation memory.

## 8. Retrieval Evaluation

The validated retrieval corpus contains **125,762 Product Documents V2**. Product and query embeddings use BGE-M3 dense vectors with dimension **1,024**, searched by exact inner product through `IndexFlatIP`.

The controlled benchmark uses a fixed seed of 42 and samples 2,000 products for each query type:

- title queries derived from the product title;
- deterministic shortened feature queries derived from product features.

| Query Type | Recall\@1 | Recall\@5 | Recall\@10 | Recall\@20 |   MRR\@20 |
| ---------- | --------: | --------: | ---------: | ---------: | --------: |
| Title      |    0.9255 |    0.9935 |     0.9960 |     0.9980 | 0.9561996 |
| Feature    |    0.4950 |    0.7060 |     0.7660 |     0.8135 | 0.5869288 |

This is a controlled known-item retrieval benchmark using weak supervision/pseudo ground truth: the product from which a query was derived is treated as the target. A missed target does not imply that every retrieved alternative is irrelevant, and these metrics should not be interpreted as human-judged recommendation quality.

## 9. Reranking Analysis

The V1 baseline applied rating and popularity signals across the full candidate set. In the fixed USB-C hub sanity analysis, this caused large rank perturbations:

- Retrieval Top-10 retained in V1 Final Top-10: **4/10**
- Mean retrieval rank in V1 Final Top-10: **25.0**

V2 first restricts reranking to the retrieval relevance pool. With a pool of 20 in the corresponding analysis:

- Retrieval Top-10 retained in V2 Final Top-10: **10/10**
- Mean retrieval rank in V2 Final Top-10: **5.5**

This improvement is partly structural by design: products outside the selected relevance pool cannot enter the final ranking. It demonstrates more controlled relevance preservation for this diagnostic query, not universal superiority across every query or business objective.

## 10. Project Structure

```text
AgentCode/
├── scripts/
│   ├── 01-16                     # offline data, embedding, index, and evaluation
│   ├── 17_retrieval_service.py
│   ├── 18_product_repository.py
│   ├── 19_recommendation_service.py       # V1 baseline
│   ├── 19_recommendation_service_v2.py    # current relevance-first baseline
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
├── data/                         # local data and generated artifacts; not committed
├── models/                       # local model weights; not committed
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The numbered files remain in place to preserve the implemented pipeline. Offline scripts, runtime services, and application components are separated conceptually but have not yet been migrated into a Python package.

## 11. Requirements and Validated Environment

AgentRec uses a two-environment workflow.

### Development environment

- Windows
- VS Code / Codex
- source editing, Git, and static checks
- no requirement to store the complete dataset, model, embeddings, or FAISS artifacts

### Validated runtime environment

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

The complete BGE-M3, embedding, FAISS, recommendation, RAG, Agent, FastAPI, and Web Demo flow has been validated in the Ubuntu GPU environment. Full GPU execution is not claimed for the lightweight Windows development workspace.

## 12. Installation

Create and activate a Python 3.12 environment on the target runtime machine.

PyTorch must be installed separately before the project requirements. Choose the build appropriate for the machine's CUDA environment by following the official PyTorch installation guide. The validated server uses PyTorch `2.12.1+cu132`; the project intentionally does not pin this platform-specific build in `requirements.txt`.

After installing the appropriate PyTorch build:

```bash
pip install -r requirements.txt
```

Before running the GPU pipeline, verify that PyTorch can access the intended CUDA device.

## 13. Dataset and Model Setup

AgentRec uses:

- **Dataset:** Amazon Reviews'23 Electronics
- **Embedding model:** BAAI/bge-m3

The Git repository does not include:

- Amazon raw reviews or metadata;
- processed interaction datasets;
- Product Documents;
- BGE-M3 model weights;
- generated product embeddings;
- FAISS indexes or evaluation artifacts.

Place the dataset and local model in the paths expected by the scripts before running the offline pipeline. Refer to the upstream dataset and model sources for their licenses and usage terms.

## 14. Artifact Layout

The formal offline pipeline produces the following files under `data/processed/rag/`:

```text
product_documents_v2.jsonl
product_embeddings.npy
product_ids.json
product_embeddings.meta.json
product_embeddings.checkpoint.json
product_embeddings.faiss
faiss_index.meta.json
```

The online retrieval/recommendation stack directly loads:

- `product_documents_v2.jsonl`
- `product_embeddings.faiss`
- `product_ids.json`
- `faiss_index.meta.json`

The embedding matrix, embedding metadata, and completed checkpoint remain part of the validated build/provenance set and are used by the offline validation and index-building stages. None of these generated artifacts should be committed to Git.

The local BGE-M3 model is expected under:

```text
models/bge-m3/
```

## 15. Running the System

AgentRec reads the DeepSeek API key from the process environment:

```bash
export DEEPSEEK_API_KEY="..."
```

Do not commit the real key. `.env.example` is documentation only: the application does not use `python-dotenv` and does not automatically load that file.

From the repository root, start the FastAPI application:

```bash
uvicorn scripts.24_fastapi_backend:app --host 0.0.0.0 --port 8000
```

The runtime loads the local model, FAISS index, product IDs, and Product Documents during application startup.

Once startup succeeds:

- Web Demo: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Health endpoint: `GET http://localhost:8000/health`

The Web Demo and API share the same origin and port.

## 16. API Example

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

### Compact response contract

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

Full recommendation objects, raw product documents, retrieval scores, and internal tool payloads are not returned by the public chat endpoint.

## 17. Web Demo

The browser demo is implemented with plain HTML, CSS, and JavaScript. It provides:

- same-origin calls to `/api/chat`;
- no API key or provider credentials in frontend code;
- safe plain-text response rendering through `textContent`;
- Enter to send and Shift+Enter for a new line;
- loading and friendly error states;
- deterministic example shopping prompts;
- a responsive single-page interface served by FastAPI.

Assistant answers support a deliberately limited Markdown subset: paragraphs and line breaks, bold text, unordered and ordered lists, and level 1-3 headings. The formatter does not use `innerHTML`; it creates allowlisted elements and inserts all LLM content through text nodes or `textContent`.

## 18. Security Notes

- `DEEPSEEK_API_KEY` is read only by the server-side process.
- The API key is never sent to or stored in the browser application.
- `.env` and `.env.*` are ignored by Git, while `.env.example` contains no real key.
- Generated data, models, embeddings, and FAISS artifacts are ignored by Git.
- The current Web Demo/API has no authentication or rate limiting.
- The backend serializes access to its shared synchronous Agent/retrieval runtime, but it is not designed as a production concurrency or distributed-serving architecture.
- Do not expose the demo directly to the public internet without authentication, rate limits, transport security, request controls, monitoring, and secret management.

## 19. Known Limitations

- Each chat request is stateless and single-turn; there is no persistent conversation memory.
- Only price, average rating, and rating count are explicit structured hard constraints.
- Arbitrary feature constraints such as HDMI, Ethernet, codec support, dimensions, or connector types are semantic/RAG-grounded preferences rather than guaranteed hard filters.
- Recommendation Service V2 is a deterministic heuristic reranker, not a learned ranker.
- The current formal artifacts assume 125,762 products, 1,024-dimensional BGE-M3 dense vectors, and `IndexFlatIP` inner-product search.
- Product embeddings are not additionally L2-normalized by the project pipeline.
- The runtime is GPU-oriented and currently targets `cuda:0` for BGE-M3 inference.
- The application has no production authentication, rate limiting, horizontal-scaling design, or distributed model serving.
- Shared model state is protected through serialized Agent execution, which limits request concurrency.
- The Web Demo Markdown formatter is intentionally limited: links, images, attributes, raw HTML, and other Markdown features are not supported.
- Controlled known-item retrieval metrics are not a substitute for human relevance or recommendation-quality evaluation.

## 20. Roadmap

- Extract structured product-feature constraints with explicit validation.
- Evaluate and train learned reranking alternatives against human or task-level judgments.
- Add opt-in multi-turn conversation memory.
- Evaluate additional safely allowlisted Markdown features without enabling raw HTML.
- Introduce production authentication, rate limiting, observability, and deployment controls.
- Gradually move runtime components into a conventional importable Python package.
- Add automated contract, unit, and API tests that do not require loading the full model stack.

## 21. Acknowledgements

AgentRec builds on:

- Amazon Reviews'23
- BAAI BGE-M3
- FAISS
- FastAPI
- DeepSeek API

Refer to each upstream project or dataset for its authoritative documentation, citation guidance, license, and terms of use.
