# RAGCore — rigorously evaluated multi-strategy RAG

A retrieval-augmented generation system with hybrid FAISS + BM25 retrieval,
automatic query routing, and a configurable Self-RAG generation path.
The pipeline is benchmarked end-to-end against FiQA-2018, with results,
failure mode documentation, and reproducible scripts committed alongside
the code.

[![CI](https://github.com/Kollipara-Hema/ragcore/actions/workflows/ci.yml/badge.svg)](https://github.com/Kollipara-Hema/ragcore/actions)
[![Coverage](https://codecov.io/gh/Kollipara-Hema/ragcore/branch/main/graph/badge.svg)](https://codecov.io/gh/Kollipara-Hema/ragcore)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

---

## Table of Contents

- [Headline Results](#headline-results)
- [Architecture](#architecture)
- [Retrieval Strategy Routing](#retrieval-strategy-routing)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Status](#status)
- [Project Artifacts](#project-artifacts)
- [What's next](#whats-next)

---

## Headline Results

Benchmarked on 50 FiQA-2018 financial Q&A queries. Retrieval metrics are
identical between strategies; Self-RAG improves faithfulness at the cost of
latency.

| Metric | Baseline | Self-RAG | Delta |
|---|---|---|---|
| hit@5 | 0.92 | 0.92 | — |
| MRR | 0.86 | 0.86 | — |
| NDCG@5 | 0.75 | 0.75 | — |
| faithfulness | 0.36 | 0.43 | +20% |
| mean latency | 4.7s | 8.5s | +81% |

Self-RAG's claim verification loop costs 1.8× the latency of basic generation
for a 20% faithfulness gain; whether that trade-off is worth it depends on how
much hallucination matters for the use case. Full per-query breakdown:
[evaluation/notebooks/baseline_vs_selfrag.ipynb](evaluation/notebooks/baseline_vs_selfrag.ipynb)

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        A([Documents<br/>PDF · DOCX · TXT · HTML]) --> B[Load & Clean]
        B --> C[Chunk<br/>fixed · semantic · hierarchical · sentence]
        C --> D[Embed<br/>BGE · MiniLM]
        D --> E[(Hybrid Index)]
    end

    subgraph Index ["Hybrid Index"]
        E1[(FAISS<br/>Dense Vectors)]
        E2[(BM25<br/>Sparse Index)]
    end

    subgraph Query Pipeline
        G([User Query]) --> H[Query Router<br/>Heuristic → LLM fallback]
        H --> I[Retrieval Executor]
        I --> J[Reranker<br/>Cross-Encoder]
        J --> K[LLM Generator<br/>+ Citations]
        K --> L([Answer])
    end

    E --> E1
    E --> E2
    E1 --> I
    E2 --> I
```

---

## Retrieval Strategy Routing

The router runs in two passes: a fast regex heuristic catches obvious patterns
(lookup, analytical, multi-hop), then falls back to a cheap LLM call
(GPT-4o-mini, temp=0) for ambiguous queries.

```mermaid
flowchart LR
    Q([Query]) --> R{Heuristic<br/>Router}
    R -->|uncertain| L[LLM Classifier<br/>gpt-4o-mini]
    L --> S{Strategy<br/>Map}
    R --> S

    S -->|Factual| HY[Hybrid<br/>BM25 + FAISS · α=0.7]
    S -->|Lookup| KW[BM25 Keyword]
    S -->|Semantic| VE[Vector Search<br/>FAISS cosine]
    S -->|Multi-Hop| MQ[Multi-Query<br/>RRF fusion]
    S -->|Analytical| HY
    S -->|Comparative| MQ

    HY --> RE[Cross-Encoder<br/>Reranker]
    KW --> RE
    VE --> RE
    MQ --> RE
    RE --> G[Generator]
```

| Query Type | Primary | Fallback | Triggered by |
|-----------|---------|----------|-------------|
| Factual | Hybrid (α=0.7) | Semantic | "What is X?" |
| Lookup | BM25 Keyword | Semantic | "Find doc by author / date" |
| Semantic | Vector (FAISS) | Hybrid | "Explain how X relates to Y" |
| Multi-Hop | Multi-Query + RRF | Hybrid | "X which then affects Y" |
| Analytical | Hybrid (α=0.7) | Semantic | "Summarize / analyze / compare" |
| Comparative | Multi-Query + RRF | Hybrid | "Differences between X and Y" |

---

## Quick Start

### FAISS (built-in, no Docker required)

The fastest way to run locally. FAISS runs in-process; no separate vector
store service needed.

```bash
git clone https://github.com/Kollipara-Hema/ragcore.git
cd ragcore
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# Edit .env:
#   VECTOR_STORE_PROVIDER=faiss
#   LLM_PROVIDER=groq    # set GROQ_API_KEY, or use openai / anthropic

uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

Weaviate and Chroma configuration keys exist in `config/settings.py` but are
not verified end-to-end. FAISS is the only vector store tested against the
full pipeline.

### Full Stack with Docker Compose

API, Celery workers, and Redis in one command.

```bash
cp .env.example .env
# Edit .env with your LLM API key and VECTOR_STORE_PROVIDER=faiss

docker-compose up --build
# API:    http://localhost:8000
# Docs:   http://localhost:8000/docs
# Flower: http://localhost:5555
```

### UI Frontends

```bash
# Streamlit — dashboard with file upload
cd ui_streamlit && streamlit run app.py

# Chainlit — conversational chat
cd ui_chainlit && chainlit run app.py
```

---

## API Reference

### Ingest

```bash
# Async (default) — returns job_id immediately, processes in background via Celery
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf" \
  -F "title=Q3 Report 2024" \
  -F "tags=finance,quarterly"

# Check job status
curl http://localhost:8000/ingest/status/<job_id>

# Synchronous — waits for completion (use for small files)
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@notes.txt" \
  -F "async_processing=false"
```

### Query

```bash
# Auto-routed query (system picks the best strategy)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the key findings in the Q3 report?", "top_k": 5}'

# With strategy override and metadata filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return policy?",
    "metadata_filter": {"doc_type": "pdf"},
    "strategy_override": "keyword"
  }'

# Streaming response (SSE)
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the main points."}' \
  --no-buffer
```

---

## Configuration

All settings are environment-variable driven. Copy `.env.example` → `.env`.

### Retrieval

| Setting | Default | Options / Effect |
|---------|---------|-----------------|
| `VECTOR_STORE_PROVIDER` | `faiss` | `weaviate` · `faiss` · `chroma` |
| `CHUNKING_STRATEGY` | `semantic` | `fixed` · `semantic` · `hierarchical` · `sentence` |
| `HYBRID_ALPHA` | `0.7` | `0` = keyword only · `1` = vector only |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Final chunks sent to LLM |
| `ENABLE_RERANKING` | `true` | Two-stage retrieval with cross-encoder |
| `ENABLE_QUERY_EXPANSION` | `false` | Multi-query paraphrasing for complex questions |

### Generation

| Setting | Default | Options / Effect |
|---------|---------|-----------------|
| `LLM_PROVIDER` | `groq` | `groq` · `openai` · `anthropic` · `ollama` · `demo` |
| `GROQ_API_KEY` | — | Required if `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | — | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | Azure API version |
| `GENERATION_STRATEGY` | `basic` | Options: `basic`, `self_rag`, `flare`, `agentic`. Self-RAG adds claim verification. |

### Evaluation

| Setting | Default | Effect |
|---------|---------|--------|
| `ENABLE_EVALUATION` | `false` | LLM-based confidence scoring per answer |
| `EVAL_STRATEGY` | `heuristic` | `heuristic` · `ragas` (requires `pip install -e ".[eval]"`) |
| `RAGAS_ENABLED` | `false` | Enable RAGAS faithfulness + relevance metrics |

### Infrastructure

| Setting | Default | Effect |
|---------|---------|--------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for Celery + long-term memory |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate host |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local Chroma persistence directory |
| `ENABLE_TRACING` | `false` | Send traces to Langfuse |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |

---

## Project Structure

<details>
<summary>Expand directory tree</summary>

```
ragcore/
├── agent/                  # LangGraph agent
│   ├── graph.py            # Node wiring and conditional edges
│   ├── state.py            # AgentState TypedDict
│   └── nodes/              # router · retriever · reranker · generator · evaluator
│
├── api/                    # FastAPI application
│   └── main.py             # /ingest · /query · /query/stream · /agent/query · /health
│
├── config/
│   └── settings.py         # Pydantic settings — all env vars with defaults
│
├── embeddings/             # BGEEmbedder · MiniLMEmbedder · embedder factory
│
├── evaluation/
│   ├── evaluator.py        # Retrieval + generation metrics (MRR, NDCG, hit rate)
│   ├── datasets/           # FiQA-2018 eval set (50 queries, seed=42)
│   ├── notebooks/          # baseline_vs_selfrag.ipynb
│   ├── results/            # Raw JSON benchmark output (basic_fiqa.json, self_rag_fiqa.json)
│   └── scripts/            # run_benchmark.py · stage_b_sanity.py
│
├── generation/
│   ├── llm_service.py      # Provider switcher: Groq / OpenAI / Anthropic / Ollama
│   ├── advanced_generation.py  # SelfRAGGenerator · FLAREGenerator (not wired to API)
│   └── prompts/            # Prompt templates
│
├── ingestion/
│   ├── chunkers/           # fixed · semantic · hierarchical · sentence
│   └── loaders/            # PDF · DOCX · TXT · HTML · GitHub · web
│
├── monitoring/
│   └── tracer.py           # NoOpTracer (default) · LangfuseTracer (optional)
│
├── reranking/
│   └── reranker.py         # CrossEncoderReranker · NoOpReranker
│
├── retrieval/
│   ├── router/
│   │   └── query_router.py # HeuristicRouter · LLMQueryClassifier · STRATEGY_MAP
│   └── strategies/
│       └── retrieval_executor.py
│
├── tests/
│   ├── unit/               # 75 tests — no external services required
│   └── integration/        # 25 tests — FAISS in-memory, mocked external services
│
├── docs/
│   └── debugging-notes.md
│
├── utils/
│   └── models.py           # Shared types: Chunk · Document · RetrievedChunk
│
└── vectorstore/
    └── vector_store.py     # FAISSVectorStore · get_vector_store() singleton
```

</details>

---

## Testing

```bash
# Unit tests only (no external services needed)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/unit/ --cov=. --cov-report=html
```

Current: 75 unit tests passing, 25 integration tests passing. Two tests in
`TestObservabilityIntegration` fail on scaffolded tracer code — pre-existing,
not blocking.

---

## Status

### Working end-to-end

- Document ingestion: PDF, DOCX, text, HTML, web, GitHub via 8 loaders and 4
  chunking strategies
- Hybrid retrieval: FAISS dense + BM25 sparse with configurable alpha fusion
- Cross-encoder reranking
- Query routing: heuristic regex + LLM fallback, dispatches to 5 of 6
  retrieval strategies (parent-child is a known stub)
- Basic and Self-RAG generation paths, configurable via `GENERATION_STRATEGY`
- Retrieval evaluation: MRR, NDCG@5, hit@5, precision@5, recall@5 — all
  correctly bounded after dedup fix (see debugging notes)
- FiQA-2018 benchmark runner and comparison notebook

### Deferred or scaffolded

- Self-RAG claim verification hardcodes `gpt-4o-mini` for the verification
  step — works if `OPENAI_API_KEY` is set, not provider-agnostic
- Monitoring: `NoOpTracer` methods are all `pass`; Langfuse tracing is off by
  default; Prometheus/Grafana appear in `docker-compose.yml` but the
  application emits no metrics
- Multi-vector-store: Weaviate/Chroma/Pinecone configs exist; only FAISS is
  verified end-to-end
- Advanced generation: FLARE and Agentic RAG code is in
  `generation/advanced_generation.py` but not wired to the API
- RAGAS-based generation metrics: currently heuristic word-overlap; install
  the `eval` extra to enable RAGAS

---

## Project Artifacts

- [AUDIT.md](AUDIT.md) — Module-by-module assessment of what's implemented
  versus what's scaffolded or deferred, with README claims rated REAL / PARTIAL /
  OVERSTATED and five previously-undocumented bugs.
- [docs/debugging-notes.md](docs/debugging-notes.md) — Real bugs caught
  during development: vector store singleton, UUID/corpus-ID mismatch, metric
  inflation from duplicate chunk IDs, Self-RAG prompt escaping. Each entry
  includes symptom, root cause, fix, and regression test.
- [evaluation/notebooks/baseline_vs_selfrag.ipynb](evaluation/notebooks/baseline_vs_selfrag.ipynb) —
  Baseline vs Self-RAG across 50 FiQA queries. Per-query faithfulness delta,
  latency and token histograms, Self-RAG internals (regen rate, claim counts).
- [evaluation/results/](evaluation/results/) — Raw benchmark JSON for both
  strategies. Reproducible:
  `python evaluation/scripts/run_benchmark.py --strategy basic`

---

## What's next

- Wire FLARE and Agentic RAG into the generation strategy dispatch (code
  exists in `advanced_generation.py`, not yet called by orchestrator)
- Make Self-RAG claim verification provider-agnostic
- Emit real Prometheus metrics from retrieval and generation steps
- Fix the two scaffolded tracer integration tests
- Expand FiQA evaluation to the full test split to reduce variance in
  faithfulness estimates
- **Deploy a live demo** (Streamlit Cloud is the likely path)
