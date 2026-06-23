# RAGCore — rigorously evaluated multi-strategy RAG

A retrieval-augmented generation system with hybrid FAISS + BM25 retrieval,
automatic query routing, and a configurable Self-RAG generation path.
The pipeline is benchmarked end-to-end against FiQA-2018, with results,
failure mode documentation, and reproducible scripts committed alongside
the code.
Two faithfulness metrics — token-overlap and LLM-judged (RAGAS) —
disagree on whether Self-RAG helps. Both deltas are statistically
significant. The disagreement, not Self-RAG itself, is the headline
finding.

[![CI](https://github.com/Kollipara-Hema/ragcore/actions/workflows/ci.yml/badge.svg)](https://github.com/Kollipara-Hema/ragcore/actions)
[![Coverage](https://codecov.io/gh/Kollipara-Hema/ragcore/branch/main/graph/badge.svg)](https://codecov.io/gh/Kollipara-Hema/ragcore)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

---

## Table of Contents

- [Live demo](#live-demo)
- [Headline Results](#headline-results)
- [Architecture](#architecture)
- [Retrieval Strategy Routing](#retrieval-strategy-routing)
- [Quick Start](#quick-start)
- [Monitoring](#monitoring)
- [API Reference](#api-reference)
- [Per-session uploads](#per-session-uploads)
- [Configuration](#configuration)
- [Security](#security)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Status](#status)
- [Project Artifacts](#project-artifacts)
- [Recent Updates](#recent-updates)
- [What's next](#whats-next)

---

## Live demo

**[ragcore.streamlit.app](https://ragcore.streamlit.app)** — try the deployed demo

Indexed on [FiQA-2018](https://huggingface.co/datasets/explodinggradients/fiqa) (380 chunks of personal-finance Q&A). Llama 3.3 70B via Groq for generation, FAISS + BM25 hybrid retrieval, ms-marco cross-encoder reranking. Optional Self-RAG verification toggle in the sidebar.

First request may take ~30s on free tier (Render cold start).

![RAGCore Demo](docs/assets/demo.png)

---
## Headline Results

Benchmarked on 50 FiQA-2018 financial Q&A queries. Retrieval metrics are
identical between strategies by construction (same retrieval path); the
meaningful comparison is faithfulness, where the two metrics disagree.

| Metric | Baseline | Self-RAG | Delta | Run range | p |
|---|---|---|---|---|---|
| hit@5 | 0.92 | 0.92 | — | — | — |
| MRR | 0.86 | 0.86 | — | — | — |
| NDCG@5 | 0.75 | 0.75 | — | — | — |
| faithfulness (word-overlap) | 0.36 | 0.42 | +0.054 | [0.032, 0.083] | 0.0002 |
| faithfulness (RAGAS, gpt-4o-mini) | 0.55 | 0.46 | −0.085 | [−0.115, −0.065] | 0.007–0.069 |
| mean latency | 5.9s | 10.6s | +79% | — | — |

The two faithfulness metrics give opposite verdicts. Word-overlap says
Self-RAG is more faithful (+15% relative). RAGAS says Self-RAG is less
faithful, with a mean delta of −0.085 across three judge-controlled
re-runs (range −0.115 to −0.065; Wilcoxon p straddles α=0.05 across
runs at 0.007, 0.049, 0.069). On baseline answers, the two metrics
negatively correlate (Spearman ρ = −0.385, p=0.006) — they are not
measuring the same thing.

The April analysis reported a single-run RAGAS delta of −0.109, within
the range observed across reruns above. The single point estimate
overstates the precision of the underlying instrument: RAGAS at n=50
has a per-query noise floor of ~0.07–0.09 mean absolute difference
between re-runs of identical data, comparable in magnitude to the
effect sizes being measured. The per-claim analysis (linked below)
characterizes this non-determinism and the structural disagreement
between Self-RAG's internal verifier and the RAGAS judge.

A plausible mechanism: Self-RAG decomposes answers into atomic claims,
which raises token overlap with contexts (helping word-overlap) but
creates more opportunities for any single claim to be flagged as
unsupported (hurting RAGAS). The per-claim analysis confirms a
structural finding: Pearson r between Self-RAG's internal verifier
and the RAGAS judge across paired queries is near zero (-0.12 to
-0.16, p > 0.3), and 34% of queries show internal-accept/RAGAS-reject
disagreement against only 3% in the reverse direction. The two
verifiers do not share a meaningful ranking signal even when they use
the same underlying model.

For the full statistical breakdown of the April run (paired bootstrap
CIs, Wilcoxon, Spearman cross-correlation), see
[evaluation/results/ragas_analysis_2026-04-26.md](evaluation/results/ragas_analysis_2026-04-26.md).
For the per-claim mechanism investigation, including the verifier-disagreement
finding and characterization of run-to-run instrument variance, see
[evaluation/results/per_claim_analysis_2026-05-05.md](evaluation/results/per_claim_analysis_2026-05-05.md).

The pre-registered analysis plan, written before the RAGAS run, is at
[docs/ragas_run_plan_2026-04-26.md](docs/ragas_run_plan_2026-04-26.md).

The methodology change made after seeing the first run (judge
max_tokens raised from default to 8192 to recover 14 verification-stage
truncations) is documented in the analysis file alongside the original
default-judge run preserved as evidence.

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        A([Documents<br/>PDF · TXT · DOCX · HTML · CSV]) --> B[Load & Clean]
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

    subgraph Monitoring
        K --> M1[Prometheus<br/>5 custom metrics + RED]
        I --> M1
        M1 --> M2[Grafana<br/>overview dashboard]
        K --> M3[Structured logs<br/>JSON + request_id]
    end

    E --> E1
    E --> E2
    E1 --> I
    E2 --> I
```
Citations include attributed_spans — the parser extracts the clause preceding each `<cite source="N">` trailing marker emitted by the LLM, allowing the UI to render inline yellow highlights with source chips.
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

Both FAISS and Chroma are verified end-to-end. Weaviate, Pinecone, and Qdrant
configuration keys exist but are not implemented; selecting them logs a warning
and falls back to FAISS.

### Full Stack with Docker Compose

API, Redis, Prometheus, and Grafana in one command. The compose file also
defines a Celery worker service from a removed async ingestion path; it
starts but receives no jobs from the HTTP API and is queued for cleanup.

```bash
cp .env.example .env
# Edit .env with your LLM API key and VECTOR_STORE_PROVIDER=faiss
# Also set REDIS_PASSWORD and GF_ADMIN_PASSWORD before running — the stack will start without them but defaults to placeholder credentials

docker-compose up --build
# API:        http://localhost:8000
# API docs:   http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

### UI Frontends

```bash
# Streamlit — dashboard with file upload
cd ui_streamlit && streamlit run app.py

# Chainlit — conversational chat
cd ui_chainlit && chainlit run app.py
```

---

## Monitoring

After `docker compose up`, Prometheus scrapes the API's `/metrics` endpoint every 15 seconds and is accessible at `http://localhost:9090`. The `ragcore_api` scrape target should show as **UP** within one scrape interval; verify at `http://localhost:9090/targets`.

Grafana is available at `http://localhost:3000`; log in with the `GF_ADMIN_PASSWORD` you set in `.env` (defaults to `CHANGE_ME_IN_PROD` if unset — see [Security](#security)). The **RAGCore Overview** dashboard is preloaded under Dashboards → RAGCore and displays request rate, p95 latency, stage durations, token usage, Self-RAG claim outcomes, process memory, and vector store disk usage.

![RAGCore Grafana Overview](docs/assets/grafana_overview.png)

---

## API Reference

### Ingest

Uploads land in a per-session corpus, gated by the `X-Session-Id` request header.
The first call without that header mints a new session token, returned in the
`X-Session-Id` response header; subsequent calls send the token back to add to
the same session corpus. Indexing is synchronous and blocks until the file is
embedded and persisted.

```bash
# First call — mints a session. -i prints headers so you can copy X-Session-Id.
curl -i -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf" \
  -F "title=Q3 Report 2024" \
  -F "tags=finance,quarterly"

# Subsequent uploads — reuse the token from the previous response.
curl -X POST http://localhost:8000/ingest/file \
  -H "X-Session-Id: <token>" \
  -F "file=@notes.txt"

# Raw text into the same session.
curl -X POST http://localhost:8000/ingest/text \
  -H "X-Session-Id: <token>" \
  -H "Content-Type: application/json" \
  -d '{"text_content": "Q3 revenue was $4.2B...", "metadata": {"title": "Q3 highlights"}}'
```

Accepted types: PDF (up to 100 pages by default) and plain text (TXT/MD). File
type is decided by magic-byte sniff — the client's `Content-Type` and filename
are ignored. Anything else returns 415. Per-session file/byte caps and an
idle-TTL eviction loop bound storage on small deploys. Stale-token behavior is
asymmetric by design: `/query` and `/retrieve` return 404 on an unknown or
expired session token because header presence commits the request to the
session path and the read path must not silently fall back to public corpora
(it would answer "what does my document say?" with chunks the user never
uploaded). `/ingest/*` is permissive instead — an absent or unknown token
mints a fresh session, because replacing a stale write-token leaks nothing.
Queries that carry a valid `X-Session-Id` are answered from that session's
corpus instead of the public FiQA index.

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

### Retrieve

`POST /retrieve` runs the router → retrieve → rerank pipeline without
generation, returning the ranked chunks and their scores. Useful for comparing
strategies, building custom downstream generators, or inspecting retrieval
quality without LLM cost.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy?", "top_k": 5}'
```

### Corpora

`GET /corpora` returns the registered corpus names and their `doc_count`
values. Clients select a corpus by name via the `corpus` field on `/query`
and `/retrieve`; the FiQA `default` corpus is selected when the field is
omitted.

```bash
curl http://localhost:8000/corpora
```

### Other endpoints

- `DELETE /documents/{doc_id}` — remove all chunks of a document from the
  active corpus.
- `GET /health`, `GET /health/live`, `GET /health/ready` — basic, liveness,
  and readiness probes. The last runs a vector-store ping, an embedder
  smoke test, and an LLM API-key presence check; returns 200 all-pass or
  503 with per-check failure reasons.
- `GET /metrics` — Prometheus scrape endpoint (5 custom `ragcore_*` metrics
  plus RED metrics on every route; see [Monitoring](#monitoring)).
- `POST /agent/query`, `GET /agent/trace/{trace_id}`, `GET /trace/{trace_id}` —
  LangGraph agent path. Status PARTIAL per [AUDIT.md](AUDIT.md); the trace
  endpoints have known security limitations documented there.

---

## Per-session uploads

The `/ingest/file` and `/ingest/text` endpoints write into a private corpus
scoped to a session token, not into the public FiQA index. The token is
minted by the server on the first call (returned as the `X-Session-Id`
response header) and supplied by the client on subsequent calls to add to
the same corpus.

### Isolation invariant

A request that carries an `X-Session-Id` reads or writes ONLY that session's
corpus. The body's `corpus` field is ignored on the session path. The same
isolation contract is threaded through the Self-RAG and FLARE mid-generation
re-retrieval paths so a verifier or iterative loop can't pull public-corpus
chunks into a session answer (see the 2026-06-04 entries in
[docs/debugging-notes.md](docs/debugging-notes.md)).

Stale-token behavior is asymmetric by design. On the **read path** (`/query`,
`/retrieve`), an unknown or expired token returns 404 — header presence
commits the request to the session corpus, and silently falling back to the
public FiQA index would answer "what does my document say?" with chunks the
user never uploaded. On the **write path** (`/ingest/*`), an absent or
unknown token mints a fresh session rather than 404ing — replacing a stale
write-token leaks nothing, so the cheaper user-experience behavior (your
upload always succeeds) is safe.

### Per-session limits (defaults)

| Setting | Default | Effect |
|---|---|---|
| `RAGCORE_SESSION_MAX_FILE_BYTES` | 1 MB | Per-file size cap; returns 413 above this. |
| `RAGCORE_SESSION_MAX_FILES` | 3 | Max files per session; returns 409 above this. |
| `RAGCORE_SESSION_MAX_CONCURRENT` | 3 | Max active sessions per process; returns 503 above this. |
| `RAGCORE_PDF_MAX_PAGES` | 100 | Per-PDF page cap (probed before embedding); returns 413 above this. |
| `RAGCORE_SESSION_IDLE_TTL_SECONDS` | 1800 | Sessions idle past this are evicted by the reaper. |
| `RAGCORE_SESSION_SWEEP_INTERVAL_SECONDS` | 300 | Reaper sweep cadence; worst-case eviction lag = TTL + sweep. |
| `RAGCORE_INGEST_MAX_BODY_BYTES` | 10 MB | Transport-level guard on `/ingest/*` request bodies; returns 413. |

The caps are scaffolding shaped for a 2 GB box. On a larger deploy, lift
them via env vars — don't edit the defaults. The TTL is idle-keyed, not
absolute: every ingest or query bumps `last_access`, so an active user is
never evicted mid-session. On eviction, the reaper drops the session's
Chroma cache entry and (on Linux/glibc) calls `malloc_trim(0)` to return
the freed pages to the OS.

### Streamlit UI workflow

The Streamlit demo (`ui_streamlit/app.py`) wires the session API into its
chat surface:

1. Upload a PDF or text file from the sidebar's "Your documents" panel.
   The first upload mints a session token; subsequent uploads in the same
   browser tab add to it up to the file/byte caps above.
2. A mode pill at the top of the chat indicates whether the next query
   will target the public FiQA corpus or the active session corpus.
3. Ask questions about the uploaded documents. Queries automatically carry
   the session token; answers cite only chunks from the session corpus.
4. "Start new session" clears the token and the upload state, returning
   the chat to public-corpus mode. The button bumps a nonce baked into
   the `file_uploader`'s `key=` so a retained file can't silently re-mint
   the cleared session — see the 2026-06-04 entry in
   [docs/debugging-notes.md](docs/debugging-notes.md) for the trap.

---

## Configuration

All settings are environment-variable driven. Copy `.env.example` → `.env`.

### Retrieval

| Setting | Default | Options / Effect |
|---------|---------|-----------------|
| `VECTOR_STORE_PROVIDER` | `faiss` | `faiss` · `chroma` |
| `CHUNKING_STRATEGY` | `semantic` | `fixed` · `semantic` · `hierarchical` · `sentence` |
| `HYBRID_ALPHA` | `0.7` | `0` = keyword only · `1` = vector only |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Final chunks sent to LLM |
| `ENABLE_RERANKING` | `true` | Two-stage retrieval with cross-encoder |
| `ENABLE_QUERY_EXPANSION` | `false` | Multi-query paraphrasing for complex questions |

### Generation

| Setting | Default | Options / Effect |
|---------|---------|-----------------|
| `LLM_PROVIDER` | `groq` | `groq` and `anthropic` dispatch to dedicated clients in `GenerationService._build_llm()`; `openai` (and Azure OpenAI when `AZURE_OPENAI_ENDPOINT` is set) runs through the OpenAI client, which is also the fallthrough default for any unrecognized value. `ollama` and `demo` are accepted enum placeholders for planned providers — not yet wired to a dedicated client. |
| `GROQ_API_KEY` | — | Required if `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | — | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | Azure API version |
| `GENERATION_STRATEGY` | `basic` | `basic`, `self_rag` (claim verification loop), `flare` (FLARE-inspired with numeric-novelty re-retrieval trigger), `agentic` (scaffolded — falls through to basic). |

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
| `FAISS_DATA_DIR` | `./faiss` | Directory for FAISS index and metadata files. Set to a persistent-disk path on platforms with ephemeral filesystems (e.g., Render). |
| `ENABLE_TRACING` | `false` | Send traces to Langfuse |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |

### CORS

| Setting | Default | Effect |
|---------|---------|--------|
| `CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed origins for cross-origin requests. Set to your frontend's URL in production. |

### API Key Authentication

API key auth is **off by default** so the Streamlit demo at [ragcore.streamlit.app](https://ragcore.streamlit.app) remains accessible to reviewers without configuration. When enabled, every non-exempt endpoint requires the `X-API-Key` request header. Exempt paths (always reachable without a key): `/health`, `/health/live`, `/health/ready`, `/metrics`. The Streamlit UI reads `RAGCORE_API_KEY` from its own environment and forwards it automatically if set.

| Setting | Default | Effect |
|---------|---------|--------|
| `RAGCORE_AUTH_ENABLED` | `false` | When `true`, all non-exempt endpoints require `X-API-Key` header. |
| `RAGCORE_API_KEY` | — | Required when `RAGCORE_AUTH_ENABLED=true`. Use a long random string (e.g. `python -c 'import secrets; print(secrets.token_urlsafe(32))'`). |

---

## Security

RAGCore is a portfolio demo; its security posture is off by default to keep the live demo reachable without configuration. Four mechanisms are opt-in for production deploys: a CORS allowlist, optional API key authentication, a per-IP rate limiter, and docker-compose credential hardening.

**CORS allowlist.** `CORSMiddleware` rejects preflight requests from origins not in the `CORS_ORIGINS` list (default `http://localhost:8501` for the Streamlit dev port). In production set `CORS_ORIGINS` to your frontend's URL. See the [Configuration](#configuration) table.

**API key authentication.** Disabled by default (`RAGCORE_AUTH_ENABLED=false`). When enabled, every non-exempt endpoint requires an `X-API-Key` header matched using constant-time comparison. Exempt paths — `/health`, `/health/live`, `/health/ready`, `/metrics` — remain reachable for monitoring tools without a key. Set `RAGCORE_API_KEY` to a long random value before enabling. See the [Configuration](#configuration) table.

**Rate limiter.** Enforces a per-IP sliding window (default 60 requests / 60-second window), configurable via `RAGCORE_RATE_LIMIT_MAX_REQUESTS` and `RAGCORE_RATE_LIMIT_WINDOW_SECONDS`. Rejected requests receive a `429` with a `Retry-After` header. By default the limiter reads `request.client.host`; set `RAGCORE_TRUST_PROXY_HEADERS=true` only when running behind a trusted reverse proxy. State is process-local — see AUDIT.md for the multi-worker caveat.

**Docker-compose credentials.** The compose stack defaults Redis and Grafana credentials to `CHANGE_ME_IN_PROD` if `REDIS_PASSWORD` and `GF_ADMIN_PASSWORD` are unset — a visible placeholder, not a silent default. Set both in `.env` before running `docker compose up`.

Five security limitations are documented and tracked in [AUDIT.md](AUDIT.md): the in-memory trace store has no eviction and will exhaust memory on long-running deploys; the trace retrieval endpoints (`/agent/trace/{id}`, `/trace/{id}`) require no authentication; several endpoints surface raw exception messages to callers; the Celery failure path returns the serialized exception object verbatim; and `RequestIdMiddleware` accepts any client-supplied `X-Request-Id` value without format validation.

---

## Deployment

The backend runs on Render as the `ragcore-api` service: Docker runtime, Standard plan, with a 5 GB persistent disk mounted at `/var/data`. `render.yaml` is the Blueprint source of truth for the plan, the disk declaration, and every managed environment variable. The three API keys (`GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are declared as `sync: false` so their secret values stay out of version control while their presence remains declared in the Blueprint. Pushes to `main` trigger autoDeploy.

**Vector store split.** The FiQA `"default"` corpus lives on FAISS at `$FAISS_DATA_DIR` (`faiss_index.idx` plus `faiss_metadata.pkl`). The 6 Apple corpora live on Chroma, one per persist directory at `$CHROMA_PERSIST_DIR/<corpus_name>/` — each containing `chroma.sqlite3`, a `bm25_state.pkl` sidecar for hybrid retrieval, and a UUID-named HNSW segment subdir. `GET /corpora` returns all 7 entries from a single backend-agnostic registry, so clients select a corpus by name and never see the FAISS/Chroma split.

**Apple corpus delivery (Setup B).** The Apple Chroma collections are built locally with `python scripts/ingest_apple_corpus.py`, committed under `data/chroma_collections/<corpus_name>/`, and bundled into the Docker image. On first boot, the lifespan handler in `api/main.py` runs `_seed_apple_collections()`, which atomically copies each per-corpus directory from `/app/data/chroma_collections/<name>` to `$CHROMA_PERSIST_DIR/<name>` via a sibling temp directory plus `os.replace` — `chroma.sqlite3` only appears at the destination once the whole tree is in place. Subsequent boots detect the existing `chroma.sqlite3` at the destination and skip the seed for that corpus. In local development, where the source and destination resolve to the same path, the function no-ops.

**Verification after deploy.** `curl https://ragcore-api.onrender.com/corpora` should return 7 entries (the FiQA `default` plus the 6 Apple corpora) with their expected `doc_count` values. Then run one `/query` per corpus with a known-good question and confirm a grounded answer with citations and `total_tokens > 0` — HTTP 200 alone is not sufficient, since a healthy registration is fully compatible with an empty-retrieval bug elsewhere in the pipeline. As of 2026-05-29, all 7 corpora are verified live through `/query`; one cosmetic issue — inline `[N]` citation markers sometimes render without the digit — is tracked separately in [docs/debugging-notes.md](docs/debugging-notes.md) and does not affect retrieval or grounding. If `/corpora` returns fewer than 7, check the boot logs for `Seed failed for <corpus>` warnings: the seed step logs-and-continues on error, so a failed seed surfaces as a silently missing corpus rather than a startup crash.

**Updating a corpus.** Re-ingest locally with `python scripts/ingest_apple_corpus.py`, commit the regenerated `data/chroma_collections/<corpus_name>/`, and push. The seed step does **not** overwrite a destination that already contains a `chroma.sqlite3`, so a push alone will not propagate the change to the live disk. The surgical workaround is to open a Render shell and remove that corpus's directory under `/var/data/chroma_db/` before the next deploy; rotating the persist root (`CHROMA_PERSIST_DIR=/var/data/chroma_db_v2`) is an alternative that re-seeds every corpus on first boot. A manifest-hash trigger that re-seeds when the in-image collection differs from the on-disk one is a planned follow-up. One related note for contributors: local queries can write back into a Chroma HNSW segment and dirty the tracked binaries; whether to gitignore the per-segment files or restore them after testing is a separate deferred decision.

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
│   └── main.py             # /ingest/* · /query · /retrieve · /query/stream · /corpora · /documents/* · /health/* · /metrics · /agent/*
│
├── config/
│   ├── settings.py         # Pydantic settings — all env vars with defaults
│   └── corpora.py          # CORPORA_CONFIG — per-corpus Chroma persist_dir + chunker
│
├── embeddings/             # BGEEmbedder · MiniLMEmbedder · embedder factory
│
├── evaluation/
│   ├── evaluator.py        # Retrieval + generation metrics (MRR, NDCG, hit rate)
│   ├── datasets/           # FiQA-2018 eval set (50 queries, seed=42)
│   ├── notebooks/          # baseline_vs_selfrag.ipynb
│   ├── results/            # Raw JSON benchmark output (basic_fiqa.json, self_rag_fiqa.json)
│   └── scripts/            # run_benchmark.py · stage_b_sanity.py · production_health_check.py
│
├── generation/
│   ├── llm_service.py      # Provider switcher: Groq / OpenAI / Anthropic
│   ├── advanced_generation.py  # SelfRAGGenerator · FLAREGenerator · AgenticRAG
│   └── prompts/            # Prompt templates
│
├── ingestion/
│   ├── chunkers/           # fixed · semantic · hierarchical · sentence
│   └── loaders/            # PDF · TXT · DOCX · HTML · CSV
│
├── monitoring/
│   ├── tracer.py           # NoOpTracer (default) · LangfuseTracer (optional)
│   ├── metrics.py          # Prometheus metric definitions (5 custom ragcore_* + RED)
│   └── logging_config.py   # structlog setup
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
│   ├── unit/               # no external services required
│   └── integration/        # FAISS in-memory, mocked external services
│
├── docs/
│   └── debugging-notes.md
│
├── utils/
│   └── models.py           # Shared types: Chunk · Document · RetrievedChunk · IngestRequest · QueryRequest
│
└── vectorstore/
    ├── vector_store.py     # FAISSVectorStore + per-corpus registry (register_corpus / get_corpus / list_corpora)
    ├── chroma_store.py     # ChromaVectorStore — backs the Apple multi-corpus
    ├── bm25_index.py       # BM25Index helper shared by FAISS and Chroma backends
    └── session_store.py    # SessionStore — per-session corpora with TTL eviction and in-flight pin
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

The unit suite runs without external services; the integration suite uses
in-memory FAISS and mocked external services. One integration test
(`test_agent_graph_with_tracing`) is known to fail on a mock-target
resolution bug in the test itself, unrelated to the graph it exercises.
For current counts, run `pytest tests/unit/ --collect-only -q` and
`pytest tests/integration/ --collect-only -q`.

---

## Status

### Working end-to-end

- Document ingestion (HTTP API): per-session uploads of PDF and plain text
  (TXT/MD) via `/ingest/file` and `/ingest/text`. File type decided by
  magic-byte sniff; client `Content-Type` and filename ignored. PDFs capped at
  100 pages by default.
- Document ingestion (offline scripts): PDF, DOCX, HTML, CSV, and plain text
  via the loader registry — used by `scripts/ingest_apple_corpus.py` and
  similar admin tooling to build the curated corpora (FiQA `default`, Apple
  10-K family) that ship with the deploy. Same 4 chunking strategies as the
  API path.
- Hybrid retrieval: FAISS dense + BM25 sparse with configurable alpha fusion
- Cross-encoder reranking
- Query routing: heuristic regex + LLM fallback, dispatches to 5 of 6
  retrieval strategies; parent-child is enumerated but not wired
- Basic and Self-RAG generation paths, configurable via `GENERATION_STRATEGY`
- FLARE-inspired iterative generation: dollar-token novelty between response and retrieved chunks triggers re-retrieval; configurable via `GENERATION_STRATEGY=flare`. Heuristic deviates from Jiang et al. 2023 (which uses token logprobs unavailable on Groq's Llama 3.3 70B endpoint); see the `FLAREGenerator` docstring for rationale.
- Retrieval evaluation: MRR, NDCG@5, hit@5, precision@5, recall@5 — all
  correctly bounded after dedup fix (see debugging notes)
- LLM-judged faithfulness via RAGAS (gpt-4o-mini judge) on the same 50 FiQA
  queries, with paired bootstrap confidence intervals and Wilcoxon signed-rank
  significance testing — pre-registered analysis plan committed before the run
- FiQA-2018 benchmark runner and comparison notebook
- Citation span attribution with inline highlights (yellowmark + numbered source chips)
- Follow-up question suggestions (LLM-generated, click to prefill chat)
- Hallucination verifier toggle (per-query Self-RAG opt-in, ~3x slower)
- Per-query pipeline section in sidebar (router, retrieve, rerank, generate, 
  latency, tokens, confidence with recalibrated thresholds)
- Real Prometheus metrics: 5 custom `ragcore_*` metrics (stage duration
  histogram, generation tokens, Self-RAG claims, process memory, vector
  store disk bytes) plus RED metrics on every endpoint via
  `prometheus-fastapi-instrumentator`. Scraped by Prometheus at 15s
  intervals; Grafana dashboard with overview panels preloaded via
  provisioning.
- Deep health checks: `/health/live` (process alive) and
  `/health/ready` (vector store ping, embedder smoke test, LLM API key
  validation). Returns 200 all-pass or 503 with structured per-check
  failures. Original `/health` retained for Render's default monitor.
- Structured JSON logging via structlog with request-ID correlation.
  `RequestIdMiddleware` honors inbound `X-Request-Id` header or
  generates UUID4; binds into context so every log line during request
  handling carries the field. Echoed in response header.
- Self-RAG verifier robustness: two latent bugs found and fixed during
  benchmark runs. Exception handler in `_verify_claim` was fail-open
  (silently promoting unparseable responses to verified) — now fail-closed.
  `_extract_claims` and `_verify_claim` did not strip markdown code
  fences before `json.loads`, causing all benchmark-shape responses
  from gpt-4o-mini to land in the exception handler. Both fixes
  covered by 6 regression tests.

### Deferred or scaffolded

- Monitoring: `NoOpTracer` methods are all `pass`; Langfuse tracing is
  off by default and never exercised end-to-end against a live
  instance. Prometheus and Grafana now wired and verified locally.
- Multi-vector-store: FAISS and Chroma are both verified end-to-end with hybrid
  retrieval via the shared BM25Index helper, and both pass the same parity test
  surface. Weaviate, Pinecone, and Qdrant remain config-only fall-throughs with
  an explicit warning log; they are not documented options.
- Agentic RAG: code is in `generation/advanced_generation.py` but not wired to the orchestrator; `GENERATION_STRATEGY=agentic` falls through to basic generation
- RAGAS evaluation runs end-to-end and produces the LLM-judged faithfulness
  numbers in the headline table. Word-overlap retained alongside RAGAS as the
  displaceable historical metric — the headline finding is that the two metrics
  disagree on Self-RAG.

---

## Project Artifacts

- [AUDIT.md](AUDIT.md) — Module-by-module assessment of what's implemented
  versus what's scaffolded or deferred, with README claims rated REAL / PARTIAL /
  OVERSTATED and five previously-undocumented bugs.
- [evaluation/results/ragas_analysis_2026-04-26.md](evaluation/results/ragas_analysis_2026-04-26.md) —
  RAGAS vs word-overlap faithfulness analysis on 50 FiQA queries. Paired
  bootstrap CIs, Wilcoxon signed-rank tests, cross-metric Spearman correlation,
  top-outlier query analysis, mechanism hypothesis, and limitations. Documents
  the metric disagreement that motivates the README headline.
- [evaluation/results/per_claim_analysis_2026-05-05.md](evaluation/results/per_claim_analysis_2026-05-05.md) —
  Per-claim faithfulness analysis under controlled conditions
  (gpt-4o-mini as generator, Self-RAG verifier, and RAGAS judge).
  Documents the structural disagreement between the two verifiers
  (Pearson r near zero across paired queries, 34% internal-accept/
  RAGAS-reject vs 3% reverse) and characterizes RAGAS instrument
  non-determinism (per-query noise floor 0.05-0.09; Wilcoxon p
  straddles α=0.05 across reruns of identical data). Self-updating
  writeup with f-string substitutions; numbers regenerate on each
  notebook run.
- [docs/ragas_run_plan_2026-04-26.md](docs/ragas_run_plan_2026-04-26.md) —
  Pre-registered analysis plan, written before the RAGAS benchmark ran. Decision
  rules committed in advance for three result scenarios; the RAGAS regression
  triggered Rule 3 (rewrite README), which is what this README does.
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
## Recent Updates

| Date | What changed |
|---|---|
| 2026-06-04 | Per-session document upload feature: synchronous `/ingest/file` and `/ingest/text` gated by `X-Session-Id`, per-session file/byte/concurrency caps, idle-TTL eviction with `malloc_trim` RAM reclamation, Streamlit upload UI |
| 2026-06-03 | Removed the async/Celery `/ingest` path from the HTTP surface; added the session-scoped vector-store registry that the per-session feature builds on |
| 2026-05-30 | Corpus-aware Streamlit UI (sidebar dropdown across 7 corpora); `POST /retrieve` endpoint (router → retrieve → rerank, no generation); chunker comparison panel |
| 2026-05-28 | Apple multi-corpus shipped: 6 Chroma corpora alongside FiQA on FAISS; `GET /corpora` endpoint translates unknown-corpus errors to HTTP 400 |
| 2026-05-27 | Per-corpus vector-store registry replaces the single-store singleton; PDF loader swapped from pdfplumber to pymupdf4llm |
| 2026-05-13 | `delete_document` FAISS index-wipe bug fixed (embeddings persisted in metadata pickle, index rebuilt on delete); production health-check cron added |
| 2026-05-08 | Grafana dashboard screenshot and Recent Updates section added |
| 2026-05-07 | AUDIT and README refreshed to reflect the past week's batch |
| 2026-05-06 | Per-claim faithfulness analysis: characterized RAGAS judge non-determinism (0.05-0.09 noise floor at n=50) and structural verifier disagreement (Pearson r ≈ -0.14 between Self-RAG verifier and RAGAS judge using the same model) |
| 2026-05-06 | Self-RAG verifier robustness: fixed fail-open exception handler and markdown fence parsing; both bugs were silently corrupting benchmark runs |
| 2026-05-04 | Structured JSON logging with request-ID correlation via middleware |
| 2026-05-03 | Grafana dashboard with overview panels (request rate, latency, stage durations, token rate, Self-RAG claims, memory, disk) |
| 2026-05-03 | Prometheus scraping the API every 15s; full docker-compose stack runnable locally |
| 2026-04-30 | Real Prometheus metrics on /metrics: 5 custom metrics + RED metrics on every endpoint |
| 2026-04-30 | Deep health checks: /health/live and /health/ready with vector store, embedder, and LLM config validation |

For full commit history: `git log --oneline -30`

---
## What's next

- Wire Agentic RAG into the generation strategy dispatch (code exists in
  `advanced_generation.py`, not yet called by orchestrator)
- Wire the Ollama provider for local inference (currently an accepted enum
  placeholder; `GenerationService._build_llm()` has no `ollama` branch and
  falls through to the OpenAI client)
- Manual claim-level annotation on the 13 internal-accept/RAGAS-reject
  queries to determine whether they reflect real grounding errors or
  overzealous RAGAS rejection (per-claim analysis investigated the
  mechanism but did not adjudicate individual claims).
- Expand to the full FiQA test split (~648 queries) to tighten confidence
  intervals on both faithfulness deltas
- Cross-model judge comparison (gpt-4o vs gpt-4o-mini) to test whether
  the verifier disagreement and noise floor findings hold under a
  stronger judge.

---
## Licensing notes

RAGCore itself is MIT-licensed. One dependency carries a copyleft license:

- PDF parsing uses pymupdf4llm (AGPL-3.0; see [LICENSE-pymupdf4llm.md](LICENSE-pymupdf4llm.md))
