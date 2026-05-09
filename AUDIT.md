# RAGCore — Self-Audit

**Date:** 2026-04-30
**Updated:** 2026-05-06 to reflect the 9-commit batch (be22906..c2e70a1).
**Auditor:** Hema Kollipara
**Scope:** Post-citation-spans state — attributed_spans backend parser, attributed_spans
frontend rendering, follow-up question generation, hallucination verifier toggle, confidence
threshold recalibration, visual rebuild.

---

## Classification Scheme

| State | Meaning |
|-------|---------|
| **WORKING** | Exercises real code paths end-to-end. Has been run against live data. |
| **PARTIAL** | Core path works; edge cases, error handling, or secondary paths are incomplete or wrong. |
| **SCAFFOLD** | Class/function exists with correct structure; not reachable from any live code path. |
| **DEFERRED** | Config key or README entry exists; no implementation behind it. |

---

## Module Status

### Vector Store

| Module | File | State | Notes |
|--------|------|-------|-------|
| FAISSVectorStore | vectorstore/vector_store.py | WORKING | IndexFlatIP + BM25 hybrid via BM25Index (f85b80a). Index and metadata path configurable via `FAISS_DATA_DIR` (default `./faiss`, commit 5ba20cf). Singleton confirmed via integration tests. |
| get_vector_store() singleton | vectorstore/vector_store.py | WORKING | Module-level `_vector_store_instance`; `reset_vector_store()` for test isolation. |
| BM25Index | vectorstore/bm25_index.py | WORKING | Extracted from FAISSVectorStore (f85b80a). `build` / `upsert` / `query` / `reset` / `save` / `load`. Shared by both FAISSVectorStore and ChromaVectorStore via composition. |
| ChromaVectorStore | vectorstore/chroma_store.py | WORKING | Full BaseVectorStore implementation: upsert, vector_search, keyword_search, hybrid_search, delete_document. BM25Index hybrid retrieval. 11 parity integration tests pass against FAISS (commit 857c007). Persistence verified on Render. Not the active production backend (VECTOR_STORE_PROVIDER=faiss); FiQA benchmark has only ever run on FAISS. chromadb 0.4.24 + NumPy 2.x shim required locally; Docker pins numpy 1.26.4. |

### Embeddings

| Module | File | State | Notes |
|--------|------|-------|-------|
| BGEEmbedder | embeddings/embedder.py | WORKING | BAAI/bge-large-en-v1.5, local, async via ThreadPoolExecutor. Default in benchmark. |
| MiniLMEmbedder | embeddings/embedder.py | WORKING | Alternative local model; selectable via `EMBEDDING_MODEL`. |
| CachedEmbedder | embeddings/embedder.py | PARTIAL | Redis wrapper; degrades silently to underlying embedder if Redis is absent. |
| CohereEmbedder | embeddings/embedder.py | PARTIAL | API key bug fixed (now passes `settings.cohere_api_key`); not exercised end-to-end. |
| MatryoshkaEmbedder / ColBERTEmbedder / EmbeddingFineTuner | embeddings/advanced_embeddings.py | SCAFFOLD | Not registered in `get_embedder()`; unreachable from any pipeline path. |
| get_embedder() singleton | embeddings/embedder.py | WORKING | Module-level `_embedder_instance`; `reset_embedder()` for test isolation (commit `2f8c9c2`). Memoized when called with default arguments; explicit arguments bypass singleton and return fresh instances (used by tests). |

### Ingestion

| Module | File | State | Notes |
|--------|------|-------|-------|
| FixedSize / Semantic / Hierarchical / SentenceWindow chunkers | ingestion/chunkers/ | WORKING | All four dispatched by `get_chunker()` and confirmed end-to-end. Three additional chunkers (Propositional, TableAware, DocumentStructure) are also registered in `get_chunker()` but untested end-to-end — see PARTIAL row below. |
| Propositional / TableAware / DocumentStructure chunkers | ingestion/chunkers/ | PARTIAL | Registered in `get_chunker()`. PropositionalChunker calls `gpt-4o-mini` to decompose paragraphs into atomic facts (OpenAI hardcoded — requires `OPENAI_API_KEY`); falls back to SemanticChunker on failure. TableAwareChunker detects markdown/HTML tables via regex; large tables split into 20-row batches preserving header. DocumentStructureChunker splits on `#` markdown headings or `<h1>`–`<h6>` HTML tags; oversized sections overflow to SemanticChunker. None benchmarked end-to-end; no dedicated integration tests. |
| PDF / DOCX / TXT / HTML / Web / GitHub loaders | ingestion/loaders/ | WORKING | All exercise real parsing libraries. |

### Retrieval

| Module | File | State | Notes |
|--------|------|-------|-------|
| HeuristicRouter | retrieval/router/query_router.py | WORKING | Regex patterns for MULTI_HOP, ANALYTICAL, LOOKUP; returns None for uncertain queries. |
| LLMQueryClassifier | retrieval/router/query_router.py | WORKING | gpt-4o-mini, json_object mode; degrades to SEMANTIC on API error. |
| QueryExpander | retrieval/router/query_router.py | PARTIAL | Code works; disabled by default (`ENABLE_QUERY_EXPANSION=false`). Not exercised in benchmark. |
| Hybrid / Keyword / Semantic / MultiQuery strategies | retrieval/strategies/retrieval_executor.py | WORKING | All four dispatch correctly. Confirmed end-to-end. |

### Reranking

| Module | File | State | Notes |
|--------|------|-------|-------|
| CrossEncoderReranker | reranking/reranker.py | WORKING | cross-encoder/ms-marco-MiniLM-L-6-v2. Verified end-to-end. |
| NoOpReranker | reranking/reranker.py | WORKING | Passthrough; assigns ranks. |

### Generation

| Module | File | State | Notes |
|--------|------|-------|-------|
| PromptBuilder | generation/prompts/prompt_builder.py | WORKING | `SYSTEM_BASE` updated to instruct `<cite source="N">` trailing markers; `FOLLOWUP_SYSTEM` and `FOLLOWUP_TEMPLATE` added. Token-budget-aware. |
| GenerationService (GroqLLM / OpenAILLM / AnthropicLLM) | generation/llm_service.py | WORKING | Three providers wired; streaming works. |
| generate_followups | generation/llm_service.py | WORKING | Separate LLM call post-answer. `_extract_followup_questions` handles multi-line Llama JSON shape (one array per line). Orchestrator wraps call in `try/except`; any failure returns `[]`. |
| Ollama / demo providers | generation/llm_service.py | DEFERRED | `_build_llm()` has no branch for ollama or demo; falls through to `OpenAILLM`. |
| LLMService class (legacy) | generation/llm_service.py | SCAFFOLD | Synchronous dict-format interface. Never called by orchestrator. |
| Azure OpenAI | generation/llm_service.py | PARTIAL | Config keys present; `OpenAILLM` builds `AzureOpenAI` client when `AZURE_OPENAI_ENDPOINT` is set. Not exercised end-to-end. |
| SelfRAGGenerator | generation/advanced_generation.py | WORKING | Wired in orchestrator. Activated via `verify_claims=True` per-query override or `GENERATION_STRATEGY=self_rag`. Two latent bugs fixed post-audit: (a) exception handler in `_verify_claim` was fail-open (returned `(True, "")`), silently promoting all unparseable responses to verified — changed to fail-closed `(False, "")` (commit `679315e`); (b) `_extract_claims` and `_verify_claim` did not strip markdown code fences before `json.loads`, causing all benchmark-shape responses from gpt-4o-mini to raise `JSONDecodeError` and land in the exception handler — `_strip_markdown_fences()` helper added at both call sites (commit `6af0589`). The combined effect during 2026-05-06 benchmark runs was 0 verified / 176 unsupported claims across 50 queries (entirely wrong direction). Both fixed; covered by 6 regression tests. |
| FLAREGenerator | generation/advanced_generation.py | WORKING | Wired in orchestrator (`elif effective_strategy == "flare":`, commit `6affffb`). Dollar-token novelty heuristic: re-retrieves when the LLM answer mentions a `$NNN` value absent from the current chunk pool. Iterates up to `FLARE_MAX_RETRIEVAL_ROUNDS` (default 3). `FLAREResult` carries `all_chunks` (deduped across rounds), `retrieval_rounds`, `total_tokens`, `novel_tokens_per_round`. 7 unit tests cover dispatch, loop termination, chunk deduplication, and query construction. |
| AgenticRAG | generation/advanced_generation.py | DEFERRED | Class implementation exists with `run()` method that hardcodes `AsyncOpenAI` (line ~579). Not yet dispatched by orchestrator; `GENERATION_STRATEGY=agentic` falls through to basic generation with no warning. Planned for future implementation per project roadmap. When wired, the OpenAI-coupling will need to be migrated through `llm_service.generate(...)` the way `SelfRAGGenerator._extract_claims` and `_verify_claim` were in commit `b03614d`. |

### Orchestrator & API

| Module | File | State | Notes |
|--------|------|-------|-------|
| RAGOrchestrator | orchestrator.py | WORKING | Full pipeline confirmed. Includes per-query `verify_claims` override, attributed span extraction, and follow-up generation. |
| _extract_attributed_spans | orchestrator.py | WORKING | Trailing `<cite source="N">` marker parser. Runs on every non-empty response. Graceful `except → (answer, [])` fallback. |
| FastAPI endpoints (/ingest, /query, /query/stream, /health) | api/main.py | WORKING | Integration-tested and smoke-tested. Two deep health endpoints added alongside (see below): `/health/live` for process-alive check and `/health/ready` for three-check readiness probe (commit `2f8c9c2`). Original `/health` left unchanged for Render's default health monitoring. |
| `/health/live` and `/health/ready` | api/main.py | WORKING | Added in commit `2f8c9c2`. `/live` returns `{"status": "alive"}` with no dependency calls. `/ready` runs three checks (vector store `ping()`, embedder `embed_query` on one token, LLM API key non-empty via `_LLM_KEY_MAP`) and returns 200 all-pass or 503 with structured per-check failure reasons. 15 unit tests in `tests/unit/test_health.py`. Known gap: a literal placeholder key like `"your-anthropic-key-here"` passes the LLM config check (out of scope per commit body). |
| /agent/query endpoint | api/main.py | PARTIAL | Double-invoke removed in commit `1a79581`. One integration test (`test_agent_graph_with_tracing` in `tests/integration/test_evaluation.py:159`) fails on mock-target resolution — test bug, not graph bug. |
| `/metrics` endpoint | api/main.py, monitoring/metrics.py | WORKING | Real Prometheus instrumentation as of commit `441133d`. Five custom metrics in `monitoring/metrics.py`: `ragcore_stage_duration_seconds` (histogram, stage/strategy labels), `ragcore_generation_tokens_total` (counter, direction/provider), `ragcore_self_rag_claims_total` (counter, outcome), `ragcore_process_memory_bytes` (gauge via psutil), `ragcore_vector_store_disk_bytes` (gauge, backend label). RED metrics on every endpoint via `prometheus-fastapi-instrumentator`. 7 unit tests + 4 integration tests. Known gap: custom `ragcore_*` metrics fire only on queries that complete the full pipeline; early-return paths (e.g., empty-index guard) emit only RED metrics. |
| Celery async ingestion | api/main.py | PARTIAL | Code correct; requires running Redis + Celery worker. Not tested in CI. |
| RequestIdMiddleware | api/middleware/request_id.py | WORKING | Added in commit `d25ae0b`. Honors inbound `X-Request-Id` header; generates UUID4 otherwise. Binds `request_id` into structlog context so every log line emitted during request handling carries the field. Echoes `X-Request-Id` in the response header. Registered last in middleware stack (runs first/outermost). 3 unit tests. |

### Agent

| Module | File | State | Notes |
|--------|------|-------|-------|
| LangGraph StateGraph | agent/graph.py | PARTIAL | Single-invoke path works after commit `1a79581`. One integration test (`test_agent_graph_with_tracing`, `tests/integration/test_evaluation.py:159`) still failing. |
| Agent nodes (router, retriever, reranker, generator, evaluator) | agent/nodes/ | WORKING | Each delegates to the same pipeline modules as the main orchestrator. |

### Monitoring

| Module | File | State | Notes |
|--------|------|-------|-------|
| NoOpTracer | monitoring/tracer.py | WORKING | All methods are pass/return. Correct placeholder. |
| LangfuseTracer | monitoring/tracer.py | PARTIAL | Singleton fix applied (commit `6e4c1b6`); `_record_step` TypeError fixed (commit `f424495`). Still never exercised end-to-end against a live Langfuse instance. Module logger migrated to structlog in commit `d25ae0b`; no functional change. |
| Prometheus metrics | monitoring/metrics.py | WORKING | Five custom `ragcore_*` metrics (histograms, counters, gauges) plus RED metrics via `prometheus-fastapi-instrumentator`. All registered at import time; `monitoring/metrics.py` is imported in `api/main.py` lifespan. Prometheus service in `docker-compose.yml` scrapes `/metrics` every 15s (commit `364af9c`). Verified end-to-end. Known: Grafana panels show "No data" locally without LLM keys until the first request completes the pipeline (expected). |
| Structured logging | monitoring/logging_config.py, api/middleware/request_id.py | WORKING | `configure_logging()` called in FastAPI lifespan startup. Processor chain: `merge_contextvars → add_log_level → add_logger_name → TimeStamper(iso, UTC) → format_exc_info → JSONRenderer`. Five representative call sites converted to native structlog kwargs; the remaining 90+ call sites emit via the stdlib bridge and gain JSON output and `request_id` correlation automatically. Static fields `service` and `environment` bound at startup via `bind_contextvars`. Quirk: `TimeStamper(fmt="iso", utc=True)` emits `Z` suffix, not `+00:00` (commit `d25ae0b`). |

### Evaluation

| Module | File | State | Notes |
|--------|------|-------|-------|
| RetrievalEvaluator | evaluation/evaluator.py | WORKING | hit@K, MRR, NDCG@K, precision@K, recall@K all correctly bounded [0,1]. |
| GenerationEvaluator (heuristic) | evaluation/evaluator.py | PARTIAL | Word-overlap proxy. Reports "faithfulness" but measures token co-occurrence, not factual grounding. |
| GenerationEvaluator (RAGAS) | evaluation/evaluator.py | WORKING | `LLMFaithfulnessEvaluator` wired into `run_benchmark.py`; results committed to `evaluation/results/`. Live API path (`EVAL_STRATEGY=ragas`) remains unvalidated end-to-end. |
| Old Evaluator class | evaluation/evaluator.py | SCAFFOLD | `evaluate()` raises `NotImplementedError`. Dead code. |
| FiQA-2018 benchmark runner | evaluation/scripts/run_benchmark.py | WORKING | 50-query eval; `date.today()` (hardcoded date removed in commit `b04070b`). Reproducible. New `--output PATH` argument added (commit `c2e70a1`); auto-named default preserved. Output JSON shape changed from flat list to `{"metadata": {...}, "results": [...]}`. metadata captures `llm_provider`, `llm_model`, `vector_store_provider`, `embedding_provider`, `eval_strategy`, `generation_strategy`, `ragas_judge`, `dataset`, `dataset_size`, `run_timestamp_utc`. Per-record `model_used` and `retrieved_contexts` fields added to support reproducible per-claim re-evaluation. Notebook's `_records()` helper handles both old flat-list and new dict shapes (backward compatible). |
| per_claim_analysis.py / PatchedFaithfulness | evaluation/scripts/per_claim_analysis.py, evaluation/notebooks/per_claim_analysis_2026-05-05.ipynb | WORKING | `PatchedFaithfulness(Faithfulness)` subclass intercepts RAGAS's `NLIStatementOutput` via a `claim_cache` side-channel before `_ascore` discards it, exposing per-claim verdicts the public RAGAS API does not surface. `run_ragas_with_claims()` and DataFrame builders for per-claim and per-query analysis. Patch is contained to this script; RAGAS package not modified. Results in `evaluation/results/per_claim_analysis_2026-05-05.md` plus two PNG plots. Self-updating writeup via f-string substitution; re-runs regenerate the markdown with current numbers (commit `c2e70a1`). |

### UI

| Module | File | State | Notes |
|--------|------|-------|-------|
| _render_answer_with_spans | ui_streamlit/app.py | WORKING | Inline `<mark>` highlights with `<sup>` source chips. Falls back to `st.markdown` when `attributed_spans` is empty or None. |
| _follow_up_chips | ui_streamlit/app.py | WORKING | Clickable pill buttons; sets `prompt_prefill` and calls `st.rerun()`. |
| Hallucination verifier toggle | ui_streamlit/app.py | WORKING | Sidebar checkbox (`key="verify_claims"`); passes `verify_claims=True` to `QueryRequest`; activates Self-RAG per-query override in orchestrator. |
| compute_confidence | ui_streamlit/app.py | WORKING | Score-gap heuristic on pre-rerank candidates. Thresholds 3.79/8.13 calibrated 2026-04-29 on n=10 live backend queries; docstring records methodology. |
| Sidebar pipeline (S4) | ui_streamlit/app.py | WORKING | Per-query `stage_timings`, `latency_ms`, `total_tokens`, and confidence; populated from `_latest_result()` in session state. |
| Visual rebuild (hero, sources grid, prompt cards) | ui_streamlit/app.py | WORKING | Design tokens, 2×2 source grid, stat cards, prompt card buttons. |

### Tests

| Module | File | State | Notes |
|--------|------|-------|-------|
| Unit tests (144) | tests/unit/ | WORKING | `grep -r "def test_" tests/unit/` returns 144. +31 since `be22906`: 3 fail-closed Self-RAG (`679315e`), 3 fence-stripping (`6af0589`), 15 health checks (`2f8c9c2`), 3 logging (`d25ae0b`), 7 metrics (`441133d`). |
| Integration tests (45) | tests/integration/ | PARTIAL | 45 total, 44 passing. +4 from `test_metrics_endpoint.py` (commit `441133d`). One test (`test_agent_graph_with_tracing`) still fails on mock-target resolution in the test itself — unchanged since April. |

---

## README Claims Audit

### Headline Results

**"Benchmarked on 50 FiQA-2018 financial Q&A queries."**
→ **REAL.** `evaluation/datasets/` has 50 queries seeded at 42. Raw JSON output in
`evaluation/results/` is committed. Numbers reproducible via
`python evaluation/scripts/run_benchmark.py --strategy basic`.

**Two-metric faithfulness table (word-overlap +0.054, RAGAS regression characterized across runs)**
→ **REAL with caveat.** Both runs are committed and reproducible. The word-overlap delta (+0.054) is from a single run and matches the committed `basic_fiqa.json` and `self_rag_fiqa.json`. The RAGAS regression is now reported as a multi-run summary in the README (mean −0.085, range [−0.115, −0.065] across three judge-controlled re-runs, Wilcoxon p straddling α=0.05 at 0.007/0.049/0.069), superseding the April single-run point estimate of −0.109 in headline framing. The April number is itself within the run-to-run range and was a real measurement; what changed is the precision being claimed. The per-claim analysis (committed in `c2e70a1`) characterizes the non-determinism and the structural verifier disagreement.

The framing change in the README is honest: the per-run p-values straddle conventional significance, the run-to-run delta variation is comparable to the effect size, and the reader is pointed at the per-claim analysis for the mechanism. The historical April single-run number is preserved (in `evaluation/results/`) and the April analysis file is unchanged; the README headline now reports the multi-run summary instead.

**"Self-RAG's claim verification loop costs 1.8× the latency of basic generation"**
→ **REAL** in spirit, but the README headline table now reports 10.6s vs 5.9s (+79%),
not 1.8×. The body text says "1.8×"; the table says "+79%". These are inconsistent
(1.8× = +80%). Minor rounding discrepancy; both figures are within measurement noise
of the same underlying data. The May 2026-05-06 benchmark runs (`baseline_fiqa_2026-05-05_gpt4o-mini.json`, `self_rag_fiqa_2026-05-05_gpt4o-mini.json`) used the post-fix verifier and capture latency under the same conditions; pre-fix runs were systematically unusable for verifier-outcome metrics (0 verified / 176 unsupported across 50 queries) but their latency measurements were unaffected.

**Retrieval metrics identical between strategies**
→ **REAL.** Both strategy runs use the same retrieval path. Identical numbers are
correct by construction.

### Live Demo

**"ragcore.streamlit.app — try the deployed demo"**
→ **REAL.** Frontend is deployed on Streamlit Cloud. Backend at
`ragcore-api.onrender.com` on Render free tier.

**"First request may take ~30s on free tier (Render cold start)"**
→ **REAL.** Render free-tier services spin down after inactivity. The caveat is
accurate.

**"Optional Self-RAG verification toggle in the sidebar"**
→ **REAL.** Sidebar checkbox (`verify_claims`) wired to `QueryRequest.verify_claims`;
orchestrator upgrades strategy to `self_rag` for that query only.

### Architecture Diagram

**"Hybrid Index: FAISS Dense Vectors + BM25 Sparse Index"**
→ **REAL.** Both are in-memory in `FAISSVectorStore`.

**"Query Router: Heuristic → LLM fallback"**
→ **REAL.** `HeuristicRouter` (regex) → `LLMQueryClassifier` (gpt-4o-mini) exactly
as drawn.

**"LLM Generator + Citations"**
→ **REAL.** Citations present. Attributed spans (inline citation highlights) are
produced by `_extract_attributed_spans` and rendered by the UI, but this is not
mentioned in the diagram or README body text — it is a gap in README coverage.

**"Monitoring subgraph: Prometheus + Grafana + structured logging"**
→ **REAL** (newly added). Prometheus `/metrics` endpoint emits 5 custom metrics + RED metrics; Prometheus service in `docker-compose.yml` scrapes at 15s; Grafana dashboard with overview panels preloaded via provisioning; structured JSON logs with request-ID correlation via `RequestIdMiddleware`. Diagram updated to reflect this in the README pass accompanying this audit.

### Retrieval Strategy Routing

**"6 query types → strategy pairs"**
→ **REAL** for 5 of 6 strategies. ParentChild is enumerated but dispatch was removed;
the README's Status section correctly notes "parent-child is enumerated but not wired."

### Pre-Registered Analysis Plan

**"The pre-registered analysis plan, written before the RAGAS run, is at
docs/ragas_run_plan_2026-04-26.md"**
→ **REAL.** The file exists and is committed. Git history confirms it predates the
RAGAS run results.

**"The methodology change made after seeing the first run (judge max_tokens raised
from default to 8192 to recover 14 verification-stage truncations) is documented
in the analysis file alongside the original default-judge run preserved as evidence."**
→ **REAL.** Both judge runs are present in `evaluation/results/`. The analysis file
documents the methodology change and its rationale.

### Configuration

**`GENERATION_STRATEGY` options: `basic`, `self_rag`, `flare`, `agentic`**
→ **PARTIAL.** `basic`, `self_rag`, and `flare` dispatch to distinct code paths. `agentic` falls through to `basic` generation with no warning; AgenticRAG implementation exists but is not wired to the orchestrator (planned).

**`LLM_PROVIDER=ollama` and `LLM_PROVIDER=demo`**
→ **OVERSTATED.** `_build_llm()` has no branch for ollama or demo. Both fall through
to `OpenAILLM`, which will fail without `OPENAI_API_KEY`.

**`VECTOR_STORE_PROVIDER=faiss|chroma`**
→ **REAL.** Both backends verified end-to-end. Unrecognized providers fall through to FAISS with a warning log. Weaviate, Pinecone, and Qdrant enum values are retained in `utils/models.py` for backward compatibility with existing `.env` files but are not documented or supported.

**`CHROMA_PERSIST_DIR=./chroma_db`**
→ **REAL.** Setting is in `config/settings.py` (`chroma_persist_dir`) and the README Infrastructure table. `ChromaVectorStore.__init__` uses it as the root for the ChromaDB persistence directory and the `bm25_state.pkl` sidecar file (commit 857c007). Default matches.

**`FAISS_DATA_DIR=./faiss`**
→ **REAL.** Setting in `config/settings.py` (`faiss_data_dir`) and used by `FAISSVectorStore.__init__` to locate the index and metadata files. Motivation was Render's ephemeral filesystem — pointing `FAISS_DATA_DIR` at a mounted persistent-disk path preserves ingested data across redeploys. Documented in the README Infrastructure table.

**`ENABLE_TRACING=true` (Langfuse)**
→ **PARTIAL.** `LangfuseTracer` is structurally correct and the singleton bug is
fixed. It has never been exercised end-to-end against a live Langfuse instance.

**`ENABLE_EVALUATION=true`, `EVAL_STRATEGY=ragas`**
→ **PARTIAL.** RAGAS runs in the benchmark script. The live API path
(`EVAL_STRATEGY=ragas` per-query) remains unvalidated.

### Status Section

The README's Status section was accurate as of 2026-04-30 but had three claims rendered false by the 9-commit batch: (1) "Prometheus/Grafana appear in `docker-compose.yml` but the application emits no metrics" — now false; 5 custom metrics + RED metrics fire. (2) "fix the two scaffolded tracer integration tests" in What's Next — actually fixed in commit `f424495` (pre-`be22906`); the tests pass. (3) Test counts 113/40-of-41 — now 144/44-of-45.

The accompanying README pass updates all three: adds four new "Working end-to-end" bullets (Prometheus, deep health checks, structured logging, Self-RAG verifier robustness), trims the Monitoring deferred bullet to remove the false claim, removes the stale tracer-integration-tests bullet from What's Next, and updates the test counts in the Testing section.

Self-RAG hardcoded to gpt-4o-mini was already noted; that limitation is unchanged. NoOpTracer remaining default is unchanged. FLARE/AgenticRAG not wired remains correctly flagged. Weaviate/Pinecone/Qdrant fall-through is unchanged. RAGAS requiring extra install is unchanged.

### Testing Section

**"144 unit tests passing, 44 of 45 integration tests passing"**
→ **REAL.** `grep -r "def test_"` returns 144 unit tests and 45 integration tests. The single failing integration test (`test_agent_graph_with_tracing`) is a mock-target resolution bug in the test itself, unchanged since April. The post-`be22906` test additions (+31 unit, +4 integration) are accounted for in the Tests table above.

> Note: the `tests/unit/` and `tests/integration/` directory tree comments in the README still show "75 tests" and "25 tests" — both were already stale in the April audit and remain so. Cosmetic; not on the critical path.

### Project Structure

The directory tree matches the actual structure. `utils/models.py` description
("Shared types: Chunk · Document · RetrievedChunk") understates what the file
contains — it also defines `QueryRequest`, `QueryResponse`, `QueryTrace`, and all
agent API models. Accurate enough not to mislead; worth noting.

---

## Known Limitations

### Intentional limitations (deferred by design)

1. **Three provider config values fall through to FAISS.** Weaviate, Pinecone,
   and Qdrant config keys exist; `get_vector_store()` logs an explicit warning
   and falls through to FAISS for all three (commit 0e6c172). FAISS and Chroma
   are both verified end-to-end with parity integration tests.

2. **AgenticRAG falls through to basic generation.** Setting `GENERATION_STRATEGY=agentic` silently runs basic generation. No warning, no error. `GENERATION_STRATEGY=flare` dispatches correctly to `FLAREGenerator` (commit `6affffb`).

3. **Faithfulness metric disagreement is characterized; resolution deferred.** The April analysis identified that word-overlap and RAGAS give opposite verdicts on Self-RAG. The 2026-05-06 per-claim analysis (commit `c2e70a1`) characterizes the mechanism: Pearson r between Self-RAG's internal verifier and the RAGAS judge across paired queries is near zero (-0.12 to -0.16, p > 0.3 across runs); 34% of queries show internal-accept/RAGAS-reject disagreement against only 3% in the reverse direction; zero queries have both verifiers agreeing the answer was unsupported. The disagreement is structural, not a calibration issue between thresholds on a shared signal. Resolution would require either (a) a shared claim-extraction pass feeding both verifiers to enable claim-level cross-reference, or (b) cross-model judge validation with a stronger model (gpt-4o). Both deferred.

4. **50-query evaluation set is small.** The faithfulness deltas on 50 queries carry
   non-trivial variance. Extending to the full FiQA test split (~648 queries) would
   tighten the CIs on both metrics.

5. **Celery async ingestion requires a running Redis and worker.** Not tested in CI.
   The synchronous ingest path works without it.

6. **chromadb pinned to 0.4.24 with a NumPy 2.x compatibility shim.**
   `vectorstore/chroma_store.py` patches five removed NumPy 2.0 attributes
   at import time so chromadb 0.4.24 imports cleanly under NumPy ≥ 2.0.
   No-op in production (Docker pins numpy 1.26.4). Upgrading chromadb to
   0.5.x removes the shim but requires API migration.

7. **Follow-up prompt is domain-specific.** `FOLLOWUP_TEMPLATE` hardcodes
   "personal-finance topics (IRAs, 401k, taxes, investing, mortgages, similar)."
   Would produce off-domain suggestions if the indexed corpus changes.

8. **LLM-judge non-determinism is comparable to effect size at n=50.** `gpt-4o-mini`'s claim extraction is non-deterministic. Two `PatchedFaithfulness` passes on identical data produce mean per-query differences of 0.05–0.09. Three notebook re-runs of the same data through the same code produced RAGAS support-rate deltas of −0.065, −0.075, −0.115 with Wilcoxon p-values of 0.069, 0.049, 0.007 — straddling α=0.05 across runs. Single-claim queries can flip 1.0 → 0.0 between runs because every claim is load-bearing for the score. The April three-decimal-place RAGAS number (−0.109) is one draw from this distribution; its precision overstates reliability by roughly one order of magnitude. Single-pass LLM-as-judge faithfulness at n=50 has effective measurement uncertainty comparable to the effect sizes being reported. Documented in commit `c2e70a1`.

9. **Verifier disagreement is structural, not calibration.** Self-RAG internal verifier support rate ≈ 0.93; RAGAS judge support rate ≈ 0.45 on the same outputs, same model. Pearson r between the two across 38 paired queries: -0.12 to -0.16, p > 0.3 in all runs. Zero queries had both verifiers agreeing the answer was unsupported. If either were used as a production gate, it would block a different set of answers with no overlap on the rejection side. Documented in commit `c2e70a1`.

10. **Partial structlog migration; bridge is intentional.** Only five representative call sites in commit `d25ae0b` were converted to native structlog keyword-argument calls; 90+ other call sites still use `logging.getLogger(__name__)` and emit through the stdlib bridge. They gain JSON output and `request_id` correlation automatically but log as string messages rather than structured key-value pairs. Full migration would tighten log structure but is not blocking — the bridge approach is intentional and preserves correctness.

11. **Dockerfile dependency pinning is a maintenance hazard.** The Dockerfile pins each dependency manually rather than reading from `pyproject.toml`. Commit `5eeb5ac` added three dependencies that had been added to `pyproject.toml` in `441133d` (prometheus-client, prometheus-fastapi-instrumentator, psutil) but were absent from the Dockerfile, causing `ModuleNotFoundError` on `docker compose up`. Switching to `pip install -e .` in the Dockerfile would fix this structurally; flagged as future work.

12. **Custom `ragcore_*` metrics fire only on completed pipeline queries.** Queries that early-return (e.g., the empty-index guard returning a canned response in <1ms) emit RED metrics from `prometheus-fastapi-instrumentator` but not the `ragcore_stage_duration_seconds` histogram or `ragcore_generation_tokens_total` counter. Fixing requires moving `stage_timings` construction earlier or restructuring to a `finally`-style instrumentation block. Documented in commit `441133d`.

13. **`/health/ready` LLM config check accepts placeholder keys.** The check confirms the API key field is non-empty but does not validate that it is not a placeholder (e.g., `"your-anthropic-key-here"` passes). No live call to the provider is made during the readiness check by design. Flagged in commit `2f8c9c2` as out of scope.

14. **Stale `weaviate_url` and `weaviate_api_key` remain in `config/settings.py`.** The Weaviate service was dropped from `docker-compose.yml` in commit `5eeb5ac`. The two settings keys remain in the codebase as dead config with no runtime impact when `VECTOR_STORE_PROVIDER=faiss`. Flagged for future cleanup.

15. **In-process singleton means Celery-ingested documents require an API restart to become queryable.** `get_vector_store()` returns a module-level singleton (`_vector_store_instance`, `vectorstore/vector_store.py:24`). FAISS loads its index from disk once in `FAISSVectorStore.__init__` and holds it in memory for the process lifetime. Celery workers run in separate OS processes; each `ingest_file_task` constructs a fresh `IngestionPipeline()` (`ingestion/pipeline.py:209`), acquires the worker-process singleton, and persists upserted chunks to disk via `.save()` inside `upsert()`. The API process's singleton is independent — populated at startup and never refreshed — so Celery-written chunks are invisible to queries until the API restarts and `FAISSVectorStore.__init__` reloads from disk. Workaround: restart the API after Celery ingest. Same-process variant fixed 2026-04-17 (`docs/debugging-notes.md`).

### Bugs discovered during this audit

16. **`verify_claims` toggle resets to `False` on page reload**
    ([ui_streamlit/app.py:620](ui_streamlit/app.py#L620)).
    Streamlit session state does not persist across page reloads. A user who enables
    the hallucination verifier and reloads will silently lose the setting. The verifier
    is the only stateful sidebar control; no UI text communicates the reset behavior.

---

## Next Steps if Continued

### Bugs found during this audit

- Either persist `verify_claims` to `st.query_params` so reload restores it, or add
  a caption noting that the setting resets on reload.

### Migrating SCAFFOLD → WORKING

- Wire `AgenticRAG` into `orchestrator.py`'s strategy dispatch. The `GENERATION_STRATEGY=agentic` option currently falls through to basic generation; the `AgenticRAG` class implementation exists but is not yet reachable.

### Migrating PARTIAL → WORKING

- Fix `test_agent_graph_with_tracing` (mock-target resolution bug in the test).
- Exercise `LangfuseTracer` end-to-end against a live Langfuse instance.
- Test Celery async ingestion in CI with a Redis service container.
- Make Self-RAG claim verification provider-agnostic.

### Extending what's working

- Expand FiQA evaluation to the full test split to tighten CI widths on both
  faithfulness deltas.
- Investigate per-claim support rates to test the metric-disagreement mechanism
  hypothesis.
- Emit real Prometheus metrics from retrieval and generation steps.
- Add integration tests for PropositionalChunker, TableAwareChunker, DocumentStructureChunker. All three are dispatched by `get_chunker()` but untested end-to-end. PropositionalChunker would also need provider-agnostic LLM access for the propositional decomposition step.

---

## Honest Verdict

At 2026-05-06 the project state has changed materially since the 2026-04-30 audit. The observability stack is no longer scaffolded: `/metrics` went from a JSON placeholder to real Prometheus instrumentation with 5 custom metrics and RED metrics on every endpoint; `/health` gained two deep sibling endpoints (`/health/live`, `/health/ready`) that actually check system state; structured JSON logging with request-ID correlation is wired; the docker-compose stack runs locally with Prometheus scrape and Grafana dashboard preloaded.

The benchmark integrity finding from the per-claim analysis is the substantive new methodological result. Three judge-controlled re-runs of identical Self-RAG outputs through `PatchedFaithfulness` produced RAGAS support-rate deltas of −0.065, −0.075, −0.115 with Wilcoxon p-values straddling α=0.05 (0.069 / 0.049 / 0.007). The April single-run −0.109 is one draw from this distribution. Pearson r between Self-RAG's internal verifier and the RAGAS judge is near zero across runs (-0.12 to -0.16, p > 0.3); the two verifiers do not share a meaningful ranking signal even with the same underlying model. Single-pass LLM-as-judge faithfulness at n=50 has effective measurement uncertainty comparable to the effect sizes under study — a finding about RAG evaluation instrumentation, not just about this project. The README headline has been updated to report the multi-run summary instead of the April point estimate.

Two latent Self-RAG verifier bugs were found and fixed during 2026-05-06 benchmark runs. The exception handler in `_verify_claim` was fail-open for over a month, silently promoting unparseable LLM responses to verified. The fence-stripping bug caused a benchmark run to return 0 verified / 176 unsupported across all 50 queries (entirely the wrong direction). Neither bug was caught by isolated unit tests; both required production-shape data through the verifier to surface. Both are now fixed and covered by 6 regression tests.

The generation and agent scaffolding perimeter is unchanged since April. AgenticRAG still runs basic generation when selected (planned, not yet wired). FLARE was wired in commit `6affffb` and now runs the dollar-token novelty heuristic with iterative re-retrieval. Three advanced chunkers (Propositional, TableAware, DocumentStructure) are registered in `get_chunker()` but untested end-to-end. Three advanced embedders (Matryoshka, ColBERT, FineTuner) remain unregistered in `get_embedder()` and unreachable from any pipeline path. The lone failing integration test (`test_agent_graph_with_tracing`) is unchanged.

The honest summary: the observability surface has gone from zero to functional, the benchmark analysis has a substantive methodological finding about LLM-as-judge instability at n=50, and the headline framing in the README now reports a multi-run summary that respects the instrument's actual precision. The generation and agent perimeter is unchanged.
