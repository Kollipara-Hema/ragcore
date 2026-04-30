# RAGCore — Self-Audit

**Date:** 2026-04-30
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
| FAISSVectorStore | vectorstore/vector_store.py | WORKING | IndexFlatIP + BM25 hybrid. Singleton confirmed via integration tests. |
| get_vector_store() singleton | vectorstore/vector_store.py | WORKING | Module-level `_vector_store_instance`; `reset_vector_store()` for test isolation. |
| Weaviate / Chroma / Pinecone | vectorstore/vector_store.py | DEFERRED | Config keys exist; `get_vector_store()` logs a warning and falls through to FAISS for any unrecognized provider. |

### Embeddings

| Module | File | State | Notes |
|--------|------|-------|-------|
| BGEEmbedder | embeddings/embedder.py | WORKING | BAAI/bge-large-en-v1.5, local, async via ThreadPoolExecutor. Default in benchmark. |
| MiniLMEmbedder | embeddings/embedder.py | WORKING | Alternative local model; selectable via `EMBEDDING_MODEL`. |
| CachedEmbedder | embeddings/embedder.py | PARTIAL | Redis wrapper; degrades silently to underlying embedder if Redis is absent. |
| CohereEmbedder | embeddings/embedder.py | PARTIAL | API key bug fixed (now passes `settings.cohere_api_key`); not exercised end-to-end. |
| MatryoshkaEmbedder / ColBERTEmbedder / EmbeddingFineTuner | embeddings/advanced_embeddings.py | SCAFFOLD | Not registered in `get_embedder()`; unreachable from any pipeline path. |

### Ingestion

| Module | File | State | Notes |
|--------|------|-------|-------|
| FixedSize / Semantic / Hierarchical / SentenceWindow chunkers | ingestion/chunkers/ | WORKING | All four dispatched by `get_chunker()`. |
| Propositional / TableAware / DocumentStructure chunkers | ingestion/chunkers/ | SCAFFOLD | Classes exist; not registered in `get_chunker()`. |
| PDF / DOCX / TXT / HTML / Web / GitHub loaders | ingestion/loaders/ | WORKING | All exercise real parsing libraries. |

### Retrieval

| Module | File | State | Notes |
|--------|------|-------|-------|
| HeuristicRouter | retrieval/router/query_router.py | WORKING | Regex patterns for MULTI_HOP, ANALYTICAL, LOOKUP; returns None for uncertain queries. |
| LLMQueryClassifier | retrieval/router/query_router.py | WORKING | gpt-4o-mini, json_object mode; degrades to SEMANTIC on API error. |
| QueryExpander | retrieval/router/query_router.py | PARTIAL | Code works; disabled by default (`ENABLE_QUERY_EXPANSION=false`). Not exercised in benchmark. |
| Hybrid / Keyword / Semantic / MultiQuery strategies | retrieval/strategies/retrieval_executor.py | WORKING | All four dispatch correctly. Confirmed end-to-end. |
| ParentChild retrieval | retrieval/strategies/retrieval_executor.py | DEFERRED | `RetrievalStrategy.PARENT_CHILD` enum value retained; removed from strategy dispatch in commit `4a7d800`. Passing it now raises `ValueError`. |

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
| _extract_first_json_array | generation/llm_service.py | SCAFFOLD | **Bug:** defined as a JSON array extractor (line 384) but never called. `generate_followups` uses `_extract_followup_questions` directly. Dead code — see Known Limitations. |
| Ollama / demo providers | generation/llm_service.py | DEFERRED | `_build_llm()` has no branch for ollama or demo; falls through to `OpenAILLM`. |
| LLMService class (legacy) | generation/llm_service.py | SCAFFOLD | Synchronous dict-format interface. Never called by orchestrator. |
| Azure OpenAI | generation/llm_service.py | PARTIAL | Config keys present; `OpenAILLM` builds `AzureOpenAI` client when `AZURE_OPENAI_ENDPOINT` is set. Not exercised end-to-end. |
| SelfRAGGenerator | generation/advanced_generation.py | WORKING | Wired in orchestrator. Activated via `verify_claims=True` per-query override or `GENERATION_STRATEGY=self_rag`. |
| FLAREGenerator | generation/advanced_generation.py | SCAFFOLD | Not dispatched by orchestrator; `GENERATION_STRATEGY=flare` silently runs basic generation with no warning. |
| AgenticRAG | generation/advanced_generation.py | SCAFFOLD | Same — `GENERATION_STRATEGY=agentic` silently runs basic generation. |

### Orchestrator & API

| Module | File | State | Notes |
|--------|------|-------|-------|
| RAGOrchestrator | orchestrator.py | WORKING | Full pipeline confirmed. Includes per-query `verify_claims` override, attributed span extraction, and follow-up generation. |
| _extract_attributed_spans | orchestrator.py | WORKING | Trailing `<cite source="N">` marker parser. Runs on every non-empty response. Graceful `except → (answer, [])` fallback. |
| FastAPI endpoints (/ingest, /query, /query/stream, /health) | api/main.py | WORKING | Integration-tested and smoke-tested. |
| /agent/query endpoint | api/main.py | PARTIAL | Double-invoke removed in commit `1a79581`. One integration test (`test_agent_graph_with_tracing`) fails on mock-target resolution — test bug, not graph bug. |
| /metrics endpoint | api/main.py | DEFERRED | Returns `{"message": "Integrate prometheus_client for production metrics."}`. No real metrics emitted. |
| Celery async ingestion | api/main.py | PARTIAL | Code correct; requires running Redis + Celery worker. Not tested in CI. |

### Agent

| Module | File | State | Notes |
|--------|------|-------|-------|
| LangGraph StateGraph | agent/graph.py | PARTIAL | Single-invoke path works after commit `1a79581`. One integration test still failing. |
| Agent nodes (router, retriever, reranker, generator, evaluator) | agent/nodes/ | WORKING | Each delegates to the same pipeline modules as the main orchestrator. |

### Monitoring

| Module | File | State | Notes |
|--------|------|-------|-------|
| NoOpTracer | monitoring/tracer.py | WORKING | All methods are pass/return. Correct placeholder. |
| LangfuseTracer | monitoring/tracer.py | PARTIAL | Singleton fix applied (commit `6e4c1b6`); `_record_step` TypeError fixed (commit `f424495`). Still never exercised end-to-end against a live Langfuse instance. |

### Evaluation

| Module | File | State | Notes |
|--------|------|-------|-------|
| RetrievalEvaluator | evaluation/evaluator.py | WORKING | hit@K, MRR, NDCG@K, precision@K, recall@K all correctly bounded [0,1]. |
| GenerationEvaluator (heuristic) | evaluation/evaluator.py | PARTIAL | Word-overlap proxy. Reports "faithfulness" but measures token co-occurrence, not factual grounding. |
| GenerationEvaluator (RAGAS) | evaluation/evaluator.py | WORKING | `LLMFaithfulnessEvaluator` wired into `run_benchmark.py`; results committed to `evaluation/results/`. Live API path (`EVAL_STRATEGY=ragas`) remains unvalidated end-to-end. |
| Old Evaluator class | evaluation/evaluator.py | SCAFFOLD | `evaluate()` raises `NotImplementedError`. Dead code. |
| FiQA-2018 benchmark runner | evaluation/scripts/run_benchmark.py | WORKING | 50-query eval; `date.today()` (hardcoded date removed in commit `b04070b`). Reproducible. |

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
| Unit tests (103) | tests/unit/ | WORKING | No external services. `pytest --collect-only` returns 103. README claims 76 — stale (see README Claims Audit). |
| Integration tests (30) | tests/integration/ | PARTIAL | `pytest --collect-only` returns 30. README claims 26/27 — stale. One test (`test_agent_graph_with_tracing`) fails on mock-target resolution in the test itself. |

---

## README Claims Audit

### Headline Results

**"Benchmarked on 50 FiQA-2018 financial Q&A queries."**
→ **REAL.** `evaluation/datasets/` has 50 queries seeded at 42. Raw JSON output in
`evaluation/results/` is committed. Numbers reproducible via
`python evaluation/scripts/run_benchmark.py --strategy basic`.

**Two-metric faithfulness table (word-overlap +0.054, RAGAS −0.109)**
→ **REAL.** Both runs are committed. Word-overlap numbers match `basic_fiqa.json` and
`self_rag_fiqa.json`. RAGAS numbers match `basic_fiqa_2026-04-26_default-judge.json`.
CIs and Wilcoxon p-values are computed in the analysis notebook. The negative Spearman
correlation between metrics on baseline answers (ρ = −0.385) is reproducible from the
same data. The framing is honest: both effects are statistically significant, the metrics
genuinely disagree on baseline answers, and pointing readers at the measurement problem
is more accurate than picking a winner. The framing does not, however, plainly say "RAGAS
shows Self-RAG is worse on its own metric" — that statement is technically derivable from
the numbers but not made explicit.

**"Self-RAG's claim verification loop costs 1.8× the latency of basic generation"**
→ **REAL** in spirit, but the README headline table now reports 10.6s vs 5.9s (+79%),
not 1.8×. The body text says "1.8×"; the table says "+79%". These are inconsistent
(1.8× = +80%). Minor rounding discrepancy; both figures are within measurement noise
of the same underlying data.

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
→ **OVERSTATED.** Only `basic` and `self_rag` dispatch to distinct code paths.
`flare` and `agentic` fall through to basic generation with no warning.

**`LLM_PROVIDER=ollama` and `LLM_PROVIDER=demo`**
→ **OVERSTATED.** `_build_llm()` has no branch for ollama or demo. Both fall through
to `OpenAILLM`, which will fail without `OPENAI_API_KEY`.

**`VECTOR_STORE_PROVIDER=weaviate|chroma`**
→ **OVERSTATED.** `get_vector_store()` falls through to FAISS for any unrecognized
provider.

**`ENABLE_TRACING=true` (Langfuse)**
→ **PARTIAL.** `LangfuseTracer` is structurally correct and the singleton bug is
fixed. It has never been exercised end-to-end against a live Langfuse instance.

**`ENABLE_EVALUATION=true`, `EVAL_STRATEGY=ragas`**
→ **PARTIAL.** RAGAS runs in the benchmark script. The live API path
(`EVAL_STRATEGY=ragas` per-query) remains unvalidated.

### Status Section

The README's Status section is accurate. It correctly flags: Self-RAG hardcoded to
gpt-4o-mini, NoOpTracer, no real Prometheus metrics, FLARE and Agentic RAG not wired
to the API, Weaviate/Chroma/Pinecone unverified, RAGAS requiring extra install.
No corrections needed in this section.

### Testing Section

**"76 unit tests passing, 26 of 27 integration tests passing"**
→ **STALE.** `pytest --collect-only` on 2026-04-30 returns **103 unit tests** and
**30 integration tests** — substantially higher than both figures. Tests were added
for attributed spans, follow-up generation, orchestrator verify_claims, and the tracer
singleton since the README was last updated. The README count needs updating.

### Project Structure

The directory tree matches the actual structure. `utils/models.py` description
("Shared types: Chunk · Document · RetrievedChunk") understates what the file
contains — it also defines `QueryRequest`, `QueryResponse`, `QueryTrace`, and all
agent API models. Accurate enough not to mislead; worth noting.

---

## Known Limitations

### Intentional limitations (deferred by design)

1. **Single vector store end-to-end.** FAISS is the only verified path. Weaviate,
   Chroma, and Pinecone configs exist but fall through to FAISS at runtime.

2. **Self-RAG claim verification is OpenAI-only.** `_extract_claims()` and
   `_verify_claim()` in `SelfRAGGenerator` hardcode `AsyncOpenAI`. Setting
   `LLM_PROVIDER=anthropic` still requires `OPENAI_API_KEY` for Self-RAG.

3. **FLARE and AgenticRAG are scaffolded.** Setting `GENERATION_STRATEGY=flare`
   or `GENERATION_STRATEGY=agentic` silently runs basic generation. No warning,
   no error.

4. **Faithfulness metric disagreement is unresolved.** Word-overlap and RAGAS give
   opposite verdicts on Self-RAG. Per-claim support-rate analysis is required to
   determine which signal is more reliable.

5. **50-query evaluation set is small.** The faithfulness deltas on 50 queries carry
   non-trivial variance. Extending to the full FiQA test split (~648 queries) would
   tighten the CIs on both metrics.

6. **Celery async ingestion requires a running Redis and worker.** Not tested in CI.
   The synchronous ingest path works without it.

7. **Follow-up prompt is domain-specific.** `FOLLOWUP_TEMPLATE` hardcodes
   "personal-finance topics (IRAs, 401k, taxes, investing, mortgages, similar)."
   Would produce off-domain suggestions if the indexed corpus changes.

### Bugs discovered during this audit

8. **`_extract_first_json_array` is dead code**
   ([generation/llm_service.py:384](generation/llm_service.py#L384)).
   Defined as a JSON array extractor for follow-up parsing. `generate_followups`
   uses `_extract_followup_questions` directly; `_extract_first_json_array` is never
   called from anywhere. A future reader tracing the follow-up path will be confused
   about which function is live. Either delete it or add a comment explaining the
   relationship.

9. **Version marker still rendering in production**
   ([ui_streamlit/app.py:682](ui_streamlit/app.py#L682)).
   The sidebar renders `"UI build: 2026-04-29-followup-v1"` on every page load of
   the deployed demo. The inline comment says "remove after cloud deploy confirmed"
   — that condition has passed.

10. **`verify_claims` toggle resets to `False` on page reload**
    ([ui_streamlit/app.py:620](ui_streamlit/app.py#L620)).
    Streamlit session state does not persist across page reloads. A user who enables
    the hallucination verifier and reloads will silently lose the setting. The verifier
    is the only stateful sidebar control; no UI text communicates the reset behavior.

---

## Next Steps if Continued

### Bugs found during this audit

- Remove `_extract_first_json_array` from `llm_service.py` (dead code, never called).
- Remove the version-marker `<p>` block from the sidebar (deploy condition has passed).
- Either persist `verify_claims` to `st.query_params` so reload restores it, or add
  a caption noting that the setting resets on reload.

### Migrating SCAFFOLD → WORKING

- Wire `FLAREGenerator` into `orchestrator.py`'s strategy dispatch. The class
  structure is complete; only the `elif settings.generation_strategy == "flare":`
  branch is missing.
- Wire `AgenticRAG` similarly. Both `GENERATION_STRATEGY` options would then
  correspond to real code paths.
- Register `PropositionalChunker`, `TableAwareChunker`, `DocumentStructureChunker`
  in `get_chunker()` if they are intended to be user-selectable.

### Migrating PARTIAL → WORKING

- Fix `test_agent_graph_with_tracing` (mock-target resolution bug in the test).
- Exercise `LangfuseTracer` end-to-end against a live Langfuse instance.
- Test Celery async ingestion in CI with a Redis service container.
- Update README test counts to 103 unit / 30 integration.
- Make Self-RAG claim verification provider-agnostic.

### Extending what's working

- Expand FiQA evaluation to the full test split to tighten CI widths on both
  faithfulness deltas.
- Investigate per-claim support rates to test the metric-disagreement mechanism
  hypothesis.
- Emit real Prometheus metrics from retrieval and generation steps.

---

## Honest Verdict

At 2026-04-30 the system has moved from "one well-tested pipeline path with surrounding
scaffolding" to a deployed interactive demo with materially more surface area. The
citation-span pipeline, follow-up generation, hallucination verifier toggle, recalibrated
confidence indicator, and visual rebuild are all live at ragcore.streamlit.app. The
benchmark pipeline now carries two faithfulness signals, not one — and their disagreement,
not Self-RAG's effect size, is the headline finding.

The deployed story is real. The frontend runs on Streamlit Cloud; the backend runs on
Render free tier at ragcore-api.onrender.com. The cold-start caveat in the README is
accurate. The stat cards in the UI (hit@5 = 0.92, MRR = 0.86) are populated from the
committed benchmark JSON, not hardcoded aspirationally.

The metric-disagreement finding signals something substantive about evaluation rigor: word-overlap
and RAGAS are measuring different constructs. The negative Spearman correlation between them on
baseline answers (ρ = −0.385) makes this concrete. The pre-registered analysis plan, dual-run
preservation, and documented methodology change mean the disagreement is surfaced rather than
buried.

Scaffolding remains where it was in April: FLARE and AgenticRAG still run basic generation
when selected, three chunkers and three embedders are unreachable, and four provider config
values still fall through to FAISS. None of this has gotten worse since the April 27 audit.

Three latent bugs found in this audit: dead code in `llm_service.py`, a visible cosmetic
artifact in the production demo, and a Streamlit session-state UX limitation on the verifier
toggle. None are in the benchmarked code path. The previously documented bugs (agent
double-invoke, LangfuseTracer TypeError, CohereEmbedder wrong key, broken ParentChild
retrieval, dead retrieval methods) were all fixed in the April 26 cleanup commits and are
not carried forward here.

The honest summary: one solid pipeline deployed end-to-end, a benchmark that is honest about
what it measures and what it doesn't, and a scaffolding perimeter that is accurately described
in the README and unchanged since April.
