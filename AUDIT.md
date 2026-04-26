# RAGCore — Self-Audit

**Date:** 2026-04-19  
**Auditor:** Hema Kollipara  
**Scope:** Post-Day-2 state — singleton fix, Self-RAG wired, mock agent deleted,
FiQA-2018 benchmark complete.

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

| Module | File | State | Notes |
|--------|------|-------|-------|
| FAISSVectorStore | vectorstore/vector_store.py | WORKING | IndexFlatIP + BM25 hybrid. Singleton confirmed via integration tests. |
| get_vector_store() singleton | vectorstore/vector_store.py | WORKING | Module-level `_vector_store_instance`. `reset_vector_store()` for test isolation. |
| Weaviate / Chroma / Pinecone | vectorstore/vector_store.py | DEFERRED | Config keys exist; `get_vector_store()` logs a warning and falls through to FAISS. |
| BGEEmbedder | embeddings/embedder.py | WORKING | BAAI/bge-large-en-v1.5, local, async via ThreadPoolExecutor. Default in benchmark. |
| MiniLMEmbedder | embeddings/embedder.py | WORKING | Alternative local model; selectable via `EMBEDDING_MODEL`. |
| CachedEmbedder | embeddings/embedder.py | PARTIAL | Redis wrapper exists; silently degrades to underlying embedder if Redis is absent. |
| CohereEmbedder | embeddings/embedder.py | PARTIAL | **Bug:** passes `settings.openai_api_key` to Cohere client instead of a Cohere key. |
| MatryoshkaEmbedder / ColBERTEmbedder / EmbeddingFineTuner | embeddings/advanced_embeddings.py | SCAFFOLD | Not registered in `get_embedder()`; not reachable from any pipeline path. |
| FixedSize / Semantic / Hierarchical / SentenceWindow chunkers | ingestion/chunkers/ | WORKING | All four dispatched by `get_chunker()`. |
| Propositional / TableAware / DocumentStructure chunkers | ingestion/chunkers/ | SCAFFOLD | Classes exist; not registered in `get_chunker()`. |
| PDF / DOCX / TXT / HTML / Web / GitHub loaders | ingestion/loaders/ | WORKING | All exercise real parsing libraries. |
| HeuristicRouter | retrieval/router/query_router.py | WORKING | Regex patterns for MULTI_HOP, ANALYTICAL, LOOKUP; returns None for uncertain. |
| LLMQueryClassifier | retrieval/router/query_router.py | WORKING | gpt-4o-mini, json_object mode. Degrades to SEMANTIC on API error. |
| QueryExpander | retrieval/router/query_router.py | PARTIAL | Code works; disabled by default (`ENABLE_QUERY_EXPANSION=false`). Not exercised in benchmark. |
| Hybrid / Keyword / Semantic / MultiQuery strategies | retrieval/strategies/retrieval_executor.py | WORKING | All four dispatch correctly. Confirmed end-to-end. |
| ParentChild retrieval | retrieval/strategies/retrieval_executor.py | SCAFFOLD | `_parent_child_search()` filters on `is_child_chunk=True` — a metadata key ingestion never sets. Returns `child_results[0]` repeated `top_k` times. |
| Dead retrieval methods | retrieval/strategies/retrieval_executor.py | SCAFFOLD | `dense_retrieval()`, `sparse_retrieval()`, `hybrid_retrieval()` reference `self.embedder` and `self.vector_store` which don't exist on the class. Would raise AttributeError. Never called by `execute()`. |
| CrossEncoderReranker | reranking/reranker.py | WORKING | cross-encoder/ms-marco-MiniLM-L-6-v2. Verified end-to-end. |
| NoOpReranker | reranking/reranker.py | WORKING | Passthrough; assigns ranks. Correct placeholder. |
| PromptBuilder | generation/prompts/prompt_builder.py | WORKING | Token-budget-aware; per-query-type templates; labels sources. |
| GenerationService (GroqLLM / OpenAILLM / AnthropicLLM) | generation/llm_service.py | WORKING | Three providers wired; streaming works. |
| Ollama / demo providers | generation/llm_service.py | DEFERRED | `_build_llm()` falls through to OpenAI for all unrecognized providers including ollama and demo. |
| LLMService class (legacy) | generation/llm_service.py | SCAFFOLD | Synchronous, dict-format chunks. Never called by orchestrator. |
| Azure OpenAI | generation/llm_service.py | PARTIAL | Config keys present; `OpenAILLM` constructor checks for azure endpoint and builds `AzureOpenAI` client. Not exercised end-to-end. |
| SelfRAGGenerator | generation/advanced_generation.py | WORKING | Wired in orchestrator. Verified end-to-end. Claim extraction and verification confirmed running post-fix. |
| FLAREGenerator | generation/advanced_generation.py | SCAFFOLD | Class present, structured. `orchestrator.py` falls through to basic generation for `GENERATION_STRATEGY=flare` with no warning. |
| AgenticRAG | generation/advanced_generation.py | SCAFFOLD | Same — not dispatched. `GENERATION_STRATEGY=agentic` silently runs basic generation. |
| RAGOrchestrator | orchestrator.py | WORKING | Full pipeline confirmed. Self-RAG and basic paths both verified. |
| FastAPI endpoints (/ingest, /query, /query/stream, /health) | api/main.py | WORKING | Tested via integration tests and manual smoke tests. |
| /agent/query endpoint | api/main.py | PARTIAL | **Bug:** calls `graph.astream_events(state)` to collect events, then `graph.ainvoke(state)` — graph executes twice per request. No comment explaining the double call. |
| /metrics endpoint | api/main.py | DEFERRED | Returns `{"message": "Integrate prometheus_client for production metrics."}`. Emits no real metrics. |
| Celery async ingestion | api/main.py | PARTIAL | Code correct; requires a running Redis and Celery worker. Not tested in CI. |
| LangGraph StateGraph | agent/graph.py | PARTIAL | Graph wires 5 nodes with retry conditional edge. Double-invoke in API endpoint means graph runs twice per request. |
| Agent nodes (router, retriever, reranker, generator, evaluator) | agent/nodes/ | WORKING | Each node delegates to the same pipeline modules as the main orchestrator. |
| NoOpTracer | monitoring/tracer.py | WORKING | All methods are pass/return immediately. Correct placeholder. |
| LangfuseTracer | monitoring/tracer.py | PARTIAL | **Bug:** `_record_step()` treats a QueryTrace dataclass as a dict — `self._traces[trace_id]["steps"]` raises TypeError at runtime. Only triggered when `ENABLE_TRACING=true` and `LANGFUSE_PUBLIC_KEY` is set. |
| RetrievalEvaluator | evaluation/evaluator.py | WORKING | hit@K, MRR, NDCG@K, precision@K, recall@K all correctly bounded [0,1] post-dedup fix. |
| GenerationEvaluator (heuristic) | evaluation/evaluator.py | PARTIAL | Word-overlap proxy is the live path. Reports "faithfulness" but measures token co-occurrence, not factual grounding. |
| GenerationEvaluator (RAGAS) | evaluation/evaluator.py | PARTIAL | Code present; requires `pip install -e "[eval]"`. Not run in benchmark — FiQA faithfulness numbers are heuristic. |
| Old Evaluator class | evaluation/evaluator.py | SCAFFOLD | `evaluate()` raises NotImplementedError with a migration message. Dead code. |
| FiQA-2018 benchmark runner | evaluation/scripts/run_benchmark.py | WORKING | 50-query eval. UUID→FiQA-ID translation, chunk dedup. Reproducible. |
| Unit tests (75) | tests/unit/ | WORKING | No external services. All passing. |
| Integration tests (25) | tests/integration/ | PARTIAL | 23 passing; 2 failing in TestObservabilityIntegration on scaffolded tracer code. |

---

## README Claims Audit

### Headline Results

**"Benchmarked on 50 FiQA-2018 financial Q&A queries."**
→ **REAL.** `evaluation/datasets/` has 50 queries seeded at 42. Raw JSON output in
`evaluation/results/` is committed. Numbers are reproducible via
`python evaluation/scripts/run_benchmark.py --strategy basic`.

**"faithfulness 0.36 → 0.43, +20%"**
→ **PARTIAL.** Numbers are real outputs from the benchmark run. However, "faithfulness"
here is word-overlap between the answer and retrieved contexts — not an LLM-judged
factual entailment score. RAGAS faithfulness is an optional install, not what produced
these numbers. The README does not state this distinction.

**"Self-RAG's claim verification loop costs 1.8× the latency of basic generation"**
→ **REAL.** 4.7s vs 8.5s is consistent with the per-query data in the benchmark JSON.

**Retrieval metrics identical between strategies (hit@5=0.92, MRR=0.86, NDCG@5=0.75)**
→ **REAL.** Both strategy runs use the same retrieval path. Retrieval metrics being
identical is correct by construction.

### Architecture Diagram

**"Hybrid Index: FAISS Dense Vectors + BM25 Sparse Index"**
→ **REAL.** Both are in-memory in FAISSVectorStore. Alpha-weighted fusion implemented.

**"Query Router: Heuristic → LLM fallback"**
→ **REAL.** HeuristicRouter (regex) → LLMQueryClassifier (gpt-4o-mini) exactly as drawn.

**"Reranker: Cross-Encoder"**
→ **REAL.** CrossEncoderReranker (ms-marco-MiniLM-L-6-v2) in the live path.

### Retrieval Strategy Routing

**"6 query types → strategy pairs"**
→ **REAL.** All 6 pairs present in STRATEGY_MAP. 5 of 6 strategies dispatch correctly.
ParentChild is the exception (see module table).

**ParentChild implied as a live strategy**
→ **OVERSTATED.** The strategy exists in STRATEGY_MAP but `_parent_child_search()`
filters on metadata that ingestion never sets. Returns the top chunk repeated `top_k`
times.

### Configuration

**`GENERATION_STRATEGY` options: `basic`, `self_rag`, `flare`, `agentic`**
→ **OVERSTATED.** Only `basic` and `self_rag` dispatch to distinct code paths.
`flare` and `agentic` fall through to basic generation in `orchestrator.py` with no
warning logged and no error raised.

**`LLM_PROVIDER=ollama` and `LLM_PROVIDER=demo`**
→ **OVERSTATED.** `_build_llm()` has no branch for ollama or demo. Both fall through
to the OpenAI client, which will fail if no `OPENAI_API_KEY` is set.

**`VECTOR_STORE_PROVIDER=weaviate|chroma`**
→ **OVERSTATED.** `get_vector_store()` logs a warning and creates a FAISSVectorStore
for any unrecognized provider.

**`ENABLE_TRACING=true` (Langfuse)**
→ **PARTIAL.** `LangfuseTracer` would fail on first traced request due to the
`_record_step()` TypeError. Tracing has never been exercised end-to-end.

**`ENABLE_EVALUATION=true`, `EVAL_STRATEGY=ragas`**
→ **PARTIAL.** RAGAS path exists; requires extra install. Not run in any committed
benchmark. Heuristic path is the default and what produces the committed results.

### Status Section

The README's Status section is the most honest part of the document. It correctly
flags:

- Self-RAG hardcoded to gpt-4o-mini for verification
- NoOpTracer, Langfuse off by default
- Prometheus/Grafana emitting no real metrics
- FLARE and Agentic RAG not wired to API
- Weaviate/Chroma/Pinecone unverified
- RAGAS requiring extra install

No corrections needed here. The Status section reads as written by someone who ran
the system, not as aspirational documentation.

### Project Structure

The directory tree in the README matches the actual structure. All listed paths exist.
The description of `agent/nodes/` as "router · retriever · reranker · generator ·
evaluator" is accurate.

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
   no error. A user following the README's config table would not know this.

4. **Faithfulness metric is word-overlap, not LLM-judged.** The 0.36/0.43
   faithfulness numbers in the README are word co-occurrence scores. RAGAS
   (semantic, LLM-judged) exists but is not run in any committed evaluation.

5. **50-query evaluation set is small.** The faithfulness delta (+20%) from
   50 queries has high variance. Extending to the full FiQA test split would
   reduce uncertainty.

6. **Celery async ingestion requires a running Redis and worker.** Not tested
   in CI. The synchronous ingest path works without it.

### Bugs discovered during this audit (to fix)

7. **Agent double-invoke** (FIXED — see below) ([api/main.py](api/main.py)). `agent_query()` calls
   `graph.astream_events(state)` to collect trace events, then calls
   `graph.ainvoke(state)` for the result. The full graph executes twice per
   `/agent/query` request. No comment explains the double call.

8. **LangfuseTracer TypeError** (FIXED — see below) ([monitoring/tracer.py](monitoring/tracer.py)).
   `_record_step()` accesses `self._traces[trace_id]["steps"]` — treating a
   `QueryTrace` dataclass as a dict. Raises `TypeError` on first traced request
   when `ENABLE_TRACING=true`.

9. **CohereEmbedder wrong API key** (FIXED — see below) ([embeddings/embedder.py](embeddings/embedder.py)).
   `CohereEmbedder` initializes with `api_key=settings.openai_api_key`. Will
   authenticate against Cohere with the wrong credential type.

10. **ParentChild retrieval is broken** (FIXED — see below)
    ([retrieval/strategies/retrieval_executor.py](retrieval/strategies/retrieval_executor.py)).
    `_parent_child_search()` filters on `is_child_chunk=True` — metadata the
    ingestion pipeline never writes. Returns `child_results[0]` repeated `top_k`
    times when called.

11. **Dead retrieval methods with wrong attributes** (FIXED — see below)
    ([retrieval/strategies/retrieval_executor.py](retrieval/strategies/retrieval_executor.py)).
    `dense_retrieval()`, `sparse_retrieval()`, and `hybrid_retrieval()` reference
    `self.embedder` and `self.vector_store`, which are not attributes of
    `RetrievalExecutor`. Would raise `AttributeError` if called. Not reachable
    from `execute()`.

### Fixed in 2026-04-26 cleanup

- Item 7 — Agent double-invoke removed; single `ainvoke` call — commit `1a79581`
- Item 8 — LangfuseTracer `_record_step` stale recording path deleted — commit `f424495`
- Item 9 — CohereEmbedder now passes `settings.cohere_api_key`; field added to Settings — commit `c376880`
- Item 10 — PARENT_CHILD removed from `_dispatch`; `_parent_child_search` deleted — commit `4a7d800`
- Item 11 — `dense_retrieval`, `sparse_retrieval`, `hybrid_retrieval` deleted — commit `9e9440e`

---

## Next Steps if Continued

### Bugs found during this audit

These are broken — they will error if the relevant code path is exercised:

- Fix agent double-invoke in `api/main.py`: replace `astream_events` + `ainvoke`
  with a single `astream_events` call that captures both events and the final state,
  or drop event collection and use `ainvoke` alone.
- Fix `LangfuseTracer._record_step()`: access the `QueryTrace` dataclass by
  attribute (`self._traces[trace_id].steps`) not by dict key.
- Fix `CohereEmbedder`: pass `settings.cohere_api_key` (add to settings if missing)
  instead of `settings.openai_api_key`.
- Fix or remove `_parent_child_search()`: either implement parent chunk metadata in
  the ingestion pipeline or remove PARENT_CHILD from STRATEGY_MAP.
- Remove or clearly gate the dead retrieval methods (`dense_retrieval`,
  `sparse_retrieval`, `hybrid_retrieval`) to prevent confusion.

### Migrating SCAFFOLD → WORKING

These are structured but not wired — lower risk, higher ROI than greenfield:

- Wire `FLAREGenerator` into `orchestrator.py`'s strategy dispatch. The class
  structure is complete; only the `elif settings.generation_strategy == "flare":`
  branch is missing.
- Wire `AgenticRAG` similarly. Then the four `GENERATION_STRATEGY` values in the
  README config table will all correspond to real code paths.
- Register `PropositionalChunker`, `TableAwareChunker`, `DocumentStructureChunker`
  in `get_chunker()` if they are intended to be user-selectable.
- Make Self-RAG claim verification provider-agnostic: pass the orchestrator's
  `_generation` service into `SelfRAGGenerator` instead of constructing a new
  `AsyncOpenAI` client internally.

### Migrating PARTIAL → WORKING

- Run RAGAS faithfulness evaluation on the committed benchmark queries and replace
  or supplement the word-overlap faithfulness numbers.
- Fix the two failing `TestObservabilityIntegration` tests.
- Exercise `LangfuseTracer` end-to-end (after fixing the TypeError) — it has
  never been called against a live Langfuse instance.
- Test Celery async ingestion in CI with a Redis service container.

### Extending what's working

- Expand FiQA evaluation to the full test split (~648 queries) to reduce
  variance in faithfulness estimates.
- Emit real Prometheus metrics from retrieval and generation steps; the
  `/metrics` stub currently returns a plain text message.
- Deploy a live demo (Streamlit Cloud is the likely path, per README).

---

## Honest Verdict

RAGCore's core retrieval and generation pipeline is real. The FAISS + BM25 hybrid
retrieval, cross-encoder reranking, and Self-RAG generation all execute against
live data, and the FiQA benchmark numbers are reproducible from committed code.
The debugging notes document real bugs and real fixes, not hypothetical ones.

The project also carries a meaningful layer of scaffolding. Three generation
strategies listed in the README config table are silent no-ops. Three chunkers and
three embedders are unreachable from any pipeline path. The vector store provider
config allows four values but only one works. This is normal for a research project
that built and benchmarked one solid path rather than five shallow ones — but the
README's Configuration table implies parity that doesn't exist.

Five bugs found during this audit were not previously documented: the agent
double-invoke, the LangfuseTracer TypeError, the CohereEmbedder wrong API key,
the broken ParentChild retrieval, and the dead retrieval methods with wrong
attribute references. These are latent, not active — none are in the benchmarked
code path — but they would surface immediately if the corresponding config options
were exercised. All five were fixed in the 2026-04-26 cleanup commits (`1a79581` through `4a7d800`).

The honest summary: one well-tested path in production shape, surrounded by
correctly-structured scaffolding and a small set of latent bugs. The benchmark
is real; the breadth is not yet there.
