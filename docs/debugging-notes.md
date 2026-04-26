# Debugging Notes

Real bugs encountered during development and how they were fixed.
Kept as a running log — useful for interview prep and for future me.

---

## 2026-04-17 — Vector store singleton bug

### Symptom

End-to-end smoke test: `POST /ingest/text` returned `{"status":"indexed","message":"Successfully indexed 1 chunks."}`. The immediately-following `POST /query` returned `"I could not find relevant information in the indexed documents"` with zero citations and `total_tokens: 0`.

### Hypotheses considered

1. Stale data from earlier Streamlit testing polluting the index.
2. Disk persistence not happening despite the "indexed" response.
3. Ingestion and retrieval using separate in-memory vector store instances.

### Tests run

1. Wiped `faiss_index.idx` and `faiss_metadata.pkl`, retried → same failure. **Ruled out (1).**
2. Checked file size before and after ingest → grew by ~4KB each time (one 1024-dim float32 vector). **Ruled out (2).**
3. Grep'd `get_vector_store()` callers → found ingestion and retrieval each call it once at construction time and cache the result.

### Root cause

`get_vector_store()` in `vectorstore/vector_store.py` was constructing a new `FAISSVectorStore()` on every call with no singleton pattern. `IngestionPipeline.__init__` and `RetrievalExecutor.__init__` each held references to different instances. Writes went to instance A; reads came from instance B. Because FAISS keeps its index in memory and only reads from disk in `__init__`, instance B was permanently stale.

### Fix

Module-level singleton in `vectorstore/vector_store.py`:

- Added `_vector_store_instance: Optional[BaseVectorStore] = None` at module level
- Rewrote `get_vector_store()` to check the cache before constructing
- Added `reset_vector_store()` for test isolation

### Regression tests

Added `tests/integration/test_vectorstore_singleton.py` with four tests. The critical one:

- `test_upserted_chunks_are_searchable_via_second_get_vector_store_call` — upserts via one `get_vector_store()` call, searches via another. Would have failed before the fix.

### Verification

After fix, the same ingest + query cycle returns a correctly grounded answer:

> "The Q3 2024 revenue was $4.2 billion, which represents a 15% growth year-over-year [Source 1]."

Score: 7.6617 (positive, high). Citations match the ingested chunk's `doc_id`.

### Commit

`953e62d` — Make get_vector_store() a module-level singleton

---

## 2026-04-17 — Known issue: Self-RAG claim verification parser

### Symptom

With `GENERATION_STRATEGY=self_rag`, queries succeed and return correct answers, but uvicorn logs print:
Claim verification failed: '"supported"'
Claim verification failed: '"supported"'

### Root cause (suspected)

`SelfRAGGenerator._verify_claim()` in `generation/advanced_generation.py` appears to parse the LLM's response assuming a bare string or boolean, but the LLM (gpt-4o-mini) is returning a JSON-quoted string (`"supported"` with literal quotes). The parsing step throws an exception that's caught and logged, and Self-RAG falls back to the initial generation without verification.

### Impact

System degrades gracefully — returns correct answers from initial generation. Verification loop is a no-op.

### Status

Logged, not yet fixed. Low priority because end-user output is correct. To fix: inspect `_verify_claim`'s response parser, add handling for JSON-quoted strings.

---

## 2026-04-18 — Self-RAG claim verification silently failing

### Symptom

With `GENERATION_STRATEGY=self_rag`, queries succeeded and returned correct
grounded answers, but uvicorn logs printed on every query:
Claim verification failed: '"supported"'
Claim verification failed: '"supported"'
One line per claim extracted. The broad `except Exception as e` in
`_verify_claim` was catching the error and logging only `str(e)`, so the
full traceback was hidden. Self-RAG was silently degrading to basic
generation — the initial answer was correct, but the verification loop
never actually ran.

### Initial hypothesis (wrong)

The error message `'"supported"'` with literal quotes looked like a JSON
parser bug: the LLM was returning a bare JSON string instead of a dict,
and `json.loads('"supported"')` → `str`, and `.get()` on a string raises
`AttributeError`. Plausible and consistent with the symptom.

### Actual root cause

The bug was one layer upstream. `VERIFICATION_PROMPT` contained the JSON
schema example `{"supported": true/false, "evidence": "..."}` as a literal
string. When this template was passed through `str.format()` to fill in
the claim and context fields, Python interpreted `{"supported": ...}` as a
format field named `"supported"` and raised `KeyError: '"supported"'` —
the exact string in the log.

The LLM was never being called. The parser wasn't the problem — the
prompt wasn't rendering.

### Why the initial hypothesis was wrong but produced the right symptom

`KeyError('"supported"')` stringifies to `'"supported"'` — identical to
what a parser failure on a bare JSON string would look like. The masking
exception handler made them indistinguishable from the log line alone.

### Fix

Three layers of defense, because debugging revealed the parser *also* had
bugs that would surface once the prompt was fixed:

1. **Prompt template** (`generation/advanced_generation.py:97`): escaped
   the JSON schema braces (`{{` and `}}`) so `str.format()` treats them as
   literal output instead of format fields.

2. **`_verify_claim` parser**: added defensive handling for bare-string
   responses and string-valued booleans (`"false"` as a string is
   truthy — the original code would silently treat it as verified).
   Positive whitelist (`"true"/"yes"/"supported"/"1"`) rather than
   negative blacklist, so unknown/gibberish responses fail closed.

3. **`_extract_claims` parser**: removed a too-permissive fallback that
   would treat any list-valued field in the response as claims. Restricted
   to explicit keys (`claims` / `statements` / `items`), falls through to
   sentence-split otherwise.

### Regression tests

`tests/unit/test_self_rag_verification.py` — 6 tests covering:
- Standard dict response with `{"supported": true, "evidence": "..."}`
- Bare-string response `"supported"` (the original failure mode)
- String-valued boolean `{"supported": "false"}` (the silent
  false-positive bug)
- Gibberish response (must fail closed → `(False, "")`)
- Dict without evidence field `{"supported": true}`
- `_extract_claims` with non-"claims" key (`statements`)

### Verification

End-to-end smoke test: ingest + query, no `Claim verification failed`
lines in logs. Self-RAG exercises the full loop — initial generation,
claim extraction, per-claim verification. Response cites both retrieved
chunks as `[Source 1][Source 2]`.

### Lesson

A broad `except Exception` that logs only `str(e)` can make two different
bugs produce identical log output. If the exception handler had logged
`type(e).__name__` alongside `str(e)`, the distinction between
`KeyError('"supported"')` and `AttributeError(...)` would have been
obvious and saved an investigation step. Worth remembering.

### Commit

`a12c06e` — Fix Self-RAG claim verification end-to-end

---

## 2026-04-18 — UUID vs FiQA-ID mismatch in benchmark retrieval

### Symptom

Initial Stage B sanity check: 0/3 test queries had any overlap between cited doc IDs and the golden set's `relevant_doc_ids`, despite the system retrieving clearly relevant-looking financial content. Hit rate appeared to be 0.0 across the board.

### Hypotheses considered

1. Retrieval was broken — wrong documents being returned entirely.
2. The golden dataset's `relevant_doc_ids` format didn't match what the system produces.
3. The system's internal document IDs and FiQA's corpus IDs were living in separate spaces with no translation layer.

### Actual root cause

Two ID spaces that never converge in the code path the evaluator reads. `IngestionPipeline.ingest_text()` assigns a fresh UUID as `Document.doc_id` at ingest time. The original FiQA corpus ID (e.g., `"44085"`) is stored in `metadata["custom"]["doc_id"]` inside the FAISS pickle — but the `Citation` object and API response surface only the internal UUID as `doc_id`. The evaluator was comparing UUID strings to integer-string corpus IDs; no match was possible.

Impact: without correction, all retrieval metrics (hit rate, MRR, NDCG, precision, recall) would silently be 0.0 in any benchmark, even with perfect retrieval.

### Fix

Implemented `_build_uuid_to_fiqa_id()` in the benchmark script: reads the FAISS pickle at startup, iterates metadata entries, maps `m["doc_id"]` (UUID) to `m["custom"]["doc_id"]` (FiQA string). Applied the translation to all cited UUIDs before computing any metric. After fix, Stage B sanity check showed 2/3 queries with correct overlap.

The translation belongs inside the system — either the ingestion pipeline should preserve source corpus IDs as first-class `doc_id`, or `Citation` should include an `external_id` field. Deferred; the benchmark script carries the workaround for now.

### Regression tests or verification

Stage B sanity check script (`evaluation/scripts/stage_b_sanity.py`) re-run after fix: 2/3 queries show cited doc overlap with ground truth. The remaining miss is expected — FiQA's qrels don't cover every plausible relevant document.

### Lesson

Watch for dual ID spaces anywhere a system assigns its own identifiers to data that already has external identifiers. The symptom — 0% overlap with ground truth — looks exactly like broken retrieval when it is actually an ID-space misalignment. The correct diagnostic question is not "why is retrieval failing?" but "are the IDs being compared even drawn from the same space?"

### Commit

`ff2aca2` — Add Stage B sanity check script and results

---

## 2026-04-18 — NDCG and recall exceeding 1.0 in Stage C benchmark

### Symptom

Baseline benchmark across 50 FiQA queries reported:
- NDCG@5 mean 0.83, **max 1.808**
- recall@5 mean 0.86, **max 2.5**

NDCG is normalized by definition — maximum possible is 1.0. Recall is bounded by |relevant ∩ retrieved| / |relevant|; it cannot exceed 1.0 for any individual query. These numbers are mathematically impossible.

### Hypotheses considered

Initial explanation offered: "artefact of small corpus." That is wrong. Small corpora don't break the mathematical bounds of DCG normalization or set-intersection recall. The answer was not in the data distribution.

### Actual root cause

The UUID→FiQA-ID translation (see entry above) collapses multiple chunks from the same source document onto the same FiQA corpus ID. A single query could return a retrieved list like `['44085', '44085', '70305', '70305', '44085']` — five chunk-level results, three mapping to the same document.

Feeding that raw list directly to the metric functions:

- **NDCG**: DCG summed gains from every duplicate occurrence of a relevant doc. IDCG normalized by unique relevant count. DCG / IDCG > 1.0.
- **Recall**: counted hits per duplicate occurrence against `|unique relevant|`. Three hits on the same document against a denominator of 1 → 3.0.

### Fix

Deduplicate the retrieved list by first-occurrence rank before scoring, in both `RetrievalEvaluator.evaluate()` (`evaluation/evaluator.py`) and `retrieval_metrics()` (`evaluation/scripts/run_benchmark.py`). First-occurrence-rank preserves the rank at which each document was first surfaced — correct for rank-based metrics. Both dedup blocks carry the same comment explaining the design decision.

After fix: hit@5 (0.92) and MRR (0.86) unchanged — unaffected by duplicates. NDCG@5 corrected to 0.745; recall@5 to 0.707; precision@5 from 0.412 (chunk-level, inflated) to 0.399 (doc-level). All values in [0, 1].

### Regression tests or verification

After fix, all 50 queries produce per-query NDCG and recall in [0, 1]. Confirmed via raw JSON output and benchmark summary table.

Why the bug was invisible until Stage C: `RetrievalEvaluator` had never been exercised against real retrieved data. The repo audit flagged it as PARTIAL. Unit tests would not have caught it — they test the formula against mocked inputs without duplicates. The bug required real ingested documents and real retrieval to manifest.

### Lesson

Metric granularity must match label granularity. FiQA labels at document level; RAG retrieves chunks. Deduplication is mandatory when bridging those two granularities. Any time a system assigns multiple internal records to the same external entity (chunks → documents, events → sessions, line items → orders), aggregation metrics computed on the raw internal records will be inflated.

### Commit

`dd33fcd` — Fix NDCG and recall exceeding 1.0 by deduplicating chunk-level retrievals to doc-level

---

## 2026-04-18 — Self-RAG VERIFICATION_PROMPT unescaped braces masked by broad exception handler

### Symptom

With `GENERATION_STRATEGY=self_rag`, queries succeeded and returned correct answers, but uvicorn logs printed on every query:

```
Claim verification failed: '"supported"'
Claim verification failed: '"supported"'
```

One line per extracted claim. Self-RAG was silently degrading to basic generation — the initial answer was correct, but the verification loop never actually ran.

### Initial hypothesis (wrong)

The literal quotes in `'"supported"'` suggested a JSON parser bug: the LLM returning a bare JSON string instead of a dict, `json.loads('"supported"')` → `str`, and `.get()` on a string raises `AttributeError`. Plausible and consistent with the log output.

### Actual root cause

The bug was one layer upstream. `VERIFICATION_PROMPT` contained a JSON schema example — `{"supported": true/false, "evidence": "..."}` — as a literal string in the template. When passed through `str.format()` to fill in `claim` and `context` fields, Python interpreted `{"supported": ...}` as a format field named `"supported"` and raised `KeyError: '"supported"'`. The LLM was never called. The parser wasn't the problem — the prompt wasn't rendering.

`KeyError('"supported"')` stringifies to `'"supported"'` — identical to what a parser failure on a bare JSON string would produce. The broad `except Exception as e: logger.warning("Claim verification failed: %s", e)` handler logged only `str(e)`, discarding the exception type. Both failure modes produced the same log line.

### Fix

Three layers, because debugging revealed the parser also had bugs that would surface once the prompt was fixed:

1. **Prompt template** (`generation/advanced_generation.py`): escaped all JSON-example braces (`{{` / `}}`) so `str.format()` treats them as literal output.
2. **`_verify_claim` parser**: switched from negative blacklist to positive whitelist (`"true"` / `"yes"` / `"supported"` / `"1"`). Unknown or gibberish responses fail closed. Handles bare-string responses and string-valued booleans (`"false"` as a string is truthy in Python — the original code would silently mark it as supported).
3. **`_extract_claims` parser**: removed the too-permissive fallback that would treat any list-valued field in the response as claims. Restricted to explicit keys (`claims` / `statements` / `items`), falls through to sentence-split otherwise.

### Regression tests or verification

`tests/unit/test_self_rag_verification.py` — 6 tests covering the standard dict response, bare-string response (original failure mode), string-valued boolean (silent false-positive), gibberish response (must fail closed), dict without evidence field, and `_extract_claims` with a `statements` key.

End-to-end: ingest + Self-RAG query with no `Claim verification failed` in logs. Verification loop runs; `self_rag_stats` in response shows `verified_claims` and `unsupported_claims` populated.

### Lesson

A broad `except Exception` that logs only `str(e)` can make two distinct exception types produce identical log output. `KeyError('"supported"')` and `AttributeError` on a string both stringify to `'"supported"'`. Had the handler logged `type(e).__name__` alongside `str(e)`, the distinction would have been immediate and saved an investigation step. Broad exception handlers should always preserve the exception type in logs.

### Commit

`a12c06e` — Fix Self-RAG claim verification end-to-end

---

## 2026-04-26 — LangfuseTracer _record_step was a stale parallel recording path

### Symptom

AUDIT.md item 8: `LangfuseTracer._record_step()` accesses `self._traces[trace_id]["steps"]` — treating a `QueryTrace` dataclass as a dict. Raises `TypeError` on first traced request when `ENABLE_TRACING=true`. Two integration tests in `TestObservabilityIntegration` were failing as a result.

### Initial hypothesis

Fix the dict-subscript: replace `self._traces[trace_id]["steps"].append(...)` with `self._traces[trace_id].steps.append(...)` to access the attribute correctly on the dataclass.

### Actual root cause

Three layers, each invalidating the previous fix direction:

1. **Dict subscript on a dataclass.** `self._traces[trace_id]` is a `QueryTrace` Pydantic model. Dict-style access raises `TypeError` immediately. This is the bug the audit described.

2. **The field doesn't exist.** Even with the correct attribute-access syntax, `QueryTrace` has no `steps` field — only `events: list[TraceEvent]`. Fixing the syntax alone would produce `AttributeError`.

3. **Even fixed, it would duplicate.** Each `log_*` method (e.g. `log_routing`, `log_retrieval`) already calls `self._traces[trace_id].events.append(event)` correctly before calling `_record_step`. A corrected `_record_step` writing to `events` would append each event twice. The `_record_step` path was a stale parallel recording mechanism — never functional, not read by anything downstream.

### Fix

Deleted the recording block from `_record_step` entirely. Kept only the debug log line:

```python
def _record_step(self, trace_id: str, step_name: str, data: dict):
    logger.debug("Trace step [%s] %s: %s", trace_id[:8], step_name, json.dumps(data))
```

Event accumulation is already correct through the `events.append` calls in each `log_*` method. The `_record_step` deletion removes a broken dead path, not a working feature.

### Regression test

`test_langfuse_tracer_trace_lifecycle` in `tests/integration/test_evaluation.py` — was failing before the fix, passes after. It calls `log_routing` and `log_retrieval` in sequence and asserts `len(trace.events) >= 2`. Because `events.append` was always working, the test now passes cleanly; the TypeError from `_record_step` was the only thing that had been causing it to fail.

### Lesson

Symptom-described bugs are sometimes shallower than the underlying problem. The audit correctly identified the observable crash site (`_record_step` dict subscript) but the right fix direction required understanding why the code existed at all — not just what it was doing wrong. The audit described what it observed; the root cause was that the whole recording path was stale and could be removed. Worth digging one layer deeper before writing the fix.

### Commit

`f424495` — Remove vestigial _record_step path from LangfuseTracer

---

## Patterns I've noticed

Across the bugs documented in this file, a recurring pattern: code in the repo's audit "PARTIAL" list consistently produces silent failures when first exercised against real data end-to-end. The singleton bug, the UUID mismatch, the NDCG/recall dedup, the `RetrievalEvaluator` never-called-in-anger, and the Self-RAG verification parser all lived in modules the audit flagged as "not fully wired" or "never called." Each was invisible until a smoke test or benchmark forced them to execute.

**Takeaway:** integration tests that exercise real workflows catch classes of bugs unit tests cannot. Unit tests verify individual functions do what their mocks claim; they cannot catch "this function is never actually reached in the real code path" or "these two correctly-written components use incompatible ID spaces."

**Specifically useful signal for future me:** if an audit or code review says a module is PARTIAL, assume it will break on first real-world use. Plan debug time accordingly. Don't trust static code review to catch cross-layer integration bugs.

The 2026-04-26 audit cleanup confirmed the PARTIAL prediction held precisely. All five latent bugs (items 7–11 in AUDIT.md) were found in modules the audit had flagged as PARTIAL or SCAFFOLD — the agent endpoint, the tracer, the Cohere embedder, and the retrieval executor's dead and broken methods. Not one surfaced from a module the audit marked WORKING. The audit's classification scheme was accurate as a forward predictor: WORKING modules were safe, PARTIAL and SCAFFOLD modules contained every bug found.
