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

## 2026-04-26 — LangfuseTracer _record_step was a stale parallel recording path masked by a TypeError

### Symptom

What AUDIT.md flagged: `_record_step()` accesses `self._traces[trace_id]['steps']` — treating a `QueryTrace` dataclass as a dict. Raises `TypeError` on first traced request when `ENABLE_TRACING` is set.

The symptom as described looked like a one-line dict-vs-dataclass fix. It was not.

### Initial hypothesis (wrong)

Replace the dict subscript with attribute access: `self._traces[trace_id]["steps"]` → `self._traces[trace_id].steps`. Plausible from the `TypeError` alone.

### Actual root cause

Three layered bugs, only the outermost visible from the audit description:

1. **Dict subscript on a dataclass** — the surface `TypeError`.
2. **The "steps" field doesn't exist on `QueryTrace`.** The dataclass has `events: list[TraceEvent]` and no `steps` field. Even with the subscript fixed to attribute access, the call would raise `AttributeError`.
3. **Even if the field existed, `_record_step` would duplicate event recording.** The `log_routing`, `log_retrieval`, etc. methods on `LangfuseTracer` already accumulate events via `events.append`. `_record_step` ran in parallel, recording the same events to a different field. Stale code from an earlier API.

The audit described the symptom truthfully. The root cause was deeper than the symptom suggested.

### Fix

Deleted the recording block from `_record_step`. Kept the debug log line. Event accumulation continues to work through the existing `log_*` methods.

Three layers needed addressing because each layer alone would have left a different bug:
- Fix layer 1 only: `AttributeError` on first traced request
- Fix layers 1 and 2 (add `steps` field): silent double-recording, every event recorded twice
- Delete the whole path: events recorded correctly via `log_*` methods alone

Deletion is the only option that produces correct behavior because the recording path was never needed.

### Regression test or verification

`test_langfuse_tracer_trace_lifecycle` in `tests/integration/test_evaluation.py` asserts `len(trace.events) >= 2` after `log_routing` and `log_retrieval` calls. Was failing before this fix (one of two failing integration tests in AUDIT.md). Now passing. The test exercises the exact path that hit `_record_step` and would catch a regression.

### Lesson

Audit reports describe what was observed, not necessarily the root cause. The `TypeError` in the audit was real, but it was the outermost layer of a three-layer problem. When a reported bug is a stale code path, the right fix is removal, not repair — repairing would have silently introduced a different bug (double-recording).

A broad `except Exception as e: logger.warning("...", e)` style handler upstream would have made the three layers indistinguishable in logs. Worth pairing with the Self-RAG `VERIFICATION_PROMPT` entry above as another instance of the same pattern: shallow symptom masking deeper structure.

### Commit

`f424495` — Remove vestigial _record_step path from LangfuseTracer

---

## 2026-04-28 — RetrievedChunk.rank not set by most retrieval strategies

### Symptom

Most retrieval strategies leave `RetrievedChunk.rank` at its dataclass default of `0`. Discovered while exposing `retrieval_candidates` on `QueryResponse` for the Stage 2 UI redesign: the candidates table has a rank column, but nearly all rows show `0`.

### Root cause

Only `_multi_query_search` in `retrieval/strategies/retrieval_executor.py` assigns ranks explicitly (via Reciprocal Rank Fusion). `_semantic_search`, `_keyword_search`, `_hybrid_search`, and `_metadata_filter_search` all return chunks without calling `enumerate`. The field defaults to `0` and is never overwritten.

### Impact

Minor — metadata hygiene, not a correctness bug. The UI sorts candidates by `score` instead of `rank`, so production retrieval is unaffected. The `rank` field is accessible but unreliable for those strategies.

### Status

Deferred. Fix: each strategy should call `enumerate(chunks, start=1)` and assign `chunk.rank` before returning.

---

## 2026-04-28 — Render free-tier 502s under rapid sequential LLM burst

### Symptom

During confidence threshold calibration (Stage 2, 2026-04-28): 10 queries attempted against the Render backend sequentially with no delay. Q09 and Q10 returned HTTP 502 Bad Gateway. Q01–Q08 all succeeded. 8 successful responses were used for p33/p66 threshold calculation.

### Pattern

502s appeared at the tail of the batch, not at the start. Rules out cold-start. Consistent with Render free-tier rate limiting or ephemeral resource exhaustion after sustained burst load (8 full LLM round-trips in rapid succession).

### Impact

Negligible under normal single-user usage. A real user wouldn't fire 8+ LLM queries in rapid succession.

### Status

No action required. If it recurs during a future calibration run: add a 2 s sleep between queries, or upgrade to Render Starter tier for sustained load.

---

## 2026-04-28 — Testing gap: orchestrator stage_timings has no unit test

### Symptom

Stage 1 added `stage_timings` population to `orchestrator.py`. There is no automated test that asserts the six timing keys (`router_ms`, `retrieve_ms`, `rerank_ms`, `prompt_ms`, `generate_ms`, `total_ms`) are all present and non-zero on a real query.

### Status

Exercised only by the manual curl pre-flight in Stage 3 (all six keys confirmed non-zero). No automated regression coverage exists.

### Deferred fix

Add a unit test (or integration test) for `orchestrator.query()` that mocks the five pipeline stages and asserts the returned `QueryResponse.stage_timings` dict contains all six keys with positive float values.

---

## 2026-04-29 — Production Llama emitted multi-line JSON arrays for follow-up questions

### Symptom

Production Render backend returned `follow_up_questions: null` on every `/query`. Local backend with the same prompt and the same model (Llama 3.3 70B via Groq) returned well-formed 3-element arrays in 4/4 test cases.

### Initial hypothesis (wrong)

Stale deploy — Render hadn't picked up the updated prompt code. Was about to add deploy-SHA logging to confirm when the actual issue surfaced.

### Actual root cause

Production Llama emitted three consecutive single-element JSON arrays on separate lines:

```
["First question?"]
["Second question?"]
["Third question?"]
```

Each line is valid JSON. The parser called `json.loads(text)`, which reads the first complete value — a 1-element list — failed the "exactly 3 strings" validation, and returned `[]`. Local Llama with the same prompt produced a single 3-element array. Same model, same prompt, different output shape in production.

The root cause was found by bumping the existing DEBUG-level error log to WARNING with `%r` format: `logger.warning("FOLLOWUP_RAW_RESPONSE: %r", text)`. The `repr()` output preserved the embedded newlines that made the multi-line shape visible. `%s` would have collapsed the lines and hidden the structure entirely.

### Fix

Two layers:

1. **Prompt**: added an explicit structural example block specifying "ONE array, not multiple" to reduce shape variation.
2. **Parser**: made robust to the multi-array case — try the happy path first (single `json.loads`), fall back to line-by-line parsing that strips non-array lines and flattens results.

Production compliance went from 0% to 100% across five varied test queries after the fix.

### Lesson

LLM output shape is not deterministic across deploys, even with the same model and the same prompt. Local testing establishes a floor on compliance, not a ceiling on shapes the model might emit in production. Build parsers that handle multiple shapes when the cost of rigidity is silent failure. Log raw responses at WARNING (not DEBUG) on validation failures — production failures must be visible without log-level changes. Use `%r` format rather than `%s` so invisible characters (newlines, quotes) are preserved in the log output.

### Commits

- `8588194` — Log raw LLM response on follow-up validation failure
- `cf68e2a` — Make follow-up parser robust to multi-line JSON arrays

---

## 2026-04-29 — Streamlit widget key collision via hash() across multi-turn chat history

### Symptom

Clicking a follow-up suggestion chip in the second response raised:

```
StreamlitDuplicateElementKey: There are multiple elements with the same key='followup_4618601950993349981'
```

The error message named the issue directly.

### Root cause

Follow-up chips used content-based hash keys: `key=f"followup_{hash(question)}"`. Streamlit re-renders the entire chat history on every interaction, including all previous follow-up chip sets. Two scenarios trigger collision: the LLM produces the same follow-up question text in two different responses, or two different strings hash to the same value. Either way, two `st.button` calls with identical keys raise `StreamlitDuplicateElementKey`.

The bug was invisible in single-turn testing because one response produces one set of follow-up chips with no possibility of duplication. Collision requires at least two rendered responses in the same session — which only happens in real multi-turn chat history.

### Fix

Position-based keys: `key=f"followup_{message_idx}_{i}"` where `message_idx` is the message's index in chat history and `i` is the chip's position within that message. Threaded `message_idx` through `_render_assistant_message` and `_follow_up_chips`. Uniqueness is guaranteed by construction, independent of question content.

### Lesson

For any framework that requires globally unique widget keys across re-renders, avoid content-based hashing. Position-based keys are simpler and provably unique. Hash-based keys appear stable but fail silently in collision cases that only surface once a feature is exercised across more than one turn. Multi-turn chat features need multi-turn testing.

### Commit

`2ba58cd` — Add hallucination verifier toggle and fix follow-up chip key collision (the chip-key fix landed bundled with the verifier toggle — both touched the same file and the collision surfaced while testing the toggle)

---

## 2026-04-30 — Llama wrapping prior: \<cite\> tags never adopted despite 4 prompt iterations

### Symptom

After four prompt iterations asking Llama 3.3 70B to wrap cited content in `<cite source="N">...</cite>` tags, local compliance was 0/5. The model consistently emitted trailing markers (`<cite source="1">` after the sentence end, not wrapping it) regardless of prompt wording.

### Initial hypothesis (wrong)

Prompt engineering issue — better examples, explicit CORRECT/WRONG contrast blocks, few-shot demonstration, or answer-completion framing would move the model to wrapping behavior. Tried all four. Each iteration produced the same trailing-marker output.

### Actual root cause

Llama 3.3 70B has heavy RLHF training on `[Source N]` trailing citation patterns. The pattern occupies a fixed slot in the model's representation of how to attribute a claim. `<cite source="N">` mapped to that same trailing slot — the prompt asked for wrapping, the model's prior for what a citation marker is produced trailing output every time. Four varied prompt rewrites hitting the same wall is the signal that the prior won't move.

### Decision

Adapt the parser rather than the model. Switched from attempting to extract spans the model was supposed to delimit, to extracting the clause preceding each trailing marker using a sentence-boundary heuristic (last `. `, `! `, or `? ` before the marker). Renamed the field from `cited_spans` to `attributed_spans` to accurately describe what's produced: the parser attributes source N to the clause preceding marker N — it is not reporting boundaries the LLM drew.

The visual output (yellow highlight + source chip per attributed clause) is identical to what wrapping would have produced. The difference is in the README, commit messages, and field name: what shipped is sentence-level attribution, not LLM-bounded citation spans.

### Lesson

When a model has strong RLHF-trained priors, 3+ varied prompt rewrites all producing the same non-target shape is the signal to stop iterating and adapt downstream instead. Prompt iteration past that point burns time without changing the prior.

Honest naming is a separate decision from adapting the implementation, but it follows from the same reasoning. `cited_spans` would have been accurate if the LLM drew the boundaries. `attributed_spans` is accurate for what the parser actually produces. The visual output was identical; the field name is where the honesty lives.

### Commits

- `8567eaa` — Add attributed_spans to QueryResponse via trailing-marker parser
- `9e49d6c` — Render attributed_spans as inline highlights with source chips

---

## 2026-04-30 — FLARE [UNCERTAIN: topic] heuristic abandoned for numeric-novelty after Llama 3.3 70B failed to emit self-signaling markers

### Symptom

The existing `FLAREGenerator`'s iterative loop was gated on detecting
`[UNCERTAIN: topic]` markers in LLM responses. Empirical test against
Llama 3.3 70B via Groq: the marker never appeared. Round 0 didn't
include the instruction at all (gated behind `if answer_parts:`); round
1+ included an explicit instruction to emit `[UNCERTAIN: topic]` when
uncertain. Both rounds returned fully-formed confident answers with
zero markers.

### Initial hypotheses (wrong)

1. Prompt wording too weak — clearer instruction would move the model.
2. The gating logic prevents the instruction from reaching the model in
   the meaningful path.

### Actual root cause

Hypothesis (2) was confirmed by reading the gating logic — the instruction
was absent from round 0. Round 1 confirmed (1) was also wrong: the
instruction reached the model and was ignored. Llama 3.3 70B's RLHF
training produces complete, confident answers; structural self-interruption
with `[UNCERTAIN: topic]` markers is not a behavior the model was trained
to emit on command.

### Fix

Abandoned the `[UNCERTAIN: topic]` heuristic entirely. Replaced with
dollar-token novelty: extract `$`-prefixed numeric tokens from the
response, diff against tokens present in the retrieved chunks,
re-retrieve when novel tokens appear. No model self-signaling required.
The new heuristic is robust to Llama's output style because it operates
on the response's content rather than structural markers the model must
voluntarily emit.

### Lesson

When a model has strong RLHF-trained priors, 2–3 varied prompt rewrites
all producing the same non-target shape is the signal to stop iterating
and adapt downstream instead. Same pattern as the 2026-04-30
attribution-spans entry above — that one abandoned `<cite>...</cite>`
wrapping for trailing-marker parsing after four prompt iterations failed
to move Llama's trailing-citation prior. Both cases: the right move is
a downstream adaptation that produces the desired observable behavior
without requiring the model to change its output shape.

### Commit

`6affffb` — Wire FLARE-inspired generator into orchestrator with numeric-novelty heuristic

---

## 2026-04-30 — FLARE non-convergent loop: regex asymmetry between response and chunk dollar-token extraction

### Symptom

End-to-end curl test of `GENERATION_STRATEGY=flare` with the query
"What is the difference between a Roth 401k and a traditional 401k in
terms of contribution limits and tax treatment?" hit the iteration cap:
4 rounds, 38s latency, 5629 tokens. The novel token set was identical
across every round: `['$18,000', '$54,000']`. Round 1 added 2 new
chunks; rounds 2 and 3 added 0. The loop ran to cap without converging.

### Hypotheses considered

1. Heuristic is over-sensitive — every query with dollar figures produces
   unverifiable tokens.
2. Corpus genuinely doesn't contain the exact figures the LLM produced.
3. Regex applied asymmetrically between response and chunk content.

### Tests run

Dumped the 2 chunks added in round 1 and grepped for `$18,000`,
`$18000`, `18,000`, `18000`, `$54,000`. Chunk `09c1ac43` contained
`"contribute up to $18k/year"` and `"$54k/year total"` — the source of
both figures. The corpus contained the facts. **Ruled out (2).**

### Actual root cause

The regex `\$\d{1,3}(?:,\d{3})*(?:\.\d+)?` matched `$18,000` in the
response but matched only `$18` in the chunk's `$18k/year` (the regex
stopped before the `k`). Result: `chunk_dollars = {'$18'}`,
`answer_dollars = {'$18,000'}` — no overlap. The heuristic flagged
`$18,000` as novel every round even though it was grounded in the corpus
all along. Asymmetric regex application against differently-formatted
text.

### Fix

Two layers:

1. **Regex normalization**: extended `_extract_dollar_tokens` to run two
   passes — the original regex (with a skip if the match ends immediately
   before `k`/`K`), plus a second pass `\$(\d+)[kK]\b` that normalizes
   abbreviated thousands (`$18k` → `$18,000`) before comparison.
2. **Zero-new-chunks exit condition**: if de-duplicating by `chunk_id`
   produces an empty `new_chunks` list after re-retrieval, the loop
   breaks before the next LLM call. This bottoms out gracefully when the
   retriever has nothing more to offer regardless of heuristic state.

Post-fix curl on the same query: 1 round, 10.5s latency, 1211 tokens.
Mortgage query unchanged at 1 round (no dollar tokens in answer).

### Regression tests or verification

`test_flare_exits_when_reretrieval_adds_no_new_chunks` covers the
zero-new-chunks exit condition explicitly.

### Lesson

When a heuristic fires repeatedly without converging, suspect asymmetric
comparison — two sets generated by processes that don't share the same
normalization. The bug is rarely "the heuristic is too sensitive"; it's
usually "the comparison is apples-to-oranges in some subtle way."
Generalizes beyond regex: any time a system computes set-difference
between two collections that pass through different extraction or
formatting paths, verify the formats are normalized before treating the
difference as meaningful.

### Commit

`6affffb` — Wire FLARE-inspired generator into orchestrator with numeric-novelty heuristic

---

## 2026-05-01 — Self-RAG verification provider-coupling removed; shell env vars override .env

### Symptom

`SelfRAGGenerator._extract_claims` and `_verify_claim` instantiated `AsyncOpenAI`
directly with `settings.openai_api_key`, hardcoded `gpt-4o-mini`, and used OpenAI's
native `response_format={"type": "json_object"}`. With `LLM_PROVIDER=groq`, the main
answer generation correctly used Groq, but the verification step silently bypassed the
configured provider and went to OpenAI. If `OPENAI_API_KEY` was unset, Self-RAG crashed
at the verification step. Audit-flagged in three places (AUDIT.md known limitation #2,
README Status, future_work.docx Tier 1.5).

### Root cause

The hardcoded client was an implementation accident, not an algorithmic requirement.
Self-RAG's verification only needs an LLM capable of: (1) reading an answer and emitting
a JSON list of claims, (2) reading a claim + context and emitting a JSON verdict. Any
reasonably capable LLM can do both. The original implementation just wrote OpenAI first
and never abstracted.

### Fix

Routed both methods through the orchestrator's existing `GenerationService` abstraction
by accepting `llm_service` as a parameter and building a `ConstructedPrompt` for each
verification call. The defensive JSON parsing from the 2026-04-18 fix already handles
cross-provider response variation (Anthropic doesn't support native JSON mode; the same
defensive parser handles its raw string output). 6 existing unit tests refactored to mock
at the LLM service interface level. 2 new tests cover Anthropic and Groq paths explicitly.
113 unit tests total.

### Verification

Local: curl with `OPENAI_API_KEY` unset, `LLM_PROVIDER=groq`,
`LLM_MODEL=llama-3.3-70b-versatile`. Returned grounded answer with
`model_used: llama-3.3-70b-versatile` and populated
`self_rag_stats.verified_claims`. Verification ran on Groq.

Production: Render auto-deployed the commit. Tested via curl to
`ragcore-api.onrender.com/query` with `verify_claims: true`. Same result —
Llama-on-Groq running the verification step. `OPENAI_API_KEY` is still set on
Render at time of writing as a not-yet-decisive proof; deleting it from Render's
environment and re-testing is the final step (queued separately).

### Lesson — primary

When a coupling is documented in three places (audit, README, future-work doc) but
the actual code change is small (~60 lines), the social cost of the limitation has
been higher than the technical cost of the fix. Removing it is also a reminder that
abstractions documented as known limitations stay limitations until someone explicitly
converts them. The audit served its purpose here as a forward predictor — it flagged
the coupling on 2026-04-30 and the fix shipped 24 hours later.

### Sub-finding 1: Shell env vars override `.env` files

During local verification, the curl test returned `model_used: gpt-4o-mini` even
though `.env` had been updated to `LLM_PROVIDER=groq`. Investigation:
`env | grep -E "^LLM_"` revealed `LLM_PROVIDER=openai` and `LLM_MODEL=gpt-4o-mini`
exported in the shell from a prior session. Pydantic-settings priority order: shell
env beats `.env` file beats code defaults. The shell vars had been silently overriding
`.env` for some unknown duration. Fix: `unset LLM_PROVIDER LLM_MODEL` in the active
shell. They were not in any rc file, so a fresh terminal would have cleared them
automatically.

### Sub-finding 2: `LLM_MODEL` and `LLM_PROVIDER` are independently set

While debugging the shell-var issue: `.env` was authored as a matched pair
(`LLM_PROVIDER=openai` + `LLM_MODEL=gpt-4o-mini`). Mixing providers without changing
the model name produces a runtime error from the new provider rejecting the unknown
model name. There is no validation that the two settings are coherent. Not a
fix-this-now issue; logged for a future commit that adds mutual-validation or
smarter default-model-per-provider logic.

### Lesson — sub-findings combined

Configuration coherence has two failure modes worth checking during debugging:
(1) shell environment leakage from prior sessions silently overriding declared config;
(2) related config values that are individually valid but jointly incoherent. Standard
debugging move when local behavior diverges from declared config: `env | grep` for the
relevant prefix before assuming the config file is being read. Standard config-design
move: validate related settings together, not just individually.

### Cross-reference

The 2026-04-18 Self-RAG verification entry fixed the prompt template (escaped braces)
and the parser (positive whitelist for support, defensive bare-string handling). This
entry removes the provider-coupling that the prompt and parser were running on. The two
entries together cover the full Self-RAG verification path: prompt correctness, parser
correctness, provider portability.

### Commit

`b03614d` — Make Self-RAG claim verification provider-agnostic

---

## Patterns I've noticed

Across the bugs documented in this file, a recurring pattern: code in the repo's audit "PARTIAL" list consistently produces silent failures when first exercised against real data end-to-end. The singleton bug, the UUID mismatch, the NDCG/recall dedup, the `RetrievalEvaluator` never-called-in-anger, and the Self-RAG verification parser all lived in modules the audit flagged as "not fully wired" or "never called." Each was invisible until a smoke test or benchmark forced them to execute.

**Takeaway:** integration tests that exercise real workflows catch classes of bugs unit tests cannot. Unit tests verify individual functions do what their mocks claim; they cannot catch "this function is never actually reached in the real code path" or "these two correctly-written components use incompatible ID spaces."

**Specifically useful signal for future me:** if an audit or code review says a module is PARTIAL, assume it will break on first real-world use. Plan debug time accordingly. Don't trust static code review to catch cross-layer integration bugs.

The 2026-04-26 audit cleanup confirmed the PARTIAL prediction held precisely. All five latent bugs (items 7–11 in AUDIT.md) were found in modules the audit had flagged as PARTIAL or SCAFFOLD — the agent endpoint, the tracer, the Cohere embedder, and the retrieval executor's dead and broken methods. Not one surfaced from a module the audit marked WORKING. The audit's classification scheme was accurate as a forward predictor: WORKING modules were safe, PARTIAL and SCAFFOLD modules contained every bug found.

The audit's PARTIAL prediction held again on a 2026-04-26 RAGAS integration. AUDIT.md flagged `GenerationEvaluator`'s RAGAS code path as PARTIAL — "code present; requires extra install." Installing the extra was the easy part; what surfaced on first end-to-end use was a chain of distinct integration issues. The benchmark runner had no RAGAS code path despite the existing class. The 0.4.x library had migrated from the API the existing code targeted (column names changed, llm wrapper deprecated). The default judge configuration truncated on long answers, dropping 14 of 50 queries with no in-band warning. Each issue was invisible until the integration was forced to execute end-to-end, exactly matching the singleton bug, the UUID mismatch, and the NDCG dedup before it. PARTIAL means "looks correct in isolation but has never been exercised against real workflow." That's the right level of skepticism to apply.

Two of the 2026-04-29 bugs share a structural pattern that's distinct from the PARTIAL-module theme: "worked locally" was the wrong test bar. Production Llama emitted multi-array JSON that local Llama never produced with the same model and prompt; the Streamlit key collision only manifested once chat history held more than one rendered response. In both cases, local testing exercised the feature against a narrower input space than production would. The multi-line JSON bug was invisible until a WARNING-level log with `%r` format surfaced the raw production response shape; the key-collision bug was invisible until a second response was rendered in the same session.

**Takeaway:** "works locally" misses shape variations that only appear at scale or with state accumulation. For features whose correctness depends on runtime behavior — LLM output shape, stateful re-renders — plan for production-shaped variation: log raw responses at WARNING (not DEBUG) on validation failures so failures are visible without log-level changes; use `%r` to preserve invisible characters; exercise multi-turn flows in feature testing, not just the single-turn happy path.

The 2026-04-30 attribution-spans investigation surfaced a separate pattern. Four varied prompt rewrites all asked Llama 3.3 70B to wrap cited content in `<cite>` tags; all four produced the same trailing-marker output. The signal was clear by iteration three — the model's RLHF-trained `[Source N]` prior wasn't moving. Adapting the parser downstream to attribute the preceding clause from trailing markers was cheaper and more reliable than a fifth iteration.

When 3+ varied prompt rewrites produce the same non-target shape, stop iterating and adapt post-processing instead. The honest corollary: when adapting changes what's actually shipped, rename to match reality. `cited_spans` would have implied the LLM drew the span boundaries; `attributed_spans` accurately describes parser-derived clause attribution. The visual output was identical; the field name is where the honesty lives.

The 2026-04-30 FLARE `[UNCERTAIN]` investigation is the second instance of the RLHF-prior pattern. Two rounds, two different failure modes — instruction absent in round 0, instruction ignored in round 1 — same output: complete confident answers with no self-interruption marker. The pivot to numeric-novelty was faster than the attribution-spans pivot because the prior entry had already established the threshold. Each recurrence narrows the generalization: it is not only `[Source N]` trailing citations that are entrenched; Llama's RLHF priors resist structural self-interruption markers of any shape. The downstream adaptation strategy now has two confirming cases.

The 2026-04-30 regex-asymmetry bug introduces a pattern distinct from the PARTIAL-module and RLHF-prior themes: asymmetric set-difference in heuristics. The `$18k` vs `$18,000` case is the archetype — two sets extracted from text that passed through different formatting conventions, compared as if normalized to the same space. The heuristic fired every round not because it was wrong about what it was measuring, but because the measure was applied inconsistently across the two sides of the comparison. Before treating a set-difference as meaningful, verify that both sets were produced by extraction paths that normalize to the same format. When a heuristic fires without converging across multiple rounds with no change in the underlying data, asymmetric normalization is the first thing to check. Generalizes beyond regex: any time two collections pass through separate extraction or formatting paths before being compared, the difference may be measuring format divergence rather than semantic novelty.
