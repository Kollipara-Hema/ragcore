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

## 2026-05-02 — Citation-rendering "regression" was per-call LLM stochasticity, not a regression

### Symptom

Reported: after the 2026-05-01 Self-RAG provider-agnostic commit, citation
chips and inline highlights "disappeared" when the hallucination verifier
was enabled — basic queries showed citations, Self-RAG queries didn't.

### Initial hypothesis (wrong)

Today's Self-RAG commit broke citation rendering. The toggle correlation
(verifier ON → broken, verifier OFF → working) suggested a Self-RAG-path
regression in `_extract_attributed_spans` or downstream rendering.

### Tests that distinguished

Three identical curl calls of the same query "How does tax impact mortgage
payoff?" with the same toggle state and same backend produced
`num_attributed_spans` of 2, 2, and 0. Same query, same configuration,
different results across calls. The toggle correlation in the original
report was a sampling accident — verifier-on queries happened to land on
calls where Llama didn't emit `<cite source="N">` markers.

### Actual finding

Llama 3.3 70B emits citation markers stochastically per LLM call, not
just per query shape. The 2026-04-30 attribution-spans entry documented
that the model resists `<cite>...</cite>` wrapping; that entry framed
the issue as a deterministic prior. Today's data adds a second dimension:
even within the trailing-marker format Llama "accepts," emission is
inconsistent across identical calls. The parser is correct per
specification — it produces spans when markers are present, returns
empty when not.

### Fix

None. No code change. The current behavior is correct given the model's
output. A parser fallback that uses sentence-to-chunk overlap when
markers are absent could close the visible inconsistency, but that's a
feature decision, not a bug fix.

### Lesson

When a "regression" report shows correlation with a toggle or input
change, test variability before treating the correlation as deterministic.
Three identical runs cost ~30 seconds and would have caught this misdiagnosis
yesterday. The correct first hypothesis when investigating a feature whose
output is LLM-mediated is: "is the LLM behaving consistently across calls?"
That diagnostic step belongs before code-reading.

### Cross-reference

The 2026-04-30 attribution-spans entry established that Llama's RLHF
prior resists structural wrapping. This entry extends that finding: even
the trailing-marker format the model "accepts" is emitted with per-call
stochasticity. The two entries together describe the full envelope of
Llama's citation-marker behavior — resistant to wrapping, inconsistent
on emission.

---

## 2026-05-03 — Stale Docker-baked FAISS files masked unpushed migration commit

### Symptom

After committing the `FAISS_DATA_DIR` migration, setting the env var on Render, and waiting for "Live" status, a smoke query returned 5 citations. The freshly-mounted persistent disk should have been empty — 0 citations was the expected result.

### Diagnostic that caught it

`grep -A 5 "def __init__" /app/vectorstore/vector_store.py` via Render Shell showed the old hardcoded-filename `__init__` signature. Locally, `git log origin/main..main --oneline` confirmed one unpushed commit.

### Root cause

Two layers. The migration commit was never pushed, so the redeploy triggered by the env-var save ran old code that hardcoded `faiss_index.idx` and `faiss_metadata.pkl` at the process working directory (`/app`). Those files had been tracked in git before this commit added `git rm --cached` and `.gitignore` entries; the previous Docker build had copied them into `/app/`, where the old code read them on startup. The 5 citations came from stale FiQA data baked into the image.

### Fix

`git push`. Render redeployed with the new code; `os.makedirs` created `/var/data/faiss/` on first boot. Smoke query returned 0 citations. Persistence test: ingest doc → restart → same doc_id returned.

### Lesson

Verifying response shape is insufficient for configuration changes. When a path or config migration deploys, the first-line checks are: `git log origin/main..main --oneline` (confirm the commit is pushed) and `ls -la` on the expected data directory via Render Shell (confirm files are where the new config says). Response shape is second-line.

### Cross-reference

Same family as 2026-05-02: a plausible-looking response masked the question "is the system running the code I think it is?" Both cases required three diagnostic lines that would have replaced an hour of wrong-direction debugging. The general pattern: when deploying a configuration change, verify deployment state before interpreting the response.

### Commit

`5ba20cf` — Make FAISS index path configurable via FAISS_DATA_DIR

---

## 2026-05-05 — Self-RAG verifier fail-open: exception handler returned (True, "") on empty LLM response

### Symptom

Benchmark run with `LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini --strategy self_rag` across
50 queries produced JSON output where every query had all extracted claims in
`verified_claims` and `0` in `unsupported_claims`. Approximately 150 log lines:

```
WARNING generation.advanced_generation: Claim verification failed: Expecting value: line 1 column 1 (char 0)
```

`Expecting value: line 1 column 1 (char 0)` is the `JSONDecodeError` message for
`json.loads("")` — gpt-4o-mini was returning empty strings for a significant fraction (~38%)
of verification calls. Despite those failures being logged, every claim was marked as
verified.

### Root cause

`SelfRAGGenerator._verify_claim()` in `generation/advanced_generation.py` had a broad
`except Exception` handler that caught the `JSONDecodeError` from the empty response and
returned `(True, "")` — fail-**open**, not fail-closed:

```python
except Exception as e:
    logger.warning("Claim verification failed: %s", e)
    return True, ""   # Assume supported if verification fails
```

`generate()` routes on the first element: `if is_supported: verified_claims.append(claim)`.
Every empty-response claim therefore went into `verified_claims`, never into
`unsupported_claims`. The log warning was emitted, but the code path immediately after it
silently promoted the claim to verified.

The 2026-04-18 fix was intended to make the parser fail closed. It did — for the non-dict
parser branch (bare string gibberish). It missed the exception path entirely because the
return value in the exception handler was not updated.

### Why the 2026-04-18 regression tests didn't catch it

`test_bare_string_gibberish_fails_closed` passed `'"xyz_gibberish_unknown"'` — a valid JSON
string. `json.loads('"xyz_gibberish_unknown"')` succeeds and returns a Python `str`. The
non-dict branch on lines 327–328 then checks whether the string is in the positive
whitelist; it is not, so the test correctly got `(False, "")`.

The empty-string case takes an entirely different code path: `json.loads("")` raises
`JSONDecodeError` before reaching the non-dict branch. The exception handler is a separate
code path from the parser branch. The 2026-04-18 test suite covered the parser branch for
gibberish JSON but did not cover the exception handler path at all. No test called
`_run_verify("")`.

**The lesson:** a test that verifies "gibberish JSON fails closed" is not the same as a
test that verifies "unparseable input fails closed." They exercise different lines of code.

### Fix

One-line change to the exception handler (`generation/advanced_generation.py:332`):

```python
# Before
return True, ""   # Assume supported if verification fails

# After
return False, ""  # Fail closed: unparseable response → unsupported
```

Also updated the log message to include `type(e).__name__` per the 2026-05-01 lesson:

```python
logger.warning("Claim verification failed: %s — type=%s", e, type(e).__name__)
```

This makes `JSONDecodeError` (empty response) distinguishable from `KeyError` (prompt
rendering failure) and `TypeError` (None response) without reading the source.

### Regression tests added

`tests/unit/test_self_rag_verification.py` — `TestVerifyClaimEmptyAndUnparseable`:

- `test_empty_string_response_fails_closed` — `_run_verify("")` → `(False, "")`. Pins the
  exact path that caused the benchmark result.
- `test_whitespace_only_response_fails_closed` — `_run_verify("   \n  ")` → `(False, "")`.
  Whitespace also raises `JSONDecodeError` via the same path.
- `test_none_answer_fails_closed` — `answer = None` → `json.loads(None)` raises `TypeError`
  → `(False, "")`. Ensures the handler catches non-string responses too.

### Cross-reference

- **2026-04-18** — This is the fix that introduced the fail-closed intent for the
  verification parser. The exception handler was overlooked. The regression tests from that
  entry covered the parser branch but not the exception handler branch.
- **2026-04-18** (second entry, same date) — The original `VERIFICATION_PROMPT` brace-escape
  fix. That bug also produced `Claim verification failed` log lines, from `KeyError` not
  `JSONDecodeError`. Had the handler logged `type(e).__name__` at that time, the two bugs
  would have been distinguishable at a glance.
- **2026-05-01** — Established the lesson "broad except handlers should log exception type
  alongside message." This fix applies it.

### Commit

`679315e` — Fix Self-RAG verifier exception handler to fail closed.

---

## 2026-05-05 — Self-RAG verifier: gpt-4o-mini wraps JSON in markdown fences with complex contexts

### Symptom

After the fail-closed fix (see 2026-05-05 entry above), a benchmark re-run with
`LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini --strategy self_rag` across 50 FiQA queries
produced 0 verified claims and 176 unsupported claims — every extracted claim landing in
`unsupported_claims`. Hundreds of log lines:

```
WARNING generation.advanced_generation: Claim verification failed: Expecting value: line 1 column 1 (char 0) — type=JSONDecodeError
```

Isolated `_verify_claim` tests with hand-crafted claims and contexts worked correctly.

### Initial hypothesis (wrong)

The earlier fail-closed entry documented this error as `json.loads("")` — interpreting
`Expecting value: line 1 column 1 (char 0)` as evidence of an empty LLM response. The
OpenAI dashboard showed no API errors, so the explanation offered was content-filter or
rate-limit transients causing silent empty responses.

That interpretation was wrong.

### Actual root cause: the error message is not specific to empty strings

`JSONDecodeError("Expecting value: line 1 column 1 (char 0)")` is Python's generic "the
first character is not a valid JSON start character" error. It is raised for:

```python
json.loads("")                              # → same error
json.loads("hello world")                  # → same error
json.loads('```json\n{"supported": true}') # → same error  ← actual case
```

The diagnostic that resolved the ambiguity: log `repr(result.answer)` before calling
`json.loads`. The raw response was:

```
'```json\n{"supported": true, "evidence": "Start as early as possible..."}\n```'
```

gpt-4o-mini was returning a correct, non-empty, correctly formatted JSON verdict — but
wrapped in a markdown code fence. The VERIFICATION_PROMPT says `"Answer ONLY with JSON"`
but does not enforce `response_format={"type": "json_object"}`, so the model was free to
add fences. It did so consistently with complex FiQA forum-post contexts but not with the
short hand-crafted contexts used in isolated tests.

### Why isolated tests passed but benchmark failed

Hand-crafted test contexts (one or two sentences) produce bare JSON from gpt-4o-mini.
Real FiQA retrieval contexts (multi-paragraph financial forum posts) nudge the model toward
markdown formatting. Same prompt, different context length/style, different output shape.
This is the same class of problem as the 2026-04-29 Llama multi-array JSON shape: LLM
output format is not deterministic across input styles, even with the same model and prompt.

### Fix

Extracted `_strip_markdown_fences(text: str) -> str` as a module-level helper in
`generation/advanced_generation.py`. Applied at both unguarded `json.loads` call sites:

- `_extract_claims` — was `json.loads(result.answer)`
- `_verify_claim` — was `json.loads(result.answer)`

Both now call `json.loads(_strip_markdown_fences(result.answer))`. The stripping regex
(`^```(?:json)?\\s*` / `\\s*```$`) is the same pattern already used in
`GenerationService.generate_followups` at `llm_service.py:556-560`.

### Regression tests added

`tests/unit/test_self_rag_verification.py` — `TestMarkdownFenceStripping`:

- `test_verify_claim_strips_markdown_fences` — passes `` ```json\n{...}\n``` ``, asserts `(True, "test")`.
  Pins gpt-4o-mini's real-world output shape with FiQA contexts.
- `test_verify_claim_strips_plain_fence_no_lang` — passes `` ```\n{...}\n``` `` (no `json` tag),
  asserts `(True, "test")`. Covers the variant without a language specifier.
- `test_extract_claims_strips_markdown_fences` — passes a fenced JSON array to `_extract_claims`,
  asserts correct claim list returned.

### Lesson

`JSONDecodeError("Expecting value: line 1 column 1 (char 0)")` is a generic parse-start
error, not an empty-string indicator. Any input that does not start with a valid JSON
character (`{`, `[`, `"`, `t`, `f`, `n`, digit, `-`) produces this exact message. Logging
`repr(text)` on parse failures — not just `str(e)` — is the diagnostic step that
distinguishes:

- Empty response (`""`)
- Markdown-fenced response (`` ```json\n{...}\n``` ``)
- Plain-text preamble (`"Based on the context, the claim is..."`)

All three produce identical exception messages. Only `repr` preserves the shape.

Generalizes: any time a parser reports "expected X at position 0" and the assumption is
that the input was empty, verify with `repr` before drawing that conclusion.

### Cross-reference

- **2026-05-05 fail-closed entry** (`_verify_claim` exception handler returned `True`
  instead of `False`): that fix corrected the routing but didn't expose the root cause —
  the responses were not empty. The `type=JSONDecodeError` log addition from that fix
  correctly narrowed the exception class but did not surface the content because `str(e)`
  doesn't include the actual input text.
- **2026-04-29 Llama multi-array JSON** (`generate_followups` parser): same pattern —
  parser assumed one output shape, model emitted another. Fix was adapting the parser
  downstream. This entry is the same pattern applied to `_verify_claim` and
  `_extract_claims`.

### Commit

`6af0589` — Strip markdown fences before parsing Self-RAG JSON responses.

---

## 2026-05-12 — production FiQA corpus never re-ingested after May 3 disk migration; surfaced May 12 by the security pass

### Symptom

Live demo at ragcore.streamlit.app returning either the canned empty-retrieval fallback or grounded-but-useless answers citing only two ad-hoc test chunks. Direct curl to production `/query` confirmed the same — sub-millisecond retrieval latency, near-zero citations from FiQA, every query routed to either the empty-retrieval fallback message or the low-confidence abstention path.

### Investigation path

Each hypothesis below was tested in order. Each was ruled out by a single diagnostic before moving to the next.

**CORS broken** — ruled out by OPTIONS preflight returning the correct `Access-Control-Allow-Origin` header. The Streamlit frontend could reach the backend; the problem was in what the backend returned, not whether it responded.

**Multi-worker singleton bug** — `VectorStore` and `BM25Index` are module-level singletons; if multiple uvicorn workers had started, each worker would have its own in-process state and requests could land on differently-initialized workers. Ruled out by `/proc/1/cmdline` showing `--workers 1`. Only one worker process was running.

**FAISS index empty** — `GET /health/ready` reports index size. Ruled out by `ntotal=2`. The index was not empty; it contained exactly two vectors.

**Metadata key mismatch** — the pipeline stores text under a configurable content key; a mismatch would produce vectors with no retrievable text. Ruled out by inspecting the stored metadata: the content key was populated correctly on both chunks.

**BM25 not rebuilt on load** — `BM25Index` rebuilds its `_bm25` attribute from `_corpus` on startup; if the pickle had been corrupted or the rebuild skipped, hybrid retrieval would fail silently. Ruled out by confirming `_corpus` held 2 strings and `_bm25` was a `BM25Okapi` instance. BM25 had been rebuilt correctly on load; whether IDF weights were numerically degenerate on a 2-document corpus was a separate open question.

**BM25Okapi IDF degenerate math on tiny corpus** — with only 2 documents, IDF weights are extreme; `query()` returned empty results, which initially appeared to confirm this. Falsified when Streamlit returned a qualitatively different abstention message than the curl path. The two query paths were hitting different failure modes, which meant the corpus size was not the cause — the cause was something that affected all query paths equally regardless of BM25 behavior. That realization redirected attention to disk state rather than index math.

### Root cause

The FiQA corpus may have been ingested to production briefly between the April 28 first-deploy and the May 3 disk migration, but no verifiable chat-history evidence confirms a successful grounded production query in that window. In any case, the corpus was definitely absent from production for at least 9 days between May 3 and May 12, when the security-pass CORS deploy forced the first end-to-end `/query` check since the migration. Render free-tier OOM events around late April motivated the `FAISS_DATA_DIR` persistent-disk migration on May 3 (commit `5ba20cf`). The migration created an empty persistent disk at `/var/data/faiss/` and was validated with a single test chunk (the "RAGCore was tested for persistence on May 3 2026 by Hema" entry, still visible on disk tonight) that survived a Render restart. The full 245-document FiQA corpus was never re-ingested to the new persistent disk. Production has been returning empty-corpus responses for every FiQA query since the May 3 migration.

The two chunks visible on disk tonight are diagnostic artifacts: one from the May 3 persistence validation, one from a May 12 debugging session ingest. Git history confirms no re-ingest occurred — only one commit touched `evaluation/` between May 3 and May 12 (`c2e70a1`, per-claim faithfulness analysis), which analyzes existing benchmark results rather than ingesting data.

### Why it stayed hidden for 9 days

Between May 3 and May 12, every commit passed CI, every Render deploy reached "Live", `/health/ready` returned 200, `/metrics` emitted data, and `AUDIT.md` said WORKING for retrieval. The production system was returning empty answers for every query the entire time.

Nothing detected it because: unit tests use mocked vector stores; integration tests use in-process FAISS with test data; all development between May 3 and May 12 ran against local Chroma (May 4 directory timestamp confirms a local-dev environment switch) or against mocked data; and the live demo was never exercised in CI.

The May 12 security-pass CORS deploy required verifying that ragcore.streamlit.app could still reach the backend — the first time the live `/query` path was exercised end-to-end since the May 3 migration. The broken state became visible immediately.

### Fix

Re-ingest the FiQA corpus to Render via HTTP. The existing `evaluation/scripts/ingest_fiqa_corpus.py` uses `IngestionPipeline` directly and cannot target a remote API. A new `ingest_fiqa_corpus_http.py` POSTs each of the 245 documents to `/ingest/text` with a 1.1-second sleep between requests to stay safely under the 60/min rate limit. No code change to production; pure data-state restoration.

Executed in two passes. First pass: 150/245 documents ingested successfully; 95 failures, all HTTP 502, clustered in the first ~10 minutes of the run. Progress-line analysis showed the failure count plateaued at 95 by document position ~110 and never incremented again across the final 135 documents — a single Render free-tier pod-restart window triggered by cold-start embedding load, not persistent instability.

502 retry logic was added to the script (one-shot, 5-second backoff) before the retry pass. Retry pass against the 95 failed doc_ids only: 95/95 successful, zero failures, ~10 minutes total. Confirmed the first-pass failures were a single pod-restart event.

Final disk state: 245 FiQA documents + 2 stale test chunks from May 3 and May 12 + 3 duplicate chunks from the smoke-test pass against the first 3 corpus documents. Duplicates are cosmetic and do not affect retrieval quality; cleanup deferred.

Phase 4 verification: 4 production queries across mortgages, 401k, capital gains tax, and backdoor Roth IRA. Two (mortgages, 401k) returned grounded answers with multiple FiQA citations and 4–9s latency. Two (capital gains tax, backdoor Roth IRA) returned honest abstentions because those specific topics are not directly covered in the 245-doc sample. The abstention behavior is correct — the explicit-abstention path is by design — not a retrieval failure.

### Lesson — primary

"Deploy succeeded" plus green health checks is compatible with "every query returns empty." None of the standard observability signals exercise the actual data path. `/health` and `/health/ready` confirm the service is running and dependencies are reachable, not that the corpus is populated. `/metrics` emits scrape data on whatever queries happen to fire, but does not distinguish between "answered with citations" and "fell through to empty-retrieval fallback."

After any change to a deployment target — env var, disk path, Dockerfile, start command, or a forced redeploy from any cause — run one curl against the production `/query` endpoint with a known-good query and verify a real answer with citations. Ten seconds. Catches the entire class of "deploy succeeded but retrieval is empty."

### Lesson — secondary

False hypotheses during systematic debugging are still data, but pattern-matching to "what changed recently" can mislead. Five hypotheses were ruled out before the real cause was found, and one (BM25Okapi IDF degenerate math) was wrongly considered confirmed before being falsified by Streamlit returning a qualitatively different abstention message than the curl path. Each diagnostic ruled out or refined a class of cause; none were wasted. The 2026-04-17 vectorstore-singleton entry's "hypothesis → distinguishing test → next hypothesis" pattern held up across all of them.

The trap to flag for future-me: when a bug surfaces after a deploy, "what changed recently" feels like the obvious starting frame, but the cause can predate the deploy by weeks. The right question is "what state would produce this symptom independent of when it became visible." In this case the symptom became visible on May 12, but the cause was a missing re-ingest step on May 3.

### Lesson — tertiary

Portfolio demos need an automated end-to-end production health check. The reason this took 9 days to surface is that the only person who ever queried production was me, and I only did so tonight because the security pass forced it. A reviewer or interviewer hitting the live demo at any point in those 9 days would have seen the broken state and drawn the wrong conclusion about the project. A 30-second cron job that hits `/query` with a known-good question and alerts on empty-citation responses would have caught this within an hour. Worth adding as a follow-up; out of scope for tonight's fix.

### Commit

`57ad6c3` — Add HTTP-based FiQA corpus ingest script for production restoration

The script being a permanent utility flips the "data state, not code" framing from the original version of this entry. The data restoration produced a reusable code artifact.

---

## 2026-05-13 — attributed_spans HTML leaks into rendered answer on dollar-amount-heavy outputs

### Symptom

A Phase 4 verification query ("How does a 401k employer match work and how should I maximize it?") returned an answer with raw HTML markup visible in the rendered Streamlit response. Fragments including `</mark>`, `<sup style="background:#6d4aab; color:white; padding:1px 5px; border-radius:3px; margin-left:2px; font-size:11px; font-weight:600;">2</sup>`, and `</div><div class="src-chunk">chunk 71d13a7f…</div></div>` appeared inline in the prose mid-sentence. The underlying text content was correct — the answer was grounded in retrieved FiQA documents. The mortgage and Roth IRA queries in the same verification batch rendered cleanly.

### Trigger pattern (suspected)

The query answer contained dollar amounts with commas (`$54,000`, `$18,000`). The `_render_answer_with_spans()` function in the Streamlit UI processes LLM output containing `<cite source="N">` trailing markers by substituting each span with `<mark>` highlight tags and `<sup>` citation links. The suspected failure mode: the regex that consumes each `<cite source="N">` token fails to match cleanly when the marker is adjacent to a numeral or comma, leaving an unclosed HTML span in the output that Streamlit's markdown renderer passes through as raw HTML. A secondary candidate: the markdown-unsafe-HTML pass closes spans in the wrong order when a sentence ends with a number followed immediately by a cite marker.

The May 30 root-cause investigation showed both suspected mechanisms above were wrong. See the 2026-05-30 math-parser entry for the actual cause.

### Impact

Cosmetic but damaging for a portfolio demo. Raw HTML in an LLM answer is the kind of artifact that signals a broken product to a reviewer, regardless of whether the underlying information is correct. Observed on one of four production queries during the May 12 Phase 4 verification batch (the 401k query). The other three rendered cleanly. The trigger rate at scale is unknown.

### Status

Resolved by the 2026-05-30 math-parser fix. Re-running the 401k trigger query confirmed the citation-chrome fragments (`</mark>`, `<sup>`) and the source-card chrome fragments (`</div><div class="src-chunk">`) all render correctly. Scope is narrow: the markup chrome fragments are fixed. Source-card excerpt content rendering still has a separate hazard observed on the same re-run — see the 2026-05-30 source-card excerpt entry below.

### Proposed fix (superseded)

Two layers were proposed at the time: tighten the cite-marker regex, and restructure into a markdown-then-injection pipeline. Neither was needed — the actual fix was a one-line `\$` escape in the renderer's text-extraction path (see the 2026-05-30 math-parser entry's Fix section). Kept here for the historical record of the hypothesis that didn't pan out.

### Cross-reference

The 2026-04-30 entry covers the attributed-spans parser design — how trailing `<cite source="N">` markers get attributed to the preceding clause, and why Llama's RLHF prior resists wrapping cited content directly in XML tags. The 2026-04-30 entry resolved the parser layer (trailing-marker attribution instead of LLM-bounded wrapping). This bug surfaces in the renderer layer that consumes those attributed spans — the next step in the same pipeline.

The 2026-05-30 math-parser entry is the closing entry for this bug.

### Commit

`2ceacea` — Make the single-corpus UI corpus-aware (same commit as the 2026-05-30 math-parser entry; that commit's `_md_escape` helper closed this bug, per the math-parser entry's verification re-run).

---

## 2026-05-13 — delete_document() wipes entire FAISS index instead of single document

### Symptom

Latent; not yet observed in production. No call to `DELETE /documents/{doc_id}` has been made against the live Render deployment. The bug was found by code-reading during the cleanup-duplicates investigation. If called, vector search and hybrid retrieval would return zero results for every subsequent query until a full re-ingest completes (~25 minutes for 245 docs). BM25-only keyword retrieval would still work because the BM25 index is rebuilt correctly.

### The bug

`delete_document()` at `vectorstore/vector_store.py:195-202`:

```python
async def delete_document(self, doc_id: UUID) -> int:
    original_count = len(self.metadata)
    self.metadata = [m for m in self.metadata if str(m.get("doc_id")) != str(doc_id)]
    if self.index is not None:
        self.index.reset()          # ← destroys all vectors
    self._bm25_index.build([m.get("content", "") for m in self.metadata])
    self.save()
    return original_count - len(self.metadata)
```

`self.index.reset()` calls `IndexFlatIP.reset()`, which empties the FAISS index entirely — it is FAISS's "clear all vectors" operation, not a per-vector delete. `IndexFlatIP` does not expose a row-level remove operation at all. After `reset()`, the metadata list has been correctly filtered to exclude the deleted document's entries, but the embeddings for every surviving document are gone. `self.save()` then writes the now-empty FAISS index to disk, making the destruction permanent. On next load, `self.metadata` has all surviving chunks and `self.index.ntotal` is 0.

### Why it stayed hidden

No code path in production calls `DELETE /documents/{doc_id}`. The endpoint exists, is HTTP-accessible, and appears in the OpenAPI schema, but it has never been exercised against a populated index. Unit tests for `delete_document` use a fresh test store with one or two test documents. They check post-delete metadata length but do not assert `index.ntotal == original_ntotal - chunks_deleted` — a check that would have failed from the first test run. If a test had asserted FAISS vector count after deletion, the bug would have been visible on day one.

### How it was found

Investigation into removing 3 duplicate Roth IRA chunks from production — the first 3 FiQA docs were double-ingested during the May 12 corpus restoration (once in the `--limit 3` smoke test, once in the full 245-doc run). Reading `delete_document()` to understand whether the endpoint deletes by internal UUID or by source `doc_id`. The `self.index.reset()` call was immediately flagged: `reset()` is FAISS's "remove all" operation, not a targeted delete. Confirmed by reading `IndexFlatIP.reset()` in the FAISS docs. The decision not to call the endpoint followed directly.

### Severity

High. The endpoint is reachable from production at any time. Any deletion attempt — manual debugging curl, an admin script, an exploratory API call — would destroy vector and hybrid retrieval for all surviving documents until a full re-ingest. Re-ingest for 245 FiQA docs takes approximately 25 minutes over HTTP on the Render free tier. The API key middleware (commit `6416224`, `RAGCORE_AUTH_ENABLED`) gates the endpoint behind auth when auth is enabled, but auth is currently off in the demo deployment.

### Proposed fix

FAISS `IndexFlatIP` does not support per-vector deletion. Four plausible paths:

1. **Store embeddings in metadata pickle.** On `delete_document`, filter metadata, then rebuild the FAISS index by re-adding the surviving embeddings directly from the pickle — no embedder calls needed. Storage cost ~4 KB per chunk × 245 chunks ≈ 1 MB. Requires a one-time format migration of the existing metadata pickle. This is the planned approach.
2. **Re-embed retained docs after deletion.** Call the embedder for each surviving chunk on every delete. Compute cost scales linearly with corpus size; impractical at scale.
3. **Switch to `IndexIDMap`.** Wrap an inner FAISS index in `IndexIDMap` or `IndexIDMap2`, which expose `remove_ids()`. Medium refactor; changes the on-disk index format.
4. **Switch to Chroma.** The codebase already includes `ChromaVectorStore`; Chroma supports per-vector delete natively. Flipping the default provider would close the bug with minimal new code.

Path 1 is the right call: store embeddings in the metadata pickle and rebuild the index from surviving entries after deletion — no new dependencies, no provider change. Planned for a separate commit later today.

### Gate until fixed

Documented in AUDIT.md as known limitation #22. Do not call `DELETE /documents/{doc_id}` on production until the fix lands. The endpoint could return 501 Not Implemented as an interim guard, but that requires a code change beyond the scope of this documentation commit.

### Cross-reference

The 2026-05-12 corpus-restoration entry describes the impact of a fully-empty production FAISS index: every query returns the empty-retrieval fallback in sub-millisecond latency with no grounded answer. Calling `DELETE /documents/{doc_id}` on production would reproduce that exact failure mode through a different mechanism.

### Commit

Single commit covering this entry and AUDIT.md limitation #22. Code fix follows in a separate commit later today.

---

## 2026-05-27 — XBRL period-span variation: Q2/Q3 quarterly values inflated by ~2x/3x

### Symptom

Apple FY2024 Q2 revenue extracted from SEC companyfacts JSON came out
as $210B in the generated quarterly CSV. Apple's actual Q2 FY2024
revenue is ~$91B. Spot-checking the CSV against publicly reported
figures revealed every Q2 row reading ~2x high and every Q3 row
reading ~3x high. Q1 rows looked correct. Annual (FY) rows looked
correct.

### Initial audit (incomplete)

The pre-implementation audit sampled the first Q1 entry from the JSON
to confirm field shape (start, end, val, form, fp). All fields present,
end - start span = 90 days, single-quarter value. The audit declared
the shape correct and the dedup-by-(end, fp) rule sufficient.

### Actual root cause

XBRL companyfacts files each quarterly monetary metric in two distinct
shapes within the same 10-Q filing, under the same accession number,
sharing the same (end, fp) key:

- Single-quarter: start..end span ~90 days (or 97 days in Apple's
  53-week fiscal years FY2018, FY2024)
- Year-to-date:   start..end span ~181 days at Q2, ~272 days at Q3

The dedup-by-(end, fp) rule with earliest-filed wins picked whichever
entry happened to sort first by filed date — in practice this was the
YTD entry for almost every Q2/Q3 period. The CSV emitted YTD values
labeled as single-quarter.

Q1 was unaffected because Q1's single-quarter span and YTD span are
identical by definition (Q1 is itself the first quarter of the year).
Sampling Q1 to validate field shape was structurally blind to the
multi-shape issue.

### Fix

Added a span filter applied before deduplication: for Q1/Q2/Q3 entries,
accept only those with `end - start <= 100 days`. The 100-day ceiling
catches the 90-day standard quarters and the 97-day 53-week-fiscal-year
quarters, and excludes the 181/272-day YTD entries. FY entries are not
span-filtered (annual entries are ~365 days by definition).

After the fix: Q2 FY2024 revenue is $90,753M — matches Apple's reported
figure. All 25 quarterly rows spot-check against external sources.

### Lesson

When auditing time-series data that may contain cumulative variants
(YTD alongside single-period), the first-period sample is structurally
inadequate. Cumulative and single-period values are mathematically
identical at the first boundary; the divergence only appears at later
boundaries. Sample at least one entry from each period boundary, not
the first one only.

### Cross-reference

- 2026-04-29 Llama multi-array JSON: parser assumed one output shape,
  model emitted another. Same family — single-example audit missed
  the shape variation.
- 2026-04-30 attribution-spans: four varied prompt rewrites produced
  the same trailing-marker shape. Same family — variation a single
  sample didn't surface.

### Commit

_(Pending — fix applied in the JSON-to-CSV extraction script. Script
lives outside the repo during corpus assembly; moves into the repo
when the corpus is integrated.)_

---

## 2026-05-27 — SemanticChunker memory blow-up on 10-K-scale documents; replaced as DocumentStructureChunker overflow

### Symptom
Running SemanticChunker.chunk(doc) on Apple's FY2025 10-K (108 pages, 219K
chars) consumed 65.2 GB of physical memory after 43 minutes of CPU. The
system was swapping heavily; the process was producing no output and had
to be killed. The same chunker produced normal output on FiQA documents
(~300 chars each) in well under a second per call.

DocumentStructureChunker's overflow path delegated to SemanticChunker for
sections larger than max_section_chars=2000. On the 10-K, 28 of 222
sections exceeded that threshold — small enough that an individual section
might not have blown up alone, but the no-headings fallback path called
SemanticChunker on the entire document.

### Suspected cause (not fully root-caused)
Not fully root-caused. Hypothesis: sentence-transformers' .encode() on
several thousand sentences in a single call holds transformer
intermediates in a way that scales much worse than the documented O(N)
memory profile suggests. The model itself is ~80 MB, the embedding matrix
for ~3K sentences at 384 dims is ~5 MB — neither alone explains 65 GB,
so the cost is in intermediates or in some N²-shaped allocation downstream
of .encode(). Pathological inputs may also include very long "sentences"
produced by the regex split on 10-K text where period-then-space is rare
inside dense table regions.

Did not investigate further. The chunker is not the right tool at this
scale regardless of the precise cause.

### Fix
DocumentStructureChunker.__init__ now uses FixedSizeChunker as its overflow
path. SemanticChunker is still mapped in the factory and remains the right
choice for short documents — its class docstring is updated to warn that
it should not be used on documents above ~50K chars.

Both DocumentStructureChunker overflow call sites are covered by the same
attribute: the oversized-section split path and the no-headings fallback
path. A heading-less PDF (scanned report, etc.) would have hit the same
OOM via the fallback path; the substitution closes that exposure too.

### Lesson
"WORKING in the existing test corpus" does not generalize to documents an
order of magnitude larger or shaped differently. SemanticChunker had
pre-existing tests on FiQA-sized inputs; none would have caught the 10-K
memory profile. Before promoting a chunker into the production path of a
new corpus, run it once standalone against a representative document from
that corpus with a memory ceiling and a wall-clock budget.

### Cross-reference
- 2026-05-27 XBRL period-span variation: same audit-was-incomplete family.
  A single sample of an "easy" input shape masked a failure mode that
  only appeared on the harder inputs the new feature would actually see.
- The "Patterns I've noticed" section at the end of this file documents
  the recurring shape: modules flagged WORKING in static audits surface
  silent failures when first exercised against a new workload. AUDIT.md
  marked SemanticChunker WORKING; the classification was accurate against
  FiQA and false against the 10-K.

---

## 2026-05-28 — Dockerfile silently missing pymupdf4llm; Render deploy succeeded green, first PDF query crashed with ModuleNotFoundError

### Symptom
Render deploy completed, container started, /health returned 200, and
/health/ready returned ready. The first request that loaded a PDF
through the ingestion path raised ModuleNotFoundError: pymupdf and
returned HTTP 500. Local development was unaffected because
`pip install -e ".[all]"` in venv311 had pulled pymupdf4llm and its
transitive pymupdf from pyproject.toml.

### Root cause
Three dependency sources in the repo, two in sync and one drifted:

- pyproject.toml lists pymupdf4llm>=1.27 as a main dependency (added
  alongside the AGPL notice commit).
- requirements.txt is a one-line shim: `-e .[all]`. It resolves
  through pyproject and gets pymupdf4llm correctly.
- Dockerfile maintains its own hardcoded `pip install` list of ~22
  packages. It COPYs requirements.txt but never sources from it —
  the install layer runs explicit `pip install <pkg>==<pin>` lines
  and ignores the COPYed file. pymupdf4llm was never added to that
  hardcoded list when the dependency was introduced.

The drift was structural, not accidental: the Dockerfile already
carried a `# TODO(AUDIT #11)` comment explicitly acknowledging that
its install list does not read from pyproject.toml and warning that
manual pins would drift over time. The TODO predicted exactly this
class of failure; nothing closed the loop on the prediction.

### Why CI didn't catch it
CI's unit-tests job runs `pip install -e ".[test]"`, which sources
from pyproject.toml — it installs pymupdf4llm correctly and the test
suite passes. CI's docker-build job builds the image (so it would
catch a Dockerfile syntax error or unresolvable pin) but does not
run the container or exercise any code path through the built image.
No CI job both (a) builds the Docker image AND (b) runs the
PDF-loading code through that image. The only environment that does
both is Render production. The drift is invisible until first deploy.

### Fix
Add two lines to the Dockerfile pip install block, next to the
existing pdfplumber line, pinning both pymupdf and pymupdf4llm
explicitly. pymupdf is pulled transitively by pymupdf4llm but is
also imported directly by the PDF loader (used for page_count), so
it is a first-class dependency in the source code and is pinned
explicitly to match. Versions match the resolved pins in venv311
(currently 1.27.2.3 for both packages, which release in lockstep).

Adding the new lines changes the RUN command string, which changes
the layer hash, which invalidates the cache for the pip install
layer and every downstream layer including the model pre-download.
The next Render deploy will re-run pip install (~3-5 min) and
re-download BGE-large + cross-encoder from HuggingFace (~1-2 min),
adding ~5-7 min over a fully-cached build. Acceptable cost.

### Deferred
The permanent fix is the existing TODO(AUDIT #11): rewire the
Dockerfile to `pip install -r requirements.txt` (or equivalent
pyproject-sourced install), so the hardcoded list goes away and
all future dependency additions to pyproject.toml automatically
propagate to Docker builds. Not included in this commit — it's a
larger refactor that benefits from being isolated and tested
against a full deploy cycle.

### Cross-reference
- 2026-05-12 production-FiQA-corpus entry: same "deploy succeeded
  green, feature broken" family. Different mechanism (data state
  drift rather than dependency drift) but identical observability
  failure: every standard signal (deploy status, /health, /metrics)
  said the system was healthy while the actual code path was broken.
  Generalizes to: green deploy is a necessary condition for "system
  works," never sufficient.
- TODO(AUDIT #11) in Dockerfile: predicted this class of drift
  explicitly. The TODO is older than this incident and identifies
  the exact structural fix. The lesson is not "we should have read
  the TODO" — it's that flagged-but-not-acted-on technical debt
  fires eventually, and the longer it sits, the more code paths
  accumulate around it as latent traps.

---

## 2026-05-29 — Heuristic metadata-hint trap: boolean where-clause filter silently zeroed Apple retrieval

### Symptom

After Setup B shipped the 6 Apple Chroma corpora to Render's persistent disk, `GET /corpora` correctly reported 7 entries with the expected `doc_count` values. Every `/query` against an Apple corpus returned the empty-retrieval fallback — `citations: []`, `total_tokens: 0`, the canned abstention text — while FiQA queries continued to work. The pattern read like a deploy or seeding failure: the data was clearly on disk (count was right), but no chunks came back.

### Investigation

Each layer was probed in isolation and ruled out by a single targeted test before moving down.

**Generation** — ruled out. `model_used` matched the configured `llama-3.3-70b-versatile` and `total_tokens` was 0, which is only emitted by `_empty_response` (the LLM is never called on that path). Generation could not be the failure point.

**Deploy / seed** — ruled out. The exact failing query reproduced locally against the in-repo `data/chroma_collections/apple_10k_document_structure` collection. Whatever was happening was not specific to the Render disk or to the first-boot seed copy.

**Store layer** — ruled out. A throwaway probe instantiated `ChromaVectorStore` directly, embedded the query through the same `BGEEmbedder` the deployed runtime uses, and ran both the raw `_collection.query(query_embeddings=[vec], n_results=5)` and the wrapper `vector_search(vec, top_k=5, metadata_filter=None)`. Both returned 5 strong matches (cosine distances 0.25–0.34, the top hit being the literal by-category sales table). The embeddings matched the stored vectors in shape and norm (both 1024-dim, both L2-normalized to 1.0). `list_collections()` showed a single collection of the right name and the right ID. The store was healthy end-to-end.

The divergence was isolated by extending the probe to drive the real router → executor pipeline in-process, with the store's search methods monkey-patched to log their incoming arguments. At each boundary the chunk count was printed. `RetrievalExecutor.execute()` returned 0 chunks for the same query that returned 5 against the raw store, and the wrapped store call logged `metadata_filter={'author': True}` — a value that did not appear when the wrapper was called directly with `metadata_filter=None`.

### Root cause

`HeuristicRouter.extract_metadata_hints` at `retrieval/router/query_router.py` emits boolean markers — `{'author': True}`, `{'doc_type': True}`, `{'date_range': True}` — when its regex patterns fire. `QueryRouter.route()` merged these into `combined_filter`, which became `RoutingDecision.metadata_filter`, which became Chroma's `where=` clause. Chroma compared boolean `True` against the actual string-valued metadata on each chunk (e.g. `doc_type='pdf'`), matched nothing, and returned zero rows. The executor's `try/except` fallback at `retrieval_executor.py:51` is gated on **exceptions**, not on empty results — a cleanly-executed query that returns `[]` propagates straight through to the orchestrator's `if not retrieval_result.chunks` branch and emits `_empty_response`.

The `author` pattern made the regression catastrophic on the Apple demo specifically. Its source pattern `by [A-Z][a-z]+ [A-Z][a-z]+` was written to match a proper-noun author name like "by Tim Cook." The call site at `query_router.py:88` passes `re.IGNORECASE`, which neuters the case classes — under that flag `[A-Z]` and `[a-z]` each match any letter — collapsing the regex to `by \w+ \w+`. Common financial phrasings such as "net sales by product category," "revenue by region," "costs by quarter" all match. Every one of them silently produced an empty result on every Apple corpus.

The other two patterns share the same defect at the boolean-emission layer: `{'doc_type': True}` fires on "in the report" / "in the document" / "in the pdf"; `{'date_range': True}` fires on "from 2025" / "since 2020" / etc. Both are common in queries about a 10-K or an earnings report.

### Why it stayed hidden

The hint fields don't exist on Apple chunk metadata at all. No chunk carries `author` or `date_range`; `doc_type` exists but is degenerate per-corpus (every chunk in a given corpus has the same value). The filter could only ever subtract correct results, never match. `count()` was a faithful 392 and `/corpora` reported the right registered set because the collection itself was healthy — the kill was a query-time filter two layers above the store, applied to a column the data didn't even have.

### Fix

Stopped merging `meta_hints` into `combined_filter` at `query_router.py`. `HeuristicRouter.extract_metadata_hints` and `METADATA_PATTERNS` are left in place — they may be revived once a corpus carries real author or date metadata that a string-valued filter could match — but `route()` no longer consumes them. A comment at the merge site records the failure mode and the IGNORECASE detail so the trap is not reintroduced. Active metadata filtering still works via the LLM classifier's `llm_meta` (a JSON object emitting real string values per its system prompt) and any caller-supplied `metadata_filter`. Verified the previously failing query plus two other trap phrasings ("earnings from 2025" and "discussed in the report") each now retrieve 20 chunks instead of zero.

### Lesson — primary

Non-zero `count()` combined with zero query results is a query-time filter or where-clause problem, not a data problem. The bisection move is to call the store directly with `metadata_filter=None` and compare to the same call through the full pipeline; the first differing argument between the two calls names the buggy layer. Applied here, the difference at the executor boundary was `{'author': True}` vs `None` — and the layer that injected it was the router. Health signals at the registration layer (`count()`, `/corpora`) tell you the data is loaded; they do not tell you anything about what queries against that data return.

### Lesson — secondary

`re.IGNORECASE` silently disables the `[A-Z]` and `[a-z]` case classes. Never combine them in the same regex. The original `by [A-Z][a-z]+ [A-Z][a-z]+` was clearly meant to require proper-noun capitalization, and the IGNORECASE flag was equally clearly meant to be permissive about the literal `"by"` — those two intentions are incompatible. If the literal word needs case-insensitivity, use a per-token case-fold or a scoped inline flag like `(?-i:[A-Z][a-z]+)` for the parts that need their case classes intact. Cheaper still: lowercase the query and write the literal word in lowercase, leaving the case classes to mean what they say.

### Cross-reference

- 2026-05-12 production-FiQA-corpus and 2026-05-28 Dockerfile-pymupdf4llm: same "green deploy + healthy-looking endpoint, every query returns empty" family. This is the third confirmed instance and the third distinct mechanism — missing data, missing dependency, query-time filter. Each was invisible until the live `/query` path was exercised end-to-end against the real corpus. The "Patterns I've noticed" section below is updated with the generalization.

### Commit

`fbe1748` — Disable heuristic metadata hints from the retrieval filter

---

## 2026-05-29 — Citation markers render blank in answer text (observed on Apple queries)

### Symptom

A deployed Apple-corpus query returned a correct grounded answer, but the inline `[N]` citation markers in the answer text were empty — the rendered output read "According to , …" and "as noted in that …" rather than "According to [1], …" or "as noted in [2] that …". The `attributed_spans` field on the response carried correct source indices, so the attribution data was intact; only the substitution into the answer prose dropped the number.

### Status

Open. Not yet investigated. Generation/renderer layer. Re-confirmed still open as of 2026-06-05 — no investigation since the original 2026-05-29 filing; per-session document work has been the focus.

### Cross-reference

Likely overlaps with two earlier entries — 2026-05-13 `attributed_spans` HTML rendering bug (renderer mishandling LLM-output HTML) and 2026-05-02 per-call Llama citation-marker stochasticity (model emits the markers inconsistently across calls). This symptom may be either of those resurfacing on the Apple corpus rather than a new bug. Confirm shape before treating as distinct.

### Commit

N/A — not yet investigated.

---

## 2026-05-29 — Follow-up questions are wrong-domain (FiQA topics on Apple queries)

### Symptom

A deployed Apple 10-K query returned `follow_up_questions` about IRAs, 401(k)s, and investment-tax topics — pure FiQA-domain content, completely unrelated to the Apple corpus or to the query that was actually asked. Suggests the follow-up generator either lacks corpus / query context at the prompt level or is running against a hardcoded FiQA-flavored template.

### Status

Open. Not yet investigated. Demo-visible — undercuts the multi-corpus story the Apple work was meant to demonstrate. Generation layer. Re-confirmed still open as of 2026-06-05 — no investigation since the original 2026-05-29 filing; per-session document work has been the focus.

### Cross-reference

The 2026-04-29 entry covered follow-up question **parsing** (Llama's multi-array JSON output shape). This is a distinct failure mode of the same feature — topically wrong content, not malformed structure. The parser is presumably succeeding; what comes out of it is just on the wrong subject.

### Commit

N/A — not yet investigated.

---

## 2026-05-30 — Streamlit markdown-it inline-math parser silently consumes citation HTML between paired `$` signs

### Symptom

During the corpus-aware UI work, the new sidebar dropdown made the six Apple corpora reachable from the Streamlit app for the first time. On Apple 10-K queries (financial-figure-heavy), the rendered answer body showed raw HTML markup inline — visible fragments like `</mark><sup style="background:#6d4aab;...">` appearing as text instead of as a rendered highlight/badge. Some `<mark>` / `<sup>` spans in the same answer rendered correctly; others did not. FiQA-default queries continued to render cleanly.

### Initial hypothesis (wrong)

Suspected a regression from the same change-set to `_render_assistant_message` (added `corpus` kwarg, gated follow-up chips). Audit ruled it out: no edits in this change-set touched `_render_answer_with_spans`, the `unsafe_allow_html=True` flag, the answer-rendering CSS, or any selector reaching the answer body. The function's HTML-building loop and final `st.markdown(final_html, unsafe_allow_html=True)` were byte-identical to the prior version.

### What the data showed

Counted dollar signs across the saved `/tmp/q_*.json` responses:

| corpus                   | `$` in answer |
|--------------------------|---------------|
| `q_fiqa.json` (FiQA)     | 0             |
| `q_default.json`         | 0             |
| `q_apple_environmental`  | 0             |
| `q_apple_10k_fixed`      | 18            |

The 10-K answer is full of `$209,586`, `$201,183`, `$33,708`, etc. After `_render_answer_with_spans` builds `final_html` and `\n` is replaced with `<br>`, the actual string handed to `st.markdown` looks like:

```
... $209,586 in 2025, $201,183 in 2024, and $200,583 in 2023</mark><sup>1</sup> . <br><mark>- Mac: $33,708 ...
```

### Root cause

Streamlit 1.32.0's `st.markdown` runs the answer through markdown-it with inline-math (`$...$`) parsing enabled by default. The parser pairs dollar signs greedily — `$209,586 in 2025, $`, then `201,183 in 2024, and $200,583 in 2023</mark><sup>1</sup>.<br><mark>- Mac: $`, etc. — and silently consumes any HTML tags falling between paired dollar signs as math content. The HTML chunks lucky enough to land outside a math pair render normally. That exactly produces the "some spans render correctly, others show raw markup" symptom; uniform breakage from a dropped `unsafe_allow_html` flag would have looked different.

### Why this stayed hidden

Latent until the corpus dropdown shipped. FiQA-default answers contained zero dollar signs, so the bug never fired in any prior verification run. Apple corpora had been ingestable for two weeks but had no UI path that reached them — the dropdown was the first surface that let a user issue a `/query` against a financial-figure-heavy corpus and see the rendered output. The classification: not a regression from the corpus-aware-UI commits, but a latent renderer bug those commits surfaced by widening the answer-content distribution.

### Fix

Added a tiny `_md_escape(text)` helper that does `html.escape(text).replace("$", r"\$")` and routed all three answer-text extraction calls in `_render_answer_with_spans` through it (the gap text between spans, the span text inside `<mark>`, and the trailing text after the last span). Also escaped `$` in the empty-spans `st.markdown(answer)` fallback so the same hazard doesn't bite if a future corpus produces an answer with no citations. The `<mark>` and `<sup>` injection sites are untouched — only the user text passing through them gets the dollar-sign escape. Markdown then renders `\$` as a literal `$` and never enters math mode.

### Cross-reference

The 2026-05-13 entry ("attributed_spans HTML leaks into rendered answer on dollar-amount-heavy outputs") observed the same renderer-layer symptom on a FiQA 401k query that contained `$54,000` and `$18,000`. That entry hypothesized a regex-anchoring bug in `_render_answer_with_spans` — close, but wrong about the layer: the regex wasn't the problem, markdown-it's math parser was. Re-ran the May 13 trigger query after this fix landed; all three observed fragment types (`</mark>`, the purple `<sup>...</sup>`, and `</div><div class="src-chunk">`) render correctly. May 13 is closed by this commit. (The source-card chrome fragment was initially expected to need a separate fix because it comes from a different `st.markdown` call in `_sources_grid`. The re-run confirmed all three fragment types render correctly; the mechanism by which the source-card chrome fragment resolved was not separately diagnosed, and is not assumed here.)

This fix does NOT touch bug 2a (blank `[N]` citation markers, still open per the 2026-05-29 entry). 2a is an attribution-substitution layer issue with a different symptom; today's fix is purely about HTML chrome surviving the markdown render. The two bugs are independent.

The post-fix re-run also surfaced a separate hazard in source-card excerpt rendering (markdown-formatting from raw chunk text) — see the 2026-05-30 source-card excerpt entry below. Same content-distribution pattern, different surface, deferred.

The 2026-04-30 attribution-spans entry remains the upstream layer (parser turning trailing `<cite>` markers into spans). This bug was always in the next stage (renderer consuming those spans for display), and that layer is now hardened against dollar-amount content.

### Lesson

Two distinct generalizations, both well-supported.

First — content-distribution dependence is a real risk class. A renderer that's been verified against one corpus's answer distribution is verified for that distribution only. When the upstream content source widens (new corpus, new domain, new LLM, different question types), the renderer's hidden assumptions about answer content get re-tested whether the work plan acknowledges it or not. Treat "first time a code path sees a new content distribution" as deserving the same verification rigor as "first time a code path runs in production." Both are integration-shaped surfaces.

Second — markdown-it has a non-obvious set of plugins enabled by default in Streamlit. Inline math (`$...$`), tables, footnotes, and others are not opt-in; user-supplied content can trigger them without warning. Before passing user-derived (or LLM-derived) text into `st.markdown`, audit it for the special tokens that activate plugins: `$` for math, leading `|` for tables, `[...]` for links/footnotes, leading `>` for blockquotes, `~~` for strikethrough. For any token whose presence in the input would be incidental rather than intentional, escape it. This is the markdown analogue of html.escape — and the standard `html.escape` does not cover it.

### Commit

`2ceacea` — Make the single-corpus UI corpus-aware (the same commit that introduced the corpus dropdown also added the `_md_escape` helper that fixes this bug).

---

## 2026-05-30 — Source-card excerpt renders raw chunk text as markdown (observed during the math-parser fix re-run)

### Symptom

On the post-fix re-run of the 401k FiQA query that validated the 2026-05-13 closure, one of the source-card excerpts in the 2×2 sources grid rendered "18k/year" in monospace font — markdown code-formatting applied to text that should be plain prose. The answer body rendered correctly (the math-parser fix landed); the artifact was confined to a source-card excerpt.

### Suspected mechanism

The `_md_escape` helper introduced in this commit hardens only the answer body — it's called from `_render_answer_with_spans`. The source-card path in `_sources_grid` passes `cite.get("excerpt")` to the rendered HTML (via the `.src-excerpt` div) without escaping markdown-trigger tokens. Any backtick pair in the retrieved chunk content survives to the rendered HTML and triggers markdown code formatting; underscores would render italics; leading pipes would start a table; and so on. The "18k/year" monospace artifact is most likely a backtick pair in the chunk content reaching the renderer unescaped.

### Status

Observed-not-diagnosed. The suspected mechanism above is inferred from symptom shape and the obvious asymmetry in escaping coverage between the two render paths; not yet confirmed by inspecting the actual chunk content or stepping through `_sources_grid`. The retrieved chunk text itself is presumed correct — the issue is rendering, not retrieval.

### Likely fix (not yet applied)

Route source-card excerpts through the same `_md_escape` (or a sources-grid-specific equivalent covering the same markdown-trigger token set). The answer body now has a documented escape contract; source-card excerpts should match. Out of scope for the corpus-aware UI commit — observation only, deferred.

### Cross-reference

Same content-distribution-dependence pattern as the 2026-05-30 math-parser entry above: a renderer surface that worked fine on FiQA-default chunk content turns out to be sensitive to incidental markdown tokens in chunk text when the corpus distribution widens. The lesson there ("audit user-derived text for markdown-trigger tokens before passing into `st.markdown`") applies here too — `_sources_grid` is a second instance of the same hazard class. The math-parser fix hardened one of the two renderer surfaces; this entry tracks the other.

### Commit

N/A — observed, not fixed.

---

## 2026-06-04 — RAM-reclamation probe read ru_maxrss (PEAK RSS, structurally cannot decrease), reported "+0 MB everywhere," nearly killed the 2 GB session-TTL design

### Symptom

While building session TTL eviction (`fc0a4df`), the reaper's RAM-reclamation step needed verification: did removing chromadb's cached `System` actually return RSS pages to the OS, or was the process retaining them? A standalone probe (`scripts/rss_probe_linux.py` in an earlier draft) created a temp Chroma collection, upserted a synthetic embedding batch, called `_release_chroma_cache_for_path` + `del` + `gc.collect()` + `malloc_trim(0)`, and read process RSS before vs after. Every delta was +0 MB or +epsilon. Repeated across batch sizes, embedding dimensions, and trim points. The reading was consistent and damning: `malloc_trim` was reclaiming nothing, the chromadb cache eviction was freeing the handle but not the pages, the 2 GB Render box would OOM after a few dozen evict cycles, and the entire session-TTL design would have to be abandoned for a much heavier process-restart model.

### Initial hypothesis (wrong)

`malloc_trim(0)` doesn't reclaim Chroma/HNSW pages. The reasoning at the time: glibc's arena bookkeeping is documented to be conservative, HNSW's allocation pattern (many small graph nodes) fragments arenas in ways `malloc_trim` can't coalesce, and the measured "+0 MB" was direct evidence that whatever was being freed wasn't reaching the OS. Plausible, consistent with the data, and aligned with what's findable on the internet about HNSW + glibc. The hypothesis explained every datapoint the probe produced.

It was wrong. The probe was reading the wrong field.

### Actual root cause

`resource.getrusage(RUSAGE_SELF).ru_maxrss` reports PEAK RSS over the process lifetime — it is monotonic non-decreasing by construction. It cannot show a drop even when the kernel has reclaimed every page. The "+0 MB delta" the probe printed was the peak holding steady; the actual current RSS may have dropped by hundreds of MB between the readings, and `ru_maxrss` is structurally blind to that. The hypothesis under test ("does `malloc_trim` reclaim?") was being measured by an instrument that can structurally never report "yes."

The correct measurement on Linux is `/proc/self/status:VmRSS` — current resident pages, which DO decrease when the allocator returns memory. Re-ran the same probe reading `VmRSS`; deltas were −85% to −95% of the upsert footprint after `_release_chroma_cache_for_path + gc.collect() + malloc_trim(0)`. `malloc_trim` had been reclaiming the entire time. The session-TTL design survived.

### Why the wrong measurement was so plausible

`resource.getrusage` is the standard-library answer to "process RSS." Its documented field is `ru_maxrss`. The word "max" sits in the field name as a small flag, but the obvious-looking import path (`import resource`) and the obvious-looking field name frame the answer as "this is the RSS." Reading `/proc/self/status` is a more platform-specific move that the obvious path doesn't suggest. The wrong measurement was reachable in 3 lines of stdlib; the right one needed a file parse the stdlib doesn't surface.

### Fix

The probe and its docstring (`scripts/rss_probe_linux.py:14-24, 42-55`) now hard-document the trap: `ru_maxrss` is the PEAK and MUST NOT be used for before/after reclamation comparisons. The current-RSS helper reads `VmRSS` from `/proc/self/status` directly. The same warning is copied verbatim into the live regression test (`tests/integration/test_session_reaper.py:303-310`) so anyone editing that file sees the trap before they edit it.

### Sub-finding: the corrected diagnostic is Linux/glibc-only

`malloc_trim(0)` is a glibc-specific hook. macOS's allocator has no equivalent — the system allocator decides on its own when to return pages, and Python user code has no API to force it. The reaper's `_release_chroma_cache_for_path` still calls `malloc_trim` under a `try/except` so macOS development falls through cleanly (the reaper just frees the cache handle without the trim call), but the RSS-reclamation regression test is skipped on non-Linux (`tests/integration/test_session_reaper.py:315`).

Production runs on Linux, so the guarantee covers production; macOS local development will see RAM accumulate across many evict cycles, which is fine for hours-of-development scale and would not be fine at production session volume. The platform-asymmetry note belongs alongside the measurement note: both shape what "RAM is reclaimed" means operationally. Two corrections — measure with `VmRSS` (not `ru_maxrss`), and the fix is glibc-only (macOS has no hook) — applied together get the right answer.

### Regression tests

`tests/integration/test_session_reaper.py::test_F1_rss_reclaimed_on_linux` exercises the full eviction path against a real Chroma collection and asserts RSS drops by ≥50% of the upserted footprint. Skipped on non-Linux. Reads `VmRSS`, not `ru_maxrss`. Pins both the measurement-tool correction and the `malloc_trim`-actually-works finding in one test, so a future "let's clean up the imports" refactor that puts `resource.getrusage` back can't silently re-introduce the wrong reading.

### Lesson — primary

A number is only as good as what it measures. "+0 MB delta" was a real number from a real syscall — it just wasn't the number that answered the question being asked. Before drawing a strong conclusion from a "the measurement says no" result, audit the measurement: what does the field actually mean? Current vs peak? Average vs sample? RSS vs virtual size? The temptation to treat the measurement layer as transparent grows with how obvious-looking the API is. `getrusage` + `ru_maxrss` is the textbook example of an obvious-looking API hiding the wrong-shape answer — the field name even labels the trap (`max`), and the trap still landed.

### Lesson — secondary

Linux exposes per-process state through `/proc` that the standard library doesn't surface idiomatically. When the standard-library answer to a Linux observability question feels wrong, `/proc/self/status` (and friends) usually has the right field two seconds away. Same applies to fd counts, thread counts, memory maps — `resource.getrusage` is rarely the final answer on a Linux box.

### Cross-reference

- The session-reaper work (`fc0a4df`) depended on this finding for correctness. Had the wrong conclusion stuck, the entire 2 GB session-TTL design would have been abandoned for a much heavier process-restart model — and the failure mode "feature was correct, the test of the feature wasn't" would never have been surfaced.
- 2026-04-29 follow-up multi-array JSON: same shape of trap at a different layer. The obvious-looking parser call (`json.loads`) produced a real result, just not the right-shape result. "Looks correct" and "answers the right question" are independent properties at every layer — parser, measurement, render, audit.

### Commit

`fc0a4df` — Add session TTL eviction, in-flight pin, and boot cleanup (probe lives in `scripts/rss_probe_linux.py`; the live regression in `tests/integration/test_session_reaper.py`).

---

## 2026-06-04 — RetrievalExecutor.execute() isolation audit: Self-RAG and FLARE mid-generation re-retrieval would have leaked public-corpus chunks into a session answer

### Symptom

No symptom. Caught by code-reading during the per-session retrieval roll-out, before any production exposure. The `/query` → `/retrieve` → `/query/stream` session routing (`8fc597d`) threaded a `store_override: BaseVectorStore` parameter through the orchestrator so requests carrying an `X-Session-Id` retrieve from the session's private store instead of the public corpus. Initial retrieval was the obvious site to thread the override through; the question was whether that was the ONLY site, or whether other callers of `RetrievalExecutor.execute()` existed that would silently keep reading the public corpus mid-request.

### Audit

Grepped every production caller of `.execute(` against the executor:

- `orchestrator.py:253` — initial retrieval (query path). Already threaded `store_override`.
- `orchestrator.py:671` — initial retrieval (retrieve-only path). Already threaded `store_override`.
- `generation/advanced_generation.py:211` — Self-RAG mid-generation re-retrieval on unsupported claims. **NOT threaded.** Would have fetched additional verification context from the public default corpus while the initial answer came from the session's private store.
- `generation/advanced_generation.py:482` — FLARE mid-generation re-retrieval on novel-numeric-token detection. **NOT threaded.** Same mechanism — the session-scoped initial answer would be augmented with chunks from the public corpus on every novel-token trigger.
- `generation/advanced_generation.py:672` — agentic search loop (LangGraph agent path). Not currently wired to any user-reachable session endpoint. Left unthreaded with an explicit warning comment.
- `agent/tools/search_docs.py:59`, `agent/nodes/retriever.py:42`, `agent/tools/summarize_doc.py:45,75` — agent sub-tools. Same classification as the agentic search loop: not currently user-reachable for session requests; left unthreaded with the understanding that wiring an agent endpoint to session routing would have to thread the parameter through first.

Nine production call sites in total. The commit message for `8fc597d` says "all ten `RetrievalExecutor.execute()` call sites"; the tenth audit surface it counts is the executor's own `async def execute` declaration at `retrieval/strategies/retrieval_executor.py:38` — the contract source, not a caller. Both numbers describe the same audit: nine call sites + one declaration = ten places where the `store_override` parameter has to be coherent.

Two sites — Self-RAG and FLARE re-retrieval — would have leaked. The rest were safe by either threading or non-reachability.

### What would have happened

A user uploads a proprietary PDF into a session, asks a factual question, the orchestrator routes initial retrieval to the session store (correct), Self-RAG flags an unsupported claim, the verification step calls `retrieval_executor.execute()` without the override (bug), public FiQA chunks join the session's chunk set, and the final answer cites public-corpus chunks the user never uploaded — in the worst case quoting unrelated public content as if it were grounded in the user's private document. Same mechanism for FLARE. Neither would have produced a visible error: the answer would look fluent, the citations would resolve to real chunks, and the contamination would be invisible without inspecting `[Source N]` provenance against what was uploaded.

### Fix

Added `store_override: Optional[BaseVectorStore] = None` to both `SelfRAGGenerator.generate` and `FLAREGenerator.generate`, threaded it through to the re-retrieval `.execute(...)` calls (`advanced_generation.py:212` and `:483`), and updated the orchestrator sites that invoke Self-RAG (`orchestrator.py:397`) and FLARE (`orchestrator.py:422`) to pass through the same override they received. Defaulting to `None` preserves the behavior of every non-session caller exactly.

The agentic search loop (`advanced_generation.py:668-672`) is the one production caller that stays at `store_override=None`. A comment at the site documents the contract explicitly: the loop is safe today ONLY because no user-reachable endpoint routes session traffic to it; wiring an agent endpoint to session traffic must thread the override first or it ships the leak.

### Why this stayed hidden until the audit

The leak is invisible without two preconditions: (1) a session exists, and (2) Self-RAG or FLARE fires its iterative re-retrieval branch on that session's query. Until the per-session retrieval work landed, condition (1) didn't exist — no caller had a "private" store to leak from. The bug was structurally latent from the moment session routing was introduced, and would have shipped with the session feature itself if the audit hadn't preceded the merge.

### Regression tests

The orchestrator-level session-routing tests assert that retrieval with `store_override` set never reads the public corpus. The Self-RAG and FLARE re-retrieval sites are now covered by the same contract because they accept and thread the same parameter; a leak would require the orchestrator to receive `store_override` and then call through to a generator that drops it on the floor — which would be a new bug rather than this one.

### Lesson — primary

A retrieval primitive's isolation is only as complete as its least-audited caller. The contract "session traffic stays in the session's store" is enforced at every `.execute(...)` site or it is enforced nowhere — one unthreaded caller is enough to leak. When a new isolation parameter is introduced on a primitive, the audit step that matters is `grep`, not type-checking: type-checking confirms each site COULD accept the parameter; only `grep` confirms each site DOES pass it. The discipline is "enumerate every caller, classify each as threaded / not-threaded / not-applicable, document why for the not-applicable ones in a comment at the call site."

### Lesson — secondary

When an isolation parameter is `Optional[X] = None`, the safe default is also the leak-prone default. Threading the parameter is opt-in; forgetting to thread it is silent. For isolation contracts specifically, the safer pattern would be a required parameter at the executor boundary so the type system enforces "every caller has thought about this" — but that's a larger refactor with churn across many call sites. The interim discipline is the audit comment at each not-threaded site explaining why it's safe to omit; future edits that wire those sites to user-reachable session paths then have to address the comment, which makes the contract enforceable by code review.

### Cross-reference

- 2026-05-13 `delete_document` FAISS wipe (AUDIT #22): same texture — bug found by code-reading during a different investigation (there, cleanup-duplicates investigation; here, per-session retrieval roll-out). Both are "the call site looks reasonable in isolation; what's wrong is what the call site doesn't do." Both were caught before production exposure because the surrounding work surfaced the relevant code. Two instances of this pattern now in the file — promoted to "Patterns I've noticed."

### Commit

`8fc597d` — Route `/query`, `/retrieve`, `/query/stream` to per-session stores when X-Session-Id present (orchestrator and `advanced_generation.py` changes shipped together).

---

## 2026-06-04 — chromadb declared as optional [chroma] extra but a hard runtime dep; CI red, masked for weeks by the Dockerfile's hand-install

### Symptom

CI's `unit-tests` job started failing on the per-session work with `ModuleNotFoundError: No module named 'chromadb'`. The traceback pointed at the FastAPI lifespan: the per-session `SessionRegistry` constructed a `ChromaVectorStore` per registered corpus on startup, which failed at import time. Locally the same code ran fine; the Docker build produced a working image; only CI's `pip install -e ".[test]"` install path was red.

### Root cause

Three install surfaces, two in sync and one drifted:

- `pyproject.toml` listed `chromadb>=0.4.24,<0.5` under `[project.optional-dependencies.chroma]`, not under `[project.dependencies]`. A bare `pip install -e .` did not pull chromadb.
- The `Dockerfile` hand-installed `chromadb==0.4.24` in its `RUN pip install` block (line 38). Production and any local Docker build picked it up.
- CI's `unit-tests` job runs `pip install -e ".[test]"`. The `[test]` extra did not depend on `[chroma]`. CI installed a chromadb-less environment.

The reason CI wasn't red earlier: prior CI never exercised a code path that imported chromadb. As long as `ChromaVectorStore` was only constructed inside opt-in tests (and the default FAISS-backed orchestrator was the lifespan's only vector store), the unit-test job could run without chromadb in scope. The per-session work moved Chroma into the startup-required path — every test that triggers the FastAPI lifespan now imports chromadb — and the misdeclaration became visible the moment a test first hit the lifespan.

### Why this stayed hidden so long

The Dockerfile install masked the misdeclaration for the entire window between when chromadb was first introduced and when it became startup-required. Production deploys were fine because the image had chromadb. Local development was fine if developers had ever run `pip install -e ".[chroma]"` or used the Docker image, which everyone had. The only environment that would have surfaced it is a fresh checkout running `pip install -e .` with no extras and then trying to boot the API — a path nobody had run.

This is a near-mirror of the 2026-05-28 Dockerfile/pymupdf4llm entry with the directions reversed. There, the Dockerfile was missing a pyproject-declared package; here, pyproject was missing a Dockerfile-installed package. Both shipped because no environment crossed both surfaces.

### Fix

Moved `chromadb==0.4.24` from `[project.optional-dependencies.chroma]` to `[project.dependencies]`. Removed the now-redundant `[chroma]` extra. Local `pip install -e .` and CI's `pip install -e ".[test]"` both pull it now; the Dockerfile's hand-install line is kept as a belt-and-braces pin and to preserve the build-layer cache.

### Why the pin is exact (==0.4.24), not >=

The session-store cache eviction reaches into a chromadb internal: `SharedSystemClient._identifer_to_system` (note the upstream typo — "identifer", not "identifier"). On a session evict, we walk that dict, find the cached `System` entry matching the session's persist directory, call `system.stop()`, and `pop` the key. Without this, chromadb retains the cached `System` indefinitely and the session-handle never frees — RSS climbs (see the 2026-06-04 `ru_maxrss` entry for the diagnostic that made this measurable in the first place).

The reach is fragile by construction: a private attribute, with a typo in the upstream name. A higher 0.4.x release that fixes the typo (or refactors the cache) would silently change the attribute name. Our code already guards with `getattr(SharedSystemClient, "_identifer_to_system", None)` and falls back to a logged warning if the cache layout is unexpected (`scripts/rss_probe_linux.py:66-69` and the matching helper in `vectorstore/session_store.py`) — which means a chromadb minor bump would NOT raise. It would log "cache layout unexpected" and the session-handle would leak silently. RSS would climb for a few dozen evict cycles, OOM, restart.

Pinning `==0.4.24` is therefore not "we like this version"; it's "we are reaching into this version's internals and any change to them fails silent." Bumping the pin requires re-verifying the attribute name (or refactoring to a public API if chromadb ever adds one). The pin's docstring at the declaration site says exactly this so a future developer can't bump it to `0.4.25` thinking "patch version, safe."

### Regression detection

The `pip install -e ".[test]"` failure that surfaced this is itself the regression detection: a bare-install path that exercises the lifespan now fails at install rather than at boot, which is the correct shape. Going forward, any future hard runtime dependency that gets misdeclared as optional will reproduce the same shape on the unit-tests job within one CI run.

### Lesson — primary

A dependency that is present in production only by transitive accident — Docker installs it, the platform image happens to ship it, an unrelated `[extra]` pulls it — is structurally undeclared. The declared dependency set is whatever a bare `pip install -e .` on a clean Python pulls; anything beyond that is environmental luck. The right surface to declare a hard runtime dependency is `[project.dependencies]`, not an extra and not the Dockerfile alone. CI should install from the declared set, exercise the startup path, and fail if anything is missing — which is the shape that surfaced this bug, but the underlying setup was "got lucky" until the per-session work forced the lifespan to import chromadb.

### Lesson — secondary

When code reaches into a library's private attributes, pinning the exact version and documenting the reach at the declaration site is the discipline that prevents silent-fail upgrades. The pin alone is insufficient: `chromadb==0.4.24` doesn't tell a future reader anything about why it's exact. A comment at the pin documenting "reaches into `SharedSystemClient._identifer_to_system`, a 0.4.24 internal" raises the question at the declaration site rather than at the silent-failure site three months later.

### Cross-reference

- 2026-05-28 Dockerfile/pymupdf4llm: same dependency-drift family, opposite direction. Both belong in the "green deploy + healthy status signals + feature actually broken" pattern. Both required the live integration path to surface them. Together with this entry, that pattern now has its fourth confirmed instance — annotated below in "Patterns I've noticed."
- 2026-04-26 PARTIAL audit pattern: the `[chroma]` extra was effectively a PARTIAL declaration — the package was wired, but never proven to be the actual install path. PARTIAL declarations on a dependency manifest are the same risk class as PARTIAL modules in source code: they look correct until something forces them to be exercised.

### Commit

`a0052bd` — Fix red CI: declare chromadb as a runtime dep and drop the dead async job.

---

## 2026-06-04 — Streamlit "Start new session" cleared the token but the file_uploader's retained file re-uploaded on next rerun and silently re-minted the session

### Symptom

In the per-session-upload UI work, the "Start new session" button was meant to invalidate the current session: clear the `session_state` token, drop the deduplication set, restore the public-corpus mode pill. End-to-end test: upload a PDF, ask a question (answer is grounded in the upload), click "Start new session." The mode pill flipped back to public-corpus mode for exactly one frame, then a new session token appeared in `session_state`, the mode pill flipped back to session mode, and the uploaded document was queryable again — under a fresh server-minted `X-Session-Id` the user never asked for. The "new session" button was, in practice, a "re-upload the same file under a fresh session" button.

### Initial hypothesis (wrong)

The button handler missed a `session_state` key — probably the dedup set, or the mode pill's cached value. Audit: the handler cleared the token, cleared the dedup set, cleared the mode cache. Every state key that the upload path read at boot was reset to its empty value. The reset was correct; the resurrection happened anyway.

### Actual root cause

`st.file_uploader` retains its own state independent of `st.session_state`. The selected file persists across reruns keyed on the widget's `key=` identity, not on any application-level state. The "Start new session" button cleared the session token and the dedup set within the application state, then Streamlit re-ran the script — and the `file_uploader`, still keyed the same way, still held the file. The post-rerun upload-handler block saw a file present, checked the dedup set (now empty — the button cleared it), didn't recognize the file, posted it to `/ingest/file`, captured the new `X-Session-Id`, and persisted it to `session_state`.

The two state layers — application state in `session_state`, and widget-internal retained state inside `st.file_uploader` — were being managed independently and could not be reset together through any application-level operation. The button's handler was correct as written; the framework's widget state was ignoring it by design.

### Fix

Nonce-keyed widget key. An integer counter `uploader_nonce` lives in `session_state` (`ui_streamlit/app.py:315-323`), starts at 0, and is baked into the widget's `key=`:

```python
key=f"file_uploader_{st.session_state.uploader_nonce}"
```

Incrementing the nonce changes the widget's identity. The next rerun renders a NEW `file_uploader` from Streamlit's perspective, with no retained file. "Start new session" (`ui_streamlit/app.py:1344-1353`) and the 404-expired-session recovery path (`ui_streamlit/app.py:1626-1631`) both bump the nonce. The application can now clear widget state from code, which is otherwise not exposed.

Inline comments at the bump sites preserve the trap: "Without this, the file_uploader's retained value silently re-mints the session — the dedup guard is cleared by the same button, so the next rerun re-posts the file under a fresh session."

### Why this stayed hidden until end-to-end testing

The application-state side of the reset (token cleared, dedup set cleared, mode pill restored) looked correct in isolation and was correct in isolation. The widget-state side was invisible from any application-level inspection: `print(st.session_state)` showed all the cleared keys; the persisted file was inside the widget, not in `session_state`, and there was no programmatic surface to see it. Only running the full "upload → answer → reset → wait one frame" sequence surfaced the resurrection — the kind of sequence that ad-hoc manual testing might skip if the button looked plausible.

### Regression tests

Manual: upload a PDF, click "Start new session," confirm the mode pill stays on public-corpus for the subsequent rerun, the `file_uploader` renders empty, and a public-corpus query returns public answers. Automated: not added; Streamlit's widget-retained state is not straightforward to exercise from headless code paths. Tracked as a manual regression item on every UI release.

### Lesson — primary

Clearing application state is not enough when a widget holds its own retained state. Reactive frameworks frequently maintain a second state layer at the widget level that survives re-renders by design — the framework's reset operation is not the application's reset operation. When an "application reset" button needs to also reset framework-level widget state, the only reliable mechanism is to change the widget's identity (its `key=`), which forces the framework to discard its retained state and render a fresh widget. Same shape applies to React's `key` prop, Vue's `key`, and any other framework with identity-keyed component state.

### Lesson — secondary

When an application-level reset looks correct in isolation but the symptom recurs, the next question is "what state survives my reset that I'm not seeing?" — and the answer is often framework-internal. Application state is the state I can `print`; framework state is the state the framework decides to keep on my behalf. The two are not the same set, and an inspection of one says nothing about the other.

### Cross-reference

- 2026-04-29 Streamlit content-hash widget key collision: same area (Streamlit widget state), distinct mechanism. There, the hazard was content-derived `key=` values colliding across re-renders of multi-turn chat history; the fix was position-based keys for uniqueness. Here, the hazard is widget-retained state outliving application-level resets; the fix is nonce-based keys for forced widget-identity changes. Two confirmed Streamlit widget-state hazards now in the file — generalized in "Patterns I've noticed."

### Commit

`2ff6796` — Add Streamlit upload UI for per-session document query.

---

## 2026-06-08 — chroma_store numpy-2.x shim runs too late to defend; the comment asserted a guarantee the code structurally couldn't provide

### Symptom

During the M4 audit (evaluating a Dockerfile `pip install -e .` swap during HF Spaces migration prep), `pip install --dry-run -e .` against the local venv resolved numpy to 2.4.6. A follow-up bare `import chromadb` in that venv raised:

```
File "venv311/lib/python3.11/site-packages/chromadb/api/types.py", line 102
    ImageDType = Union[np.uint, np.int_, np.float_]
AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.
```

This directly contradicted the comment at `vectorstore/chroma_store.py:38-42`, which claimed "local environments running NumPy 2.x need it to import chromadb without AttributeError." The shim, in fact, does not enable a bare `import chromadb` under numpy 2.x.

### Root cause

The shim lives inside `ChromaVectorStore.__init__`, executing just before its local `import chromadb`. The patching logic (`setattr(_np, "float_", _np.float64)` etc.) only takes effect if this `__init__` runs BEFORE the process's first import of chromadb.

In any environment where chromadb is imported first — a REPL session, a bare script, a test file with a top-level `from chromadb.api.client import SharedSystemClient` (which both `tests/unit/test_session_store.py:89` and `tests/integration/test_session_reaper.py:257` have), or any module higher in the import graph that pulls chromadb transitively — the shim runs too late: chromadb's `chromadb.api.types` hits `np.float_` during its own load, before `ChromaVectorStore.__init__` is ever called, and the import fails.

The comment claimed a protection ("import chromadb without AttributeError") that the code's POSITION in the call graph structurally could not deliver. The shim is correct as patches; what's wrong is the runtime ordering it implicitly depends on, and the absence of that dependency from the documentation.

### Why this never broke production

The Docker image pins `numpy==1.26.4` ([Dockerfile:28](../Dockerfile#L28)). Under numpy 1.x the aliases the shim patches are still present; the `if not hasattr(_np, _attr)` guard makes every patch a no-op. Production has never run on numpy 2.x. The shim's "protection under numpy 2.x" branch has never actually protected anything in a deploy — it's load-bearing for an environment we don't have. The dead-code state was invisible until an audit forced an install path that would have created a numpy-2.x deploy.

### What the audit also revealed

The numpy 1.x ceiling lives ONLY in the Dockerfile, not in `pyproject.toml`. pyproject.toml line 54 is `numpy>=1.26` — open-ended. Any install via `pip install -e .` (the M4 work would have made this the production install path) pulls numpy 2.4.6 today, and chromadb 0.4.24 fails at boot import. The Dockerfile chain is the only thing holding numpy to 1.x; the shim cannot substitute for that pin.

### Fix

Rewrote the comment at `vectorstore/chroma_store.py:38-47` to state the actual protection model:
- The `numpy==1.26.4` Dockerfile pin is the real defense.
- Under numpy 1.x the shim is a no-op (hasattr guard skips every patch).
- Under numpy 2.x the shim only takes effect if this `__init__` runs before any other chromadb import in the process — verified failing for bare `import chromadb`.
- The Dockerfile numpy pin must never be dropped; pyproject's open `numpy>=1.26` does NOT enforce it.

The patch logic itself was left intact. It correctly patches the aliases when it runs in time; the bug was in the comment, not the patches. The pyproject pin-tightening is deferred to the M4 lockfile project (see [hf-migration-phase1-plan-2026-06-08.md](hf-migration-phase1-plan-2026-06-08.md) for the M4-pulled rationale).

### Why not "fix" the shim by hoisting it to module-level

Tempting: hoist the patches to top-of-file in `vectorstore/chroma_store.py` so they run on first import of THIS module. That would broaden the working window — but not close it. The tests and `scripts/rss_probe_linux.py` import chromadb without ever importing `vectorstore/chroma_store.py`, so hoisting wouldn't help them. It would also signal a guarantee that's still false in those cases. The honest fix is the comment: the numpy 1.x pin is the protection; the shim is a partial belt-and-braces inside its narrow runtime window.

### Lesson — primary

Defensive code that runs too late to defend is worse than no defense: a no-op is silent and gives no false confidence; a too-late patch with a confident comment asserts a guarantee that the code's position in the call graph cannot provide. The reader trusts the comment, doesn't audit the import ordering, and the next deploy off a `pip install -e .` path silently regresses.

When defensive code depends on RUNTIME ORDER — when it must execute before some other code path — the comment must name the order it depends on AND the limits of what it protects when the order isn't met. "This shim runs before this `import chromadb`" is precise; "this shim lets local NumPy 2.x environments import chromadb" is not, and that imprecision is what survived unchecked into the audit.

### Lesson — secondary

A dependency pin in only one of N install surfaces (here, Dockerfile but not pyproject) is a pin that holds by accident of which install path is exercised. The chromadb runtime-dep entry from 2026-06-04 documents the same shape in reverse (pyproject-declared, Dockerfile-installed). Both are PARTIAL declarations in the dependency-manifest sense. The matching discipline is: pins consumed by the runtime live in pyproject; the Dockerfile reads from pyproject. M4 is the project that achieves this; it is now scoped as a lockfile-based post-migration task, NOT as a hand-tightening of pyproject pins (the audit showed Shape A re-implements a lockfile by hand and carries the same drift risk).

### Cross-reference

- 2026-06-04 chromadb-declared-as-optional-extra: same dependency-surface drift family, opposite direction. Together they bound the problem — pins in only ONE surface (whichever one) leak silently into deploys that happen to exercise the other surface.
- M4 deferral and lockfile-shape decision: [hf-migration-phase1-plan-2026-06-08.md](hf-migration-phase1-plan-2026-06-08.md).

### Commit

Pending — will be filled when the shim-comment fix lands.

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

The 2026-04-30 FLARE `[UNCERTAIN]` investigation is the second instance of the RLHF-prior pattern. Two rounds, two different failure modes — instruction absent in round 0, instruction ignored in round 1 — same output: complete confident answers with no self-interruption marker. The pivot to numeric-novelty was faster than the attribution-spans pivot because the prior entry had already established the threshold. Each recurrence narrows the generalization: it is not only `[Source N]` trailing citations that are entrenched; Llama's RLHF priors resist structural self-interruption markers of any shape. The downstream adaptation strategy now has two confirming cases. The 2026-05-02 citation-rendering misdiagnosis adds a second dimension: even the trailing-marker format Llama accepts is emitted with per-call stochasticity — the prior produces unreliable compliance across identical calls, not only structural resistance to the target format.

The 2026-04-30 regex-asymmetry bug introduces a pattern distinct from the PARTIAL-module and RLHF-prior themes: asymmetric set-difference in heuristics. The `$18k` vs `$18,000` case is the archetype — two sets extracted from text that passed through different formatting conventions, compared as if normalized to the same space. The heuristic fired every round not because it was wrong about what it was measuring, but because the measure was applied inconsistently across the two sides of the comparison. Before treating a set-difference as meaningful, verify that both sets were produced by extraction paths that normalize to the same format. When a heuristic fires without converging across multiple rounds with no change in the underlying data, asymmetric normalization is the first thing to check. Generalizes beyond regex: any time two collections pass through separate extraction or formatting paths before being compared, the difference may be measuring format divergence rather than semantic novelty.

The 2026-05-29 metadata-hint trap is the third confirmed instance of "green deploy + healthy status signals + every query returns empty," after the 2026-05-12 missing FiQA corpus and the 2026-05-28 missing pymupdf4llm dependency. The mechanism differs each time — missing data, missing dependency, query-time filter — but the observability gap is identical: no standard signal (deploy status, `/health`, `/health/ready`, `/corpora`, `count()`) exercises the actual retrieve → generate path with a real query against the real corpus. Each layer reports healthy in isolation, and the integration silently returns empty. The 2026-05-12 entry's standing lesson is therefore now triply confirmed and worth promoting from suggestion to standing operating procedure: **after any change to a deployment — env var, disk path, Dockerfile, start command, routing code, or a forced redeploy from any cause — run one real `/query` against production with a known-good question for each corpus, and verify a grounded answer with citations.** Ten seconds per corpus. Catches the entire class.

The 2026-06-04 chromadb-misdeclaration entry is the fourth confirmed instance of the same family, with a new wrinkle: declared-vs-installed surface drift, not just code or data drift. Production was healthy for weeks because the Dockerfile hand-installed `chromadb` alongside the pyproject manifest that marked it `[optional]`; CI failed only when the per-session work moved Chroma into the startup-required path, and even then it failed at install (CI) rather than at deploy (Render). The mechanism generalizes the family from "the deployment shipped without a thing it needed" to "the deployment shipped because the deployment surface installs a thing the declared dependency manifest doesn't require." Both shapes are invisible to standard health signals. The standing-operating-procedure remedy still applies — run one real `/query` after any deployment change — and adds a sibling: **the declared dependency set is whatever a bare `pip install -e .` on a clean Python pulls. Anything beyond that, including everything the Dockerfile hand-installs, is environmental luck and must be re-declared in pyproject the moment any non-Docker environment (CI, fresh checkout, alternate platform) needs to boot the app.** Same observability gap, sister discipline.

The 2026-06-04 `RetrievalExecutor` isolation-audit entry is the second confirmed instance of "latent bug discovered by code-reading while implementing the feature that would have exposed it," after the 2026-05-13 `delete_document` FAISS wipe (AUDIT #22). Both bugs would have surfaced as silent wrong-shape behavior in production — `delete_document` would wipe the entire FAISS index on first call, the executor isolation gap would mix public-corpus chunks into session answers — and neither produced an exception, log line, or health-check anomaly. Both were found because the surrounding work brought the relevant code into reading distance: in May, an investigation into duplicate Roth IRA chunks pulled `delete_document` into view; in June, the per-session retrieval roll-out forced an audit of every `.execute(...)` caller. The pattern: **when implementing a feature that introduces a new contract (deletion semantics, isolation semantics), the audit is not "does the new code uphold the contract?" — it is "does every existing caller of the affected primitive uphold the contract?" `grep` the primitive, classify every caller, and write the not-applicable cases down at the call site.** Two instances of this pattern is enough to promote it from coincidence to discipline; the discipline is to schedule the grep before the merge, not after the symptom.

The 2026-06-04 Streamlit `file_uploader` resurrection entry is the second confirmed Streamlit widget-state hazard, after the 2026-04-29 content-hash key collision. The two mechanisms are different — the April bug was about widget keys colliding across re-renders, the June bug was about widget state outliving an application-level reset — but the common surface is `st.<widget>(key=...)`, and the common failure mode is "the `key=` parameter does more than identify the widget; it controls the framework's state-lifecycle hooks for that widget, and the application's understanding of the widget's lifecycle is not the framework's understanding by default." Generalizes: **every Streamlit widget that retains state across reruns needs a `key=` strategy that maps the application's intended state lifecycle, not just the widget's visual identity. Position-based keys (April fix) and nonce-based keys (June fix) are both expressions of "we are now using `key=` as a state-control lever, not a label."** The trap is treating `key=` as a cosmetic; the discipline is treating it as the application's only programmatic handle on framework-internal widget state. Worth assuming for every widget that holds state (`file_uploader`, `text_input`, `selectbox`, `data_editor` — anything whose value persists across reruns) until proven otherwise.
