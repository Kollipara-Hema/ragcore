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

