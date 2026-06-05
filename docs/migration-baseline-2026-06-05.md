# Pre-migration baseline — 2026-06-05

A focused snapshot of the deployment surface as it exists on Render today, to be
re-run item-by-item after the HF migration to catch regressions.

**Scope is migration-regression only** — env injection, persistence, boot
sequence, health checks, runtime dependencies, end-to-end probe. NOT code
quality, architecture, or test coverage (those don't change in a platform
port).

Current deployment: Render Standard plan, single uvicorn worker, Docker
runtime, 5 GB persistent disk mounted at `/var/data`, autoDeploy from `main`.
`render.yaml` is the Blueprint source of truth.

---

## 1. Environment variables & configuration

Every env var the app actually reads from `config/settings.py:61-229`, with its
production value (from `render.yaml`) or its in-process default.

### Set on Render via `render.yaml`

| Env var | Render value | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Drives `_build_llm()` branch. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | The Groq live model identifier. |
| `EMBEDDING_PROVIDER` | `bge` | Resolves to `BAAI/bge-large-en-v1.5`. |
| `VECTOR_STORE_PROVIDER` | `faiss` | FAISS for the FiQA `default` corpus; Chroma is used for Apple corpora regardless of this setting. |
| `ENVIRONMENT` | `production` | Surfaced by `/health`. |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranker is on. |
| `CHROMA_PERSIST_DIR` | `/var/data/chroma_db` | Apple corpora live here. |
| `FAISS_DATA_DIR` | `/var/data/faiss` | FiQA index + metadata pickle live here. |
| `GROQ_API_KEY` | `sync: false` (set in dashboard) | Required for the active LLM path. |
| `OPENAI_API_KEY` | `sync: false` (set in dashboard) | Used by `LLMQueryClassifier` (gpt-4o-mini), Self-RAG verifier (when wired to OpenAI), RAGAS judge. |
| `ANTHROPIC_API_KEY` | `sync: false` (set in dashboard) | Currently unused on the live serving path; declared for completeness. |

### Read by code but **NOT** set in `render.yaml` (uses defaults)

These take the `config/settings.py` defaults in production. Any HF re-deploy
that fails to inject the same defaults — or worse, injects different ones —
will silently change behavior.

| Setting | Default | Risk if changed |
|---|---|---|
| `CORS_ORIGINS` | `["http://localhost:8501"]` | Streamlit demo at `ragcore.streamlit.app` would 403 preflight. The live deploy must override this with the Streamlit URL — verify it's set on Render's dashboard env vars (not in `render.yaml`). |
| `RAGCORE_AUTH_ENABLED` | `false` | Demo accessibility. Flip to `true` requires `RAGCORE_API_KEY` set or boot fails (`api/main.py:425-429`). |
| `RAGCORE_API_KEY` | `None` | Only required when auth is enabled. |
| `RAGCORE_RATE_LIMIT_MAX_REQUESTS` | `60` | Per-IP, per `RAGCORE_RATE_LIMIT_WINDOW_SECONDS`. |
| `RAGCORE_RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window. |
| `RAGCORE_TRUST_PROXY_HEADERS` | `false` | If HF puts a proxy in front, may need `true` for accurate client IP. |
| `RAGCORE_INGEST_MAX_BODY_BYTES` | `10485760` (10 MB) | ASGI body cap on `/ingest/*`. |
| `RAGCORE_SESSION_ROOT` | `./data/sessions` | Per-session corpus root. **Must NOT overlap `FAISS_DATA_DIR` or `CHROMA_PERSIST_DIR`** — boot asserts and refuses to start (`api/main.py:215-241`). |
| `RAGCORE_SESSION_MAX_FILE_BYTES` | `1048576` (1 MB) | Per-file cap inside a session. |
| `RAGCORE_SESSION_MAX_FILES` | `3` | Files per session. |
| `RAGCORE_SESSION_MAX_CONCURRENT` | `3` | Active sessions per process. |
| `RAGCORE_PDF_MAX_PAGES` | `100` | Pages per uploaded PDF. |
| `RAGCORE_SESSION_IDLE_TTL_SECONDS` | `1800` (30 min) | Idle-keyed; bumps on every ingest/query. |
| `RAGCORE_SESSION_SWEEP_INTERVAL_SECONDS` | `300` (5 min) | Reaper cadence. |
| `EMBEDDING_MODEL_BGE` | `BAAI/bge-large-en-v1.5` | Embedder downloaded from HF Hub on boot. |
| `EMBEDDING_DIMENSION` | `1024` | Must match BGE-large; mismatch crashes FAISS. |
| `EMBEDDING_BATCH_SIZE` | `64` | |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking. |
| `RERANK_TOP_K` | `5` | Final chunks sent to LLM. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker downloaded from HF Hub on boot. |
| `HYBRID_ALPHA` | `0.7` | |
| `CHUNKING_STRATEGY` | `semantic` | |
| `CHUNK_SIZE` | `512` | |
| `CHUNK_OVERLAP` | `64` | |
| `GENERATION_STRATEGY` | `basic` | |
| `LLM_TEMPERATURE` | `0.1` | |
| `LLM_MAX_TOKENS` | `2048` | |
| `ENABLE_TRACING` | `false` | |

### Verify post-migration

```bash
# On the migrated host (or via a debug endpoint if available):
env | grep -E "^(LLM_|EMBEDDING_|VECTOR_|CHROMA_|FAISS_|RAGCORE_|CORS_|GROQ_|OPENAI_|ANTHROPIC_|ENVIRONMENT|ENABLE_)" | sort
```

Compare against the tables above. Any drift — value or absence — is a
regression candidate.

---

## 2. Persistence & paths

This is the #1 migration risk: HF Spaces have **ephemeral** disk by default.
What's currently on the Render persistent disk is what makes the live deploy
work; lose any of it and `/query` either 404s the corpus or returns the
empty-retrieval fallback.

### What MUST persist across restart

| Path | What's there | Size | Source if lost |
|---|---|---|---|
| `/var/data/chroma_db/apple_10k_fixed/` | Chroma sqlite + HNSW segment + `bm25_state.pkl` | ~9 MB | Re-seeded from `data/chroma_collections/apple_10k_fixed/` baked into the Docker image (lifespan `_seed_apple_collections`). |
| `/var/data/chroma_db/apple_10k_hierarchical/` | Same shape | ~13 MB | Same — bundled in image, seeded on first boot. |
| `/var/data/chroma_db/apple_10k_document_structure/` | Same shape | ~8 MB | Same. |
| `/var/data/chroma_db/apple_environmental/` | Same shape | ~13 MB | Same. |
| `/var/data/chroma_db/apple_earnings_html/` | Same shape | ~4.5 MB | Same. |
| `/var/data/chroma_db/apple_financial_csvs/` | Same shape | ~4.5 MB | Same. |
| `/var/data/faiss/faiss_index.idx` | FiQA FAISS index | small | **Not bundled.** Must be re-ingested via the HTTP ingest script (`evaluation/scripts/ingest_fiqa_corpus_http.py`) — see the 2026-05-12 debugging-notes entry. |
| `/var/data/faiss/faiss_metadata.pkl` | FiQA chunk metadata + embeddings | small | Same — paired with the index. |

**Critical asymmetry:** the Apple corpora self-heal on first boot because they
ship in the image (`data/chroma_collections/` — 53 MB total, 6 subdirs). The
FiQA `default` corpus does **not** ship in the image and must be re-ingested
via HTTP after deploy to a fresh disk. The 2026-05-12 entry documents the
9-day production outage this caused last time.

### What's ephemeral (correctly)

| Path | What | Why it's safe to lose |
|---|---|---|
| `./data/sessions/<token>/` | Per-session corpora | Boot purges orphaned dirs by design (`_purge_orphaned_session_dirs`). Tokens are in-memory only. |
| `/tmp/...` | Upload staging (`tempfile.NamedTemporaryFile` in `/ingest/file`) | Deleted in the request's `finally`. |
| Embedder/reranker model cache | HF Hub downloads | Re-downloaded on cold start. ~80 MB BGE-large + ~80 MB cross-encoder = ~160 MB; ~1–2 min over the network. |

### Verify post-migration

```bash
# After first boot on the new platform, before sending traffic:
curl https://<new-host>/corpora
# Expected: JSON listing 7 corpora — "default" (FiQA) + 6 Apple corpora —
# each with doc_count > 0.

# If FiQA is missing, run:
python evaluation/scripts/ingest_fiqa_corpus_http.py --target https://<new-host>
# Expected: 245 docs ingested over ~25 min. Retries 502s once after 5s.
```

If the new platform has ephemeral disk, the seeding still works on cold start
because the Apple corpora are baked into the image — but every cold start will
re-seed (~53 MB of file I/O). For HF specifically: verify whether the Space
has a persistent volume option, and if not, accept the cold-start re-seed cost
or block on a persistence story before migrating production traffic.

---

## 3. Boot sequence

Lifespan logic at `api/main.py:279-385`. Steps run in this exact order; each
depends on prior steps having succeeded.

| # | Step | What it does | Failure mode |
|---|---|---|---|
| 1 | `configure_logging()` | Initializes structlog JSON renderer. | Logs go to stdout broken. |
| 2 | `_assert_session_root_isolated()` | Refuses to boot if `RAGCORE_SESSION_ROOT` overlaps `FAISS_DATA_DIR` or `CHROMA_PERSIST_DIR` (`api/main.py:215-241`). | `RuntimeError`; FastAPI never starts. **Migration risk:** if HF's working directory differs and the defaults resolve to overlapping paths, boot fails. |
| 3 | `register_corpus("default", FAISSVectorStore())` | Registers the FiQA-FAISS corpus unconditionally. Construct reads the index from `FAISS_DATA_DIR`. | Empty FAISS → empty `default` corpus; `/query` returns the empty-retrieval fallback. |
| 4 | `_seed_apple_collections()` | Atomic copytree from image (`data/chroma_collections/<name>/`) to disk (`/var/data/chroma_db/<name>/`). Idempotent — skips per-corpus if `chroma.sqlite3` already at dest. | Logs `Seed failed for <name>` and continues; corpus skipped at step 5. |
| 5 | Loop `CORPORA_CONFIG` and register each Chroma corpus | Constructs `ChromaVectorStore(persist_dir, collection_name)`; skips if missing or `count()==0`. | Missing corpora silently absent from `/corpora`; check Render logs for `Corpus <name> not found at <path>, skipping`. |
| 6 | `orchestrator = RAGOrchestrator()` | Loads cross-encoder reranker (~80 MB) and BGE embedder (~80 MB) from HF Hub. **~10–30 s on cold start.** | Hangs/timeouts if HF Hub unreachable from the new platform. |
| 7 | `ingestion_pipeline = IngestionPipeline()` | Lightweight — chunker factory + embedder singleton. | Trivial. |
| 8 | `session_store = SessionStore(root=RAGCORE_SESSION_ROOT)` | Creates the session root dir if missing. | Permission error if path is read-only. |
| 9 | `_purge_orphaned_session_dirs(...)` | rm -rf every direct child of `session_root`. Safe because step 2 asserted no overlap. | None expected. |
| 10 | `asyncio.create_task(reaper_loop(...))` | Background TTL eviction task. Cancelled on shutdown. | None expected. |
| 11 | `logger.info("RAG system ready...")` | Marker line for log parsing. | — |

Total boot time on Render cold start: **~21 s** (measured via the production
health check workflow's 502→retry-after-30s logic).

### Verify post-migration

Watch the boot log for:
1. No `RuntimeError: ragcore_session_root (...) overlaps with ...` line — step 2 passed.
2. Six `Seeding apple_<name>: ... -> /var/data/chroma_db/...` lines on the **first** boot, then `Seed skipped: ... already seeded` on **subsequent** boots.
3. Six `Registered corpus apple_<name> from /var/data/chroma_db/<name>` lines.
4. `RAG system ready` marker.

Any missing step ⇒ regression. If step 2 fires the assertion, the path
resolution differs from Render — fix the env vars before retrying.

---

## 4. Health endpoints

Three probes with distinct guarantees and timing characteristics.

### `/health` — fast, dependency-free

```bash
curl https://ragcore-api.onrender.com/health
# {"status":"ok","version":"1.0.0","environment":"production"}
```

Returns 200 unconditionally as long as the FastAPI process is up. Used by
Render's healthCheckPath (`render.yaml:7`).

### `/health/live` — liveness probe

```bash
curl https://ragcore-api.onrender.com/health/live
# {"status":"alive"}
```

Returns 200 immediately. No dependency calls. Use as the platform liveness
probe on the new host.

### `/health/ready` — readiness probe (slow on cold start)

```bash
curl -i https://ragcore-api.onrender.com/health/ready
# 200 {"status":"ready","checks":{"vector_store":{"ok":true},"embedder":{"ok":true},"llm_config":{"ok":true}}}
# or
# 503 {"status":"not_ready","checks":{"<name>":{"ok":false,"reason":"<msg>"}, ...}}
```

Three checks (`api/main.py:520-557`):
1. `get_vector_store().ping()` — FAISS store is constructed and readable.
2. `get_embedder().embed_query("a")` — runs the embedder; embedder must be loaded.
3. `_check_llm_config()` — confirms the active LLM provider's API key field is non-empty (no live call; placeholder strings like `"your-key-here"` pass — see AUDIT #13).

On cold start, this can take **15–25 s** because the embedder check waits for
the model download in step 6 of boot to complete. After warm-up, it's
sub-100 ms.

### `production_health_check.py` — the real end-to-end probe

Runs on a 4-hour cron (`.github/workflows/production_health_check.yml`), or
manually via `workflow_dispatch`.

```bash
TARGET_URL=https://ragcore-api.onrender.com \
  python evaluation/scripts/production_health_check.py
```

POSTs `{"query": "What is a Roth IRA?", "top_k": 5}` to `/query` and verifies
**five** signals:

| Check | Pass condition |
|---|---|
| HTTP 200 | Status code matches. |
| Answer grounded | `answer` does NOT start with `"I could not find this in the provided documents"`. |
| Citations present | `len(citations) > 0`. |
| Tokens non-zero | `total_tokens > 0` (proves LLM was called, not the empty-index fast path). |
| Latency realistic | `latency_ms >= 100` (sub-100 ms signals empty-index fast path). |

On a 502 it sleeps 30 s and retries once (Render cold-start handler).

### Verify post-migration

```bash
# Liveness — should be instant
curl -w "%{time_total}\n" -o /dev/null -s https://<new-host>/health/live

# Readiness after warm-up — should be sub-100ms
curl -w "%{time_total}\n" -o /dev/null -s https://<new-host>/health/ready

# Full end-to-end probe
TARGET_URL=https://<new-host> python evaluation/scripts/production_health_check.py
# Must print "RESULT: PASS" with all five checks PASS.
```

---

## 5. External dependencies at runtime

| Dependency | Source | When fetched | Cost |
|---|---|---|---|
| `groq` API | `https://api.groq.com` | Every `/query` (generation). | Network roundtrip; rate-limited by Groq's free tier. |
| `openai` API | `https://api.openai.com` | Every `/query` (router's LLM classifier — `gpt-4o-mini` at temp=0). Plus Self-RAG verifier if explicitly using OpenAI. | Two short LLM calls per query when both router classifier and verifier run. |
| `BAAI/bge-large-en-v1.5` | HuggingFace Hub | **First boot only**, cached to `~/.cache/huggingface/`. | ~80 MB download, ~1 min on first cold start. Re-downloaded if cache wiped. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace Hub | **First boot only**, cached. | ~80 MB download, ~1 min on first cold start. |
| `chromadb==0.4.24` | PyPI / pip | Image build time. | Pinned exactly — see the 2026-06-04 debugging-notes entry. A higher 0.4.x silently renames `SharedSystemClient._identifer_to_system` and leaks RAM. |
| `pymupdf==1.27.2.3` + `pymupdf4llm==1.27.2.3` | PyPI | Image build. | AGPL-3.0; declared in [LICENSE-pymupdf4llm.md](../LICENSE-pymupdf4llm.md). |

### Verify post-migration

```bash
# 1. From the new host, confirm Groq is reachable:
curl -s -o /dev/null -w "%{http_code}\n" https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
# Expected: 200.

# 2. Confirm HF Hub reachable (model downloads):
curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/config.json
# Expected: 200 or 302.

# 3. Confirm the chromadb pin is exact (==0.4.24, not a range):
python -c "import chromadb; print(chromadb.__version__)"
# Expected: 0.4.24 exactly.

# 4. Confirm the SharedSystemClient internal attribute is still where the
#    session-store reaches for it (the typo'd name):
python -c "from chromadb.api.client import SharedSystemClient; print(hasattr(SharedSystemClient, '_identifer_to_system'))"
# Expected: True. If False, session eviction will silently leak RAM
# (logged warning + cache layout unexpected fallback).
```

---

## 6. Known-good end-to-end baseline

The exact production_health_check probe output that constitutes the live
known-good state to reproduce post-migration. Capture **before** migrating,
then re-run **after** and diff.

### Probe input

```
POST https://ragcore-api.onrender.com/query
Content-Type: application/json

{"query": "What is a Roth IRA?", "top_k": 5}
```

### Expected response shape

- HTTP 200
- `answer`: prose grounded in FiQA chunks. Must NOT start with `"I could not find this in the provided documents"`.
- `citations`: non-empty list of `{doc_id, chunk_id, score, ...}` entries.
- `total_tokens`: integer > 0.
- `latency_ms`: float >= 100 (typical 4000–9000 ms on Render free tier).
- `model_used`: `llama-3.3-70b-versatile`.
- `strategy_used`: one of the routed strategies (typically `hybrid` for this query).

### Run before migration to capture exact baseline

```bash
curl -s -X POST https://ragcore-api.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Roth IRA?", "top_k": 5}' \
  | tee /tmp/baseline-2026-06-05.json \
  | python -c 'import sys, json; r=json.load(sys.stdin); print("answer[0:160]:", r["answer"][:160]); print("citations:", len(r["citations"])); print("total_tokens:", r["total_tokens"]); print("latency_ms:", r["latency_ms"]); print("model:", r.get("model_used")); print("strategy:", r.get("strategy_used"))'
```

Commit the file at `/tmp/baseline-2026-06-05.json` to a private gist or local
note — do NOT commit to the repo (the answer text is non-deterministic but
recognizable; useful as a reference).

### Re-run post-migration

```bash
curl -s -X POST https://<new-host>/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Roth IRA?", "top_k": 5}' \
  | tee /tmp/postmigration-<date>.json \
  | python -c 'import sys, json; r=json.load(sys.stdin); print("answer[0:160]:", r["answer"][:160]); print("citations:", len(r["citations"])); print("total_tokens:", r["total_tokens"]); print("latency_ms:", r["latency_ms"]); print("model:", r.get("model_used")); print("strategy:", r.get("strategy_used"))'
```

Compare. The answer prose will vary (Groq is non-deterministic), but:
- Both must have `citations: 5` (or close to it).
- Both must have `total_tokens` in a similar range.
- `model` and `strategy` must match.
- The answer must reference Roth IRAs specifically and cite at least one of the same FiQA `doc_id`s as the baseline (FiQA corpus IDs are stable; `bisect /tmp/baseline-*.json /tmp/postmigration-*.json` for overlap).

If any of those fail: the migration changed retrieval or generation in a way
that needs investigation before cutting over traffic.

---

## Migration checklist — re-run order

When the new host is up but before cutting over DNS:

1. **Env vars** — `env | grep` per section 1. No drift.
2. **Persistence** — `curl /corpora` returns 7 entries with non-zero `doc_count`. If FiQA missing, run the HTTP ingest script.
3. **Boot logs** — All 11 lifespan steps logged in order; no assertion fires.
4. **Health endpoints** — `/health/live` instant, `/health/ready` 200 with all three checks `ok: true`.
5. **External deps** — Groq + HF Hub reachable; chromadb pin exact; `_identifer_to_system` attribute present.
6. **End-to-end** — `production_health_check.py` PASSes all 5 checks; the "Roth IRA" query returns grounded, cited, non-trivial.

Any FAIL stops the cutover. Each item maps to a specific code path or
artifact — investigate at the named file/line.
