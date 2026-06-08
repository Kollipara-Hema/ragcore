# HF Spaces migration — Phase 0 decision memo (2026-06-06)

Before any code moves. Five load-bearing decisions, each with the option set
considered, the recommendation, and the tradeoff that recommendation accepts.
The persistence decision (#1) drives the rest — read it first.

Companion to [migration-baseline-2026-06-05.md](migration-baseline-2026-06-05.md),
which enumerates the regression surface and the exact env/path/boot/health
contract this memo must preserve.

---

## 1. Persistence — the central decision

**What must survive a restart** ([baseline §2](migration-baseline-2026-06-05.md)):
- 6 Apple Chroma corpora — 53 MB total, already baked into the image at
  [data/chroma_collections/](../data/chroma_collections/) and copytreed to disk
  by `_seed_apple_collections` ([api/main.py:152-208](../api/main.py#L152-L208)).
- FiQA FAISS — index 1.5 MB + metadata 3.6 MB = **5.1 MB total**, currently
  at [faiss/](../faiss/) in the dev tree, **not bundled** into the image, and
  not seeded by the lifespan. This is the artifact that caused the 9-day
  production outage on 2026-05-12 when the Render disk needed re-ingest.

### Options

| Option | Cost | Cold start | Heals FiQA outage? | Update flow |
|---|---|---|---|---|
| **A. HF Persistent Storage** (mounts at `/data`, 20 GB) | $5/mo | Fast (disk warm) | **No** — same bug as Render: fresh disk = FiQA gone, HTTP re-ingest needed | Live, no rebuild |
| **B. HF Dataset + `snapshot_download` on boot** | Free | +5–15 s per cold start (58 MB pull from HF Hub) | Yes — if FiQA is in the dataset | Push to Dataset; re-pull on next boot |
| **C. Bake Apple + FiQA into the Docker image** | Free | Fast (copytree from `/app` → working path) | **Yes** — both corpora ship in image | Image rebuild required |
| D. Re-ingest on boot | Free | ~25 min for FiQA per [baseline §2](migration-baseline-2026-06-05.md) | N/A — boot doesn't complete in time | — |

D stays rejected on the same grounds as before — embedding 245 FiQA docs
through BGE on every cold start is multiple minutes of wall time even with the
embedder already loaded.

### Recommendation: **C — bake both corpora into the image**

Why this and not A:
- **A preserves the bug we're migrating to escape.** The 9-day outage cause
  was FiQA not self-healing onto a fresh disk. Buying persistent storage on
  HF rents around the bug; rebuilding the image with FiQA bundled fixes it.
  The next time HF wipes the disk (rebuild, region move, paid-tier downgrade),
  A re-creates the exact failure mode. C does not.
- **Size is not a tradeoff worth weighing.** Image is already gigabytes —
  torch 2.2.1 + sentence-transformers + BGE-large (pre-baked at
  [Dockerfile:63-68](../Dockerfile#L63-L68)) + cross-encoder. Adding 5.1 MB of
  FiQA to the 53 MB of Apple already in the image is rounding error.
- **Update friction is hypothetical.** FiQA is a static benchmark and the
  Apple 10-K data is from a published filing. Neither has changed in the
  lifetime of this project. The "rebuild to update" cost is paid never.
- **Free.** No $5/mo storage line item.

Why not B:
- Adds a new runtime failure mode (HF Hub auth/network on boot), and the same
  health-check workflow that flagged 502s on Render would now also flag
  Hub-timeout boots. Trading a solved problem for an unsolved one.
- Cold-start network pull happens every restart, not just first boot — the
  ephemeral-disk model means the local cache is also wiped.

### Tradeoff accepted

Updating a corpus requires a Space rebuild instead of a live re-ingest. For
the Apple/FiQA workload this is the right tradeoff — but the moment a corpus
becomes dynamic (e.g. user-curated, periodically refreshed), revisit. The
per-session uploads at `data/sessions/` are unaffected — those are ephemeral
by design and already work on ephemeral disk.

### What this changes in code (Phase 1, not now)

- Move [faiss/faiss_index.idx](../faiss/faiss_index.idx) and
  [faiss/faiss_metadata.pkl](../faiss/faiss_metadata.pkl) into a bundled path
  (e.g. `data/faiss_seed/`) inside the image.
- Add a FAISS-seed step to the lifespan analogous to `_seed_apple_collections`:
  on boot, if `FAISS_DATA_DIR` is empty, copy the bundled seed into it.
- This is the only structural code change Phase 0 implies. It does not change
  the FAISS contract — it makes FiQA self-heal the same way the Apple corpora
  already do.

---

## 2. Space type & topology

### Current production topology

Two processes, two platforms:
- **FastAPI backend** on Render, Docker runtime.
- **Streamlit UI** on Streamlit Cloud at `ragcore.streamlit.app`, talking to
  the backend over HTTPS via `RAGCORE_BACKEND_URL`
  ([ui_streamlit/app.py:1](../ui_streamlit/app.py)). Streamlit Cloud is free.

Streamlit is **not** on Render. That matters because the migration is
backend-only.

### HF Spaces options

| Topology | Mechanism | Fit |
|---|---|---|
| **A. Docker Space for backend; Streamlit stays on Streamlit Cloud** | Existing Dockerfile, mostly unchanged | Best — one thing changes |
| B. Docker Space for backend + second HF Space (SDK Streamlit) for UI | Two HF Spaces, cross-Space HTTPS | Migrates a working thing for no reason |
| C. Single Docker Space running both uvicorn + streamlit under supervisord | One process tree, two ports — HF Spaces only exposes one port publicly | Loses the two-platform deployability; HF exposes a single port per Space |

HF SDK Spaces (Gradio/Streamlit) only support their own entry point and port —
they cannot host a FastAPI backend. So the backend **must** be a Docker Space
regardless. The only real question is whether the UI moves too.

### Recommendation: **A — backend → HF Docker Space; Streamlit stays put**

- Streamlit Cloud is free, already deployed, and not part of the outage
  history. Migrating it adds risk for zero benefit.
- The only Streamlit-side change is updating its `RAGCORE_BACKEND_URL` env
  var from the Render URL to the new HF URL.

### Tradeoff accepted

Two-platform deployment continues — slightly more operational surface than
a single-platform deploy. But this is the surface today, and it works.

---

## 3. Secrets & env

### How HF differs from Render

Render: `render.yaml` declares non-secret vars in-repo; `sync: false` vars
(API keys) are set in the dashboard. The Blueprint is the source of truth and
re-applies on every push.

HF Spaces: no in-repo declaration. Every var is set in **Settings → Variables
and Secrets** in the Space UI. "Variables" = plain text, visible in Space
metadata. "Secrets" = encrypted, masked in logs. Both are injected as
process env vars on container start. There is no Blueprint equivalent — once
set, drift between the repo's expected vars and the live vars is silent.

### What MUST be set on HF (not optional)

Mirror the 11 vars from [render.yaml:14-36](../render.yaml#L14-L36):
- **Variables**: `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_PROVIDER`,
  `VECTOR_STORE_PROVIDER`, `ENVIRONMENT`, `ENABLE_RERANKING`,
  `CHROMA_PERSIST_DIR`, `FAISS_DATA_DIR`.
- **Secrets**: `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
  (declared on Render but unused on the live serving path; mirror for parity).

### What the baseline flagged as "silently takes a default" — HF-critical

These are NOT in `render.yaml` today but matter on HF:

| Var | Code default | Why it matters on HF |
|---|---|---|
| `CORS_ORIGINS` | `["http://localhost:8501"]` | Streamlit Cloud UI will hit 403 preflight on every call. **Must explicitly set** to include `https://ragcore.streamlit.app`. |
| `RAGCORE_TRUST_PROXY_HEADERS` | `false` | HF puts a reverse proxy in front of every Space. With this false, the rate limiter sees every request as coming from the proxy's IP — one shared bucket for all clients. **Set to `true`** on HF. |
| `RAGCORE_SESSION_ROOT` | `./data/sessions` | Relative path; resolves against WORKDIR. See #4 — set to an absolute path. |
| `CHROMA_PERSIST_DIR` / `FAISS_DATA_DIR` | — | Render values `/var/data/...` don't exist on HF. Must repoint to the in-image bundled paths (decision #1, option C). |

### Recommendation

Mirror all 11 render.yaml vars on HF as Variables + Secrets, plus the four
HF-critical defaults above. Keep a `docs/hf-env-vars.md` checklist as the
out-of-repo source of truth, since HF gives us no Blueprint analog. Verify
after first boot with the `env | grep` snippet from
[baseline §1.3](migration-baseline-2026-06-05.md).

### Tradeoff accepted

Drift detection is now manual — there is no `render.yaml`-style file the next
person can grep to know what should be set. Mitigation: the checklist file.

---

## 4. Path & boot risk

### Where the boot can fail

`_assert_session_root_isolated` at
[api/main.py:215-241](../api/main.py#L215-L241) refuses to boot when
`RAGCORE_SESSION_ROOT` overlaps `FAISS_DATA_DIR` or `CHROMA_PERSIST_DIR`. The
check uses `Path.resolve()` + `Path.is_relative_to` (both directions, plus
equality), so a relative `./data/sessions` resolves against current working
directory before comparison.

On HF Docker Spaces:
- WORKDIR is whatever the Dockerfile sets — currently `/app`
  ([Dockerfile:22](../Dockerfile#L22)). HF doesn't override this.
- The container **must run as non-root** on HF (UID 1000 convention).
  Current Dockerfile has no `USER` directive — running as root will fail HF's
  Space launch.
- HF Persistent Storage (if used) mounts at `/data` and is writable by UID
  1000. Not relevant under decision #1's option C (no persistent storage).

### Does the assertion fire?

Walk the four-corner cases with decision #1's recommended paths:
`FAISS_DATA_DIR=/app/data/faiss`, `CHROMA_PERSIST_DIR=/app/data/chroma_db`,
`RAGCORE_SESSION_ROOT=/app/data/sessions`:

- `/app/data/sessions` vs `/app/data/faiss` — siblings under `/app/data`,
  neither is relative to the other. **Safe.**
- `/app/data/sessions` vs `/app/data/chroma_db` — same, siblings. **Safe.**
- All three resolve absolutely; no cwd dependency. **Safe.**

The assertion fires only if someone sets `FAISS_DATA_DIR=/app/data` (the
parent of `sessions`) or vice versa. Easy to avoid with explicit absolute
values in the HF env config.

The riskier configuration is the **current default** — `./data/sessions`
relative, with no explicit absolute. On HF if WORKDIR ever differs from the
Dockerfile's `/app` (e.g. someone sets a HF-recommended `/home/user/app`
later), the implicit-cwd resolution silently shifts. Removing the
implicit-cwd dependency is one env-var setting away.

### Recommendation

Set `RAGCORE_SESSION_ROOT`, `FAISS_DATA_DIR`, and `CHROMA_PERSIST_DIR` to
**explicit absolute paths under `/app/data/`** in the HF env config. All three
become siblings; the assertion passes. Add a `USER 1000` directive to the
Dockerfile during Phase 1 to satisfy HF's non-root requirement, plus the
required `chown` on `/app/data` so the runtime user can write.

### Tradeoff accepted

Two more env vars to manage explicitly (well, one — `RAGCORE_SESSION_ROOT`
isn't currently in render.yaml). Worth it to remove the implicit-cwd
dependency that's the most plausible HF-specific boot failure.

---

## 5. Cost

### Bottom line

| Compute | Storage | Total |
|---|---|---|
| HF Spaces CPU Basic (free tier) | None — corpora in image, sessions ephemeral | **$0/mo** |

### Compute justification

The prior RAM work proved the feature runs bounded on the current Render
Standard footprint. HF CPU Basic gives 2 vCPU + 16 GB RAM — strictly more
RAM headroom than Render Standard's 2 GB. No paid compute tier needed.

### Storage justification

Decision #1 option C means **no persistent storage line item** ($0 vs $5/mo
for HF Persistent Storage Small). The $5/mo tier only earns its keep if the
persistence choice were option A — which we rejected on bug-grounds, not
cost-grounds.

### Tradeoffs accepted

- **Cold start latency.** HF free Spaces sleep after inactivity and pay a
  cold-start cost on first request after wake. For demo/portfolio traffic
  this is acceptable. The existing health-check workflow's 502→retry-after-30s
  logic ([baseline §4](migration-baseline-2026-06-05.md)) already tolerates
  Render's ~21 s cold start; HF's is similar order.
- **No SLA.** Free tier has no uptime guarantee. Same as Render free tier;
  Render Standard had no real SLA for this workload either.

### Net vs current

Current Render Standard cost goes to $0. Migration is net cost-down even
before the engineering value of fixing the FiQA self-heal bug.

---

## Summary — five decisions, one sentence each

1. **Persistence**: bake FiQA into the image alongside the existing Apple
   bundle. Free, eliminates the 9-day-outage cause, no persistent storage
   needed.
2. **Topology**: backend → HF Docker Space, Streamlit stays on Streamlit
   Cloud. Only one thing changes.
3. **Env**: mirror 11 render.yaml vars as HF Variables/Secrets + explicitly
   set `CORS_ORIGINS`, `RAGCORE_TRUST_PROXY_HEADERS`, `RAGCORE_SESSION_ROOT`,
   and the two corpus dirs. Maintain an out-of-repo checklist since HF has no
   Blueprint analog.
4. **Path/boot**: absolute paths for all three corpus/session dirs as
   siblings under `/app/data/`; add `USER 1000` + chown to the Dockerfile in
   Phase 1.
5. **Cost**: $0/mo. Free CPU Basic + image-baked corpora; no $5 storage tier.

Nothing has been ported yet. Phase 1 (the actual port) is gated on
acceptance of these five decisions.
