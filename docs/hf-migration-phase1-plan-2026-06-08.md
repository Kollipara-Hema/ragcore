# HF Spaces migration — Phase 1 sequencing plan (2026-06-08)

Implementation order + gating for porting the FastAPI backend off Render onto a
HuggingFace Docker Space. Phase 0 decisions (persistence, topology, env, paths,
cost) are settled in [hf-migration-phase0-decisions-2026-06-06.md](hf-migration-phase0-decisions-2026-06-06.md);
this doc sequences the work that implements them.

The regression contract is [migration-baseline-2026-06-05.md](migration-baseline-2026-06-05.md);
Phase 3 of this plan is when that contract gets verified against HF.

## Operating principle

Every step below is its own reviewable unit. Steps in the same phase can land in
any order. Phases gate strictly: don't cross a phase boundary until every step
in the prior phase is green. The whole sequence is engineered around parallel
cutover: **Render serves the live demo until Phase 3 passes against HF, and
nothing in Phases 1–3 affects live traffic.**

---

## Phase 1A — Repo-only changes, mergeable to main, zero Render behavior change

These all land on `main` and ship to Render without changing anything live.
They're prerequisites for HF and several are harmless improvements regardless.

> **M4 (Dockerfile `pip install -e .` swap, originally scoped here as 1A.1)
> was pulled from this sequence after the 2026-06-08 audit.** The audit
> revealed M4 as an independent ~20-package dependency-reconciliation
> project, not a one-line refactor: `pip install -e .` against the current
> pyproject resolves numpy to 2.4.6 (chromadb 0.4.24 fails to import) and
> brings major-version drift in torch, sentence-transformers, openai,
> fastapi, pandas, structlog, and others. The Dockerfile's `pip install`
> chain is precisely the part that must stay FROZEN during the migration —
> Phase 3 only isolates platform regressions if the dependency tree is
> identical to current production. Bundling M4 would inject a confounding
> variable. M4 is filed as a standalone post-migration task, to be done as a
> LOCKFILE (uv / pip-compile / poetry) — the audit showed the install-order
> sequence in the current Dockerfile is hand-locking the transitive tree,
> and Shape A (hand-tightening pyproject pins) re-implements a lockfile by
> hand and carries the same drift risk. Audit details:
> [debugging-notes 2026-06-08 entry](debugging-notes.md).

**1A.1 — XFF parser fix** (decouples H2 from later proxy-trust enablement)
- Change leftmost-segment parsing to rightmost-trusted-hop (or add
  `RAGCORE_PROXY_HOP_COUNT` if a chain). Behavior on Render is unchanged
  because `RAGCORE_TRUST_PROXY_HEADERS=false` today — the new parser only
  matters when trust is on. Flag-flip happens in Phase 2 on HF only.
- Must land before the `hf-migration` branch is cut (see Phase 1B seam).
- **Gate to 1A.2:** unit tests cover {single proxy, no proxy, spoofed leftmost,
  IPv6, malformed header}; Render production cron still green.

**1A.2 — Bundle FiQA seed artifact into the image**
- Move [faiss/faiss_index.idx](../faiss/faiss_index.idx) +
  [faiss/faiss_metadata.pkl](../faiss/faiss_metadata.pkl) into a versioned
  in-image path (e.g. `data/faiss_seed/`). Confirm `.dockerignore` does NOT
  exclude it (parallel to [data/chroma_collections/](../data/chroma_collections/)
  handling).
- 5.1 MB total; commit directly to git.
- No behavior change yet — nothing consults this path until 1A.3 lands.

**1A.3 — FAISS self-heal seed step in the lifespan** (the boot-path change)

Mirrors the structural shape of `_seed_apple_collections`
([api/main.py:152-208](../api/main.py#L152-L208)): idempotent skip when the
destination index already exists, atomic tmp+os.replace for the copy.

**Fail-hard semantics — explicitly diverges from the Apple seed.** Per the
9-day-outage post-mortem, an empty-but-serving `default` corpus IS the outage
mode: `/health/ready` passes (vector_store ping succeeds on an empty store),
`/query` returns the empty-retrieval fallback, monitoring sees nothing wrong,
demo silently serves no FiQA answers. The Apple seed soft-fallbacks (logs
"Seed failed for X" and continues) because losing one of six explicit-name
Apple corpora is a degraded-but-honest state — the missing corpus is absent
from `/corpora`, queries for it 404, the other five work. **FiQA cannot be
treated the same way** because it's the default corpus every unscoped query
hits, and "default corpus exists but is empty" is the documented silent-failure
mode.

Behavior: if the FiQA seed step fails for any reason (source missing, source
corrupted, atomic-replace fails, dest unwritable), the step raises a
RuntimeError and the lifespan dies. Container enters restart loop. Loud. The
Apple seed continues to soft-fallback unchanged — the asymmetry is intentional
and stated in the new seed step's docstring so future consistency-pressure
doesn't quietly unify them.

**Step ordering inside the lifespan is load-bearing:**
- After `_assert_session_root_isolated` (step 2 in [baseline §3](migration-baseline-2026-06-05.md)) — so we know the dest isn't a session root.
- Before `register_corpus("default", FAISSVectorStore())` (currently step 3) — the FAISS constructor reads the index; seeding after construction is a no-op for the in-memory store.

Race surface: lifespan is single-threaded async; no concurrent boots; reaper
isn't started until step 10. No mid-seed mutation possible.

Path-overlap assertion: source is `data/faiss_seed/` (read-only in image), dest
is `FAISS_DATA_DIR`; neither is a session root if step 2 passed. No new
assertion needed.

**Local verification before merge — four cases:**
1. Fresh: rm -rf the FAISS dest dir, boot, verify seed fires, log line emitted,
   index loads, `/corpora` shows `default` with non-zero doc_count.
2. Repeat: boot again, verify seed skips with idempotency log, `default` still
   loads from disk.
3. Corrupted source: rename the seed source file, boot, verify the lifespan
   raises RuntimeError and the process exits non-zero (NOT a silent empty
   fallback).
4. Unwritable dest: chmod the dest dir read-only, boot, verify same hard-fail
   behavior.

**Render impact at merge time:** zero. Existing `/var/data/faiss/faiss_index.idx`
already exists on Render's persistent disk → seed skips on every boot. The
fail-hard branch is never exercised on Render. The PR is a no-op for live
traffic.

**Bonus:** this PR alone closes the 9-day-outage hole. Once on Render's main,
the next disk wipe self-heals — and if it can't self-heal, the deploy fails
loudly instead of serving empty.

**Gate to 1A.4:** all four local verification cases pass; Render redeploy is a
no-op (seed skips); cron green.

**1A.4 — Sweep hardcoded `ragcore-api.onrender.com` from ui_streamlit/**
- Two known instances: the comparison-mode descriptive text and the
  `[API docs](...)` markdown link. Grep the full `ui_streamlit/` tree for
  any others.
- Replace with values derived from `BACKEND_URL` (already an env var read at
  [ui_streamlit/app.py:1](../ui_streamlit/app.py)). API-docs link becomes
  `f"{BACKEND_URL}/docs"`.
- Streamlit Cloud auto-redeploys with the same `RAGCORE_BACKEND_URL` it has
  today → zero behavior change. The cleanup just removes the dead-link risk
  before cutover.
- **Gate to the branch seam:** Streamlit Cloud redeploys; UI still loads; the
  API-docs link still resolves (to Render now, to HF after cutover).

---

## Branch seam — cut `hf-migration` ONLY after all of 1A.1–1A.4 are on main

Explicit ordering: 1B does not start until every commit from 1A.1 through
1A.4 has merged to `main` and the post-merge cron is green. Cutting the branch
earlier — specifically, before 1A.1 — would carry an outdated XFF parser into
HF, and Phase 2.2 would then flip `RAGCORE_TRUST_PROXY_HEADERS=true` against
the vulnerable code, activating H2 on the new platform. The branch seam is
where this is enforced.

Verification before cutting: `git log main` shows all four 1A commits; `git
diff main..HEAD` on a freshly-cut branch shows zero diff.

---

## Phase 1B — HF-only Dockerfile changes, branch-isolated

The remaining Dockerfile work breaks Render and is therefore branch-isolated
until cutover. New branch `hf-migration` off `main`. HF Space deploys from
this branch; Render continues from `main`.

**1B.1 — `USER 1000` + chown in Dockerfile + repoint corpus paths**
- Add a non-root user (UID 1000) and chown `/app/data` so the runtime user can
  write the FAISS seed step's atomic-replace output and the orphan-purge can
  rm-rf session subdirs.
- Set image-build defaults for `FAISS_DATA_DIR` / `CHROMA_PERSIST_DIR` /
  `RAGCORE_SESSION_ROOT` to absolute paths under `/app/data/` (three siblings,
  no overlap — satisfies the path-isolation assertion).
- **Why branch-isolated:** Render's `/var/data` persistent disk is root-owned
  by default; switching the process to UID 1000 will break Render writes
  until the disk is chowned via an init script. Cleaner to keep `main`
  Render-safe and treat the merge-to-main moment as Render's death.
- **Gate to Phase 2:** branch builds locally; container can start under UID
  1000; the FAISS seed step writes successfully under the non-root user.

---

## Phase 2 — HF Space stand-up, parallel to live Render

Render is still serving the demo throughout. HF runs alongside but Streamlit
Cloud still points at Render.

**2.1 — Create the HF Space + calibrate `RAGCORE_PROXY_HOP_COUNT`**
- Docker Space type. Point at the `hf-migration` branch. Free CPU Basic. No
  Persistent Storage tier.
- **Calibration probe (gate for 2.2):** with the Space up but
  `RAGCORE_TRUST_PROXY_HEADERS` still **false**, hit a non-exempt endpoint
  from a known external IP (e.g. `curl https://<space>/query -d '...'` from
  your machine). Inspect the request log for the raw `X-Forwarded-For` header
  value and `request.client.host`. The number of comma-separated XFF segments
  is the hop count. The 1A.1 code defaults `RAGCORE_PROXY_HOP_COUNT=1`; if
  the probe shows ≥ 2, set the env var on the Space to the observed value
  BEFORE flipping trust=true. If the trust flag flips with a wrong hop_count,
  the rate limiter resolves all clients to an internal proxy IP and the
  per-IP bucketing collapses — the calibration warning at
  [api/middleware/rate_limit.py](../api/middleware/rate_limit.py)
  (`client_ip_resolved_to_non_global`) fires immediately on the first
  request as a safety net, but the gate exists to prevent that path entirely.
- **Gate to 2.2:** Space exists; calibration probe captured the actual XFF
  hop count from a real request; `RAGCORE_PROXY_HOP_COUNT` is set to match.

**2.2 — Configure env vars on the Space**
- Mirror the 11 from [render.yaml:14-36](../render.yaml#L14-L36) as Variables +
  Secrets per [decision memo §3](hf-migration-phase0-decisions-2026-06-06.md#3-secrets--env).
- Set the four HF-critical explicits: `CORS_ORIGINS` (Streamlit Cloud URL),
  `RAGCORE_TRUST_PROXY_HEADERS=true` (safe because 1A.1 is on the branch),
  `RAGCORE_SESSION_ROOT=/app/data/sessions`, `CHROMA_PERSIST_DIR=/app/data/chroma_db`,
  `FAISS_DATA_DIR=/app/data/faiss`.
- Persist this list in `docs/hf-env-vars.md` as the out-of-repo source of
  truth the memo §3 anticipated.
- **Gate to 2.3:** env-var list matches the decision-memo checklist; nothing
  missing.

**2.3 — First HF deploy + boot-log inspection**
- Trigger the Space build; watch logs end-to-end.
- Verify all 11 lifespan steps from [baseline §3](migration-baseline-2026-06-05.md#3-boot-sequence)
  fire in order, **plus the new FAISS seed step** firing with its first-boot
  path (`Seeding default: data/faiss_seed → /app/data/faiss`).
- Verify `/health/live` instantaneous, `/health/ready` 200 after warm-up.
- **Gate to Phase 3:** Space is up; `/health/ready` returns all-ok; no
  assertion fired; both seed loops (FAISS + Apple) logged their first-boot
  path.

---

## Phase 3 — The migration pass condition

**The single outcome that defines migration success:** a fresh HF box, booted
with `FAISS_DATA_DIR` empty before boot and no human running an ingest script,
serves a grounded answer to the production-health-check FiQA query (`{"query":
"What is a Roth IRA?", "top_k": 5}`) with citations from the FiQA corpus,
tokens > 0, and latency >= 100 ms.

This is the migration's reason for existing — it proves the FiQA self-heal
path closed the 9-day-outage hole. Everything else in this phase is table
stakes verifying that the supporting invariants from the baseline didn't
regress.

### Table-stakes checks (must also pass; all sourced from [baseline](migration-baseline-2026-06-05.md))

- §1 env-var grep — no drift from the table.
- §2 `/corpora` — 7 entries, all with `doc_count > 0`. (The FiQA entry being
  present without a manual ingest is the explicit confirmation of the pass
  condition above.)
- §3 boot logs — all steps logged including the new FiQA seed.
- §4 health endpoints — `/health/live` instant, `/health/ready` sub-100 ms
  warm, full `production_health_check.py` PASSes all five signals.
- §5 external deps — Groq reachable; chromadb pin 0.4.24 exact;
  `_identifer_to_system` attribute present. HF Hub model downloads are not
  exercised at runtime because models are pre-baked in the Dockerfile
  ([Dockerfile:63-68](../Dockerfile#L63-L68)) — note this divergence from the
  baseline's "first boot only" framing.

### One HF-specific check not in the baseline

- **Rate-limit attribution under proxy:** hit `/health` from two different
  external IPs; confirm the rate limiter buckets them separately. If they
  share a bucket, 1A.1 didn't land correctly on the branch and H2 is
  exploitable on HF.

**Gate to Phase 4: the pass condition above is met AND every table-stakes
check is PASS. Any FAIL → fix on the `hf-migration` branch and re-run Phase 3
from scratch. Do not advance partial-state.**

---

## Phase 4 — Cutover (the only step that affects live traffic)

**4.1 — Atomic cutover: Streamlit BACKEND_URL + CI TARGET_URL flip together**

Both pointers move to HF in the same window so the monitored backend always
matches the live backend — no period where the cron is checking a backend that
isn't serving the UI.

- Prepare the git commit that updates `.github/workflows/production_health_check.yml`'s
  `TARGET_URL` from the Render URL to the HF URL. Stage it for merge.
- Update Streamlit Cloud's `RAGCORE_BACKEND_URL` secret to the HF URL.
  Streamlit Cloud auto-redeploys (~1–2 min).
- Merge the workflow commit at the same moment the Streamlit redeploy
  starts. The next cron tick (or a manual `workflow_dispatch`) hits HF.
- **Gate to 4.2:** Streamlit Cloud redeploy complete; one `workflow_dispatch`
  run of the health-check against HF passes; UI loads end-to-end in a real
  browser (corpus list, a query returns grounded + cited, comparison view,
  API-docs link resolves to `<hf-url>/docs`).

**4.2 — Pause Render auto-deploy**
- Set Render's `autoDeploy: false` so a stray push doesn't resurrect a phantom
  backend. Don't delete the service yet. Render keeps serving its own URL but
  it's orphaned from the demo and from monitoring.

**4.3 — Watch one full cron cycle pass green against HF**
- Let one full 4h cron cycle pass without alarm.
- **Gate to Phase 5:** one full cron cycle green against HF; Streamlit UI
  still working; no errors in HF Space logs.

---

## Phase 5 — Merge + decommission (after rollback window)

**5.1 — Merge `hf-migration` → `main`**
- HF Space switches to deploying from `main`. Render is paused so the
  Dockerfile-incompatible-with-Render-disk change doesn't matter.

**5.2 — Hold a rollback window** (suggest 3–7 days)
- Render service still exists, paused. If something surfaces on HF that
  baseline + Phase 3 missed, re-enable Render's auto-deploy as the rollback.

**5.3 — Delete Render service + persistent disk**
- After the rollback window, delete the service and the 5 GB persistent disk.
  Cost goes to $0. Update `render.yaml` — either delete or add a `# DEPRECATED`
  header for historical reference.
- Sweep remaining `ragcore-api.onrender.com` references in non-UI files
  (README, docs, CI workflows, scripts).

---

## What's NOT in this sequence

- Two-platform → one-platform consolidation (combining FastAPI + Streamlit
  into one Space). [Decision memo §2](hf-migration-phase0-decisions-2026-06-06.md#2-space-type--topology)
  said don't migrate things that aren't broken.
- M4 (Dockerfile `pip install -e .` swap). Pulled to a standalone post-
  migration project, to be done as a lockfile rather than hand-pinning.
  Rationale in the Phase 1A intro above.
- AUDIT items unrelated to the platform port. Independent of the migration;
  shouldn't fight for review attention during it.
- Prometheus scrape config for HF (no metrics-port concept in HF Spaces).
  Post-migration item if wanted.
- Hardening `/health/ready`'s vector_store ping to require non-zero doc_count
  for `default`. Not needed under 1A.3's fail-hard semantics (empty FiQA
  cannot reach readiness). Worth revisiting if 1A.3's fail-hard is ever
  weakened.

## How the four sharpenings landed

1. **1A.3 fail-hard for FiQA seed**: hard-fail-to-boot on any seed failure;
   explicitly diverges from the Apple seed's soft fallback; rationale and
   asymmetry stated in the seed step's docstring so future consistency-pressure
   doesn't quietly unify them.
2. **Phase 3 pass condition promoted**: the fresh-box-serves-grounded-FiQA
   outcome is the named gate at the top of Phase 3; baseline checklist items
   are framed as table stakes verifying supporting invariants.
3. **4.1 CI TARGET_URL flip moved up**: now happens in lockstep with Streamlit
   BACKEND_URL flip in step 4.1, not in 4.3. No monitoring gap.
4. **Branch seam pinned after 1A.1–1A.4**: explicit "Branch seam" section
   between Phase 1A and Phase 1B enforces that the XFF fix is on `main` before
   the HF-targeted branch is cut and TRUST_PROXY_HEADERS=true is set.
