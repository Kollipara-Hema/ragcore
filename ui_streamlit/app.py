"""
RAGCore Streamlit Demo — visual rebuild.
Build file: new_app.py. Swap to app.py after approval.
"""
from __future__ import annotations

import base64
import html
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from comparison import CHUNKER_COMPARISON_CORPORA, fetch_chunker_comparison

# ─── Design tokens ────────────────────────────────────────────────────────────
ACCENT        = "#6d4aab"
ACCENT_TINT   = "#f5f3ff"
ACCENT_DARK   = "#3c2a72"
CORAL         = "#e07a5f"
CORAL_TINT    = "#fde8e1"
TEAL          = "#2d8e8e"
TEAL_TINT     = "#ccfbf1"
AMBER         = "#bf8b3a"
AMBER_TINT    = "#fcd9b6"
SUCCESS_BG    = "#dcfce7"
SUCCESS_DARK  = "#166534"
ERROR_BG      = "#fee2e2"
ERROR_DARK    = "#991b1b"
BORDER        = "#e5e7eb"
BG_PAGE       = "#ffffff"
BG_SIDEBAR    = "#f8fafc"
ASSISTANT_BG  = "#fcfcfc"
TEXT_PRIMARY   = "#1f2937"
TEXT_SECONDARY = "#6b7280"
TEXT_MUTED     = "#9ca3af"

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RAGCore",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Avatar SVG helper ────────────────────────────────────────────────────────
def _avatar_svg(letter: str, bg: str = ACCENT) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">'
        f'<circle cx="16" cy="16" r="16" fill="{bg}"/>'
        f'<text x="16" y="21" text-anchor="middle" font-family="sans-serif" '
        f'font-size="14" font-weight="600" fill="#ffffff">{letter}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


USER_AVATAR = _avatar_svg("Y")
ASST_AVATAR = _avatar_svg("R")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Layout ── */
[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR};
    border-right: 1px solid {BORDER};
}}
/* Tighten sidebar spacing */
[data-testid="stSidebar"] .element-container {{
    margin-bottom: 0.25rem !important;
}}

[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
    gap: 0.25rem !important;
}}
#MainMenu {{visibility: hidden;}}
footer    {{visibility: hidden;}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-left: 0.6rem;
    padding-right: 0.6rem;
}}
[data-testid="stSidebar"] .stCaption {{
    margin-bottom: 0rem !important;
}}
/* ── Status pill ── */
.status-pill {{
    display: inline-block; padding: 5px 12px; border-radius: 8px;
    font-size: 13px; font-weight: 500; width: 100%; box-sizing: border-box; margin: 4px 0;
}}
.status-ok  {{ background: {SUCCESS_BG}; color: {SUCCESS_DARK}; }}
.status-err {{ background: {ERROR_BG};  color: {ERROR_DARK};  }}

/* ── Section labels ── */
.sec-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; color: {TEXT_MUTED}; margin: 12px 0 6px; display: block;
}}

/* ── Strategy radio ── */
[data-testid="stRadio"] label {{
    padding: 2px 8px; border-radius: 6px; border: 1px solid transparent;
    cursor: pointer; margin-bottom: 0;
}}
[data-testid="stRadio"] > div > label + label {{ margin-top: 1px; }}
[data-testid="stRadio"] label:has(input:checked) {{
    background: {ACCENT_TINT}; border-color: {ACCENT};
}}

/* ── Pipeline rows ── */
.pipeline-total {{ font-size: 12px; color: {TEXT_PRIMARY}; margin-bottom: 4px; font-weight: 700; }}
.pipeline-row {{
    display: flex; align-items: center; padding: 1px 0;
    font-size: 13px; color: {TEXT_SECONDARY};
}}
.pipeline-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0; margin-right: 8px;
}}
.pipeline-label {{ flex: 1; }}
.conf-pill {{ padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 500; }}
.conf-high {{ background: {SUCCESS_BG}; color: {SUCCESS_DARK}; }}
.conf-med  {{ background: {AMBER_TINT};  color: {AMBER}; }}
.conf-low  {{ background: {ERROR_BG};   color: {ERROR_DARK}; }}

/* ── Chat message bubbles ── */
[data-testid="stChatMessage"] {{
    border-radius: 12px; border: 1px solid {BORDER};
    padding: 16px; margin-bottom: 8px; background: {ASSISTANT_BG};
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: {ACCENT_TINT}; border: 0.5px solid {ACCENT};
}}

/* ── Verified badge ── */
.verified-badge {{
    display: inline-block; background: {TEAL_TINT}; color: {TEAL};
    padding: 4px 10px; border-radius: 99px;
    font-size: 11px; font-weight: 500; margin-left: 8px;
}}

/* ── Sources 2×2 grid ── */
/* minmax(0, 1fr) lets a wide child shrink below its min-content size,
   so a long unbroken string (raw CSV chunk) can't push the column past
   the grid container and produce a horizontal scrollbar. */
.src-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; margin: 8px 0; }}
.src-card {{ background: {BG_PAGE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 12px; }}
.src-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; background: {ACCENT}; color: #fff;
    border-radius: 6px; font-size: 12px; font-weight: 600; margin-bottom: 6px;
}}
.src-title {{
    font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY};
    margin-bottom: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.src-score  {{ font-family: monospace; font-size: 11px; color: {TEXT_SECONDARY}; margin-bottom: 3px; }}
.src-excerpt {{
    font-size: 12px; color: {TEXT_SECONDARY}; line-height: 1.5;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 3px;
    /* Allow long unbroken strings (raw CSV chunk content) to wrap inside
       the card instead of forcing horizontal overflow. */
    overflow-wrap: anywhere; word-break: break-word;
}}
.src-chunk {{ font-family: monospace; font-size: 10px; color: {TEXT_MUTED}; }}

/* ── Self-RAG chips ── */
.srag-v {{
    display: inline-block; background: {SUCCESS_BG}; color: {SUCCESS_DARK};
    border-radius: 4px; padding: 2px 8px; font-size: 11px; margin: 2px 3px;
}}
.srag-u {{
    display: inline-block; background: {AMBER_TINT}; color: {AMBER};
    border-radius: 4px; padding: 2px 8px; font-size: 11px; margin: 2px 3px;
}}

/* ── Abstention / error cards ── */
.abstention-card {{
    background: #fff8e1; border: 1px solid #ffc107;
    border-radius: 8px; padding: 16px; margin: 8px 0; font-size: 14px;
}}
.error-card {{
    background: #fff5f5; border: 1px solid #feb2b2;
    border-radius: 8px; padding: 16px; margin: 8px 0; font-size: 14px;
}}

/* ── Prompt card buttons (main area only) ── */
.block-container div[data-testid="stButton"] > button {{
    text-align: left !important; white-space: normal !important; height: auto !important;
    padding: 14px !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; background: {BG_PAGE} !important;
    color: {TEXT_PRIMARY} !important; font-size: 14px !important; line-height: 1.5 !important;
    transition: background 0.12s, border-color 0.12s !important;
}}
.block-container div[data-testid="stButton"] > button:hover {{
    background: {ACCENT_TINT} !important; border-color: {ACCENT} !important;
}}

/* ── Sidebar buttons (Clear chat) ── */
[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
    background: {BG_PAGE} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; padding: 8px 16px !important;
    color: {TEXT_SECONDARY} !important; font-size: 13px !important;
    text-align: center !important; white-space: normal !important;
    transition: background 0.12s, border-color 0.12s !important;
}}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
    background: {ACCENT_TINT} !important; border-color: {ACCENT} !important;
    color: {TEXT_PRIMARY} !important;
}}

/* ── Follow-up chips (inside chat messages only) ── */
.followup-label {{
    color: {TEXT_MUTED}; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.5px; margin-top: 16px; margin-bottom: 8px;
}}
[data-testid="stChatMessage"] div[data-testid="column"] [data-testid="stButton"] > button {{
    background: white !important; border: 1px solid {BORDER} !important;
    border-radius: 99px !important; padding: 6px 14px !important;
    font-size: 13px !important; line-height: 1.4 !important;
    text-align: left !important; white-space: normal !important;
    height: auto !important; color: {TEXT_PRIMARY} !important;
    transition: all 0.15s ease !important;
}}
[data-testid="stChatMessage"] div[data-testid="column"] [data-testid="stButton"] > button:hover {{
    background: {ACCENT_TINT} !important; border-color: {ACCENT} !important;
    cursor: pointer !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("RAGCORE_BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("RAGCORE_API_KEY", "").strip()


def _auth_headers() -> dict[str, str]:
    """Return X-API-Key header if RAGCORE_API_KEY is set, else empty dict."""
    return {"X-API-Key": API_KEY} if API_KEY else {}

# Starter prompt cards per corpus. The three 10-K variants share one set
# (same underlying document, different chunkers). Each set's lead question
# is the known-good one Hema confirmed answers cleanly; the other two are
# plausible follow-ons against the same document. Anything not in this dict
# falls back to the FiQA set.
_APPLE_10K_STARTERS = [
    ("REVENUE",      CORAL,  "What were Apple's net sales by product category?"),
    ("GEOGRAPHY",    ACCENT, "How did Apple's net sales break down by geographic segment?"),
    ("RISK FACTORS", TEAL,   "What are the principal risks Apple flagged in fiscal 2025?"),
]

PROMPT_STARTERS: dict[str, list[tuple[str, str, str]]] = {
    "default": [
        ("MORTGAGES",     CORAL,  "Should I pay off my mortgage early or invest the extra cash?"),
        ("RETIREMENT",    ACCENT, "What are the contribution limits for a Roth IRA vs Traditional IRA?"),
        ("EMPLOYER 401K", TEAL,   "How does a 401k employer match work and how should I maximize it?"),
    ],
    "apple_10k_fixed":              _APPLE_10K_STARTERS,
    "apple_10k_hierarchical":       _APPLE_10K_STARTERS,
    "apple_10k_document_structure": _APPLE_10K_STARTERS,
    "apple_environmental": [
        ("EMISSIONS",    CORAL,  "What are Apple's carbon emissions reduction goals?"),
        ("RENEWABLES",   ACCENT, "How much of Apple's energy comes from renewable sources?"),
        ("SUPPLY CHAIN", TEAL,   "What environmental commitments has Apple set for its supply chain?"),
    ],
    "apple_earnings_html": [
        ("HIGHLIGHTS",   CORAL,  "What were Apple's Q4 2025 earnings highlights?"),
        ("GUIDANCE",     ACCENT, "Did Apple provide any guidance for the next quarter?"),
        ("MARGINS",      TEAL,   "What was Apple's gross margin in Q4 2025?"),
    ],
    "apple_financial_csvs": [
        ("REVENUE",      CORAL,  "What was Apple's quarterly revenue?"),
        ("EARNINGS",     ACCENT, "How has Apple's earnings per share trended over recent quarters?"),
        ("CASH FLOW",    TEAL,   "What were Apple's operating cash flows by quarter?"),
    ],
}

# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = None
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "prompt_prefill" not in st.session_state:
    st.session_state.prompt_prefill = ""
if "verify_claims" not in st.session_state:
    st.session_state.verify_claims = False
if "session_id" not in st.session_state:
    # X-Session-Id minted by the backend on first /ingest. None = public-
    # corpus mode. Written ONLY by _call_api (CONTRACT 1), the "Start new
    # session" button, and the 404-on-query handler. Grep "session_id =" to
    # find every site that touches it; anything outside those three is a
    # regression.
    st.session_state.session_id = None
if "uploaded_file_keys" not in st.session_state:
    # Dedup set for the file_uploader's rerun loop. st.file_uploader
    # retains its value across reruns, so without this guard a single
    # file selection would POST on every rerun and burn the session's
    # 3-file cap immediately.
    st.session_state.uploaded_file_keys = set()
if "upload_409_active" not in st.session_state:
    # Raised by _handle_upload on 409 so the sidebar can prominent the
    # "Start new session" button. Cleared when the button is clicked.
    st.session_state.upload_409_active = False
if "uploader_nonce" not in st.session_state:
    # Counter baked into st.file_uploader's `key=` so that incrementing it
    # forces Streamlit to render a fresh widget identity — guaranteed empty,
    # no carry-over of a retained UploadedFile. This is the only reliable
    # way to clear a file_uploader from code: pop()-ing the widget key is
    # allowed but the runtime can resurrect the UploadedFile via internal
    # file_id storage. Bumped by the "Start new session" button and the
    # 404-on-query handler.
    st.session_state.uploader_nonce = 0

# ─── Backend helpers (verbatim) ───────────────────────────────────────────────
@st.cache_data(ttl=30)
def _check_backend_cached(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", headers=_auth_headers(), timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _call_api(
    method: str,
    path: str,
    *,
    json: Optional[dict] = None,
    files: Optional[dict] = None,
    timeout: float = 60,
) -> tuple[int, dict, dict]:
    """Single-seam HTTP wrapper. Returns (status_code, body_dict, response_headers).

    Outbound: adds X-Session-Id from st.session_state when set, plus _auth_headers().
    Inbound (CONTRACT 1): if the response carries X-Session-Id, it overwrites
    st.session_state.session_id. This is the ONLY place that writes the
    session_id from a response — an expired token gets replaced by the fresh
    one the server mints, so TTL eviction or restart doesn't strand the
    client.

    Transport failures map to status_code=0 with detail in body — callers
    use a single status-based branching pattern for both HTTP errors and
    network failures.
    """
    headers = {**_auth_headers()}
    if st.session_state.session_id:
        headers["X-Session-Id"] = st.session_state.session_id
    if json is not None and files is None:
        headers["Content-Type"] = "application/json"

    try:
        r = requests.request(
            method, f"{BACKEND_URL}{path}",
            json=json, files=files, headers=headers, timeout=timeout,
        )
    except requests.RequestException as exc:
        return 0, {"detail": str(exc)}, {}

    # CONTRACT 1: capture the server-minted (or echoed) token from EVERY
    # response that carries it. Backend currently emits it from the ingest
    # endpoints; if future endpoints emit it too, this code already handles
    # them.
    minted = r.headers.get("X-Session-Id")
    if minted:
        st.session_state.session_id = minted

    try:
        body = r.json()
    except ValueError:
        body = {"detail": r.text or "no response body"}
    return r.status_code, body, dict(r.headers)


def ask_question(query: str, strategy: str, verify_claims: bool = False, corpus: str = "default") -> tuple[int, dict]:
    """POST /query. Returns (status_code, body). The caller branches on
    status — 200/404/400/5xx all have distinct UX paths.

    DECISION: when a session is active, the `corpus` field is OMITTED from
    the payload entirely. The backend ignores it on the session path
    (Commit 5 isolation), but omitting is defense-in-depth: a future
    routing-precedence bug that consulted `corpus` on the session path
    would yield a clean error rather than a silent public-corpus answer.
    """
    payload: dict = {
        "query": query,
        "strategy_override": strategy if strategy != "auto" else None,
        "verify_claims": verify_claims,
    }
    if st.session_state.session_id is None:
        payload["corpus"] = corpus
    status, body, _ = _call_api("POST", "/query", json=payload, timeout=90)
    return status, body


# Human-readable labels for the corpus dropdown. Machine names returned by
# /corpora map to these; unknown corpora fall back to a Title-Case slug.
# For apple_10k_document_structure (chunker key "structure") we use the
# corpus-name framing — "document structure" — so the label is one consistent
# term and the structure/document_structure split stays hidden from users.
CORPUS_LABELS = {
    "default":                       "FiQA (financial Q&A)",
    "apple_10k_fixed":               "Apple 10-K — fixed chunking",
    "apple_10k_hierarchical":        "Apple 10-K — hierarchical",
    "apple_10k_document_structure":  "Apple 10-K — document structure",
    "apple_environmental":           "Apple Environmental Report",
    "apple_earnings_html":           "Apple Q4 earnings (HTML)",
    "apple_financial_csvs":          "Apple financial CSVs",
}

# Aggregate eval metrics exist ONLY for corpora that have been run through
# the eval harness. Apple corpora were never benchmarked, so a missing entry
# means "show facts, not fabricated metrics" — the stat-card section keys
# off membership here to decide eval vs. facts layout.
EVAL_METRICS = {
    "default": {
        "chunks":   380,
        "hit_at_5": 0.92,
        "mrr":      0.86,
        "label":    "FiQA-2018",
    },
}


# Pretty source names for the hero sentence + Source stat card. These describe
# the underlying document (or document group), not the chunker variant — so the
# three 10-K corpora share one entry. Anything not in this dict falls back to
# the live filename (or "N files" for comma-joined multi-source corpora).
CORPUS_SOURCE_LABELS = {
    "apple_10k_fixed":               "Apple 10-K (2025)",
    "apple_10k_hierarchical":        "Apple 10-K (2025)",
    "apple_10k_document_structure":  "Apple 10-K (2025)",
    "apple_environmental":           "Apple Environmental Report (2025)",
    "apple_earnings_html":           "Apple Q4 Earnings Release (2025)",
    "apple_financial_csvs":          "Apple Financial Metrics (2026)",
}


def _corpus_label(name: str) -> str:
    return CORPUS_LABELS.get(name, name.replace("_", " ").title())


def _corpus_source_label(name: str, fallback_src: str) -> str:
    """Pretty source name with filename fallback. `fallback_src` is the raw
    `source` field from /corpora (single path, or comma-joined for
    apple_financial_csvs)."""
    pretty = CORPUS_SOURCE_LABELS.get(name)
    if pretty:
        return pretty
    if "," in fallback_src:
        return f"{len(fallback_src.split(','))} files"
    if fallback_src:
        return Path(fallback_src).name
    return "—"


@st.cache_data(ttl=60)
def _fetch_corpora(url: str) -> list[dict]:
    """GET /corpora; returns [] on any failure so the caller can fall back.

    Stays on direct requests (not _call_api) because @st.cache_data wraps
    this function — the cached body never sees a session header on warm
    hits, which is the right behavior for a public-corpus listing.
    """
    try:
        r = requests.get(f"{url}/corpora", headers=_auth_headers(), timeout=5)
        if r.status_code == 200:
            return r.json().get("corpora") or []
    except Exception:
        pass
    return []


# Vector-store backend per corpus, for the M5 footer. /corpora does NOT
# expose this — it's a UI-side assumption mirroring api/main.py lifespan:
# "default" is registered as FAISSVectorStore; every other corpus is a
# ChromaVectorStore in the Apple-seeding loop. Update this map if the
# backend split changes (e.g. a new FAISS-backed corpus is added).
CORPUS_VECTOR_STORE: dict[str, str] = {
    "default": "FAISS",
}


def _corpus_vector_store(name: str) -> str:
    return CORPUS_VECTOR_STORE.get(name, "Chroma")


# Per-answer retrieval-strategy label. Strategy is a PER-QUERY fact (auto
# router can pick any of these), so it lives next to each assistant message,
# not in the static footer. Mechanism descriptions verified against
# retrieval/strategies/retrieval_executor.py:99-115:
#   semantic → dense vector search only
#   keyword  → BM25 only
#   hybrid   → dense + BM25
# Other RetrievalStrategy values (metadata / multi_query / parent_child)
# fall through to the bare string so the caption doesn't fabricate a
# mechanism description for code paths we haven't audited.
RETRIEVAL_STRATEGY_LABELS: dict[str, str] = {
    "hybrid":   "hybrid (dense + BM25)",
    "keyword":  "keyword (BM25)",
    "semantic": "semantic (dense)",
}


def _strategy_label(strategy: Optional[str]) -> Optional[str]:
    if not strategy:
        return None
    return RETRIEVAL_STRATEGY_LABELS.get(strategy, strategy)


def _file_dedup_key(file) -> str:
    """Stable key for an UploadedFile used to skip already-processed uploads
    across reruns. Streamlit's `file_id` is its internal handle for a single
    upload session — equal across reruns for the same file selection, new
    on a fresh selection. The name+size fallback is a defense in case the
    private attribute disappears in a future Streamlit version."""
    fid = getattr(file, "file_id", None)
    if fid:
        return fid
    return f"{file.name}:{file.size}"


def _handle_upload(file) -> None:
    """Single-fire upload handler. Dedups against
    st.session_state.uploaded_file_keys so a file selected once doesn't
    re-POST on every rerun (which would burn the 3-file session cap
    instantly). Maps every backend status to a readable UI message."""
    key = _file_dedup_key(file)
    if key in st.session_state.uploaded_file_keys:
        return
    # Mark BEFORE the request so even an exception inside doesn't loop on
    # reruns — the user can clear and retry deliberately if they want.
    st.session_state.uploaded_file_keys.add(key)

    with st.spinner(f"Uploading {file.name}…"):
        status, body = upload_document(file)
    detail = body.get("detail") or body.get("message") or "no detail"

    if status == 200:
        msg = body.get("message", f"Indexed {file.name}.")
        st.success(msg)
    elif status == 413:
        st.error(detail)
    elif status == 415:
        st.error(detail)
    elif status == 409:
        # CONTRACT 2: surface the limit message; the "Start new session"
        # button below the indicator gives the user the means to recover.
        st.error(detail)
        st.session_state.upload_409_active = True
    elif status == 503:
        st.warning(f"{detail} Try again in a moment.")
    elif status == 0:
        st.error(f"Upload failed: {detail}")
    else:
        st.error(f"Upload failed ({status}): {detail}")


def upload_document(file) -> tuple[int, dict]:
    """POST /ingest/file. Returns (status_code, body). The caller maps
    status codes to UI messages:
      200 → success, X-Session-Id was captured by _call_api
      413 → file size or PDF page-count exceeded
      415 → unsupported file type (bytes-sniff rejected)
      409 → session file-count limit reached → "Start new session"
      503 → server at concurrent-session capacity
    """
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    status, body, _ = _call_api(
        "POST", "/ingest/file", files=files, timeout=120,
    )
    return status, body


# ─── compute_confidence (verbatim from commit 98f0f8e) ───────────────────────
def compute_confidence(retrieval_candidates: Optional[list]) -> str:
    """
    Confidence = f(score gap between top-1 and median pre-rerank candidate).

    Thresholds calibrated 2026-04-29 against 10 queries on the deployed
    backend using live pre-rerank hybrid
    (FAISS+BM25) scores from retrieval_candidates. All 10 queries returned
    non-empty candidates (n=20 each). Hybrid scores are unbounded and may
    include negatives — range is wider than the previous cross-encoder proxy.

    Observed gaps (sorted): 1.05, 2.85, 3.70, 3.79, 6.12, 7.12, 8.20,
                             8.23, 13.15, 13.83
    high   >= 8.13  (66th–100th pct)
    medium  3.79–8.12  (33rd–66th pct)
    low    < 3.79   (bottom third)
    """
    cands = retrieval_candidates or []
    if not cands:
        return "unknown"
    scores = sorted([c["score"] for c in cands], reverse=True)
    n = len(scores)
    if n < 2:
        return "unknown"
    gap = scores[0] - scores[n // 2]
    if gap >= 8.13:
        return "high"
    elif gap >= 3.79:
        return "medium"
    else:
        return "low"


# ─── UI helpers ───────────────────────────────────────────────────────────────
def _confidence_badge_html(level: str) -> str:
    """Returns confidence badge HTML string; empty string for 'unknown'."""
    if level == "unknown":
        return ""
    cls = {"high": "conf-high", "medium": "conf-med", "low": "conf-low"}[level]
    return f'<span class="conf-pill {cls}">{level}</span>'


def _pipeline_row(dot_color: str, label: str, value: str) -> str:
    return (
        f'<div class="pipeline-row">'
        f'<span class="pipeline-dot" style="background:{dot_color}"></span>'
        f'<span class="pipeline-label">{label}</span>'
        f'<span style="font-weight:600;color:{dot_color}">{html.escape(value)}</span>'
        f'</div>'
    )


def _sources_grid(citations: list) -> None:
    """P2 — 2×2 sources grid, always visible below the answer."""
    if not citations:
        return
    cards_html = ""
    for i, cite in enumerate(citations):
        excerpt_raw = cite.get("excerpt") or ""
        tokens = excerpt_raw.split()
        title = " ".join(tokens[:7]) + ("..." if len(tokens) > 7 else "")
        if not title:
            title = "Source"
        score = cite.get("score", 0)
        excerpt_short = html.escape(excerpt_raw[:200])
        chunk_id = cite.get("chunk_id", "")
        chunk_label = html.escape(f"chunk {chunk_id[:8]}…") if chunk_id else ""
        cards_html += (
            f'<div class="src-card">'
            f'<div class="src-num">{i + 1}</div>'
            f'<div class="src-title">{html.escape(title)}</div>'
            f'<div class="src-score">cross-encoder: {score:.3f}</div>'
            f'<div class="src-excerpt">{excerpt_short}</div>'
            f'<div class="src-chunk">{chunk_label}</div>'
            f'</div>'
        )
    st.markdown(
        f'<span class="sec-label" style="margin-top:16px">Sources</span>'
        f'<div class="src-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def _follow_up_chips(follow_ups: list[str], message_idx: int = 0) -> None:
    if not follow_ups:
        return
    st.markdown('<div class="followup-label">FOLLOW UP</div>', unsafe_allow_html=True)
    cols = st.columns(len(follow_ups))
    for i, (col, question) in enumerate(zip(cols, follow_ups)):
        with col:
            if st.button(
                question,
                key=f"followup_{message_idx}_{i}",
                use_container_width=True,
            ):
                st.session_state.prompt_prefill = question
                st.rerun()


def _retrieval_expander(retrieval_candidates: Optional[list], citations: list) -> None:
    """Retrieval candidates with rerank-score histogram and dual-score table."""
    cands = retrieval_candidates or []
    if not cands:
        return
    has_pre = any(c.get("pre_rerank_score") is not None for c in cands)
    with st.expander(f"Retrieval candidates — {len(cands)} chunks before reranking", expanded=False):
        if has_pre:
            st.caption(
                "Each candidate carries two scores on **different scales** — they are not "
                "directly comparable. **Retrieval (hybrid)** is FAISS+BM25 fusion (~0–1). "
                "**Rerank (cross-encoder)** is an unbounded logit (~−10 to +4). "
                "Rows are sorted by retrieval order; the ✅ marks chunks that survived "
                "into the final answer — scattered ✅ rows mean the cross-encoder reordered."
            )
        else:
            st.caption(
                "Rerank cross-encoder scores shown (unbounded logit). "
                "Pre-rerank hybrid scores unavailable for this response."
            )
        scores = [c["score"] for c in cands]
        df_hist = pd.DataFrame({"score": scores})
        hist = (
            alt.Chart(df_hist)
            .mark_bar(color=ACCENT, opacity=0.75)
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=12), title="Rerank score (cross-encoder logit)"),
                alt.Y("count():Q", title="Chunks"),
            )
            .properties(height=200, title="Rerank score distribution")
        )
        st.altair_chart(hist, use_container_width=True)
        if has_pre:
            sorted_cands = sorted(
                cands,
                key=lambda x: (x["pre_rerank_score"] if x.get("pre_rerank_score") is not None else -1e9),
                reverse=True,
            )
        else:
            sorted_cands = sorted(cands, key=lambda x: x["score"], reverse=True)
        df_cands = pd.DataFrame([
            {
                "pre_rerank_score": c.get("pre_rerank_score"),
                "score": round(c["score"], 4),
                "used": "✅" if c.get("used_in_answer", False) else "—",
                "source": Path(c.get("source", "")).name or c.get("source", ""),
                "excerpt": (c.get("excerpt") or "")[:120],
            }
            for c in sorted_cands
        ])
        st.dataframe(
            df_cands,
            use_container_width=True,
            hide_index=True,
            column_config={
                "pre_rerank_score": st.column_config.NumberColumn("Retrieval (hybrid)",     format="%.4f", width="small"),
                "score":            st.column_config.NumberColumn("Rerank (cross-encoder)", format="%.4f", width="small"),
                "used":             st.column_config.TextColumn(  "Used",                                  width="small"),
                "source":           st.column_config.TextColumn(  "Source",                                width="medium"),
                "excerpt":          st.column_config.TextColumn(  "Excerpt",                               width="large"),
            },
        )


def _clean_chunk_excerpt(text: str) -> str:
    """Neutralize raw markdown/HTML tokens leaked from source chunks so the
    excerpt renders as plain readable text. Strips <br> and other tags,
    drops markdown bold markers, replaces table pipes with spaces, collapses
    whitespace. Does NOT reformat the underlying table — just removes the
    markers so a viewer reads words, not syntax.

    The dangling-partial-tag strip handles the case where the upstream
    200-char truncation cuts a tag mid-name (e.g. "<br" with no closing
    ">"); without it, the <br> regex above can't match and a stray "<br"
    leaks into the rendered output.
    """
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[a-z][^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*$", "", text)
    text = text.replace("**", "").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cmp_stat_card(label: str, value: str) -> str:
    """Compact stat card for vertical stacking inside a comparison column.
    Same visual language as _stat_card (ACCENT_TINT bg, ACCENT_DARK label,
    bold value over uppercase small-caps label) but tighter padding and no
    fixed min-height so three stack cleanly in one column."""
    return (
        f'<div style="background:{ACCENT_TINT};border-radius:8px;'
        f'padding:12px 14px;margin-bottom:8px">'
        f'<div style="font-size:10.5px;font-weight:600;letter-spacing:0.5px;'
        f'text-transform:uppercase;color:{ACCENT_DARK};margin-bottom:4px">'
        f'{html.escape(label)}</div>'
        f'<div style="font-size:22px;font-weight:700;color:{TEXT_PRIMARY};'
        f'line-height:1.2;word-break:break-word">{html.escape(value)}</div>'
        f'</div>'
    )


def _comparison_column(corpus_name: str, summary: dict, doc_count: Optional[int]) -> None:
    """One column of the chunker comparison panel: stacked stat cards plus a
    cleaned top-chunk excerpt. Text is shown verbatim (markup neutralized)
    so a high rerank score on a tiny fragment stays visible next to the
    fragment itself — no winner highlight, no rank ordering."""
    st.markdown(
        f'<p style="font-size:13px;font-weight:600;color:{TEXT_PRIMARY};'
        f'margin:0 0 10px 0">{html.escape(_corpus_label(corpus_name))}</p>',
        unsafe_allow_html=True,
    )
    if "error" in summary:
        st.error(f"Failed: {summary['error']}")
        return

    top_score = summary.get("top_rerank_score")
    top_score_str = f"{top_score:.3f}" if top_score is not None else "—"
    lat_ms = summary.get("latency_retrieval") or 0.0
    lat_str = f"{lat_ms / 1000:.1f} s"

    st.markdown(
        _cmp_stat_card(
            "Chunks Indexed",
            str(doc_count) if doc_count is not None else "—",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _cmp_stat_card("Top Rerank Score (cross-encoder)", top_score_str),
        unsafe_allow_html=True,
    )
    st.markdown(
        _cmp_stat_card("Latency (retrieve + rerank)", lat_str),
        unsafe_allow_html=True,
    )

    top = summary.get("top_chunk") or {}
    excerpt = top.get("excerpt") or ""
    top_chunk_score = top.get("score")
    score_str = f"{top_chunk_score:.3f}" if top_chunk_score is not None else "—"
    cleaned = _clean_chunk_excerpt(excerpt)
    if cleaned:
        st.markdown(
            f'<div style="font-size:10.5px;font-weight:600;letter-spacing:0.5px;'
            f'text-transform:uppercase;color:{TEXT_MUTED};margin:14px 0 6px 0">'
            f'Top chunk · {html.escape(score_str)}</div>'
            f'<div style="border:0.5px solid #e5e7eb;border-radius:8px;'
            f'padding:10px 12px;background:#fafafa;min-height:160px">'
            f'<div style="font-size:12.5px;color:{TEXT_PRIMARY};line-height:1.55">'
            f'{html.escape(cleaned)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No top chunk returned.")


def _build_differs_html(cmp_data: dict, corpora_by_name: dict) -> Optional[str]:
    """Compute the body HTML for the 'what differs' summary row. Returns
    None when there are fewer than two successful columns to compare —
    in that case the caller skips the row entirely.

    The text is generated from the actual fetched data so it stays honest
    to whatever query ran. Two narratives:
      - Fragment pattern: the highest-scoring chunker's top excerpt is
        markedly shorter than the others' — name it, surface the tradeoff.
      - Otherwise: a factual numeric line — scores side by side, chunk-
        count range, and a reminder that cross-chunker scores aren't
        directly comparable.
    No winner, no rank, no failure highlight.
    """
    def _short(corpus_name: str) -> str:
        label = _corpus_label(corpus_name)
        return label.split(" — ", 1)[1] if " — " in label else label

    results = cmp_data.get("results") or {}
    rows = []
    for corpus_name in CHUNKER_COMPARISON_CORPORA:
        r = results.get(corpus_name) or {}
        if "error" in r:
            continue
        top = r.get("top_chunk") or {}
        info = corpora_by_name.get(corpus_name) or {}
        rows.append({
            "name":        corpus_name,
            "doc_count":   info.get("doc_count") or 0,
            "top_score":   r.get("top_rerank_score"),
            "top_excerpt": top.get("excerpt") or "",
        })

    if len(rows) < 2:
        return None

    scored = [r for r in rows if r["top_score"] is not None]
    if not scored:
        return (
            "Each chunker returned results; the substantive difference is in "
            "the top-chunk text above — rerank scores aren't directly "
            "comparable across chunkers of different granularity."
        )

    by_score = max(scored, key=lambda r: r["top_score"])
    others = [r for r in scored if r["name"] != by_score["name"]]

    # Boundary-alignment signal: does the top-scoring column's excerpt begin
    # mid-passage or at a clean boundary? Length-based detection is defeated
    # by the upstream 200-char cap that clips every excerpt to the same
    # length. A chunk that was carved out mid-passage almost always begins
    # with a lowercase letter (the back half of a word the chunk boundary
    # split), whereas a chunk aligned to a structural boundary opens with a
    # capital letter, a digit, a heading marker (#), or a table pipe (|).
    # IMPORTANT: this signal supports "begins mid-passage," NOT "is
    # incomplete." It correlates with the fragment hazard the comparison is
    # meant to surface, but the user-facing text must not assert a
    # completeness verdict from a first-character heuristic — let the
    # reader's eyes on the excerpts make that call.
    def _starts_mid_context(text: str) -> bool:
        text = text.lstrip()
        return bool(text) and text[0].islower()

    fragment_pattern = (
        len(others) >= 1
        and _starts_mid_context(by_score["top_excerpt"])
        and not all(_starts_mid_context(r["top_excerpt"]) for r in others)
    )

    if fragment_pattern:
        top_em = f"<strong>{html.escape(_short(by_score['name']))}</strong>"
        return (
            f"{top_em} scored highest "
            f"({by_score['top_score']:.2f}), but its top chunk begins "
            f"mid-passage rather than at a clean boundary. A high rerank "
            f"score on a tightly-matched fragment isn't the same as a "
            f"complete answer — compare the top-chunk text above to see "
            f"which actually answers the question."
        )

    score_bits = ", ".join(
        f"{html.escape(_short(r['name']))} {r['top_score']:.2f}"
        for r in scored
    )
    counts = sorted(r["doc_count"] for r in rows)
    top_name = html.escape(_short(by_score["name"]))
    return (
        f"Top rerank scores: {score_bits}. Chunk counts range "
        f"{counts[0]}–{counts[-1]}. The highest score ({top_name}, "
        f"{by_score['top_score']:.2f}) reflects retrieval granularity, not "
        f"completeness — different chunkers' scores aren't directly "
        f"comparable. Read each top-chunk text above to see which would "
        f"actually answer the question."
    )


def _comparison_differs_row(cmp_data: dict, corpora_by_name: dict) -> None:
    """Full-width 'what differs' summary row below the comparison columns.
    Honest narrative computed from the fetched data plus a standing one-line
    label clarifying what the 'hierarchical' column actually is."""
    differs_html = _build_differs_html(cmp_data, corpora_by_name)
    if differs_html is None:
        return
    hierarchical_note = (
        "Note: \"hierarchical\" here is fine-grained fixed-size child chunks "
        "(256-char windows), not true parent-child retrieval — the project "
        "label was kept for transparency."
    )
    st.markdown(
        f'<div style="margin:6px 0 0 0;padding:14px 16px;border-radius:8px;'
        f'background:{ACCENT_TINT};border:0.5px solid {ACCENT}">'
        f'<div style="font-size:10.5px;font-weight:600;letter-spacing:0.5px;'
        f'text-transform:uppercase;color:{ACCENT_DARK};margin-bottom:6px">'
        f'What differs</div>'
        f'<div style="font-size:13px;color:{TEXT_PRIMARY};line-height:1.55;'
        f'margin-bottom:8px">{differs_html}</div>'
        f'<div style="font-size:11.5px;color:{TEXT_SECONDARY};line-height:1.5;'
        f'font-style:italic">{hierarchical_note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _trace_expander(stage_timings: Optional[dict]) -> None:
    """Per-stage latency as horizontal stacked Plotly bar — diagnostic expander."""
    if not stage_timings:
        return
    stages = [
        ("router_ms",   "Router",   CORAL),
        ("retrieve_ms", "Retrieve", ACCENT),
        ("rerank_ms",   "Rerank",   TEAL),
        ("prompt_ms",   "Prompt",   AMBER),
        ("generate_ms", "Generate", "#e74c3c"),
    ]
    total = stage_timings.get("total_ms", 0)
    with st.expander(f"Stage timings — {total:.0f} ms total", expanded=False):
        fig = go.Figure()
        for key, label, color in stages:
            ms = stage_timings.get(key, 0)
            fig.add_trace(go.Bar(
                name=label, x=[ms], y=["Pipeline"], orientation="h",
                marker_color=color,
                text=f"{ms:.0f} ms",
                textposition="inside" if ms > 50 else "outside",
                hovertemplate=f"{label}: {ms:.1f} ms<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack", height=110,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)


def _self_rag_chips(self_rag_stats: Optional[dict]) -> None:
    """Per-claim verification chips — diagnostic expander."""
    if not self_rag_stats:
        return
    verified    = self_rag_stats.get("verified_claims") or []
    unsupported = self_rag_stats.get("unsupported_claims") or []
    faithfulness = self_rag_stats.get("faithfulness_score")
    regenerated  = self_rag_stats.get("regenerated", False)
    chips_html = (
        "".join(f'<span class="srag-v">✅ {html.escape(c[:70])}</span>' for c in verified)
        + "".join(f'<span class="srag-u">⚠️ {html.escape(c[:70])}</span>' for c in unsupported)
    )
    if chips_html or faithfulness is not None:
        with st.expander("Self-RAG verification", expanded=False):
            if faithfulness is not None:
                note = " · answer regenerated with additional context" if regenerated else ""
                st.caption(f"Faithfulness: {faithfulness:.2f}{note}")
            if chips_html:
                st.markdown(chips_html, unsafe_allow_html=True)


def _md_escape(text: str) -> str:
    """HTML-escape + backslash-escape `$` so Streamlit's markdown-it parser
    doesn't treat dollar-amount sequences (e.g. "$209,586 ... $201,183") as
    inline LaTeX math, which silently consumes any HTML tags falling between
    the paired dollar signs — visible as raw `</mark><sup ...>` text in the
    rendered answer. Latent until the corpus dropdown exposed
    financial-figure-heavy Apple answers."""
    return html.escape(text).replace("$", r"\$")


def _render_answer_with_spans(answer: str, attributed_spans: list[dict]) -> None:
    if not attributed_spans:
        # No HTML to inject, but `$` still needs escaping or markdown-it
        # will math-parse the answer body.
        st.markdown(answer.replace("$", r"\$"))
        return

    spans_sorted = sorted(attributed_spans, key=lambda s: s["start"])

    html_parts = []
    last_end = 0
    for span in spans_sorted:
        html_parts.append(_md_escape(answer[last_end:span["start"]]))
        html_parts.append(
            f'<mark style="background:#fff8c5; '
            f'border-bottom:2px solid #d4a017; '
            f'padding:0 2px; border-radius:2px;">'
            f'{_md_escape(span["text"])}'
            f'</mark>'
        )
        html_parts.append(
            f'<sup style="background:{ACCENT}; color:white; '
            f'padding:1px 5px; border-radius:3px; margin-left:2px; '
            f'font-size:11px; font-weight:600;">{span["source"]}</sup>'
        )
        last_end = span["end"]

    html_parts.append(_md_escape(answer[last_end:]))

    final_html = "".join(html_parts).replace("\n", "<br>")
    st.markdown(final_html, unsafe_allow_html=True)


def _abstention_card() -> None:
    st.markdown("""
<div class="abstention-card">
  <strong>Not enough information in the corpus</strong><br>
  <span style="font-size:13px;color:#6c757d">
    The indexed documents don't contain reliable information to answer this question.
    Try rephrasing, or ask about FiQA-2018 topics: IRAs, 401k, mortgages, taxes, investing.
  </span>
</div>""", unsafe_allow_html=True)


def _render_assistant_message(
    result: dict, show_retry: bool = False, message_idx: int = 0,
    corpus: str = "default",
) -> None:
    """Render a complete assistant turn (no routing strip; sources via P2 grid).

    `corpus` is the corpus the answer was generated against — used to gate
    the follow-up chip row, which the backend only generates correctly for
    the default (FiQA) corpus (see 2b stopgap below).
    """
    if "error" in result and result["error"]:
        st.markdown(f"""
<div class="error-card">
  <strong>Something went wrong</strong><br>
  <span style="font-size:13px;color:#6c757d">
    The backend returned an error. Please try again in a moment.
  </span>
</div>""", unsafe_allow_html=True)
        with st.expander("Technical details", expanded=False):
            st.code(result["error"])
        if show_retry:
            if st.button("Retry ↺", key=f"retry_{int(time.time() * 1000)}"):
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages.pop()
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        st.session_state.prompt_prefill = msg["content"]
                        break
                st.rerun()
        return

    citations            = result.get("citations") or []
    total_tokens         = result.get("total_tokens", 0)
    retrieval_candidates = result.get("retrieval_candidates")
    self_rag_stats       = result.get("self_rag_stats")
    stage_timings        = result.get("stage_timings")
    follow_up_questions  = result.get("follow_up_questions") or []
    answer               = result.get("answer", "")
    is_abstention        = not citations and total_tokens == 0

    # Header: name + verified badge
    verified_count = len((self_rag_stats or {}).get("verified_claims") or []) if self_rag_stats else 0
    badge_html = (
        f'<span class="verified-badge">Verified · {verified_count} sources</span>'
        if self_rag_stats and verified_count > 0 else ""
    )
    st.markdown(
        f'<p style="font-size:13px;font-weight:500;color:{TEXT_PRIMARY};margin-bottom:8px">'
        f'RAGCore{badge_html}</p>',
        unsafe_allow_html=True,
    )

    if is_abstention:
        _abstention_card()
    else:
        attributed = result.get("attributed_spans") or []
        _render_answer_with_spans(answer, attributed)
        _sources_grid(citations)
        # 2b stopgap pending backend fix: the follow-up generator returns
        # FiQA-domain questions regardless of which corpus was queried, so
        # for Apple corpora the chips are off-topic noise. The session-mode
        # exclusion is the same bug exposed on a different path — the LLM
        # generates finance-flavored follow-ups even when the answer came
        # from the user's uploaded doc, and the corpus dropdown typically
        # still reads "default" while the session header drives routing.
        # Suppress the entire block — header included — until the backend
        # conditions follow-ups on the actual answered content.
        if corpus == "default" and st.session_state.session_id is None:
            _follow_up_chips(follow_up_questions, message_idx)

    # Per-answer retrieval strategy that actually ran. Reflects Auto's
    # resolved choice, not the sidebar setting. Outside the abstention/
    # normal split because both cases pass through the router and the
    # field is populated in either case.
    strategy_label = _strategy_label(result.get("strategy_used"))
    if strategy_label:
        st.caption(f"retrieval: {strategy_label}")

    _retrieval_expander(retrieval_candidates, citations)
    _trace_expander(stage_timings)
    _self_rag_chips(self_rag_stats)


# ─── Stat card renderer ───────────────────────────────────────────────────────
def _stat_card(bg: str, label_color: str, label: str, value: str,
               sublabel: str = "", value_size: int = 28) -> str:
    """Returns the HTML for one stat card. Used by both eval-metric (default)
    and facts-only (Apple) layouts; the choice between layouts is made
    structurally upstream based on EVAL_METRICS membership."""
    sub = (
        f'<div style="font-size:11px;font-style:italic;color:{TEXT_SECONDARY};'
        f'margin-top:4px">{html.escape(sublabel)}</div>'
        if sublabel else ""
    )
    return (
        f'<div style="background:{bg};border-radius:10px;padding:16px;min-height:140px;'
        f'display:flex;flex-direction:column;justify-content:center">'
        f'<div style="font-size:11px;font-weight:600;letter-spacing:0.5px;'
        f'text-transform:uppercase;color:{label_color};margin-bottom:6px">'
        f'{html.escape(label)}</div>'
        f'<div style="font-size:{value_size}px;font-weight:700;color:{TEXT_PRIMARY};'
        f'word-break:break-word;line-height:1.2">{html.escape(value)}</div>'
        f'{sub}'
        f'</div>'
    )


# ─── Helpers for sidebar S4 ───────────────────────────────────────────────────
def _latest_result() -> Optional[dict]:
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and "result" in msg:
            return msg["result"]
    return None


def _repo_url() -> Optional[str]:
    try:
        url = subprocess.check_output(
            ["git", "config", "remote.origin.url"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):].removesuffix(".git")
        return url
    except Exception:
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # S1. Brand block
    st.markdown(
        f'<p style="font-size:18px;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:2px">RAGCore</p>'
        f'<p style="font-size:12px;color:{TEXT_SECONDARY};margin-top:0">'
        f'Retrieval-Augmented Generation demo</p>',
        unsafe_allow_html=True,
    )

    # S2. Backend status pill
    is_ok = _check_backend_cached(BACKEND_URL)
    st.session_state.backend_ok = is_ok
    if is_ok:
        st.markdown('<div class="status-pill status-ok">✓ Backend online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-err">✕ Backend offline</div>', unsafe_allow_html=True)

    # S2b. Corpus selector — populated from GET /corpora at app load. On
    # failure (backend offline, network error) we synthesize a single
    # "default" entry so the dropdown stays functional and ask_question
    # still has a valid corpus name to send.
    raw_corpora = _fetch_corpora(BACKEND_URL)
    corpora_ok = bool(raw_corpora)
    if not raw_corpora:
        raw_corpora = [{"name": "default", "doc_count": 0, "chunker": None, "source": None}]
    corpora_by_name = {c["name"]: c for c in raw_corpora}
    corpus_names_sorted = ["default"] + [n for n in corpora_by_name if n != "default"]
    corpus_names_sorted = [n for n in corpus_names_sorted if n in corpora_by_name]

    st.markdown('<span class="sec-label" style="margin-top:8px">Corpus</span>', unsafe_allow_html=True)
    selected_corpus = st.selectbox(
        "corpus",
        options=corpus_names_sorted,
        format_func=_corpus_label,
        label_visibility="collapsed",
    )
    if not corpora_ok:
        st.caption("⚠ Could not load corpus list — using default")
    if st.session_state.session_id:
        # When a session is active, the backend ignores this field
        # (Commit 5 routing) — make that visible rather than letting the
        # user wonder why flipping corpora doesn't change behavior.
        st.caption("Ignored while a session is active — uploaded docs take precedence.")
    selected_info = corpora_by_name.get(selected_corpus, {})

    st.divider()

    # S3. Retrieval strategy
    st.markdown('<span class="sec-label">Retrieval Strategy</span>', unsafe_allow_html=True)
    strategy = st.radio(
        "strategy",
        options=["Auto", "Hybrid", "Semantic", "Keyword"],
        index=0,
        label_visibility="collapsed",
    ).lower()

    st.markdown('<span class="sec-label" style="margin-top:10px">Verification</span>', unsafe_allow_html=True)
    st.checkbox("Hallucination verifier", key="verify_claims")
    st.caption("Per-claim grounding check (~3x slower)")
    # Chunker-comparison view toggle. Gated so the ~30s 3-way fetch doesn't
    # fire on every page load. Lives next to the verifier as a sibling
    # diagnostic instrument.
    st.checkbox("Comparison", value=False, key="cmp_on")
    st.caption("Run the same query against three chunkers, side by side")

    st.divider()

    # S4. Pipeline
    latest          = _latest_result()
    stage_timings_s = latest.get("stage_timings")        if latest else None
    latency_ms_s    = latest.get("latency_ms")           if latest else None
    total_tokens_s  = latest.get("total_tokens")         if latest else None
    ret_cands_s     = latest.get("retrieval_candidates") if latest else None

    total_val    = f'{stage_timings_s["total_ms"]:.0f}ms'          if stage_timings_s else "—"
    router_val   = f'{stage_timings_s.get("router_ms",   0):.0f}ms' if stage_timings_s else "—"
    retrieve_val = f'{stage_timings_s.get("retrieve_ms", 0):.0f}ms' if stage_timings_s else "—"
    rerank_val   = f'{stage_timings_s.get("rerank_ms",   0):.0f}ms' if stage_timings_s else "—"
    generate_val = f'{stage_timings_s.get("generate_ms", 0):.0f}ms' if stage_timings_s else "—"
    latency_val  = f"{latency_ms_s:.0f}ms" if latency_ms_s   is not None else "—"
    tokens_val   = str(total_tokens_s)      if total_tokens_s is not None else "—"

    st.markdown('<span class="sec-label">Pipeline</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pipeline-total">Total: {total_val}</div>',
        unsafe_allow_html=True,
    )

    rows_html  = _pipeline_row(CORAL,  "Router",   router_val)
    rows_html += _pipeline_row(ACCENT, "Retrieve", retrieve_val)
    rows_html += _pipeline_row(TEAL,   "Rerank",   rerank_val)
    rows_html += _pipeline_row(AMBER,  "Generate", generate_val)
    rows_html += _pipeline_row(ACCENT, "Latency",  latency_val)
    rows_html += _pipeline_row(TEAL,   "Tokens",   tokens_val)

    level = compute_confidence(ret_cands_s) if ret_cands_s else "unknown"
    if level != "unknown":
        dot_color = SUCCESS_DARK if level == "high" else (ERROR_DARK if level == "low" else AMBER)
        badge = _confidence_badge_html(level)
        rows_html += (
            f'<div class="pipeline-row">'
            f'<span class="pipeline-dot" style="background:{dot_color}"></span>'
            f'<span class="pipeline-label">Confidence</span>'
            f'{badge}'
            f'</div>'
        )
    else:
        rows_html += _pipeline_row(ACCENT, "Confidence", "—")

    st.markdown(rows_html, unsafe_allow_html=True)
    if latest is None:
        st.markdown(
            f'<p style="font-size:11px;color:{TEXT_MUTED};font-style:italic;margin-top:8px">'
            f'Ask a question to see live metrics.</p>',
            unsafe_allow_html=True,
        )
    st.divider()

    # S5. Your documents — file uploader + active session indicator
    st.markdown('<span class="sec-label">Your documents</span>', unsafe_allow_html=True)
    # key includes uploader_nonce — incrementing it forces a fresh widget
    # identity, which is how the "Start new session" button clears a
    # retained UploadedFile without relying on pop()-ing widget state.
    uploaded = st.file_uploader(
        "Upload PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_nonce}",
    )
    # CRITICAL: file_uploader retains its value across reruns. _handle_upload
    # dedups by file_id, so this call is safe to fire on every rerun — only
    # the first POSTs.
    if uploaded is not None:
        _handle_upload(uploaded)

    if st.session_state.session_id:
        prefix = st.session_state.session_id[:6]
        n_files = len(st.session_state.uploaded_file_keys)
        st.caption(f"Session: {prefix}… · {n_files} file(s)")

        # CONTRACT 2: "Start new session" affordance. Promoted to primary
        # styling after a 409 so it's the obvious next step.
        btn_type = "primary" if st.session_state.upload_409_active else "secondary"
        if st.button("Start new session", use_container_width=True, type=btn_type):
            st.session_state.session_id = None
            st.session_state.uploaded_file_keys = set()
            st.session_state.upload_409_active = False
            # Bump nonce so the file_uploader renders with a fresh key on
            # the next rerun → no retained UploadedFile → _handle_upload
            # does NOT fire → no auto-mint of a new session. Without this
            # bump, the retained file would re-upload immediately after the
            # rerun and resurrect the session we just cleared.
            st.session_state.uploader_nonce += 1
            # Don't clear messages — past chat history remains visible.
            st.rerun()
    else:
        st.caption("No session — upload a file to start one.")

    st.divider()

    # S6. Clear chat button
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # S7. Version marker (temporary — removed after cloud deploy confirmed)

    # S8. About expander
    with st.expander("About"):
        if selected_corpus == "default":
            corpus_about = "FiQA-2018 financial Q&A corpus (380 chunks). "
        else:
            chunk_n = selected_info.get("doc_count") or 0
            corpus_about = f"{_corpus_label(selected_corpus)} ({chunk_n} chunks). "
        st.markdown(
            "Llama 3.3 70B via Groq for generation. FAISS dense + BM25 sparse hybrid retrieval. "
            "ms-marco cross-encoder reranking. " + corpus_about +
            "Self-RAG verification when enabled."
        )
        repo = _repo_url()
        if repo:
            st.markdown(f"[GitHub repo]({repo})")
        st.markdown(f"[API docs]({BACKEND_URL}/docs)")


# ─── Main area ────────────────────────────────────────────────────────────────

# M1. Hero block
if selected_corpus == "default":
    hero_corpus_sentence = (
        f'Built for grounded answers, not guesswork. Explore questions in personal '
        f'finance using the <a href="https://huggingface.co/datasets/vibrantlabsai/fiqa" '
        f'target="_blank" style="color:{ACCENT};text-decoration:underline">FiQA-2018</a> dataset.'
    )
else:
    src_raw = selected_info.get("source") or ""
    src_phrase = _corpus_source_label(selected_corpus, src_raw)
    # Lead with the pretty source label only. The dropdown selection and
    # the Source/Chunker stat cards already show the chunker variant, so
    # the previous "<dropdown_label> (<source_label>)" pattern duplicated
    # the doc name (e.g. "Apple 10-K — fixed chunking (Apple 10-K (2025))")
    # and wrapped awkwardly.
    hero_corpus_sentence = (
        f'Built for grounded answers, not guesswork. Now querying the '
        f'<strong>{html.escape(src_phrase)}</strong>.'
    )

st.markdown(f"""
<div style="text-align:center;margin-top:48px;margin-bottom:32px">
  <h1 style="font-size:36px;font-weight:600;letter-spacing:-0.02em;
             color:{TEXT_PRIMARY};margin-bottom:0;line-height:1.2">
    Ask hard questions.
  </h1>
  <h1 style="font-size:36px;font-weight:600;letter-spacing:-0.02em;
             color:{ACCENT};margin:0 0 20px;line-height:1.2">
    Get cited answers.
  </h1>
  <p style="font-size:15px;color:{TEXT_SECONDARY};line-height:1.5;
            max-width:540px;margin:0 auto">
    A production-grade RAG system combining query routing, hybrid retrieval,
    cross-encoder reranking, inline citations, and hallucination verification.
    {hero_corpus_sentence}
  </p>
</div>
""", unsafe_allow_html=True)

eval_data = EVAL_METRICS.get(selected_corpus)
if eval_data:
    benchmark_subtitle = (
        f"Aggregate benchmark on the {eval_data['label']} corpus. "
        f"Per-query metrics in sidebar after each response."
    )
else:
    benchmark_subtitle = (
        "No aggregate benchmark for this corpus. "
        "Per-query metrics in sidebar after each response."
    )
st.markdown(
    f'<p style="text-align:center;font-size:11px;font-style:italic;color:{TEXT_MUTED};'
    f'max-width:540px;margin:12px auto 16px">{benchmark_subtitle}</p>',
    unsafe_allow_html=True,
)

# M2. Stat cards — eval metrics for corpora in EVAL_METRICS (FiQA only today),
# facts-only (doc_count / chunker / source) for everything else. Suppressing
# HIT@5 / MRR when there's no real eval number keeps the demo honest about
# what we've actually benchmarked.
_sc1, _sc2, _sc3 = st.columns(3, gap="small")
if eval_data:
    # Stat values from evaluation/results/basic_fiqa_2026-04-26.json
    # (50-query FiQA benchmark, baseline strategy, run 2026-04-26).
    with _sc1:
        st.markdown(
            _stat_card(ACCENT_TINT, ACCENT_DARK, "Chunks Indexed", str(eval_data["chunks"])),
            unsafe_allow_html=True,
        )
    with _sc2:
        st.markdown(
            _stat_card(
                TEAL_TINT, "#176363", "HIT@5", f"{eval_data['hit_at_5']:.2f}",
                "fraction of queries with a correct chunk in top 5",
            ),
            unsafe_allow_html=True,
        )
    with _sc3:
        st.markdown(
            _stat_card(
                AMBER_TINT, "#8a6520", "MRR", f"{eval_data['mrr']:.2f}",
                "mean reciprocal rank — higher means correct chunks rank earlier",
            ),
            unsafe_allow_html=True,
        )
else:
    doc_count = selected_info.get("doc_count") or 0
    chunker   = selected_info.get("chunker") or "—"
    src_raw   = selected_info.get("source") or ""
    source_label = _corpus_source_label(selected_corpus, src_raw)
    with _sc1:
        st.markdown(
            _stat_card(ACCENT_TINT, ACCENT_DARK, "Chunks Indexed", str(doc_count)),
            unsafe_allow_html=True,
        )
    with _sc2:
        st.markdown(
            _stat_card(
                TEAL_TINT, "#176363", "Chunker", chunker,
                "chunking strategy used at ingest",
            ),
            unsafe_allow_html=True,
        )
    with _sc3:
        st.markdown(
            _stat_card(
                AMBER_TINT, "#8a6520", "Source", source_label,
                "input document", value_size=18,
            ),
            unsafe_allow_html=True,
        )

# M3 + M4. Starter prompts — corpus-aware FiQA/Apple curated lists.
# Suppressed entirely in session mode: PROMPT_STARTERS has no "session" key
# and the FiQA fallback would prime users with finance questions above a
# chat that's actually querying their uploaded doc. Same justification as
# the follow-up chip suppression below — static FiQA noise on a non-FiQA
# session. The label and cards are wrapped together so we don't leave an
# orphan "Try asking" heading with no buttons beneath it.
if st.session_state.session_id is None:
    # M3. Try asking label
    st.markdown(
        f'<p style="font-size:11px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;'
        f'color:{TEXT_MUTED};margin-top:32px;margin-bottom:12px">'
        f'Try asking — or write your own</p>',
        unsafe_allow_html=True,
    )

    # M4. Prompt cards — corpus-aware. Falls back to the FiQA set if the
    # selected corpus has no curated starters. Button keys include the corpus
    # name to avoid collisions when categories repeat across corpora
    # (e.g. REVENUE for both apple_10k_* and apple_financial_csvs).
    active_starters = PROMPT_STARTERS.get(selected_corpus, PROMPT_STARTERS["default"])
    _pc1, _pc2, _pc3 = st.columns(3, gap="small")
    for col, (cat, color, question) in zip([_pc1, _pc2, _pc3], active_starters):
        with col:
            st.markdown(
                f'<span style="color:{color};font-size:10px;font-weight:600;'
                f'letter-spacing:0.5px;text-transform:uppercase;'
                f'display:block;margin-bottom:4px">{cat}</span>',
                unsafe_allow_html=True,
            )
            if st.button(question, key=f"pc_{selected_corpus}_{cat.lower().replace(' ', '_')}",
                         use_container_width=True):
                st.session_state.prompt_prefill = question
                st.rerun()

# M5. Stack subtitle
st.markdown(
    f'<p style="text-align:center;font-size:11px;color:{TEXT_MUTED};margin-top:24px">'
    f'Powered by Llama 3.3 70B via Groq · {_corpus_vector_store(selected_corpus)} · '
    f'ms-marco cross-encoder</p>',
    unsafe_allow_html=True,
)

# ─── Chat history replay ──────────────────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=ASST_AVATAR):
            # Use the corpus the answer was generated against, not the
            # currently selected one — that way switching corpora after
            # asking questions doesn't retroactively toggle follow-ups
            # on past Apple answers.
            _render_assistant_message(
                message.get("result", {}),
                show_retry=False, message_idx=i,
                corpus=message.get("corpus", "default"),
            )

# ─── Chat input + query dispatch ─────────────────────────────────────────────
prefill = st.session_state.get("prompt_prefill", "")
if prefill:
    del st.session_state["prompt_prefill"]

# Mode indicator: visible signal of whether the next query routes to the
# user's uploaded docs (session header) or to the public corpus dropdown.
# Without this, a user who uploaded a file and then flipped the corpus
# dropdown would have no signal that their dropdown change is ignored.
if st.session_state.session_id:
    prefix = st.session_state.session_id[:6]
    st.markdown(
        f'<div style="background:{ACCENT_TINT};border:1px solid {ACCENT};'
        f'border-radius:8px;padding:8px 14px;margin:0 0 8px 0;font-size:13px;'
        f'color:{TEXT_PRIMARY}">'
        f'<strong>Querying your uploaded document(s)</strong> '
        f'<span style="color:{TEXT_SECONDARY}">· session {prefix}…</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.caption(f"Querying: {_corpus_label(selected_corpus)}")

placeholder = (
    "Ask a follow-up..." if st.session_state.messages
    else "Ask about IRAs, 401k, mortgages, investing…"
)
typed = st.chat_input(placeholder)
active_prompt = prefill or typed

if active_prompt:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(active_prompt)
    st.session_state.messages.append({"role": "user", "content": active_prompt})

    with st.chat_message("assistant", avatar=ASST_AVATAR):
        if not st.session_state.backend_ok:
            result = {
                "error": (
                    f"Backend not reachable at {BACKEND_URL}. "
                    "If the HF Space is waking from sleep, wait ~30 s and retry."
                )
            }
            status = 0
        else:
            verify = st.session_state.get("verify_claims", False)
            spinner_text = (
                "Generating verified answer (slower with per-claim check)…"
                if verify else
                "Searching corpus and generating answer…"
            )
            with st.spinner(spinner_text):
                status, body = ask_question(
                    active_prompt, strategy,
                    verify_claims=verify, corpus=selected_corpus,
                )
            if status == 200:
                result = body
            elif status == 404 and st.session_state.session_id:
                # CRITICAL — session expired / unknown on the backend. Do
                # NOT silently re-issue this query against the public
                # corpus: the user asked about their private doc, and
                # answering from FiQA would violate the isolation trust
                # we built in Commit 5. Clear the session so the NEXT
                # query is public-mode, warn this turn, then stop.
                st.session_state.session_id = None
                st.session_state.uploaded_file_keys = set()
                # Bump nonce — same rationale as the "Start new session"
                # button. Without this, the file_uploader's retained
                # UploadedFile would auto-reupload on the next rerun and
                # silently mint a fresh session. We want the user to
                # deliberately re-upload, not have it happen invisibly.
                st.session_state.uploader_nonce += 1
                st.warning(
                    "Your session expired on the server. Re-upload your "
                    "document(s) to continue, or ask a new question against "
                    "the public corpus."
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "[Session expired — please re-upload to continue.]",
                    "result": {"error": "session_expired"},
                    "corpus": "session",
                })
                st.rerun()
                # st.rerun() never returns; the guard below is defensive.
                result = {"error": "session_expired"}
            else:
                detail = body.get("detail") or body.get("error") or "no detail"
                result = {"error": f"Query failed ({status}): {detail}"}

        _render_assistant_message(
            result, show_retry=True,
            message_idx=len(st.session_state.messages),
            corpus=selected_corpus,
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", result.get("error", "")),
            "result": result,
            "corpus": selected_corpus,
        })
        st.rerun()


# ─── Chunker comparison panel (part 2 of arc B) ──────────────────────────────
# One bordered panel containing: a title bar with a three-bar columns glyph,
# a one-line caption, the query input, the Run button, and three side-by-side
# columns (same query, hybrid strategy held constant, only the chunker varies).
# Result cached in session_state so unrelated reruns don't refire the fetch.
# Part 3 will add the "what differs" row + honesty panel + full ranked list.
if st.session_state.get("cmp_on", False):
    st.divider()
    with st.container(border=True):
        # Title bar: three-bar columns glyph + label + one-line caption.
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:2px 0 4px 0">'
            f'<span style="display:inline-flex;gap:2px;align-items:center;margin-right:10px">'
            f'<span style="width:3px;height:13px;background:{ACCENT};border-radius:1px"></span>'
            f'<span style="width:3px;height:13px;background:{ACCENT};border-radius:1px"></span>'
            f'<span style="width:3px;height:13px;background:{ACCENT};border-radius:1px"></span>'
            f'</span>'
            f'<span style="font-size:15px;font-weight:600;color:{TEXT_PRIMARY}">'
            f'Chunker comparison</span>'
            f'</div>'
            f'<p style="font-size:12px;color:{TEXT_SECONDARY};margin:0 0 14px 0">'
            f'Same query · hybrid strategy held constant · only the chunker varies</p>',
            unsafe_allow_html=True,
        )

        cmp_q = st.text_input(
            "Query",
            value="What were Apple's net sales by product category?",
            key="cmp_q",
        )
        if st.button("Run comparison", key="cmp_run"):
            with st.spinner("Running three sequential queries (~30-45s)…"):
                st.session_state["cmp_data"] = fetch_chunker_comparison(
                    cmp_q, BACKEND_URL, API_KEY,
                )

        cmp_data = st.session_state.get("cmp_data")
        if cmp_data:
            results = cmp_data.get("results") or {}
            cols = st.columns(len(CHUNKER_COMPARISON_CORPORA))
            for col, corpus_name in zip(cols, CHUNKER_COMPARISON_CORPORA):
                with col:
                    summary = results.get(corpus_name) or {"error": "no response"}
                    info = corpora_by_name.get(corpus_name) or {}
                    _comparison_column(corpus_name, summary, info.get("doc_count"))
            # "What differs" summary row — full-width inside the panel,
            # honest narrative computed from the data, with the standing
            # hierarchical-label caveat.
            _comparison_differs_row(cmp_data, corpora_by_name)
            # Bottom breathing room so the row doesn't run into the panel's
            # bottom border.
            st.markdown(
                '<div style="height:12px"></div>',
                unsafe_allow_html=True,
            )

    # Latency honesty — caption sits outside the panel so it reads as a
    # footnote on the instrument, not part of it.
    st.caption(
        "Latency varies per run with server load and cold-start state — "
        "it's not a fixed property of the chunker."
    )
