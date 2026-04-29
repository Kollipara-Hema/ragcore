"""
RAGCore Streamlit Demo — visual rebuild.
Build file: new_app.py. Swap to app.py after approval.
"""
from __future__ import annotations

import base64
import html
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

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
#MainMenu {{visibility: hidden;}}
footer    {{visibility: hidden;}}

/* ── Status pill ── */
.status-pill {{
    display: inline-block; padding: 8px 12px; border-radius: 8px;
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
    cursor: pointer; display: block; margin-bottom: 0;
}}
[data-testid="stRadio"] > div > label + label {{ margin-top: 1px; }}
[data-testid="stRadio"] label:has(input:checked) {{
    background: {ACCENT_TINT}; border-color: {ACCENT};
}}

/* ── Pipeline rows ── */
.pipeline-total {{ font-size: 12px; color: {TEXT_PRIMARY}; margin-bottom: 4px; font-weight: 700; }}
.pipeline-row {{
    display: flex; align-items: center; padding: 4px 0;
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
.src-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 8px 0; }}
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
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("RAGCORE_BACKEND_URL", "http://localhost:8000")

PROMPT_CARDS = [
    ("MORTGAGES",     CORAL,  "Should I pay off my mortgage early or invest the extra cash?"),
    ("RETIREMENT",    ACCENT, "What are the contribution limits for a Roth IRA vs Traditional IRA?"),
    ("EMPLOYER 401K", TEAL,   "How does a 401k employer match work and how should I maximize it?"),
]

# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = None
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "prompt_prefill" not in st.session_state:
    st.session_state.prompt_prefill = ""

# ─── Backend helpers (verbatim) ───────────────────────────────────────────────
@st.cache_data(ttl=30)
def _check_backend_cached(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ask_question(query: str, strategy: str) -> dict:
    try:
        payload = {
            "query": query,
            "strategy_override": strategy if strategy != "auto" else None,
        }
        r = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=90,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def upload_document(file) -> dict:
    try:
        files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
        r = requests.post(f"{BACKEND_URL}/ingest/file", files=files, timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── compute_confidence (verbatim from commit 98f0f8e) ───────────────────────
def compute_confidence(retrieval_candidates: Optional[list]) -> str:
    """
    Confidence = f(score gap between top-1 and median pre-rerank candidate).

    Thresholds calibrated 2026-04-29 against 10 queries on deployed backend
    (https://ragcore-api.onrender.com) using live pre-rerank hybrid
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


def _retrieval_expander(retrieval_candidates: Optional[list], citations: list) -> None:
    """Pre-rerank candidates with Altair histogram — diagnostic expander."""
    cands = retrieval_candidates or []
    if not cands:
        return
    with st.expander(f"Retrieval candidates — {len(cands)} chunks before reranking", expanded=False):
        st.caption(
            "Scores are pre-rerank hybrid (FAISS dense + BM25 sparse). "
            "Citation scores above are post-rerank cross-encoder — different scale."
        )
        scores = [c["score"] for c in cands]
        df_hist = pd.DataFrame({"score": scores})
        hist = (
            alt.Chart(df_hist)
            .mark_bar(color=ACCENT, opacity=0.75)
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=12), title="Hybrid score"),
                alt.Y("count():Q", title="Chunks"),
            )
            .properties(height=200, title="Score distribution (top-K pre-rerank)")
        )
        st.altair_chart(hist, use_container_width=True)
        df_cands = pd.DataFrame([
            {
                "score": round(c["score"], 4),
                "used": "✅" if c.get("used_in_answer", False) else "—",
                "source": Path(c.get("source", "")).name or c.get("source", ""),
                "excerpt": (c.get("excerpt") or "")[:120],
            }
            for c in sorted(cands, key=lambda x: x["score"], reverse=True)
        ])
        st.dataframe(
            df_cands,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score":   st.column_config.NumberColumn("Score",   format="%.4f", width="small"),
                "used":    st.column_config.TextColumn("Used",      width="small"),
                "source":  st.column_config.TextColumn("Source",    width="medium"),
                "excerpt": st.column_config.TextColumn("Excerpt",   width="large"),
            },
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


def _abstention_card() -> None:
    st.markdown("""
<div class="abstention-card">
  <strong>Not enough information in the corpus</strong><br>
  <span style="font-size:13px;color:#6c757d">
    The indexed documents don't contain reliable information to answer this question.
    Try rephrasing, or ask about FiQA-2018 topics: IRAs, 401k, mortgages, taxes, investing.
  </span>
</div>""", unsafe_allow_html=True)


def _render_assistant_message(result: dict, show_retry: bool = False) -> None:
    """Render a complete assistant turn (no routing strip; sources via P2 grid)."""
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
        st.markdown(answer)
        _sources_grid(citations)

    _retrieval_expander(retrieval_candidates, citations)
    _trace_expander(stage_timings)
    _self_rag_chips(self_rag_stats)


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

    st.divider()

    # S3. Retrieval strategy
    st.markdown('<span class="sec-label">Retrieval Strategy</span>', unsafe_allow_html=True)
    strategy = st.radio(
        "strategy",
        options=["Auto", "Hybrid", "Semantic", "Keyword"],
        index=0,
        label_visibility="collapsed",
    ).lower()

    st.divider()

    # S4. Pipeline (only when an active response exists)
    latest = _latest_result()
    if latest:
        stage_timings_s   = latest.get("stage_timings")
        latency_ms_s      = latest.get("latency_ms") or 0
        total_tokens_s    = latest.get("total_tokens") or 0
        ret_cands_s       = latest.get("retrieval_candidates")

        st.markdown('<span class="sec-label">Pipeline</span>', unsafe_allow_html=True)

        if stage_timings_s:
            total_ms_s = stage_timings_s.get("total_ms", latency_ms_s)
            st.markdown(
                f'<div class="pipeline-total">Total: {total_ms_s:.0f}ms</div>',
                unsafe_allow_html=True,
            )

        rows_html = ""
        if stage_timings_s:
            rows_html += _pipeline_row(CORAL,  "Router",   f'{stage_timings_s.get("router_ms",   0):.0f}ms')
            rows_html += _pipeline_row(ACCENT, "Retrieve", f'{stage_timings_s.get("retrieve_ms", 0):.0f}ms')
            rows_html += _pipeline_row(TEAL,   "Rerank",   f'{stage_timings_s.get("rerank_ms",   0):.0f}ms')
            rows_html += _pipeline_row(AMBER,  "Generate", f'{stage_timings_s.get("generate_ms", 0):.0f}ms')

        rows_html += _pipeline_row(ACCENT, "Latency", f"{latency_ms_s:.0f}ms")
        rows_html += _pipeline_row(TEAL,   "Tokens",  str(total_tokens_s))

        # Row 7: Confidence (only when retrieval_candidates available)
        if ret_cands_s:
            level = compute_confidence(ret_cands_s)
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

        st.markdown(rows_html, unsafe_allow_html=True)
        st.divider()

    # S5. Doc upload note
    st.markdown(
        f'<p style="font-size:11px;color:{TEXT_MUTED};font-style:italic;margin-top:16px">'
        f'Document upload available when running locally — see README.</p>',
        unsafe_allow_html=True,
    )

    # S6. Clear chat button
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # S7. About expander
    with st.expander("About"):
        st.markdown(
            "Llama 3.3 70B via Groq for generation. FAISS dense + BM25 sparse hybrid retrieval. "
            "ms-marco cross-encoder reranking. FiQA-2018 financial Q&A corpus (380 chunks). "
            "Self-RAG verification when enabled."
        )
        repo = _repo_url()
        if repo:
            st.markdown(f"[GitHub repo]({repo})")
        st.markdown("[API docs](https://ragcore-api.onrender.com/docs)")


# ─── Main area ────────────────────────────────────────────────────────────────

# M1. Hero block
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
    Built for grounded answers, not guesswork. Explore questions in personal
    finance using the <a href="https://huggingface.co/datasets/vibrantlabsai/fiqa"
    target="_blank" style="color:{ACCENT};text-decoration:underline">FiQA-2018</a> dataset.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<p style="text-align:center;font-size:11px;font-style:italic;color:{TEXT_MUTED};'
    f'max-width:540px;margin:12px auto 16px">'
    f'Aggregate benchmark on the FiQA-2018 corpus. Per-query metrics in sidebar after each response.</p>',
    unsafe_allow_html=True,
)

# M2. Stat cards
# Stat values from evaluation/results/basic_fiqa_2026-04-26.json
# (50-query FiQA benchmark, baseline strategy, run 2026-04-26).
# Chunk count from faiss_metadata.pkl.
_sc1, _sc2, _sc3 = st.columns(3, gap="small")
with _sc1:
    st.markdown(f"""
<div style="background:{ACCENT_TINT};border-radius:10px;padding:16px;min-height:140px;
            display:flex;flex-direction:column;justify-content:center">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;
              color:{ACCENT_DARK};margin-bottom:6px">Chunks Indexed</div>
  <div style="font-size:28px;font-weight:700;color:{TEXT_PRIMARY}">380</div>
</div>""", unsafe_allow_html=True)
with _sc2:
    st.markdown(f"""
<div style="background:{TEAL_TINT};border-radius:10px;padding:16px;min-height:140px;
            display:flex;flex-direction:column;justify-content:center">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;
              color:#176363;margin-bottom:6px">HIT@5</div>
  <div style="font-size:28px;font-weight:700;color:{TEXT_PRIMARY}">0.92</div>
  <div style="font-size:11px;font-style:italic;color:{TEXT_SECONDARY};margin-top:4px">
    fraction of queries with a correct chunk in top 5</div>
</div>""", unsafe_allow_html=True)
with _sc3:
    st.markdown(f"""
<div style="background:{AMBER_TINT};border-radius:10px;padding:16px;min-height:140px;
            display:flex;flex-direction:column;justify-content:center">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;
              color:#8a6520;margin-bottom:6px">MRR</div>
  <div style="font-size:28px;font-weight:700;color:{TEXT_PRIMARY}">0.86</div>
  <div style="font-size:11px;font-style:italic;color:{TEXT_SECONDARY};margin-top:4px">
    mean reciprocal rank — higher means correct chunks rank earlier</div>
</div>""", unsafe_allow_html=True)

# M3. Try asking label
st.markdown(
    f'<p style="font-size:11px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;'
    f'color:{TEXT_MUTED};margin-top:32px;margin-bottom:12px">'
    f'Try asking — or write your own</p>',
    unsafe_allow_html=True,
)

# M4. Prompt cards
_pc1, _pc2, _pc3 = st.columns(3, gap="small")
for col, (cat, color, question) in zip([_pc1, _pc2, _pc3], PROMPT_CARDS):
    with col:
        st.markdown(
            f'<span style="color:{color};font-size:10px;font-weight:600;'
            f'letter-spacing:0.5px;text-transform:uppercase;'
            f'display:block;margin-bottom:4px">{cat}</span>',
            unsafe_allow_html=True,
        )
        if st.button(question, key=f"pc_{cat.lower().replace(' ', '_')}",
                     use_container_width=True):
            st.session_state.prompt_prefill = question
            st.rerun()

# M5. Stack subtitle
st.markdown(
    f'<p style="text-align:center;font-size:11px;color:{TEXT_MUTED};margin-top:24px">'
    f'Powered by Llama 3.3 70B via Groq · FAISS · ms-marco cross-encoder</p>',
    unsafe_allow_html=True,
)

# ─── Chat history replay ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=ASST_AVATAR):
            _render_assistant_message(message.get("result", {}), show_retry=False)

# ─── Chat input + query dispatch ─────────────────────────────────────────────
prefill = st.session_state.get("prompt_prefill", "")
if prefill:
    del st.session_state["prompt_prefill"]

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
                    "If this is the Render demo, wait ~30 s and retry."
                )
            }
        else:
            with st.spinner("Searching corpus and generating answer…"):
                result = ask_question(active_prompt, strategy)

        _render_assistant_message(result, show_retry=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", result.get("error", "")),
            "result": result,
        })
        st.rerun()
