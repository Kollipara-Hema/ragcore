"""
RAGCore Streamlit Demo — single chat surface with full pipeline observability.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RAGCore",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid='stSidebar'] {
    background-color: #ffffff;
    border-right: 1px solid #e9ecef;
}
[data-testid='stChatMessage'] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    padding: 4px;
    margin-bottom: 8px;
}
.citation-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-left: 3px solid #4a6fa5;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 13px;
}
.citation-source {
    color: #4a6fa5;
    font-weight: 600;
    font-size: 12px;
}
.citation-excerpt {
    color: #6c757d;
    font-size: 12px;
    margin-top: 4px;
}
.stat-badge {
    display: inline-block;
    background: #e8f4fd;
    color: #1a6fa8;
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 12px;
    margin-right: 6px;
}
.strategy-tag {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
}
.routing-strip {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    color: #495057;
    margin-bottom: 6px;
}
.confidence-high {
    display: inline-block;
    background: #d4edda;
    color: #155724;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.confidence-medium {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.confidence-low {
    display: inline-block;
    background: #f8d7da;
    color: #721c24;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.abstention-card {
    background: #fff8e1;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    font-size: 14px;
}
.error-card {
    background: #fff5f5;
    border: 1px solid #feb2b2;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    font-size: 14px;
}
.self-rag-verified {
    display: inline-block;
    background: #d4edda;
    color: #155724;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 2px 3px;
}
.self-rag-unverified {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 2px 3px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("RAGCORE_BACKEND_URL", "http://localhost:8000")

SAMPLE_PROMPTS = [
    "Should I pay off my mortgage early or invest the extra cash?",
    "What are the contribution limits for a Roth IRA vs Traditional IRA?",
    "How does a 401k employer match work and how should I maximize it?",
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


# ─── Backend helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _check_backend_cached(url: str) -> bool:
    """Health-check the backend; result cached 30 s to avoid re-checking on every rerun."""
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


# ─── Confidence badge (Pattern M) ────────────────────────────────────────────

def compute_confidence(retrieval_candidates: Optional[list]) -> str:
    """
    Confidence = f(score gap between top-1 and median pre-rerank candidate).

    Thresholds calibrated 2026-04-28 against 8 queries on Render backend
    (https://ragcore-api.onrender.com). Citation scores (cross-encoder,
    post-rerank) used as proxy — retrieval_candidates not yet deployed at
    calibration time. Recalibrate after Stage 3 deploy using actual
    retrieval_candidates (pre-rerank hybrid scores).

    Observed gaps: 401k=9.81(H), invest=5.41(H), taxes=3.28(H),
                   ira=2.12(M), weather=1.76(M), banking=1.70(M),
                   stock=1.07(L), mortgage=0.39(L)
    high   >= 3.28  (33rd–100th pct)
    medium  1.70–3.27  (0th–33rd pct)
    low    < 1.70   (bottom third)

    NOTE: Initial calibration (2026-04-28) used post-rerank cross-encoder
    scores as proxy because retrieval_candidates wasn't yet deployed.
    Recalibrate against pre-rerank hybrid (FAISS+BM25) gaps once Stage 1
    backend changes are live. Hybrid score range is wider and includes
    negatives — these thresholds may shift meaningfully.
    """
    cands = retrieval_candidates or []
    if not cands:
        return "unknown"
    scores = sorted([c["score"] for c in cands], reverse=True)
    n = len(scores)
    if n < 2:
        return "unknown"
    gap = scores[0] - scores[n // 2]
    if gap >= 3.28:
        return "high"
    elif gap >= 1.70:
        return "medium"
    else:
        return "low"


# ─── UI rendering helpers ─────────────────────────────────────────────────────

def _routing_strip(result: dict) -> None:
    """Pattern N — inline routing strip above the answer."""
    qtype = result.get("query_type", "")
    strategy = result.get("strategy_used", "")
    cached = result.get("cached", False)
    parts = []
    if strategy:
        parts.append(f"retrieval: <span class='strategy-tag'>{strategy}</span>")
    if qtype:
        parts.append(f"query type: <span class='strategy-tag'>{qtype}</span>")
    if cached:
        parts.append("<span class='stat-badge'>cached ⚡</span>")
    if parts:
        st.markdown(
            f"<div class='routing-strip'>Routed to: {' · '.join(parts)}</div>",
            unsafe_allow_html=True,
        )


def _confidence_badge(level: str) -> None:
    """Pattern M — confidence badge above the answer body."""
    if level == "unknown":
        return
    labels = {"high": "confidence: high ✓", "medium": "confidence: medium ~", "low": "confidence: low ↓"}
    st.markdown(
        f"<span class='confidence-{level}'>{labels[level]}</span>",
        unsafe_allow_html=True,
    )


def _citations_expander(citations: list) -> None:
    """Pattern A + B — sources panel with per-citation cross-encoder score."""
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for i, cite in enumerate(citations):
            source = cite.get("source", "Unknown")
            title = cite.get("title") or Path(source).name or source
            excerpt = (cite.get("excerpt") or "")[:200]
            score = cite.get("score", 0)
            chunk_id = cite.get("chunk_id", "")
            st.markdown(f"""
<div class='citation-card'>
  <span class='citation-source'>[{i+1}] {title}</span>
  <span style='color:#adb5bd;font-size:11px;margin-left:8px'>
    cross-encoder: {score:.3f}
  </span>
  <p class='citation-excerpt'>"{excerpt}"</p>
  <p style='font-size:10px;color:#adb5bd;margin:2px 0 0'>chunk {chunk_id[:8]}…</p>
</div>""", unsafe_allow_html=True)
        st.caption(
            "Citation scores are post-rerank cross-encoder — "
            "different scale from retrieval scores below."
        )


def _retrieval_expander(retrieval_candidates: Optional[list], citations: list) -> None:
    """Pattern D + E — pre-rerank candidates with score histogram."""
    cands = retrieval_candidates or []
    if not cands:
        return
    with st.expander(f"Retrieval candidates — {len(cands)} chunks before reranking", expanded=False):
        # Addition B: score scale annotation
        st.caption(
            "Scores are pre-rerank hybrid (FAISS dense + BM25 sparse). "
            "Citation scores above are post-rerank cross-encoder — different scale."
        )
        # Pattern E: score distribution histogram (Altair)
        scores = [c["score"] for c in cands]
        df_hist = pd.DataFrame({"score": scores})
        hist = (
            alt.Chart(df_hist)
            .mark_bar(color="#4a6fa5", opacity=0.75)
            .encode(
                alt.X("score:Q", bin=alt.Bin(maxbins=12), title="Hybrid score"),
                alt.Y("count():Q", title="Chunks"),
            )
            .properties(height=110, title="Score distribution (top-K pre-rerank)")
        )
        st.altair_chart(hist, use_container_width=True)

        # Pattern D: candidates table with used_in_answer indicator
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
                "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "used": st.column_config.TextColumn("Used", width="small"),
                "source": st.column_config.TextColumn("Source", width="medium"),
                "excerpt": st.column_config.TextColumn("Excerpt", width="large"),
            },
        )


def _trace_expander(stage_timings: Optional[dict]) -> None:
    """Pattern G — per-stage latency as a horizontal stacked bar (Plotly)."""
    if not stage_timings:
        return
    stages = [
        ("router_ms",   "Router",   "#4a6fa5"),
        ("retrieve_ms", "Retrieve", "#e07b39"),
        ("rerank_ms",   "Rerank",   "#6aaa64"),
        ("prompt_ms",   "Prompt",   "#9b59b6"),
        ("generate_ms", "Generate", "#e74c3c"),
    ]
    total = stage_timings.get("total_ms", 0)
    with st.expander(f"Stage timings — {total:.0f} ms total", expanded=False):
        fig = go.Figure()
        for key, label, color in stages:
            ms = stage_timings.get(key, 0)
            fig.add_trace(go.Bar(
                name=label,
                x=[ms],
                y=["Pipeline"],
                orientation="h",
                marker_color=color,
                text=f"{ms:.0f} ms",
                textposition="inside" if ms > 50 else "outside",
                hovertemplate=f"{label}: {ms:.1f} ms<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            height=110,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)


def _self_rag_chips(self_rag_stats: Optional[dict]) -> None:
    """Pattern L — per-claim verification chips for Self-RAG mode."""
    if not self_rag_stats:
        return
    verified = self_rag_stats.get("verified_claims") or []
    unsupported = self_rag_stats.get("unsupported_claims") or []
    faithfulness = self_rag_stats.get("faithfulness_score")
    regenerated = self_rag_stats.get("regenerated", False)

    chips_html = "".join(
        f"<span class='self-rag-verified'>✅ {c[:70]}</span>" for c in verified
    ) + "".join(
        f"<span class='self-rag-unverified'>⚠️ {c[:70]}</span>" for c in unsupported
    )

    if chips_html or faithfulness is not None:
        with st.expander("Self-RAG verification", expanded=False):
            if faithfulness is not None:
                note = " · answer regenerated with additional context" if regenerated else ""
                st.caption(f"Faithfulness: {faithfulness:.2f}{note}")
            if chips_html:
                st.markdown(chips_html, unsafe_allow_html=True)


def _abstention_card() -> None:
    """Pattern P — graceful refusal when retrieval finds nothing relevant."""
    st.markdown("""
<div class='abstention-card'>
  <strong>Not enough information in the corpus</strong><br>
  <span style='font-size:13px;color:#6c757d'>
    The indexed documents don't contain reliable information to answer this question.
    Try rephrasing, or ask about FiQA-2018 topics: IRAs, 401k, mortgages, taxes, investing.
  </span>
</div>
""", unsafe_allow_html=True)


def _render_assistant_message(result: dict, show_retry: bool = False) -> None:
    """Render a complete assistant turn with all observability panels."""
    if "error" in result and result["error"]:
        st.markdown(f"""
<div class='error-card'>
  <strong>Something went wrong</strong><br>
  <span style='font-size:13px;color:#6c757d'>
    The backend returned an error. Please try again in a moment.
  </span>
</div>""", unsafe_allow_html=True)
        with st.expander("Technical details", expanded=False):
            st.code(result["error"])
        if show_retry:
            if st.button("Retry ↺", key=f"retry_{int(time.time()*1000)}"):
                # Remove the error message and the user message, re-prefill
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                    st.session_state.messages.pop()
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        st.session_state.prompt_prefill = msg["content"]
                        break
                st.rerun()
        return

    citations = result.get("citations") or []
    total_tokens = result.get("total_tokens", 0)
    retrieval_candidates = result.get("retrieval_candidates")
    self_rag_stats = result.get("self_rag_stats")
    stage_timings = result.get("stage_timings")
    answer = result.get("answer", "")

    is_abstention = not citations and total_tokens == 0

    # Pattern N: routing strip
    _routing_strip(result)

    # Pattern M: confidence badge
    confidence = compute_confidence(retrieval_candidates)
    _confidence_badge(confidence)

    if is_abstention:
        # Pattern P: abstention card
        _abstention_card()
    else:
        # Stats row
        parts = []
        if result.get("latency_ms"):
            parts.append(f"<span class='stat-badge'>{result['latency_ms']:.0f} ms</span>")
        if total_tokens:
            parts.append(f"<span class='stat-badge'>{total_tokens} tokens</span>")
        if result.get("model_used"):
            parts.append(f"<span class='stat-badge'>{result['model_used']}</span>")
        if parts:
            st.markdown("".join(parts), unsafe_allow_html=True)

        # Pattern C: answer body
        st.markdown(answer)

        # Pattern A + B: citations
        _citations_expander(citations)

    # Pattern D + E: retrieval candidates
    _retrieval_expander(retrieval_candidates, citations)

    # Pattern G: stage timings
    _trace_expander(stage_timings)

    # Pattern L: Self-RAG verification chips
    _self_rag_chips(self_rag_stats)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## RAGCore")
    st.caption("Retrieval-Augmented Generation demo")
    st.divider()

    # Passive backend health check on every load (cached 30 s)
    is_ok = _check_backend_cached(BACKEND_URL)
    st.session_state.backend_ok = is_ok
    if is_ok:
        st.success("Backend online", icon="✅")
    else:
        st.warning("Backend offline", icon="⚠️")
        st.caption(f"URL: `{BACKEND_URL}`")
        st.caption("Render free-tier may need ~30 s to wake up.")

    st.divider()

    # Retrieval strategy selector
    st.markdown("**Retrieval strategy**")
    strategy = st.selectbox(
        "Strategy",
        options=["auto", "hybrid", "semantic", "keyword"],
        index=0,
        help=(
            "auto — router picks best strategy per query\n"
            "hybrid — FAISS dense + BM25 sparse (recommended)\n"
            "semantic — vector search only\n"
            "keyword — BM25 only"
        ),
        label_visibility="collapsed",
    )

    st.divider()

    # Conditional document upload
    if os.getenv("STREAMLIT_ENABLE_UPLOAD", "false").lower() == "true":
        st.markdown("**Upload documents**")
        st.caption("PDF · DOCX · TXT · MD · HTML · CSV")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md", "html", "csv"],
            label_visibility="collapsed",
        )
        if uploaded_files:
            if st.button("Index documents", type="primary", use_container_width=True):
                for file in uploaded_files:
                    with st.spinner(f"Indexing {file.name}…"):
                        res = upload_document(file)
                    if "error" in res:
                        st.error(f"{file.name}: {res['error']}")
                    else:
                        st.success(f"Indexed: {file.name}")
                        st.session_state.indexed_docs.append(file.name)

        if st.session_state.indexed_docs:
            st.divider()
            st.markdown("**Indexed documents**")
            for doc in st.session_state.indexed_docs:
                st.caption(f"📄 {doc}")
    else:
        st.caption("Document upload available when running locally — see README.")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("About"):
        st.markdown("""
**RAGCore** is an open-source RAG reference implementation.

**Pipeline:**
1. Query router — classifies intent, selects strategy
2. Hybrid retrieval — FAISS dense + BM25 sparse
3. Cross-encoder reranking — ms-marco-MiniLM-L-6-v2
4. Prompt construction — token-budget aware
5. LLM generation — Groq / OpenAI / Anthropic

**Corpus:** FiQA-2018 financial Q&A (10k+ posts)

**Generation mode:** basic (single LLM call).
Self-RAG verification (Pattern L) renders only when
`GENERATION_STRATEGY=self_rag` is set on the backend.
Current deployment uses `basic`.
""")


# ─── Main chat area ───────────────────────────────────────────────────────────

st.markdown("## RAGCore")
st.caption(
    "Ask questions about personal finance · "
    "indexed on [FiQA-2018](https://huggingface.co/datasets/explodinggradients/fiqa) · "
    "pipeline details in each response ↓"
)

# Cold-start warning (shown once backend comes back online)
if not st.session_state.backend_ok and st.session_state.messages:
    st.info(
        "Backend may be starting up (Render free-tier cold start: ~30 s). "
        "Your question will be sent once it's ready.",
        icon="⏳",
    )

# Sample prompts — shown only on empty chat
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SAMPLE_PROMPTS))
    for col, sample in zip(cols, SAMPLE_PROMPTS):
        with col:
            if st.button(sample, use_container_width=True):
                st.session_state.prompt_prefill = sample
                st.rerun()

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            _render_assistant_message(message.get("result", {}), show_retry=False)

# ─── Chat input ───────────────────────────────────────────────────────────────

# Consume prefill (set by sample-prompt buttons) before rendering chat_input
prefill = st.session_state.get("prompt_prefill", "")
if prefill:
    del st.session_state["prompt_prefill"]

typed = st.chat_input("Ask about IRAs, 401k, mortgages, investing…")
active_prompt = prefill or typed

if active_prompt:
    with st.chat_message("user"):
        st.markdown(active_prompt)
    st.session_state.messages.append({"role": "user", "content": active_prompt})

    with st.chat_message("assistant"):
        if not st.session_state.backend_ok:
            result = {"error": f"Backend not reachable at {BACKEND_URL}. If this is the Render demo, wait ~30 s and retry."}
        else:
            with st.spinner("Searching corpus and generating answer…"):
                result = ask_question(active_prompt, strategy)

        _render_assistant_message(result, show_retry=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", result.get("error", "")),
            "result": result,
        })
