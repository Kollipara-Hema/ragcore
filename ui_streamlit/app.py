"""
=============================================================================
ui_streamlit/app.py — Streamlit Chat UI for RAG System
=============================================================================
HOW TO RUN:
    cd ui_streamlit
    streamlit run app.py

WHAT THIS HAS:
    - Proper chat bubbles (st.chat_message) — looks like a real chat app
    - API key input in sidebar — user pastes key, it is never saved anywhere
    - PDF/DOCX/TXT upload with progress bar
    - Citations shown as expandable cards below each answer
    - Retrieval strategy selector
    - Stats panel (latency, tokens, strategy used)
    - Works locally AND on Streamlit Cloud
=============================================================================
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path

# =============================================================================
# PAGE CONFIG — must be first Streamlit command
# =============================================================================

st.set_page_config(
    page_title="DocIntel — RAG System",
    page_icon="docs/favicon.png" if Path("docs/favicon.png").exists() else "📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — proper chat bubbles and clean styling
# =============================================================================

st.markdown("""
<style>
    /* Hide default Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp { background-color: #f8f9fa; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        padding: 4px;
        margin-bottom: 8px;
    }

    /* User message bubble */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #e8f4fd;
    }

    /* Citation card */
    .citation-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-left: 3px solid #4a90d9;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 13px;
    }

    .citation-source {
        color: #4a90d9;
        font-weight: 600;
        font-size: 12px;
    }

    .citation-excerpt {
        color: #6c757d;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Stats badges */
    .stat-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6fa8;
        border-radius: 99px;
        padding: 2px 10px;
        font-size: 12px;
        margin-right: 6px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 8px;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        border-radius: 24px;
    }

    /* Strategy tag */
    .strategy-tag {
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 4px;
        padding: 1px 8px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BACKEND URL — where your FastAPI is running
# =============================================================================

# When running locally in Codespaces: http://localhost:8000
# When both run in same Codespace: http://localhost:8000
BACKEND_URL = "http://localhost:8000"

# =============================================================================
# SESSION STATE — persists across reruns
# =============================================================================

# Chat history — list of {"role": "user"/"assistant", "content": "...", "meta": {...}}
if "messages" not in st.session_state:
    st.session_state.messages = []

# API key — stored in session only, never saved to disk
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Whether backend is reachable
if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = False

# Indexed documents list
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_backend():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def upload_document(file, api_key: str) -> dict:
    """
    Upload a document to the backend for indexing.
    Returns the response dict with doc_id and chunk count.
    """
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {
            "title": file.name,
            "async_processing": "false",   # Wait for indexing to complete
        }
        # Pass API key in header so backend can use it
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        response = requests.post(
            f"{BACKEND_URL}/ingest/file",
            files=files,
            data=data,
            headers=headers,
            timeout=120,    # Large files can take time
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ask_question(query: str, strategy: str, api_key: str) -> dict:
    """
    Send a question to the backend and return the response.
    """
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        payload = {
            "query": query,
            "strategy_override": strategy if strategy != "auto" else None,
        }

        response = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            headers=headers,
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def format_citations(citations: list) -> str:
    """Build HTML for citation cards shown below the answer."""
    if not citations:
        return ""

    html = "<div style='margin-top: 12px;'>"
    html += "<p style='font-size:12px;color:#6c757d;margin-bottom:6px;'>Sources used:</p>"

    for i, cite in enumerate(citations):
        source = cite.get("source", "Unknown")
        title = cite.get("title", source)
        excerpt = cite.get("excerpt", "")[:150]
        score = cite.get("score", 0)

        # Only show filename, not full path
        filename = Path(source).name if source else title

        html += f"""
        <div class='citation-card'>
            <span class='citation-source'>[{i+1}] {filename}</span>
            <span style='color:#adb5bd;font-size:11px;margin-left:8px'>
                relevance: {score:.0%}
            </span>
            <p class='citation-excerpt'>"{excerpt}..."</p>
        </div>
        """

    html += "</div>"
    return html


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## DocIntel RAG")
    st.markdown("*Ask questions about your documents*")
    st.divider()

    # ── API Key input ────────────────────────────────────────────────────────
    st.markdown("### API Key")
    st.caption(
        "Your key is only stored in this browser session. "
        "It is never saved to disk or sent anywhere except your own backend."
    )

    api_key_input = st.text_input(
        "Paste your Groq or Anthropic key",
        value=st.session_state.api_key,
        type="password",        # Masks the key — shows dots instead of characters
        placeholder="gsk_... or sk-ant-...",
        help="Get a free Groq key at console.groq.com",
    )

    # Save key to session state when user types it
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.success("Key saved for this session")

    st.divider()

    # ── Backend status ────────────────────────────────────────────────────────
    st.markdown("### Backend Status")
    if st.button("Check connection", use_container_width=True):
        st.session_state.backend_ok = check_backend()

    if st.session_state.backend_ok:
        st.success("Backend running")
    else:
        st.error("Backend not reachable")
        st.caption(f"Expected at: {BACKEND_URL}")
        st.caption("Run: `python -m uvicorn api.main:app --port 8000`")

    st.divider()

    # ── Document upload ───────────────────────────────────────────────────────
    st.markdown("### Upload Documents")
    st.caption("Supported: PDF, DOCX, TXT, MD, HTML, CSV")

    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md", "html", "csv"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Index documents", type="primary", use_container_width=True):
            if not st.session_state.api_key:
                st.warning("Please paste your API key above first")
            else:
                for file in uploaded_files:
                    with st.spinner(f"Indexing {file.name}..."):
                        result = upload_document(file, st.session_state.api_key)

                    if "error" in result:
                        st.error(f"Failed: {file.name} — {result['error']}")
                    else:
                        st.success(f"Indexed: {file.name}")
                        st.caption(f"Chunks: {result.get('message', '')}")
                        st.session_state.indexed_docs.append(file.name)

    # Show indexed documents
    if st.session_state.indexed_docs:
        st.divider()
        st.markdown("### Indexed Documents")
        for doc in st.session_state.indexed_docs:
            st.caption(f"📄 {doc}")

    st.divider()

    # ── Retrieval strategy ───────────────────────────────────────────────────
    st.markdown("### Retrieval Strategy")
    strategy = st.selectbox(
        "Strategy",
        options=["auto", "hybrid", "semantic", "keyword"],
        index=0,
        help=(
            "auto = system picks best strategy per query\n"
            "hybrid = vector + keyword search\n"
            "semantic = vector search only\n"
            "keyword = keyword (BM25) only"
        ),
        label_visibility="collapsed",
    )

    st.divider()

    # ── Clear chat ───────────────────────────────────────────────────────────
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.markdown("## Ask your documents")

# Show welcome message when no messages yet
if not st.session_state.messages:
    st.info(
        "**Getting started:**\n\n"
        "1. Paste your API key in the sidebar\n"
        "2. Upload a PDF or document\n"
        "3. Click Index documents\n"
        "4. Ask a question below",
        icon="👋",
    )

# ── Render existing chat history ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show citations and stats for assistant messages
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]

            # Stats row — latency, tokens, strategy
            stats_html = ""
            if meta.get("latency_ms"):
                stats_html += f"<span class='stat-badge'>{meta['latency_ms']:.0f}ms</span>"
            if meta.get("total_tokens"):
                stats_html += f"<span class='stat-badge'>{meta['total_tokens']} tokens</span>"
            if meta.get("strategy_used"):
                stats_html += f"<span class='strategy-tag'>{meta['strategy_used']}</span>"
            if meta.get("cached"):
                stats_html += "<span class='stat-badge'>cached</span>"

            if stats_html:
                st.markdown(stats_html, unsafe_allow_html=True)

            # Citations
            if meta.get("citations"):
                with st.expander(f"View {len(meta['citations'])} sources"):
                    st.markdown(
                        format_citations(meta["citations"]),
                        unsafe_allow_html=True,
                    )


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):

    # Validate — need API key
    if not st.session_state.api_key:
        st.error("Please paste your API key in the sidebar first.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            start = time.time()
            result = ask_question(prompt, strategy, st.session_state.api_key)
            elapsed = time.time() - start

        if "error" in result:
            # Show error message
            error_msg = f"Sorry, something went wrong: {result['error']}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "meta": {},
            })

        else:
            # Show the answer
            answer = result.get("answer", "No answer returned.")
            st.markdown(answer)

            # Build meta for stats + citations
            meta = {
                "latency_ms": result.get("latency_ms", elapsed * 1000),
                "total_tokens": result.get("total_tokens", 0),
                "strategy_used": result.get("strategy_used", strategy),
                "cached": result.get("cached", False),
                "citations": result.get("citations", []),
                "query_type": result.get("query_type", ""),
            }

            # Stats row
            stats_html = ""
            if meta["latency_ms"]:
                stats_html += f"<span class='stat-badge'>{meta['latency_ms']:.0f}ms</span>"
            if meta["total_tokens"]:
                stats_html += f"<span class='stat-badge'>{meta['total_tokens']} tokens</span>"
            if meta["strategy_used"]:
                stats_html += f"<span class='strategy-tag'>{meta['strategy_used']}</span>"
            if meta["cached"]:
                stats_html += "<span class='stat-badge'>cached</span>"

            if stats_html:
                st.markdown(stats_html, unsafe_allow_html=True)

            # Citations expander
            if meta["citations"]:
                with st.expander(f"View {len(meta['citations'])} sources"):
                    st.markdown(
                        format_citations(meta["citations"]),
                        unsafe_allow_html=True,
                    )

            # Save assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": meta,
            })
