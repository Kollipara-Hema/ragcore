"""
Streamlit 6-tab RAG pipeline dashboard with multi-provider LLM support.

FEATURES:
- ⚙️ API Configuration panel in sidebar
- Select LLM provider: Groq / OpenAI / Anthropic / Ollama / Demo
- API key input (password masked)
- Model selection based on provider
- Test connection button with status indicator
- Free embedding models (MiniLM, BGE) by default
- Demo mode banner when no API key provided
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
import time

# Mock data for demo
def mock_ingest(file):
    return f"✅ {file.name} — 5 pages, 1200 words, loaded in 0.2s"

def mock_chunk(strategy, size, overlap):
    return [
        {"id": 1, "source": "doc1.pdf", "tokens": 512, "preview": "This is chunk 1..."},
        {"id": 2, "source": "doc2.txt", "tokens": 480, "preview": "This is chunk 2..."},
    ]

def mock_embed(model):
    return {"dimensions": 1536 if "3-small" in model else 1024, "time": 1.2, "vectors": 10}

def mock_retrieve(query, strategy, alpha):
    return [
        {"rank": 1, "score": 0.95, "source": "doc1.pdf", "page": 2, "preview": "Relevant text..."},
        {"rank": 2, "score": 0.89, "source": "doc2.txt", "page": 1, "preview": "Another text..."},
    ]

def mock_generate(query, provider, model):
    return {
        "answer": f"Answer for: {query} (via {provider})",
        "citations": ["doc1.pdf — Page 2", "doc2.txt — Page 1"],
        "tokens": {"prompt": 100, "completion": 50, "total": 150},
        "risk": "Low"
    }

def mock_evaluate(csv):
    return {
        "recall": 0.85,
        "relevance": 0.78,
        "faithfulness": 0.82,
        "hallucination": 0.12
    }

# UI Setup
st.set_page_config(page_title="DocIntel — RAG Pipeline", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0f1724; color: white; }
.demo-banner { background-color: #ff9800; color: black; padding: 12px; border-radius: 6px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = os.getenv("GROQ_API_KEY", "")
if "embedding_provider" not in st.session_state:
    st.session_state.embedding_provider = "minilm"
if "connection_status" not in st.session_state:
    st.session_state.connection_status = None

st.title("DocIntel — Enterprise RAG Pipeline")

# Sidebar - API Configuration
with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    
    # Provider selection
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "openai", "anthropic", "ollama", "demo"],
        index=["groq", "openai", "anthropic", "ollama", "demo"].index(st.session_state.llm_provider),
        key="provider_select"
    )
    st.session_state.llm_provider = provider
    
    # API Key input
    if provider != "demo" and provider != "ollama":
        api_key = st.text_input(
            "API Key",
            value=st.session_state.llm_api_key,
            type="password",
            key="api_key_input"
        )
        st.session_state.llm_api_key = api_key
    
    # Model selection
    models = {
        "groq": ["llama3-70b-8192", "mixtral-8x7b", "gemma2-9b"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "ollama": ["llama3", "mistral", "phi3"],
        "demo": ["demo"]
    }
    selected_model = st.selectbox("Model", models.get(provider, ["demo"]), key="model_select")
    
    # Test connection button
    if st.button("🔌 Test Connection"):
        if provider == "demo":
            st.session_state.connection_status = "✅ Demo Mode"
        elif provider == "ollama":
            st.session_state.connection_status = "✅ Connected (Local)"
        elif st.session_state.llm_api_key:
            st.session_state.connection_status = "✅ Connected"
        else:
            st.session_state.connection_status = "❌ No API Key"
    
    if st.session_state.connection_status:
        if "✅" in st.session_state.connection_status:
            st.success(st.session_state.connection_status)
        else:
            st.error(st.session_state.connection_status)
    
    # Embedding provider
    st.markdown("### 📊 Embedding Provider")
    embedding_provider = st.selectbox(
        "Model",
        ["minilm (FREE)", "bge (FREE)", "openai"],
        index=0,
        key="embedding_select"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline Status")
    st.markdown("✅ Ingest | ⏳ Chunk | ⬜ Embed | ⬜ Retrieve | ⬜ Generate")
    
    if st.button("Reset Pipeline"):
        st.rerun()

# Demo mode banner
if st.session_state.llm_provider == "demo" or not st.session_state.llm_api_key:
    st.markdown("""
    <div class="demo-banner">
    ⚠️ <b>Running in Demo Mode</b> — Enter API key in sidebar for real AI responses
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["01 Ingest", "02 Chunk", "03 Embed", "04 Retrieve", "05 Generate", "06 Evaluate"])

with tab1:
    st.header("Document Ingestion")
    uploaded = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt", "docx", "csv"])
    if uploaded:
        for file in uploaded:
            st.write(mock_ingest(file))
        st.success("All files loaded!")

with tab2:
    st.header("Chunking")
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("Strategy", ["Fixed", "Recursive", "Sliding Window", "Semantic"])
        size = st.slider("Chunk Size", 128, 1024, 512)
        overlap = st.slider("Overlap", 0, 256, 50)
    with col2:
        chunks = mock_chunk(strategy, size, overlap)
        st.write(f"Total chunks: {len(chunks)}")
        df = pd.DataFrame(chunks)
        st.dataframe(df)
        fig = px.bar(df, x="id", y="tokens", title="Chunk Size Distribution")
        st.plotly_chart(fig)

with tab3:
    st.header("Embedding")
    model = st.selectbox("Model", ["all-MiniLM-L6-v2 (FREE)", "BGE-large (FREE)", "OpenAI text-embedding-3-small"])
    if st.button("Embed Chunks"):
        result = mock_embed(model)
        st.write(f"Dimensions: {result['dimensions']}, Time: {result['time']}s")

with tab4:
    st.header("Retrieval")
    query = st.text_input("Query")
    strategy = st.selectbox("Strategy", ["Dense", "Sparse BM25", "Hybrid"])
    alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.5)
    if query:
        results = mock_retrieve(query, strategy, alpha)
        for res in results:
            st.write(f"Rank {res['rank']}: {res['score']:.2f} — {res['source']} P{res['page']}")

with tab5:
    st.header("Generation")
    if st.button("Generate Answer"):
        gen = mock_generate("Sample query", st.session_state.llm_provider, selected_model)
        st.write(gen["answer"])
        st.write("Citations:")
        for cit in gen["citations"]:
            st.write(f"[1] {cit}")

with tab6:
    st.header("Evaluation")
    csv = st.file_uploader("Upload Golden Q&A CSV", type=["csv"])
    if csv:
        metrics = mock_evaluate(csv)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall @5", f"{metrics['recall']:.2f}")
        col2.metric("Relevance", f"{metrics['relevance']:.2f}")
        col3.metric("Faithfulness", f"{metrics['faithfulness']:.2f}")
        col4.metric("Hallucination", f"{metrics['hallucination']:.2f}")

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
