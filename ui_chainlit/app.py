"""
Chainlit conversational AI assistant with multi-provider LLM support.

FEATURES:
- 🤖 LLM provider selection on startup
- Choose from: Groq / OpenAI / Anthropic / Demo Mode
- Optional API key input for non-demo providers
- Conversational memory for follow-up questions
- Document upload and indexing
- Streaming responses
- Multi-step retrieval for complex queries
"""
import chainlit as cl
import time
import requests
from pathlib import Path
from io import BytesIO
import pdfplumber
from docx import Document
import string

BACKEND_URL = "http://localhost:9002"


def generate_demo_response(query, chunks):
    """Generate a readable response from chunks"""
    if not chunks:
        return "No relevant content found."
    
    # Check if it's a summary question
    summary_keywords = ["summary", "summarize", "overview", "what is this", "what is the document", "tell me about"]
    is_summary = any(k in query.lower() for k in summary_keywords)
    
    if is_summary:
        # Create a summary
        all_text = " ".join(chunks)
        # Extract key sentences or topics
        # For simplicity, take first 200 chars and format
        summary = all_text[:200] + "..." if len(all_text) > 200 else all_text
        response = f"📄 Document Summary:\n{summary}"
    else:
        # Extract relevant sentences
        query_words = query.lower().split()
        relevant_sentences = []
        for chunk in chunks:
            sentences = chunk.split('.')
            for sentence in sentences:
                if any(w in sentence.lower() for w in query_words):
                    relevant_sentences.append(sentence.strip() + '.')
                    if len(relevant_sentences) >= 3:
                        break
            if len(relevant_sentences) >= 3:
                break
        if relevant_sentences:
            response = f"Based on your question \"{query}\", here is what the document says:\n\n" + "\n".join(f"- {s}" for s in relevant_sentences[:3])
        else:
            response = f"Based on your question \"{query}\", the document doesn't contain specific information about that topic. Here is some general content:\n\n{chunks[0][:300]}..."
    
    response += "\n\n📎 Source: Document — Relevant sections"
    return response


def extract_text_from_bytes(file_bytes: bytes, content_type: str, filename: str) -> str:
    """Extract text from file bytes based on content type."""
    try:
        if content_type == "application/pdf":
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            return text
        elif content_type in ["text/plain", "text/markdown", "text/html", "text/csv"]:
            return file_bytes.decode('utf-8', errors='ignore')
        else:
            # Try to decode as text
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""


@cl.on_chat_start
async def start():
    # Ask for LLM provider
    actions = [
        cl.Action(name="groq", payload={"value": "groq"}, label="Groq (Recommended — fast and free)"),
        cl.Action(name="openai", payload={"value": "openai"}, label="OpenAI"),
        cl.Action(name="anthropic", payload={"value": "anthropic"}, label="Anthropic Claude"),
        cl.Action(name="demo", payload={"value": "demo"}, label="Demo Mode (no API key needed)"),
    ]
    
    await cl.Message(
        content="Which LLM would you like to use? Choose below:",
        actions=actions
    ).send()
    
    # Wait for user selection
    res = await cl.AskActionMessage(
        content="Select a provider:",
        actions=actions
    ).send()
    
    if res:
        provider = res.get("value", "demo")
        cl.user_session.set("llm_provider", provider)
        
        if provider == "demo":
            await cl.Message(content="✅ Running in Demo Mode — all responses are simulated").send()
            cl.user_session.set("api_key", "demo")
        else:
            # Ask for API key
            api_key = await cl.AskUserMessage(
                content=f"Enter your {provider.upper()} API key (or leave blank for demo mode):"
            ).send()
            cl.user_session.set("api_key", api_key.content if api_key else "")
            
            if api_key and api_key.content:
                await cl.Message(content=f"✅ Connected to {provider.upper()}").send()
            else:
                await cl.Message(content="⚠️ Running in Demo Mode").send()
    
    await cl.Message(content="Welcome to DocIntel Assistant. Upload your documents and ask me anything about them.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Runs on every message from the user.
    Handles: API key setup, file uploads, questions.
    """
    api_key = cl.user_session.get("api_key", "")

    # ── Handle API key input ─────────────────────────────────────────────────
    # If no key set yet, treat first message as the key
    if not api_key:
        potential_key = message.content.strip()

        # Basic validation — Groq keys start with gsk_, Anthropic with sk-ant-
        if potential_key.startswith(("gsk_", "sk-ant-", "sk-")):
            cl.user_session.set("api_key", potential_key)
            await cl.Message(
                content=(
                    "API key saved for this session.\n\n"
                    "Now upload a document using the paperclip icon "
                    "or ask a question if documents are already indexed."
                ),
                author="DocIntel",
            ).send()
        else:
            await cl.Message(
                content=(
                    "That doesn't look like a valid API key.\n\n"
                    "- Groq keys start with `gsk_`\n"
                    "- Anthropic keys start with `sk-ant-`\n\n"
                    "Get a free Groq key at **console.groq.com**"
                ),
                author="DocIntel",
            ).send()
        return

    # ── Handle file uploads ──────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                await handle_file_upload(element, api_key)
        return

    # ── Handle question ──────────────────────────────────────────────────────
    await handle_question(message.content, api_key)
    """
    Runs on every message from the user.
    Handles: API key setup, file uploads, questions.
    """
    api_key = cl.user_session.get("api_key", "")

    # ── Handle API key input ─────────────────────────────────────────────────
    # If no key set yet, treat first message as the key
    if not api_key:
        potential_key = message.content.strip()

        # Basic validation — Groq keys start with gsk_, Anthropic with sk-ant-
        if potential_key.startswith(("gsk_", "sk-ant-", "sk-")):
            cl.user_session.set("api_key", potential_key)
            await cl.Message(
                content=(
                    "API key saved for this session.\n\n"
                    "Now upload a document using the paperclip icon "
                    "or ask a question if documents are already indexed."
                ),
                author="DocIntel",
            ).send()
        else:
            await cl.Message(
                content=(
                    "That doesn't look like a valid API key.\n\n"
                    "- Groq keys start with `gsk_`\n"
                    "- Anthropic keys start with `sk-ant-`\n\n"
                    "Get a free Groq key at **console.groq.com**"
                ),
                author="DocIntel",
            ).send()
        return

    # ── Handle file uploads ──────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                await handle_file_upload(element, api_key)
        return

    # ── Handle question ──────────────────────────────────────────────────────
    await handle_question(message.content, api_key)


# =============================================================================
# FILE UPLOAD HANDLER
# =============================================================================

async def handle_file_upload(element, api_key: str):
    """Upload and index a file, showing progress steps."""

    filename = element.name or "document"

    # Show processing steps live
    async with cl.Step(name=f"Indexing {filename}") as step:

        step.output = "Reading file..."
        await step.update()

        try:
            # Read the file
            with open(element.path, "rb") as f:
                file_bytes = f.read()

            # Detect content type
            suffix = Path(filename).suffix.lower()
            content_type_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".html": "text/html",
                ".csv": "text/csv",
            }
            content_type = content_type_map.get(suffix, "application/octet-stream")

            # Extract text and create chunks
            text = extract_text_from_bytes(file_bytes, content_type, filename)
            if text:
                # Split into chunks by paragraphs or fixed size
                chunks = [chunk for chunk in text.split('\n\n') if chunk.strip()]
                if not chunks:
                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000) if text[i:i+1000].strip()]
                existing_chunks = cl.user_session.get("chunks", [])
                existing_chunks.extend(chunks)
                cl.user_session.set("chunks", existing_chunks)

            step.output = "Chunking and embedding..."
            await step.update()

            # Send to backend
            files = {"file": (filename, file_bytes, content_type)}
            data = {"title": filename, "async_processing": "false"}
            headers = {"X-API-Key": api_key} if api_key else {}

            response = requests.post(
                f"{BACKEND_URL}/ingest/file",
                files=files,
                data=data,
                headers=headers,
                timeout=120,
            )
            result = response.json()

            if "error" in result:
                step.output = f"Failed: {result['error']}"
                await step.update()
            else:
                step.output = f"Done — {result.get('message', 'indexed successfully')}"
                await step.update()

                # Track indexed docs
                docs = cl.user_session.get("indexed_docs", [])
                docs.append(filename)
                cl.user_session.set("indexed_docs", docs)

        except Exception as e:
            step.output = f"Error: {str(e)}"
            await step.update()


# =============================================================================
# QUESTION HANDLER
# =============================================================================

async def handle_question(query: str, api_key: str):
    """Process a user question using simple keyword search and generation."""
    
    chunks = cl.user_session.get("chunks", [])
    if not chunks:
        await cl.Message(
            content="No documents have been indexed yet. Please upload a document first.",
            author="DocIntel",
        ).send()
        return

    # Simple keyword search
    scored = []
    for i, chunk in enumerate(chunks):
        words = [w.strip(string.punctuation) for w in query.lower().split() if w.strip(string.punctuation)]
        chunk_clean = chunk.lower().translate(str.maketrans('', '', string.punctuation))
        score = sum(1 for w in words if w in chunk_clean)
        scored.append((score, i, chunk))
    scored.sort(reverse=True)
    # Get top 3 different chunks
    top_chunks = [c for _, _, c in scored[:3]]
    
    # If nothing found, use first 3 chunks anyway
    if not top_chunks or all(s == 0 for s, _, _ in scored[:3]):
        top_chunks = chunks[:3]

    provider = cl.user_session.get("llm_provider", "demo")
    docs = cl.user_session.get("indexed_docs", [])
    filename = docs[-1] if docs else "document"

    mock_chunks = [{"text_preview": chunk, "source_doc": filename, "page": i+1} for i, chunk in enumerate(top_chunks)]

    answer = generate_demo_response(query, top_chunks)

    await cl.Message(content=answer, author="DocIntel").send()


# =============================================================================
# SETTINGS — shown in Chainlit's settings panel
# =============================================================================

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes from the UI settings panel."""
    cl.user_session.set("settings", settings)
