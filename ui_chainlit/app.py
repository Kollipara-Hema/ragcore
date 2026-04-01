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

BACKEND_URL = "http://localhost:9002"


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
                await cl.Message(
                    content=f"Failed to index **{filename}**: {result['error']}",
                    author="DocIntel",
                ).send()
            else:
                step.output = f"Done — {result.get('message', 'indexed successfully')}"
                await step.update()

                # Track indexed docs
                docs = cl.user_session.get("indexed_docs", [])
                docs.append(filename)
                cl.user_session.set("indexed_docs", docs)

                await cl.Message(
                    content=(
                        f"**{filename}** has been indexed.\n\n"
                        f"{result.get('message', '')}\n\n"
                        "You can now ask questions about this document."
                    ),
                    author="DocIntel",
                ).send()

        except Exception as e:
            step.output = f"Error: {str(e)}"
            await step.update()
            await cl.Message(
                content=f"Error uploading file: {str(e)}",
                author="DocIntel",
            ).send()


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
    query_words = query.lower().split()
    scored_chunks = []
    for chunk in chunks:
        score = sum(1 for word in query_words if word in chunk.lower())
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True)
    top_chunks = [c for _, c in scored_chunks[:3]]
    
    # If nothing found, use first 3 chunks anyway
    if not top_chunks or all(s == 0 for s, _ in scored_chunks[:3]):
        top_chunks = chunks[:3]

    provider = cl.user_session.get("llm_provider", "demo")
    docs = cl.user_session.get("indexed_docs", [])
    filename = docs[-1] if docs else "document"

    if provider == "demo":
        content = "Based on the document, here is what I found:\n\n"
        for i, chunk in enumerate(top_chunks):
            content += f"**Chunk {i+1}:**\n{chunk[:500]}{'...' if len(chunk) > 500 else ''}\n\n"
        content += f"Sources: {filename} — Chunk 1"
        await cl.Message(content=content, author="DocIntel").send()
    else:
        # Call LLM with retrieved context
        try:
            from generation.llm_service import LLMService
            llm = LLMService(provider=provider, api_key=api_key)
            context = "\n\n".join(top_chunks)
            prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            answer = llm.generate(prompt)
            await cl.Message(content=answer, author="DocIntel").send()
        except Exception as e:
            await cl.Message(
                content=f"Error generating answer: {str(e)}",
                author="DocIntel",
            ).send()


# =============================================================================
# SETTINGS — shown in Chainlit's settings panel
# =============================================================================

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes from the UI settings panel."""
    cl.user_session.set("settings", settings)
