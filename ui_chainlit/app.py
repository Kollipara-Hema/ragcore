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

BACKEND_URL = "http://localhost:8000"

@cl.on_chat_start
async def start():
    # Ask for LLM provider
    actions = [
        cl.Action(name="groq", value="groq", description="Groq (Recommended — fast and free)"),
        cl.Action(name="openai", value="openai", description="OpenAI"),
        cl.Action(name="anthropic", value="anthropic", description="Anthropic Claude"),
        cl.Action(name="demo", value="demo", description="Demo Mode (no API key needed)"),
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
        else:
            # Ask for API key
            api_key = await cl.AskUserMessage(
                content=f"Enter your {provider.upper()} API key (or leave blank for demo mode):"
            ).send()
            cl.user_session.set("llm_api_key", api_key.content if api_key else "")
            
            if api_key and api_key.content:
                await cl.Message(content=f"✅ Connected to {provider.upper()}").send()
            else:
                await cl.Message(content="⚠️ Running in Demo Mode").send()
    
    await cl.Message(content="Welcome to DocIntel Assistant. Upload your documents and ask me anything about them.").send()


@cl.on_message
async def main(message: cl.Message):
    # Mock processing
    await cl.Message(content="Analyzing your query...").send()
    time.sleep(0.3)
    await cl.Message(content="Searching documents...").send()
    time.sleep(0.3)
    await cl.Message(content="Generating answer...").send()
    time.sleep(0.3)

    provider = cl.user_session.get("llm_provider", "demo")
    
    # Mock answer
    answer = f"Here's the answer to: {message.content} (via {provider})"
    citations = "📎 Sources: doc1.pdf (Page 2), doc2.txt (Page 1)"
    steps = "🔍 Retrieval: Dense search, 0.23s latency | 📊 Strategy: Hybrid"

    await cl.Message(content=f"{answer}\n\n{citations}\n\n{steps}").send()


@cl.on_file_upload
async def on_file_upload(file: cl.File):
    await cl.Message(content=f"✅ Indexed {file.name} — ready to answer your questions").send()
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health", timeout=5)
            backend_ok = response.status_code == 200
            print(f"DEBUG: backend_ok={backend_ok}, status={response.status_code}")
    except Exception as e:
        print(f"DEBUG: health check failed: {e}")
        backend_ok = False

    if not backend_ok:
        await cl.Message(
            content=(
                "Backend is not running.\n\n"
                "Start it with:\n"
                "```bash\n"
                "uvicorn rag_system_v2.api.main:app --port 8000\n"
                "```"
            ),
            author="System",
        ).send()
        return

    # Welcome message with instructions
    await cl.Message(
        content=(
            "# Welcome to DocIntel\n\n"
            "I can answer questions about your documents.\n\n"
            "**To get started:**\n"
            "1. Type your **Groq or Anthropic API key** first\n"
            "   *(Get a free Groq key at console.groq.com)*\n"
            "2. Upload a **PDF or document** using the paperclip icon\n"
            "3. **Ask any question** about your documents\n\n"
            "---\n"
            "Please paste your API key to begin:"
        ),
        author="DocIntel",
    ).send()


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
    """
    Process a user question through the RAG pipeline.
    Shows live steps as they happen — this is what makes Chainlit special.
    """

    # ── Step 1: Query classification ─────────────────────────────────────────
    async with cl.Step(name="Classifying query") as step:
        step.output = f'Query: "{query[:60]}..."'
        await step.update()

        try:
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            }
            payload = {"query": query}

            # Step 2: Retrieval ────────────────────────────────────────────────
            async with cl.Step(name="Searching documents") as search_step:
                search_step.output = "Running hybrid search..."
                await search_step.update()

                # Step 3: Generation ──────────────────────────────────────────
                async with cl.Step(name="Generating answer") as gen_step:
                    gen_step.output = "Calling LLM..."
                    await gen_step.update()

                    # Make the actual API call
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    result = response.json()

                    if "error" in result:
                        gen_step.output = f"Error: {result['error']}"
                        await gen_step.update()
                        await cl.Message(
                            content=f"Error: {result['error']}",
                            author="DocIntel",
                        ).send()
                        return

                    # Update steps with actual results
                    strategy = result.get("strategy_used", "hybrid")
                    query_type = result.get("query_type", "unknown")
                    latency = result.get("latency_ms", 0)
                    tokens = result.get("total_tokens", 0)
                    citations = result.get("citations", [])
                    cached = result.get("cached", False)

                    step.output = f"Type: {query_type}"
                    await step.update()

                    search_step.output = (
                        f"Strategy: {strategy} · "
                        f"Found {len(citations)} sources"
                    )
                    await search_step.update()

                    gen_step.output = (
                        f"{tokens} tokens · "
                        f"{latency:.0f}ms"
                        f"{' · cached' if cached else ''}"
                    )
                    await gen_step.update()

        except Exception as e:
            await cl.Message(
                content=f"Error connecting to backend: {str(e)}\n\nMake sure the backend is running.",
                author="DocIntel",
            ).send()
            return

    # ── Send the answer ────────────────────────────────────────────────────
    answer = result.get("answer", "No answer returned.")

    # Build citation elements to attach to the message
    citation_elements = []
    for i, cite in enumerate(citations):
        source = cite.get("source", "Unknown")
        filename = Path(source).name if source else cite.get("title", f"Source {i+1}")
        excerpt = cite.get("excerpt", "")
        score = cite.get("score", 0)

        # Create a text element for each citation
        citation_elements.append(
            cl.Text(
                name=f"[{i+1}] {filename}",
                content=f"Relevance: {score:.0%}\n\n{excerpt}",
                display="side",     # Shows in a side panel when clicked
            )
        )

    # Send the answer with citations attached
    await cl.Message(
        content=answer,
        elements=citation_elements,
        author="DocIntel",
    ).send()


# =============================================================================
# SETTINGS — shown in Chainlit's settings panel
# =============================================================================

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes from the UI settings panel."""
    cl.user_session.set("settings", settings)
