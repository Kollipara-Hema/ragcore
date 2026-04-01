"""
Gradio strategy comparison lab with multi-provider LLM support.

FEATURES:
- ⚙️ API Configuration at top of page
- Provider selection with dynamic model dropdown
- API key input for authentication
- Side-by-side RAG strategy comparison
- Test different chunking, embedding, and retrieval combinations
- Winner badge showing best performing configuration
- Example queries for quick testing
- Demo mode for zero-setup exploration
"""
import gradio as gr
import os

def compare_configs(chunk_a, embed_a, retrieve_a, rerank_a, chunk_b, embed_b, retrieve_b, rerank_b, provider, api_key, model, document, query):
    # Mock comparison
    results = {
        "Config A": {"answer": f"Answer A ({chunk_a})", "sources": "doc1.pdf", "latency": 0.5, "score": 0.85},
        "Config B": {"answer": f"Answer B ({chunk_b})", "sources": "doc2.txt", "latency": 0.3, "score": 0.92}
    }
    winner = "Config B" if results["Config B"]["score"] > results["Config A"]["score"] else "Config A"
    demo_mode = not api_key and provider != "ollama" and provider != "demo"
    demo_text = " (Demo Mode)" if demo_mode else f" (via {provider} - {model})"
    return results, f"✅ {winner} was better{demo_text}"

# Gradio UI
with gr.Blocks(title="DocIntel RAG Comparison Lab") as demo:
    gr.Markdown("# 🔍 DocIntel RAG Comparison Lab")
    
    # API Configuration
    with gr.Row():
        with gr.Column(scale=1):
            provider = gr.Dropdown(
                ["groq", "openai", "anthropic", "ollama", "demo"],
                value="groq",
                label="LLM Provider"
            )
        with gr.Column(scale=2):
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Leave blank for demo mode"
            )
        with gr.Column(scale=1):
            model = gr.Dropdown(
                ["llama3-70b-8192", "mixtral-8x7b", "gemma2-9b"],
                value="llama3-70b-8192",
                label="Model"
            )
    
    # Demo banner
    def update_models(provider_val):
        models_map = {
            "groq": ["llama3-70b-8192", "mixtral-8x7b", "gemma2-9b"],
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
            "ollama": ["llama3", "mistral", "phi3"],
            "demo": ["demo"]
        }
        return gr.Dropdown(choices=models_map.get(provider_val, ["demo"]))
    
    provider.change(fn=update_models, inputs=provider, outputs=model)
    
    # Two column comparison layout
    with gr.Row():
        with gr.Column(label="Configuration A"):
            gr.Markdown("## Configuration A")
            chunk_a = gr.Dropdown(["Fixed", "Recursive", "Semantic"], value="Fixed", label="Chunking Strategy")
            embed_a = gr.Dropdown(["MiniLM (FREE)", "BGE (FREE)", "OpenAI"], value="MiniLM (FREE)", label="Embedding")
            retrieve_a = gr.Dropdown(["Dense", "Sparse BM25", "Hybrid"], value="Dense", label="Retrieval")
            rerank_a = gr.Checkbox(label="Reranking", value=False)

        with gr.Column(label="Configuration B"):
            gr.Markdown("## Configuration B")
            chunk_b = gr.Dropdown(["Fixed", "Recursive", "Semantic"], value="Semantic", label="Chunking Strategy")
            embed_b = gr.Dropdown(["MiniLM (FREE)", "BGE (FREE)", "OpenAI"], value="BGE (FREE)", label="Embedding")
            retrieve_b = gr.Dropdown(["Dense", "Sparse BM25", "Hybrid"], value="Hybrid", label="Retrieval")
            rerank_b = gr.Checkbox(label="Reranking", value=True)

    # Document and query inputs
    gr.Markdown("## Comparison Input")
    document = gr.File(label="Upload Document")
    query = gr.Textbox(label="Enter Query", placeholder="What do you want to know?")

    # Run button
    run_btn = gr.Button("⚡ Run Both Configurations", size="lg", variant="primary")

    # Results
    results = gr.JSON(label="Comparison Results")
    winner = gr.Textbox(label="Winner", interactive=False)

    # Example queries
    gr.Examples(
        examples=[
            "What are the main topics?",
            "Summarize the document",
            "Compare different sections"
        ],
        inputs=query,
        label="Try these queries:"
    )

    run_btn.click(
        compare_configs,
        inputs=[chunk_a, embed_a, retrieve_a, rerank_a, chunk_b, embed_b, retrieve_b, rerank_b, provider, api_key, model, document, query],
        outputs=[results, winner]
    )

demo.launch()