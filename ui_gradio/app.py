"""
Gradio strategy comparison lab.
"""
import gradio as gr
import pandas as pd

def compare_configs(config_a, config_b, document, query):
    # Mock comparison
    results = {
        "Config A": {"answer": "Answer A", "sources": "doc1.pdf", "latency": 0.5, "score": 0.85},
        "Config B": {"answer": "Answer B", "sources": "doc2.txt", "latency": 0.3, "score": 0.92}
    }
    winner = "Config B" if results["Config B"]["score"] > results["Config A"]["score"] else "Config A"
    return results, f"✅ {winner} was better"

with gr.Blocks() as demo:
    gr.Markdown("# DocIntel RAG Comparison Lab")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Configuration A")
            chunk_a = gr.Dropdown(["Fixed", "Recursive", "Semantic"], label="Chunking")
            embed_a = gr.Dropdown(["OpenAI", "BGE", "MiniLM"], label="Embedding")
            retrieve_a = gr.Dropdown(["Dense", "Sparse", "Hybrid"], label="Retrieval")
            rerank_a = gr.Checkbox(label="Reranking")

        with gr.Column():
            gr.Markdown("## Configuration B")
            chunk_b = gr.Dropdown(["Fixed", "Recursive", "Semantic"], label="Chunking")
            embed_b = gr.Dropdown(["OpenAI", "BGE", "MiniLM"], label="Embedding")
            retrieve_b = gr.Dropdown(["Dense", "Sparse", "Hybrid"], label="Retrieval")
            rerank_b = gr.Checkbox(label="Reranking")

    document = gr.File(label="Document")
    query = gr.Textbox(label="Query")

    run_btn = gr.Button("⚡ Run Both Configurations")

    results = gr.JSON(label="Results")
    winner = gr.Textbox(label="Winner")

    run_btn.click(compare_configs, inputs=[chunk_a, embed_a, retrieve_a, rerank_a, chunk_b, embed_b, retrieve_b, rerank_b, document, query], outputs=[results, winner])

demo.launch()