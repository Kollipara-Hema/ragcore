"""
Sample golden dataset for RAG evaluation.

This file contains example Q&A pairs with ground truth answers and relevant document IDs.
Used for evaluating retrieval quality, answer faithfulness, and overall RAG performance.

Format:
- query: The user's question
- ground_truth: Expected answer (for faithfulness evaluation)
- relevant_doc_ids: List of document IDs that should be retrieved
- context: Optional additional context for the question
"""

from typing import List, Dict, Any

# Sample evaluation dataset
# In a real scenario, this would be loaded from a JSON/CSV file
SAMPLE_GOLDEN_DATASET: List[Dict[str, Any]] = [
    {
        "query": "What is the main purpose of this RAG system?",
        "ground_truth": "The main purpose is to provide accurate answers to user questions by retrieving and synthesizing information from indexed documents using advanced retrieval strategies.",
        "relevant_doc_ids": ["doc_readme_001", "doc_api_002"],
        "context": "System documentation and API docs"
    },
    {
        "query": "How does the hybrid retrieval work?",
        "ground_truth": "Hybrid retrieval combines vector similarity search with BM25 keyword matching, using a configurable alpha parameter to weight the two approaches.",
        "relevant_doc_ids": ["doc_retrieval_003", "doc_config_004"],
        "context": "Retrieval strategy documentation"
    },
    {
        "query": "What are the supported LLM providers?",
        "ground_truth": "The system supports OpenAI, Anthropic, Groq, and Ollama as LLM providers.",
        "relevant_doc_ids": ["doc_config_004", "doc_llm_005"],
        "context": "Configuration and LLM service docs"
    },
    {
        "query": "How is memory implemented in the agent?",
        "ground_truth": "The agent uses short-term memory for conversation context (up to 10 turns) and Redis-backed long-term memory for persistent information storage.",
        "relevant_doc_ids": ["doc_agent_006", "doc_memory_007"],
        "context": "Agent architecture and memory system docs"
    },
    {
        "query": "What evaluation metrics are available?",
        "ground_truth": "Available metrics include retrieval recall/MRR/NDCG, generation faithfulness/relevance, latency percentiles, and cost tracking per query.",
        "relevant_doc_ids": ["doc_eval_008", "doc_monitoring_009"],
        "context": "Evaluation and monitoring documentation"
    }
]

def get_sample_dataset() -> List[Dict[str, Any]]:
    """Return the sample golden dataset for evaluation."""
    return SAMPLE_GOLDEN_DATASET.copy()

def load_golden_dataset_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load golden dataset from a JSON file.

    Expected format:
    [
        {
            "query": "What is X?",
            "ground_truth": "Answer...",
            "relevant_doc_ids": ["doc1", "doc2"],
            "context": "Optional context"
        }
    ]
    """
    import json
    with open(filepath, 'r') as f:
        return json.load(f)