# DocIntel — Production-Grade Multi-Strategy RAG System

A scalable document intelligence system with intelligent retrieval routing,
hybrid search, reranking, and grounded answer generation.

---

## Architecture Overview

```
Documents → Ingest → Clean → Chunk → Embed → Hybrid Index (Vector + BM25)
                                                        ↓
Query → Understand → Route → Retrieve → Rerank → Prompt → Generate → Answer + Citations
```

### Retrieval Strategies (auto-selected per query)

| Query Type      | Primary Strategy    | When Used |
|----------------|---------------------|-----------|
| Factual         | Hybrid              | "What is X?" |
| Lookup          | Keyword (BM25)      | "Find document by author/date" |
| Semantic        | Vector              | "Explain how X relates to Y" |
| Multi-hop       | Multi-query         | Questions requiring chaining |
| Analytical      | Hybrid + Rerank     | "Compare A vs B" |
| Comparative     | Multi-query         | "Differences between X and Y" |

---

## Quick Start

### 1. Local setup (no Docker)

```bash
# Clone and install
git clone <repo>
cd rag-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add OPENAI_API_KEY at minimum

# Start Weaviate (requires Docker for the vector store)
docker run -d -p 8080:8080 semitechnologies/weaviate:1.24.1 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e DEFAULT_VECTORIZER_MODULE=none

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run API
uvicorn rag_system.api.main:app --reload --port 8000
```

### 2. Full stack with Docker Compose

```bash
cp .env.example .env
# Edit .env with your API keys

docker-compose up --build

# Services:
#   API:        http://localhost:8000
#   API docs:   http://localhost:8000/docs
#   Flower:     http://localhost:5555  (Celery monitoring)
#   Grafana:    http://localhost:3000  (admin/admin)
#   Weaviate:   http://localhost:8080
```

---

## API Usage

### Ingest a document

```bash
# Async (default — returns job_id immediately)
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf" \
  -F "title=Q3 Report 2024" \
  -F "tags=finance,quarterly"

# Check status
curl http://localhost:8000/ingest/status/<job_id>

# Synchronous (small files)
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@notes.txt" \
  -F "async_processing=false"
```

### Query

```bash
# Standard query (auto-routes to best strategy)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the key findings in the Q3 report?",
    "top_k": 5
  }'

# With metadata filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return policy?",
    "metadata_filter": {"doc_type": "pdf"},
    "strategy_override": "keyword"
  }'

# Streaming
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the main points of the report."}' \
  --no-buffer
```

---

## Configuration

All settings are environment-variable driven (see `.env.example`).

Key knobs:

| Setting | Default | Effect |
|---------|---------|--------|
| `CHUNKING_STRATEGY` | `semantic` | `fixed/semantic/hierarchical/sentence` |
| `HYBRID_ALPHA` | `0.7` | 0=keyword only, 1=vector only |
| `ENABLE_RERANKING` | `true` | Two-stage retrieval with cross-encoder |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Final chunks sent to LLM |
| `ENABLE_QUERY_EXPANSION` | `true` | Multi-query for complex questions |

---

## Testing

```bash
# Unit tests (no external services needed)
pytest tests/unit/ -v

# Integration tests (requires Redis; vector store is mocked)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ --cov=rag_system --cov-report=html
```

---

## Evaluation

```python
from rag_system.evaluation.evaluator import RAGEvaluator, EvalSample
from rag_system.orchestrator import RAGOrchestrator
import asyncio

golden_dataset = [
    {
        "query": "What was the Q3 revenue?",
        "ground_truth": "Q3 revenue was $4.2B, up 15% YoY.",
        "relevant_doc_ids": ["doc-abc123"],
    },
    # ... more examples
]

async def main():
    orch = RAGOrchestrator()
    evaluator = RAGEvaluator()
    report = await evaluator.run_against_orchestrator(golden_dataset, orch)
    print(report.summary())

asyncio.run(main())
```

---

## Cloud Deployment

### AWS ECS (Fargate)

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t docintel .
docker tag docintel:latest <account>.dkr.ecr.<region>.amazonaws.com/docintel:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/docintel:latest

# Deploy with Terraform or AWS CDK (see infra/ directory)
```

Recommended AWS architecture:
- **API**: ECS Fargate (2 vCPU, 4GB) × 2–4 tasks behind ALB
- **Workers**: ECS Fargate (4 vCPU, 8GB) × 2–8 tasks
- **Vector DB**: Weaviate on ECS or Pinecone managed
- **Cache**: ElastiCache Redis (r7g.large)
- **Storage**: S3 for raw documents, EFS for model weights

### Scaling the vector database

**Weaviate horizontal scaling:**
```yaml
# weaviate-cluster.yml
replicationFactor: 3    # replicas for read scaling
shards: 4               # shards for write scaling
```

**Pinecone**: managed scaling — just create a larger pod type or use serverless.

---

## Optimization Roadmap

### Phase 1 — Chunking improvements
- [ ] Implement `PropositionalChunker` (extract atomic propositions)
- [ ] `TableAwareChunker` (detect and preserve table structure)
- [ ] Document-structure-aware splitting (respect section headings)

### Phase 2 — Embedding improvements
- [ ] Fine-tune BGE on domain-specific Q&A pairs
- [ ] Matryoshka embeddings (variable-dimension at query time)
- [ ] Late interaction models (ColBERT) for higher precision

### Phase 3 — Hybrid search tuning
- [ ] BM25 parameter tuning (k1, b) via grid search on eval set
- [ ] Adaptive alpha (tune `hybrid_alpha` per query type from eval data)
- [ ] Sparse + dense fusion alternatives (SPLADE)

### Phase 4 — Reranking improvements
- [ ] Fine-tune cross-encoder on domain data
- [ ] MonoT5 reranker for better multilingual support
- [ ] LLM-based reranking (use GPT-4o-mini to score pairs)

### Phase 5 — Generation
- [ ] Self-RAG: generate answer, then verify claims against retrieved context
- [ ] FLARE: generate → check → retrieve more if uncertain
- [ ] Agentic RAG: multi-turn with tool use for complex analytical queries

### Phase 6 — Infrastructure
- [ ] GraphRAG: build knowledge graph from documents for multi-hop
- [ ] Incremental indexing (avoid re-indexing unchanged chunks)
- [ ] A/B testing framework for strategy comparison
