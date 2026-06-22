FROM python:3.11-slim

# System deps for pdfplumber, OCR, scientific computing, and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in dependency order to avoid conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    # Core scientific stack first (most constrained dependencies)
    pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir torch==2.2.1 && \
    pip install --no-cache-dir pandas==2.2.1 && \
    pip install --no-cache-dir faiss-cpu==1.8.0 && \
    pip install --no-cache-dir sentence-transformers==2.6.1 && \
    # Core data science and ML
    pip install --no-cache-dir rank-bm25==0.2.2 && \
    pip install --no-cache-dir "langgraph>=1.0.0" && \
    pip install --no-cache-dir typing_extensions && \
    # Data processing and vector stores
    pip install --no-cache-dir chromadb==0.4.24 && \
    pip install --no-cache-dir redis==5.0.3 && \
    # Document processing
    pip install --no-cache-dir pdfplumber==0.10.3 && \
    pip install --no-cache-dir pymupdf==1.27.2.3 && \
    pip install --no-cache-dir pymupdf4llm==1.27.2.3 && \
    pip install --no-cache-dir python-docx==1.1.0 && \
    pip install --no-cache-dir beautifulsoup4==4.12.3 && \
    # API and web framework
    pip install --no-cache-dir pydantic==2.6.3 && \
    pip install --no-cache-dir pydantic-settings==2.2.1 && \
    pip install --no-cache-dir fastapi==0.110.0 && \
    pip install --no-cache-dir uvicorn[standard]==0.27.1 && \
    pip install --no-cache-dir python-multipart==0.0.9 && \
    pip install --no-cache-dir httpx==0.27.0 && \
    pip install --no-cache-dir openai==1.14.0 && \
    pip install --no-cache-dir "groq>=0.5" && \
    pip install --no-cache-dir prometheus-client==0.20.0 && \
    pip install --no-cache-dir prometheus-fastapi-instrumentator==6.1.0 && \
    pip install --no-cache-dir psutil==5.9.8 && \
    pip install --no-cache-dir structlog==24.4.0 && \
    # TODO(AUDIT #11): pinned manually because the Dockerfile does not read from
    # pyproject.toml; remove this line when that structural fix is made.
    pip install --no-cache-dir "celery[redis]>=5.3"

# Bake the model cache under /app so the non-root runtime user can read it.
# Build runs as root; without HF_HOME the models cache to /root/.cache and
# UID 1000 would re-download from HF Hub at boot, breaking the pre-baked
# invariant (baseline §5 / Phase 3 §5). chown of /app/.cache (below) makes it
# readable + lets hf_hub write its lock files at runtime.
ENV HF_HOME=/app/.cache/huggingface

# Pre-download ML models — eliminates HuggingFace Hub downloads at runtime
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-large-en-v1.5'); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Confirm HF_HOME took effect for the pre-download — models must live under
# /app/.cache/huggingface, not stranded under /root/.cache. Fail the build if
# the cache is empty.
RUN test -d /app/.cache/huggingface && ls -la /app/.cache/huggingface \
    && [ -n "$(ls -A /app/.cache/huggingface)" ] || (echo "HF cache empty — HF_HOME did not take effect" && exit 1)

COPY . .

# HF Spaces requires non-root (UID 1000 convention). Named user via useradd -m
# so /home/user exists as a real writable $HOME for any library that reaches
# for it (torch ~/.cache, hf_hub token/lock files) beyond what HF_HOME covers.
RUN useradd -m -u 1000 user

# Absolute path defaults as siblings under /app/data so
# _assert_session_root_isolated passes (no overlap). Also set as Space vars in
# Phase 2 — both layers, reconciling the Phase-0/Phase-1 doc wording.
ENV FAISS_DATA_DIR=/app/data/faiss \
    CHROMA_PERSIST_DIR=/app/data/chroma_db \
    RAGCORE_SESSION_ROOT=/app/data/sessions

# Create the writable dests, then chown both the writable dirs AND the seed
# SOURCES the runtime user reads (faiss_seed, chroma_collections — both under
# /app/data) plus the model cache. Single recursive chown reaches all five.
RUN mkdir -p /app/data/faiss /app/data/chroma_db /app/data/sessions \
    && chown -R user:user /app/data /app/.cache

USER user

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
