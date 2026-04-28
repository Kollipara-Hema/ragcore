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
    pip install --no-cache-dir "groq>=0.5"

# Pre-download ML models — eliminates HuggingFace Hub downloads at runtime
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-large-en-v1.5'); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
