FROM python:3.11-slim

# System deps for pdfplumber, OCR, and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
# Install core dependencies matching CI plus API packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.27.1 \
    pydantic==2.6.3 \
    pydantic-settings==2.2.1 \
    python-multipart==0.0.9 \
    httpx==0.27.0 \
    openai==1.14.0 \
    sentence-transformers==2.6.1 \
    torch==2.2.1 \
    pdfplumber==0.10.3 \
    python-docx==1.1.0 \
    beautifulsoup4==4.12.3 \
    numpy==1.26.4 \
    chromadb==0.4.24 \
    redis==5.0.3 \
    pandas==2.2.1 \
    rank-bm25==0.2.2 \
    faiss-cpu==1.8.0 \
    langgraph==0.0.40 \
    typing_extensions

COPY . .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
