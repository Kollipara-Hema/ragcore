"""
HTTP-level integration test for GET /corpora.

Drives the endpoint through FastAPI TestClient and verifies:
- All registered corpora appear in the response.
- source / chunker come from CORPORA_CONFIG for entries that have one.
- source / chunker are null for the runtime-bound "default" corpus.
- doc_count is read live from each store via count().

TestClient is constructed WITHOUT the `with` context manager so the
lifespan startup (which would build the orchestrator and try to register
production CORPORA_CONFIG entries) does not run. Each test's fixture
registers its own corpora explicitly.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from utils.models import Chunk
from vectorstore.chroma_store import ChromaVectorStore
from vectorstore.vector_store import register_corpus, reset_corpus_registry


def _upsert(store: ChromaVectorStore, n: int) -> None:
    doc_id = uuid4()
    chunks = [
        Chunk(content=f"chunk {i}", doc_id=doc_id, embedding=[0.1 + 0.01 * i] * 8)
        for i in range(n)
    ]
    asyncio.run(store.upsert(chunks))


@pytest.fixture
def client_with_corpora(tmp_path, monkeypatch):
    """Register a default Chroma store + two Apple-shaped corpora with
    CORPORA_CONFIG entries; return a TestClient."""
    # CORPORA_CONFIG is read by /corpora to populate static metadata.
    fake_config = {
        "apple_alpha": {
            "source": "Test source — Apple alpha (2025)",
            "ingest_path": "data/apple_demo/alpha.pdf",
            "chunker": "fixed",
            "persist_dir": str(tmp_path / "apple_alpha"),
        },
        "apple_csvs": {
            "source": "Test source — Apple CSVs (quarterly + annual)",
            "ingest_paths": [
                "data/apple_demo/q.csv",
                "data/apple_demo/a.csv",
            ],
            "chunker": "fixed",
            "persist_dir": str(tmp_path / "apple_csvs"),
        },
    }
    monkeypatch.setattr("api.main.CORPORA_CONFIG", fake_config)

    reset_corpus_registry()

    default_store = ChromaVectorStore(persist_dir=str(tmp_path / "default"), collection_name="default")
    apple_alpha = ChromaVectorStore(persist_dir=str(tmp_path / "apple_alpha"), collection_name="apple_alpha")
    apple_csvs  = ChromaVectorStore(persist_dir=str(tmp_path / "apple_csvs"),  collection_name="apple_csvs")

    _upsert(default_store, 5)
    _upsert(apple_alpha, 7)
    _upsert(apple_csvs, 2)

    register_corpus("default", default_store)
    register_corpus("apple_alpha", apple_alpha)
    register_corpus("apple_csvs", apple_csvs)

    from api.main import app
    return TestClient(app)


def test_corpora_lists_default_and_registered(client_with_corpora):
    resp = client_with_corpora.get("/corpora")
    assert resp.status_code == 200
    names = sorted(item["name"] for item in resp.json()["corpora"])
    assert names == ["apple_alpha", "apple_csvs", "default"]


def test_corpora_returns_correct_doc_counts(client_with_corpora):
    resp = client_with_corpora.get("/corpora")
    counts = {item["name"]: item["doc_count"] for item in resp.json()["corpora"]}
    assert counts == {"default": 5, "apple_alpha": 7, "apple_csvs": 2}


def test_corpora_returns_source_and_chunker_for_apple_entries(client_with_corpora):
    resp = client_with_corpora.get("/corpora")
    by_name = {item["name"]: item for item in resp.json()["corpora"]}
    assert by_name["apple_alpha"]["source"] == "Test source — Apple alpha (2025)"
    assert by_name["apple_alpha"]["chunker"] == "fixed"
    # `source` is a single provenance label even for multi-file corpora; the
    # per-file local paths live in `ingest_paths` and are not exposed here.
    assert by_name["apple_csvs"]["source"] == "Test source — Apple CSVs (quarterly + annual)"
    assert by_name["apple_csvs"]["chunker"] == "fixed"


def test_corpora_returns_null_metadata_for_default(client_with_corpora):
    resp = client_with_corpora.get("/corpora")
    by_name = {item["name"]: item for item in resp.json()["corpora"]}
    assert by_name["default"]["source"] is None
    assert by_name["default"]["chunker"] is None


def test_corpora_only_default_registered(tmp_path, monkeypatch):
    """With only the default corpus registered and an empty CORPORA_CONFIG,
    /corpora returns a single entry with null metadata and doc_count 0."""
    monkeypatch.setattr("api.main.CORPORA_CONFIG", {})

    reset_corpus_registry()
    default_store = ChromaVectorStore(persist_dir=str(tmp_path / "default"), collection_name="default")
    register_corpus("default", default_store)

    from api.main import app
    client = TestClient(app)
    resp = client.get("/corpora")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"corpora": [{"name": "default", "source": None, "chunker": None, "doc_count": 0}]}
