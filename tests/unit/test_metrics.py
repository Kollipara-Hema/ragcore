"""Unit tests for monitoring/metrics.py."""
import pytest


def test_stage_duration_label_names():
    from monitoring.metrics import stage_duration
    assert stage_duration._labelnames == ("stage", "strategy")


def test_generation_tokens_label_names():
    from monitoring.metrics import generation_tokens
    assert generation_tokens._labelnames == ("direction", "provider")


def test_self_rag_claims_label_names():
    from monitoring.metrics import self_rag_claims
    assert self_rag_claims._labelnames == ("outcome",)


def test_resource_collector_registers_gauge_names():
    from prometheus_client import REGISTRY
    import monitoring.metrics  # noqa: F401 — ensures collector is registered
    names = {m.name for m in REGISTRY.collect()}
    assert "ragcore_process_memory_bytes" in names
    assert "ragcore_vector_store_disk_bytes" in names


def test_disk_bytes_remote_backend_returns_zero():
    from monitoring.metrics import _disk_bytes
    assert _disk_bytes("weaviate") == 0.0
    assert _disk_bytes("pinecone") == 0.0
    assert _disk_bytes("qdrant") == 0.0


def test_disk_bytes_missing_path_returns_zero(tmp_path, monkeypatch):
    from monitoring import metrics as m
    from config.settings import settings
    monkeypatch.setattr(settings, "faiss_data_dir", str(tmp_path / "nonexistent"))
    assert m._disk_bytes("faiss") == 0.0


def test_disk_bytes_existing_path(tmp_path, monkeypatch):
    from monitoring import metrics as m
    from config.settings import settings
    (tmp_path / "index.faiss").write_bytes(b"x" * 1024)
    monkeypatch.setattr(settings, "faiss_data_dir", str(tmp_path))
    assert m._disk_bytes("faiss") == 1024.0


def test_retrieval_empty_label_names():
    from monitoring.metrics import retrieval_empty
    assert retrieval_empty._labelnames == ("strategy",)


def test_retrieval_empty_increments():
    from monitoring.metrics import retrieval_empty
    before = retrieval_empty.labels(strategy="hybrid")._value.get()
    retrieval_empty.labels(strategy="hybrid").inc()
    assert retrieval_empty.labels(strategy="hybrid")._value.get() == before + 1
