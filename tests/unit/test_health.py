"""Unit tests for /health/live and /health/ready endpoints."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# /health/live
# =============================================================================

class TestHealthLive:
    def test_returns_200(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_body_has_alive_status(self, client):
        resp = client.get("/health/live")
        assert resp.json()["status"] == "alive"


# =============================================================================
# /health/ready — happy path
# =============================================================================

class TestHealthReady:
    """Each test patches only what it needs; _check_llm_config is mocked by
    default so tests that don't exercise the LLM check stay focused."""

    def test_happy_path_returns_200(self, client):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert all(v["ok"] for v in body["checks"].values())

    def test_vector_store_check_passes(self, client):
        mock_store = MagicMock()
        mock_store.ping.return_value = None
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        assert resp.json()["checks"]["vector_store"]["ok"] is True

    def test_embedder_check_passes(self, client):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        assert resp.json()["checks"]["embedder"]["ok"] is True

    def test_llm_config_check_passes(self, client):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        assert resp.json()["checks"]["llm_config"]["ok"] is True

    # -------------------------------------------------------------------------
    # Failure paths
    # -------------------------------------------------------------------------

    def test_embedder_failure_returns_503_with_structured_body(self, client):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(side_effect=RuntimeError("model not loaded"))

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["embedder"]["ok"] is False
        assert "model not loaded" in body["checks"]["embedder"]["reason"]
        # Other checks still ran and passed
        assert body["checks"]["vector_store"]["ok"] is True
        assert body["checks"]["llm_config"]["ok"] is True

    def test_vector_store_failure_returns_503(self, client):
        mock_store = MagicMock()
        mock_store.ping.side_effect = RuntimeError("FAISS data directory missing: ./faiss")
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["vector_store"]["ok"] is False
        assert "FAISS data directory missing" in body["checks"]["vector_store"]["reason"]

    def test_all_checks_run_even_if_first_fails(self, client):
        """A failure in check 1 must not short-circuit checks 2 and 3."""
        mock_store = MagicMock()
        mock_store.ping.side_effect = RuntimeError("store down")
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config"):
            resp = client.get("/health/ready")

        body = resp.json()
        assert "vector_store" in body["checks"]
        assert "embedder" in body["checks"]
        assert "llm_config" in body["checks"]


# =============================================================================
# _check_llm_config() helper — tested directly
# =============================================================================

class TestCheckLlmConfig:
    def test_passes_when_groq_key_set(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings:
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = "sk-test"
            _check_llm_config()  # must not raise

    def test_passes_when_openai_key_set(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings:
            mock_settings.llm_provider = LLMProvider.OPENAI
            mock_settings.openai_api_key = "sk-openai-test"
            _check_llm_config()

    def test_passes_when_anthropic_key_set(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings:
            mock_settings.llm_provider = LLMProvider.ANTHROPIC
            mock_settings.anthropic_api_key = "sk-ant-test"
            _check_llm_config()

    def test_raises_when_groq_key_missing(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings:
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = None
            with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
                _check_llm_config()

    def test_raises_when_groq_key_empty_string(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings:
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = ""
            with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
                _check_llm_config()

    def test_raises_for_unconfigured_provider(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings:
            mock_settings.llm_provider = LLMProvider.TOGETHER
            with pytest.raises(RuntimeError, match="not fully configured"):
                _check_llm_config()
