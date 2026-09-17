"""Unit tests for /health/live and /health/ready endpoints."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app

# _check_llm_config now returns a detail dict that the endpoint merges into the
# llm_config check body; the bare MagicMock default is no longer a valid stand-in.
_LLM_OK = {"model": "test-model", "model_verified": True}


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
             patch("api.main._check_llm_config", return_value=_LLM_OK):
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
             patch("api.main._check_llm_config", return_value=_LLM_OK):
            resp = client.get("/health/ready")

        assert resp.json()["checks"]["vector_store"]["ok"] is True

    def test_embedder_check_passes(self, client):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config", return_value=_LLM_OK):
            resp = client.get("/health/ready")

        assert resp.json()["checks"]["embedder"]["ok"] is True

    def test_llm_config_check_passes(self, client):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.1])

        with patch("api.main.get_vector_store", return_value=mock_store), \
             patch("api.main.get_embedder", return_value=mock_embedder), \
             patch("api.main._check_llm_config", return_value=_LLM_OK):
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
             patch("api.main._check_llm_config", return_value=_LLM_OK):
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
             patch("api.main._check_llm_config", return_value=_LLM_OK):
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
             patch("api.main._check_llm_config", return_value=_LLM_OK):
            resp = client.get("/health/ready")

        body = resp.json()
        assert "vector_store" in body["checks"]
        assert "embedder" in body["checks"]
        assert "llm_config" in body["checks"]


# =============================================================================
# _check_llm_config() helper — tested directly
# =============================================================================

class TestCheckLlmConfig:
    @pytest.fixture(autouse=True)
    def _clear_model_cache(self):
        """The model-availability cache is module state; leaking it across tests
        would let one test's verdict satisfy another's provider call."""
        import api.main

        api.main._model_check_cache.clear()
        yield
        api.main._model_check_cache.clear()

    def test_passes_when_groq_key_set_and_model_served(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", return_value={"openai/gpt-oss-120b"}):
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = "sk-test"
            mock_settings.llm_model = "openai/gpt-oss-120b"
            assert _check_llm_config()["model_verified"] is True

    def test_passes_when_openai_key_set_and_model_served(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", return_value={"gpt-4o-mini"}):
            mock_settings.llm_provider = LLMProvider.OPENAI
            mock_settings.openai_api_key = "sk-openai-test"
            mock_settings.llm_model = "gpt-4o-mini"
            assert _check_llm_config()["model_verified"] is True

    def test_passes_when_anthropic_key_set_and_model_served(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", return_value={"claude-sonnet-4"}):
            mock_settings.llm_provider = LLMProvider.ANTHROPIC
            mock_settings.anthropic_api_key = "sk-ant-test"
            mock_settings.llm_model = "claude-sonnet-4"
            assert _check_llm_config()["model_verified"] is True

    def test_raises_when_model_no_longer_served(self):
        """The September 2026 outage: valid key, model decommissioned upstream."""
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", return_value={"openai/gpt-oss-120b"}):
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = "sk-test"
            mock_settings.llm_model = "llama-3.3-70b-versatile"
            with pytest.raises(RuntimeError, match="is not served by groq"):
                _check_llm_config()

    def test_raises_when_provider_rejects_the_key(self):
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        class AuthenticationError(Exception):
            pass

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", side_effect=AuthenticationError("401")):
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = "sk-revoked"
            mock_settings.llm_model = "openai/gpt-oss-120b"
            with pytest.raises(RuntimeError, match="was rejected by groq"):
                _check_llm_config()

    def test_unreachable_provider_reports_unverified_without_failing(self):
        """A provider blip must not flap readiness and restart healthy containers."""
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", side_effect=ConnectionError("timeout")):
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = "sk-test"
            mock_settings.llm_model = "openai/gpt-oss-120b"
            detail = _check_llm_config()

        assert detail["model_verified"] is False
        assert "could not reach groq" in detail["reason"]

    def test_verified_result_is_cached(self):
        """Readiness can be polled often; a verified model must not re-hit the provider."""
        from api.main import _check_llm_config
        from config.settings import LLMProvider

        with patch("api.main.settings") as mock_settings, \
             patch("api.main._provider_model_ids", return_value={"m"}) as lister:
            mock_settings.llm_provider = LLMProvider.GROQ
            mock_settings.groq_api_key = "sk-test"
            mock_settings.llm_model = "m"
            _check_llm_config()
            _check_llm_config()

        assert lister.call_count == 1

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
