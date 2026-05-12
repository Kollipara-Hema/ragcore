"""Unit tests for CORS configuration."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_allows_configured_origin(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_rejects_unconfigured_origin(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    acao = response.headers.get("access-control-allow-origin")
    assert acao != "https://evil.example.com"


def test_cors_default_settings_value():
    from config.settings import settings

    assert settings.cors_origins == ["http://localhost:8501"]


def test_cors_env_override(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "  https://a.com , https://b.com  ")
    from config.settings import Settings

    fresh = Settings()
    assert fresh.cors_origins == ["https://a.com", "https://b.com"]
