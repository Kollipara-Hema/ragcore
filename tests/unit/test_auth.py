"""Unit tests for API key authentication middleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.auth import APIKeyAuthMiddleware, EXEMPT_PATHS


def _authed_client(api_key: str) -> TestClient:
    """Minimal FastAPI app with APIKeyAuthMiddleware for isolated testing."""
    test_app = FastAPI()

    @test_app.get("/protected")
    async def protected_route():
        return {"ok": True}

    test_app.add_middleware(APIKeyAuthMiddleware, api_key=api_key)
    return TestClient(test_app)


def test_auth_disabled_by_default():
    """When RAGCORE_AUTH_ENABLED is unset/false, routes work without X-API-Key."""
    from api.main import app

    client = TestClient(app)
    # POST /query with empty body → 422 (validation error), not 401
    response = client.post("/query", json={})
    assert response.status_code != 401


def test_auth_enabled_requires_key():
    """Missing key returns 401 with WWW-Authenticate: ApiKey header."""
    client = _authed_client("test-key-123")
    response = client.get("/protected")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "ApiKey"


def test_auth_enabled_wrong_key_rejected():
    """Wrong key returns 401."""
    client = _authed_client("test-key-123")
    response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_auth_enabled_correct_key_accepted():
    """Correct key passes through to the handler."""
    client = _authed_client("test-key-123")
    response = client.get("/protected", headers={"X-API-Key": "test-key-123"})
    assert response.status_code == 200


def test_auth_exempt_paths_bypass():
    """Auth-gated requests to exempt paths are let through without a key."""
    client = _authed_client("secret")
    for path in EXEMPT_PATHS:
        resp = client.get(path)
        # Middleware passes exempt paths to the handler; handler may 404 if the
        # route isn't registered in this test app, but must not return 401.
        assert resp.status_code != 401, f"{path} should be exempt but returned 401"


def test_auth_enabled_without_api_key_fails_startup():
    """APIKeyAuthMiddleware with empty api_key raises ValueError at construction."""
    with pytest.raises(ValueError, match="non-empty api_key"):
        APIKeyAuthMiddleware(app=None, api_key="")
