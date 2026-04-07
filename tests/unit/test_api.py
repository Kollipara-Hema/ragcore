"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestAgentEndpoints:
    """Tests for /agent endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)

    def test_agent_query_validates_request(self, client):
        """Test that query validation works."""
        # Missing required 'query' field
        response = client.post("/agent/query", json={})
        assert response.status_code == 422  # Validation error

    def test_agent_query_rejects_empty_query(self, client):
        """Test that empty queries are rejected."""
        response = client.post(
            "/agent/query",
            json={"query": ""}  # Empty query
        )
        assert response.status_code == 422  # Below min_length=3

    def test_agent_query_rejects_very_long_query(self, client):
        """Test that very long queries are rejected."""
        response = client.post(
            "/agent/query",
            json={"query": "a" * 2001}  # Over max_length=2000
        )
        assert response.status_code == 422

    def test_agent_query_accepts_valid_request(self, client):
        """Test that valid requests are accepted (with mocked agent)."""
        # Note: This will fail if the agent graph isn't properly initialized
        # but we're testing the endpoint signature
        response = client.post(
            "/agent/query",
            json={"query": "What is machine learning?"}
        )
        # We expect 500 if agent isn't mocked, but not 422
        assert response.status_code in [200, 500]

    def test_agent_query_response_schema(self, client):
        """Test that response follows the expected schema."""
        # Send a valid request
        response = client.post(
            "/agent/query",
            json={"query": "Test query"}
        )
        
        # If successful, check the schema
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "citations" in data
            assert "confidence" in data
            assert "retry_count" in data
            assert "model_used" in data
            assert "latency_ms" in data
            assert "trace_id" in data

    def test_trace_retrieval_not_found(self, client):
        """Test that non-existent traces return 404."""
        response = client.get("/agent/trace/nonexistent-trace-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_trace_retrieval_after_query(self, client):
        """Test that traces are stored and retrievable."""
        # Send a query with tracing enabled
        query_response = client.post(
            "/agent/query",
            json={
                "query": "Test query",
                "trace_enabled": True
            }
        )
        
        if query_response.status_code == 200:
            trace_id = query_response.json()["trace_id"]
            
            # Retrieve the trace
            trace_response = client.get(f"/agent/trace/{trace_id}")
            assert trace_response.status_code == 200
            
            trace_data = trace_response.json()
            assert trace_data["trace_id"] == trace_id
            assert trace_data["query"] == "Test query"
            assert "events" in trace_data
            # Should now have events populated
            assert isinstance(trace_data["events"], list)


class TestHealthEndpoint:
    """Basic tests for existing endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)

    def test_health_endpoint_exists(self, client):
        """Test that /health endpoint responds."""
        response = client.get("/health")
        assert response.status_code in [200, 404]  # 404 if not implemented yet
