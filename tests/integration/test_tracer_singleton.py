"""
Integration tests for the get_tracer() singleton.

These tests reproduce the exact production bug:
  - get_tracer() used to construct a new LangfuseTracer() on every call
  - The orchestrator and the /trace/{trace_id} endpoint each called it once
  - Traces written by the orchestrator were invisible to the endpoint's instance
  - Result: silent 404 on every /trace lookup when ENABLE_TRACING=true

Each test calls reset_tracer() via the autouse fixture so the singleton
is clean at the start and end of every test.
"""
from __future__ import annotations
import asyncio
import pytest


# ---------------------------------------------------------------------------
# Fixture: reset the singleton before and after every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_singleton():
    from monitoring.tracer import reset_tracer

    reset_tracer()
    yield
    reset_tracer()


# ---------------------------------------------------------------------------
# Test 1 (structural): two get_tracer() calls return the same instance
# ---------------------------------------------------------------------------

class TestGetTracerSingleton:

    def test_get_tracer_returns_same_instance(self):
        """Two calls to get_tracer() must return the exact same object."""
        from monitoring.tracer import get_tracer

        tracer_a = get_tracer()
        tracer_b = get_tracer()

        assert tracer_a is tracer_b, (
            "get_tracer() returned two different instances. "
            "Traces written by the orchestrator are invisible to the API endpoint."
        )

    def test_reset_clears_singleton(self):
        """reset_tracer() must cause the next call to construct a fresh instance."""
        from monitoring.tracer import get_tracer, reset_tracer

        tracer_a = get_tracer()
        reset_tracer()
        tracer_b = get_tracer()

        assert tracer_a is not tracer_b, (
            "After reset_tracer(), get_tracer() should return a new instance."
        )


# ---------------------------------------------------------------------------
# Test 2 (behavioral): a trace written via one get_tracer() call is
# retrievable via a second get_tracer() call — the test that would have
# failed before the fix.
# ---------------------------------------------------------------------------

class TestTraceVisibleAcrossCalls:

    def test_trace_written_by_first_call_readable_by_second_call(self, monkeypatch):
        """
        Reproduces the production bug:
          1. Simulate the orchestrator path — get the tracer once, start a trace.
          2. Simulate the API endpoint path — call get_tracer() again independently.
          3. Assert get_trace(trace_id) on the second handle returns the trace.

        With the old code (no singleton) this test returns None → 404.
        With the fix it must return the QueryTrace written in step 1.
        """
        from config.settings import settings
        import monitoring.tracer as tracer_module

        monkeypatch.setattr(settings, "enable_tracing", True)
        monkeypatch.setattr(settings, "langfuse_public_key", "test-key")

        # ── Orchestrator path ─────────────────────────────────────────────────
        orchestrator_tracer = tracer_module.get_tracer()
        trace_id = asyncio.run(orchestrator_tracer.start_trace("what is RAG?"))

        # ── API endpoint path ─────────────────────────────────────────────────
        endpoint_tracer = tracer_module.get_tracer()
        assert endpoint_tracer is orchestrator_tracer, "pre-condition: same instance"

        result = endpoint_tracer.get_trace(trace_id)

        assert result is not None, (
            "get_trace() returned None on the second get_tracer() handle. "
            "This is the singleton bug: each call held a separate _traces dict."
        )
        assert result.query == "what is RAG?"
