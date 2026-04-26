"""
Monitoring and tracing layer.

Integrates with:
  - Langfuse: LLM-specific observability (traces, token costs, latency)
  - OpenTelemetry: standard distributed tracing
  - Structured logging: JSON logs for aggregation in ELK/Datadog

Usage:
    tracer = get_tracer()
    trace_id = await tracer.start_trace(query)
    await tracer.log_routing(trace_id, decision)
    ...
    await tracer.end_trace(trace_id, latency_ms)
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from typing import Optional

from config.settings import settings
from utils.models import QueryTrace, TraceEvent

logger = logging.getLogger(__name__)


class NoOpTracer:
    """Default tracer when monitoring is disabled."""
    async def start_trace(self, query: str) -> str: return str(uuid.uuid4())
    async def log_routing(self, trace_id, decision): pass
    async def log_retrieval(self, trace_id, result): pass
    async def log_reranking(self, trace_id, chunks): pass
    async def log_generation(self, trace_id, result): pass
    async def log_error(self, trace_id, error): pass
    async def end_trace(self, trace_id, latency_ms): pass
    def get_trace(self, trace_id: str) -> Optional[QueryTrace]: return None


class LangfuseTracer:
    """
    Sends trace data to Langfuse for LLM observability.
    Tracks: latency, token usage, retrieval scores, model costs.
    """

    def __init__(self):
        self._client = None
        self._traces: dict[str, QueryTrace] = {}

    def _get_client(self):
        if self._client is None:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        return self._client

    async def start_trace(self, query: str) -> str:
        from datetime import datetime
        trace_id = str(uuid.uuid4())
        trace = QueryTrace(
            trace_id=trace_id,
            query=query,
            start_time=datetime.utcnow(),
            events=[],
        )
        self._traces[trace_id] = trace
        try:
            client = self._get_client()
            lf_trace = client.trace(
                id=trace_id,
                name="rag_query",
                input={"query": query},
                metadata={"environment": settings.environment},
            )
            # Store langfuse trace for later updates
            trace.metadata = {"langfuse_id": lf_trace.id}
        except Exception as e:
            logger.debug("Langfuse trace start failed: %s", e)
        return trace_id

    async def log_routing(self, trace_id, decision):
        if trace_id in self._traces:
            event = TraceEvent(
                node="router",
                duration_ms=0.0,  # Not measured
                status="success",
                details={
                    "query_type": decision.query_type.value,
                    "strategy": decision.primary_strategy.value,
                    "expanded_queries": len(decision.expanded_queries),
                }
            )
            self._traces[trace_id].events.append(event)
        self._record_step(trace_id, "routing", event.details if 'event' in locals() else {})

    async def log_retrieval(self, trace_id, result):
        if trace_id in self._traces:
            event = TraceEvent(
                node="retriever",
                duration_ms=result.latency_ms,
                status="success",
                details={
                    "chunks_retrieved": len(result.chunks),
                    "strategy": result.request.strategy.value,
                    "fallback_used": result.fallback_used,
                    "top_scores": [round(c.score, 4) for c in result.chunks[:5]],
                }
            )
            self._traces[trace_id].events.append(event)
        self._record_step(trace_id, "retrieval", event.details if 'event' in locals() else {})

    async def log_reranking(self, trace_id, chunks):
        if trace_id in self._traces:
            event = TraceEvent(
                node="reranker",
                duration_ms=0.0,  # Not measured
                status="success",
                details={
                    "chunks_after_rerank": len(chunks),
                    "top_scores": [round(c.score, 4) for c in chunks[:3]],
                }
            )
            self._traces[trace_id].events.append(event)
        self._record_step(trace_id, "reranking", event.details if 'event' in locals() else {})

    async def log_generation(self, trace_id, result):
        if trace_id in self._traces:
            event = TraceEvent(
                node="generator",
                duration_ms=result.latency_ms,
                status="success",
                details={
                    "model": result.model_used,
                    "total_tokens": result.total_tokens,
                    "cached": result.cached,
                }
            )
            self._traces[trace_id].events.append(event)
        self._record_step(trace_id, "generation", event.details if 'event' in locals() else {})

    async def log_error(self, trace_id, error: str):
        if trace_id in self._traces:
            event = TraceEvent(
                node="error",
                duration_ms=0.0,
                status="error",
                details={"error": error}
            )
            self._traces[trace_id].events.append(event)
        self._record_step(trace_id, "error", {"error": error})

    async def end_trace(self, trace_id, latency_ms: float):
        if trace_id not in self._traces:
            return

        from datetime import datetime
        trace = self._traces[trace_id]
        trace.end_time = datetime.utcnow()
        trace.total_duration_ms = latency_ms
        trace.status = "completed"

        # Emit structured log
        logger.info(
            "RAG query completed",
            extra={
                "trace_id": trace_id,
                "total_latency_ms": round(latency_ms, 2),
                "events": [e.dict() for e in trace.events],
            }
        )

        try:
            langfuse_id = trace.metadata.get("langfuse_id") if trace.metadata else None
            if langfuse_id and self._client:
                self._client.trace(id=langfuse_id).update(
                    output={"events": [e.dict() for e in trace.events]},
                    metadata={"total_latency_ms": latency_ms},
                )
        except Exception as e:
            logger.debug("Langfuse trace end failed: %s", e)

    def get_trace(self, trace_id: str) -> Optional[QueryTrace]:
        """Retrieve a completed trace by ID."""
        return self._traces.get(trace_id)

    def _record_step(self, trace_id: str, step_name: str, data: dict):
        logger.debug("Trace step [%s] %s: %s", trace_id[:8], step_name, json.dumps(data))


def get_tracer() -> NoOpTracer | LangfuseTracer:
    if settings.enable_tracing and settings.langfuse_public_key:
        return LangfuseTracer()
    return NoOpTracer()
