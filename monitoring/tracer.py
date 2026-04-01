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

from config.settings import settings

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


class LangfuseTracer:
    """
    Sends trace data to Langfuse for LLM observability.
    Tracks: latency, token usage, retrieval scores, model costs.
    """

    def __init__(self):
        self._client = None
        self._traces: dict[str, dict] = {}

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
        trace_id = str(uuid.uuid4())
        self._traces[trace_id] = {
            "query": query,
            "start_time": time.monotonic(),
            "steps": [],
        }
        try:
            client = self._get_client()
            trace = client.trace(
                id=trace_id,
                name="rag_query",
                input={"query": query},
                metadata={"environment": settings.environment},
            )
            self._traces[trace_id]["langfuse_trace"] = trace
        except Exception as e:
            logger.debug("Langfuse trace start failed: %s", e)
        return trace_id

    async def log_routing(self, trace_id, decision):
        self._record_step(trace_id, "routing", {
            "query_type": decision.query_type.value,
            "strategy": decision.primary_strategy.value,
            "expanded_queries": len(decision.expanded_queries),
        })

    async def log_retrieval(self, trace_id, result):
        self._record_step(trace_id, "retrieval", {
            "chunks_retrieved": len(result.chunks),
            "strategy": result.request.strategy.value,
            "latency_ms": result.latency_ms,
            "fallback_used": result.fallback_used,
            "top_scores": [round(c.score, 4) for c in result.chunks[:5]],
        })

    async def log_reranking(self, trace_id, chunks):
        self._record_step(trace_id, "reranking", {
            "chunks_after_rerank": len(chunks),
            "top_scores": [round(c.score, 4) for c in chunks[:3]],
        })

    async def log_generation(self, trace_id, result):
        self._record_step(trace_id, "generation", {
            "model": result.model_used,
            "total_tokens": result.total_tokens,
            "cached": result.cached,
            "latency_ms": result.latency_ms,
        })

    async def log_error(self, trace_id, error: str):
        self._record_step(trace_id, "error", {"error": error})

    async def end_trace(self, trace_id, latency_ms: float):
        trace_data = self._traces.pop(trace_id, {})
        steps = trace_data.get("steps", [])

        # Emit structured log regardless of Langfuse availability
        logger.info(
            "RAG query completed",
            extra={
                "trace_id": trace_id,
                "total_latency_ms": round(latency_ms, 2),
                "steps": steps,
            }
        )

        try:
            langfuse_trace = trace_data.get("langfuse_trace")
            if langfuse_trace:
                langfuse_trace.update(
                    output={"steps": steps},
                    metadata={"total_latency_ms": latency_ms},
                )
        except Exception as e:
            logger.debug("Langfuse trace end failed: %s", e)

    def _record_step(self, trace_id: str, step_name: str, data: dict):
        if trace_id in self._traces:
            self._traces[trace_id]["steps"].append({
                "step": step_name,
                "data": data,
                "ts": time.monotonic(),
            })
        # Always log at debug level
        logger.debug("Trace step [%s] %s: %s", trace_id[:8], step_name, json.dumps(data))


def get_tracer() -> NoOpTracer | LangfuseTracer:
    if settings.enable_tracing and settings.langfuse_public_key:
        return LangfuseTracer()
    return NoOpTracer()
