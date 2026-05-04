"""Prometheus custom metrics for ragcore."""
import os
import logging

import psutil
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_client.core import GaugeMetricFamily

from config.settings import settings

logger = logging.getLogger(__name__)

stage_duration = Histogram(
    "ragcore_stage_duration_seconds",
    "RAG pipeline stage latency in seconds",
    ["stage", "strategy"],
)

generation_tokens = Counter(
    "ragcore_generation_tokens_total",
    "LLM tokens consumed",
    ["direction", "provider"],
)

self_rag_claims = Counter(
    "ragcore_self_rag_claims_total",
    "Self-RAG claim verification outcomes",
    ["outcome"],
)


class _ResourceCollector:
    def collect(self):
        try:
            rss = float(psutil.Process().memory_info().rss)
        except Exception:
            rss = 0.0
        g = GaugeMetricFamily(
            "ragcore_process_memory_bytes",
            "RSS memory used by the API process",
        )
        g.add_metric([], rss)
        yield g

        backend = settings.vector_store_provider.value
        g2 = GaugeMetricFamily(
            "ragcore_vector_store_disk_bytes",
            "Disk space used by the vector store",
            labels=["backend"],
        )
        g2.add_metric([backend], _disk_bytes(backend))
        yield g2


def _disk_bytes(backend: str) -> float:
    try:
        if backend == "faiss":
            path = settings.faiss_data_dir
        elif backend == "chroma":
            path = settings.chroma_persist_dir
        else:
            return 0.0
        if not os.path.exists(path):
            return 0.0
        total = 0
        for dirpath, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
        return float(total)
    except Exception:
        return 0.0


try:
    REGISTRY.register(_ResourceCollector())
except ValueError:
    pass  # Already registered — happens on module reimport during pytest
