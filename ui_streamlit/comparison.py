"""Chunker comparison data layer.

Fires concurrent /query calls (one per Apple 10-K chunker variant), pins
strategy to hybrid, and returns just the retrieval metrics the comparison
view needs. Part 1 of the comparison-view arc: data only, no rendering.
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import requests

CHUNKER_COMPARISON_CORPORA: tuple[str, ...] = (
    "apple_10k_fixed",
    "apple_10k_hierarchical",
    "apple_10k_document_structure",
)


def _post_query(
    query: str,
    corpus: str,
    backend_url: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {
        "query": query,
        "strategy_override": "hybrid",
        "verify_claims": False,
        "corpus": corpus,
    }
    try:
        r = requests.post(
            f"{backend_url}/query",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _summarize(resp: dict[str, Any]) -> dict[str, Any]:
    if "error" in resp:
        return {"error": resp["error"]}
    cits = resp.get("citations") or []
    cands = resp.get("retrieval_candidates") or []
    stage = resp.get("stage_timings") or {}
    top_cit = cits[0] if cits else None
    return {
        "chunks_retrieved":  len(cands),
        # Post-rerank cross-encoder logit. Max over citations[].score, which
        # is set in prompt_builder from rc.score AFTER reranker overwrites it.
        "top_rerank_score":  max((c.get("score", 0.0) for c in cits), default=None),
        "latency_retrieval": (stage.get("retrieve_ms") or 0.0)
                            + (stage.get("rerank_ms") or 0.0),
        "top_chunk": {
            "chunk_id": top_cit.get("chunk_id"),
            "score":    top_cit.get("score"),
            "excerpt":  top_cit.get("excerpt"),
        } if top_cit else None,
        "ranked_list":   cands,
        "strategy_used": resp.get("strategy_used"),
        "raw_response":  resp,
    }


def fetch_chunker_comparison(
    query: str,
    backend_url: str,
    api_key: str = "",
    corpora: tuple[str, ...] = CHUNKER_COMPARISON_CORPORA,
    timeout: int = 90,
) -> dict[str, Any]:
    """Fire concurrent /query calls (one per chunker variant), strategy pinned
    to hybrid, return per-corpus retrieval metrics plus concurrency metadata.

    Per-call errors/timeouts surface as ``{"error": ...}`` for that corpus
    only; the other corpora still return. Cold-start safe — default per-call
    timeout is 90s (Render dynos can take 30–60s to wake).
    """
    t0 = time.monotonic()
    results: dict[str, dict[str, Any]] = {}

    def _one(corpus: str) -> tuple[str, dict[str, Any], float, float]:
        c_start = time.monotonic()
        resp = _post_query(query, corpus, backend_url, api_key, timeout)
        c_end = time.monotonic()
        return corpus, resp, c_start - t0, c_end - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(corpora)) as ex:
        futures = {ex.submit(_one, c): c for c in corpora}
        for fut in concurrent.futures.as_completed(futures):
            corpus = futures[fut]
            try:
                _name, resp, started_s, finished_s = fut.result(timeout=timeout + 5)
                summary = _summarize(resp)
                summary["timing"] = {
                    "started_s":  round(started_s, 3),
                    "finished_s": round(finished_s, 3),
                    "duration_s": round(finished_s - started_s, 3),
                }
                results[corpus] = summary
            except concurrent.futures.TimeoutError:
                results[corpus] = {"error": f"timeout after {timeout}s"}
            except Exception as e:
                results[corpus] = {"error": str(e)}

    wall_s = time.monotonic() - t0
    sum_durations = sum(
        (r.get("timing", {}).get("duration_s") or 0.0)
        for r in results.values()
    )
    return {
        "query":   query,
        "results": results,
        "meta": {
            "wall_seconds":       round(wall_s, 2),
            "sum_durations":      round(sum_durations, 2),
            "concurrent_speedup": round(sum_durations / wall_s, 2) if wall_s > 0 else None,
            "strategies_used":    {c: r.get("strategy_used") for c, r in results.items()},
        },
    }
