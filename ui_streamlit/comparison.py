"""Chunker comparison data layer.

Fires sequential /query calls (one per Apple 10-K chunker variant), pins
strategy to hybrid, and returns just the retrieval metrics the comparison
view needs.

Why sequential rather than parallel: three concurrent /query calls against
a single HF Space container (running embedding + cross-encoder per request) plus
shared Groq tokens-per-day limits made the parallel version drop random
columns. Sequential is slower (~30-45s wall) but each call gets resources
to itself and completes reliably.
"""
from __future__ import annotations

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
) -> tuple[int, dict[str, Any]]:
    """POST /query and return (status_code, body). Body is a dict either
    way — wraps transport errors in {"error": ...} with status_code 0."""
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
            f"{backend_url}/retrieve",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except Exception as e:
        return 0, {"error": str(e)}
    try:
        body = r.json()
    except Exception as e:
        body = {"error": f"non-JSON response: {e}", "_text": r.text[:200]}
    return r.status_code, body


def _summarize(status: int, resp: dict[str, Any]) -> dict[str, Any]:
    """Project the full /query response down to the comparison-view fields.

    Detects three error shapes so a failed call surfaces as a real error in
    the UI instead of a silently-empty column:
      1. Transport error from _post_query ({"error": ...}, status 0)
      2. FastAPI 4xx/5xx body ({"detail": "..."}, status >= 400)
      3. 200 with the QueryResponse shape (no error)
    """
    if "error" in resp:
        return {"error": resp["error"]}
    if status >= 400:
        detail = resp.get("detail")
        msg = detail if isinstance(detail, str) else str(resp)[:300]
        return {"error": f"HTTP {status}: {msg[:280]}"}
    if "strategy_used" not in resp:
        return {"error": f"unexpected response shape: keys={list(resp.keys())[:6]}"}

    cands = resp.get("retrieval_candidates") or []
    stage = resp.get("stage_timings") or {}
    # /retrieve returns no citations (no LLM was called). Source the top
    # rerank score and the top-chunk excerpt from retrieval_candidates
    # instead. Prefer the backend's convenience top_rerank_score field;
    # fall back to max(retrieval_candidates) if it's missing (older deploys
    # or non-/retrieve responses).
    top_cand = max(
        cands,
        key=lambda c: c.get("score") if c.get("score") is not None else float("-inf"),
        default=None,
    ) if cands else None
    top_rerank_score = resp.get("top_rerank_score")
    if top_rerank_score is None and cands:
        top_rerank_score = max(c.get("score", 0.0) for c in cands)
    return {
        "chunks_retrieved":  len(cands),
        "top_rerank_score":  top_rerank_score,
        "latency_retrieval": (stage.get("retrieve_ms") or 0.0)
                            + (stage.get("rerank_ms") or 0.0),
        "top_chunk": {
            "chunk_id": top_cand.get("chunk_id"),
            "score":    top_cand.get("score"),
            "excerpt":  top_cand.get("excerpt"),
        } if top_cand else None,
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
    """Fire sequential /query calls (one per chunker variant), strategy pinned
    to hybrid, return per-corpus retrieval metrics.

    Per-call errors (transport failures, HTTP 4xx/5xx, unexpected response
    shapes) are isolated to that corpus's entry; the loop keeps going so
    the other corpora still get a chance to return. Cold-start safe — per-call
    timeout defaults to 90s (HF Spaces can take 30–60s to wake).
    """
    t0 = time.monotonic()
    results: dict[str, dict[str, Any]] = {}

    for corpus in corpora:
        c_start = time.monotonic()
        try:
            status, resp = _post_query(query, corpus, backend_url, api_key, timeout)
            summary = _summarize(status, resp)
        except Exception as e:
            summary = {"error": str(e)}
        c_end = time.monotonic()
        summary["timing"] = {
            "started_s":  round(c_start - t0, 3),
            "finished_s": round(c_end - t0, 3),
            "duration_s": round(c_end - c_start, 3),
        }
        results[corpus] = summary

    wall_s = time.monotonic() - t0
    return {
        "query":   query,
        "results": results,
        "meta": {
            "wall_seconds":    round(wall_s, 2),
            "mode":            "sequential",
            "strategies_used": {c: r.get("strategy_used") for c, r in results.items()},
        },
    }
