"""
End-to-end production health check for RAGCore.

POSTs a known-good financial query to the live /query endpoint and fails
if any of five signals indicate an empty index or degraded retrieval.

Environment:
    TARGET_URL  Base URL of the RAGCore API
                (default: https://ragcore-api.onrender.com)
    API_KEY     Value for X-API-Key header; omitted when empty
"""
from __future__ import annotations

import os
import sys
import time

import requests

DEFAULT_TARGET_URL = "https://ragcore-api.onrender.com"
HEALTH_QUERY = "What is a Roth IRA?"
TOP_K = 5
TIMEOUT = 90
COLD_START_SLEEP = 30
FALLBACK_PREFIX = "I could not find this in the provided documents"


def run_query(endpoint: str, headers: dict[str, str]) -> requests.Response:
    payload = {"query": HEALTH_QUERY, "top_k": TOP_K}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=TIMEOUT)
    if resp.status_code == 502:
        print(f"  502 received — waiting {COLD_START_SLEEP}s for Render cold start, retrying...")
        time.sleep(COLD_START_SLEEP)
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=TIMEOUT)
    return resp


def main() -> None:
    target_url = os.environ.get("TARGET_URL", DEFAULT_TARGET_URL).rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    endpoint = f"{target_url}/query"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    print("=== RAGCore Production Health Check ===")
    print(f"Endpoint:  {endpoint}")
    print(f"Query:     {HEALTH_QUERY!r}")
    print()

    try:
        resp = run_query(endpoint, headers)
    except requests.RequestException as e:
        print(f"FAIL  request error: {e}")
        sys.exit(1)

    failures: list[str] = []

    # Check 1: HTTP status
    if resp.status_code != 200:
        failures.append(f"HTTP status {resp.status_code} (expected 200)")

    body: dict = {}
    body_parsed = False
    if resp.status_code == 200:
        try:
            body = resp.json()
            body_parsed = True
        except Exception as e:
            failures.append(f"response is not valid JSON ({type(e).__name__}): {e}")

    answer: str = body.get("answer") or ""
    citations: list = body.get("citations") or []
    total_tokens: int = body.get("total_tokens") or 0
    latency_ms: float = body.get("latency_ms", 0.0)

    if body_parsed:
        # Check 2: answer is not the empty-index fallback
        if answer.startswith(FALLBACK_PREFIX):
            failures.append(f"answer is the empty-index fallback: {answer[:120]!r}")

        # Check 3: citations present
        if not citations:
            failures.append("citations field is empty, null, or missing")

        # Check 4: token count non-zero (LLM was actually called)
        if not total_tokens:
            failures.append("total_tokens is 0 or missing")

        # Check 5: latency_ms >= 100 (sub-100ms signals empty-index fast path)
        if latency_ms < 100:
            failures.append(f"latency_ms={latency_ms:.1f} < 100 (empty-index fast path suspected)")

    answer_preview = answer[:120].replace("\n", " ") if answer else "(empty)"
    print(f"answer:       {answer_preview!r}")
    print(f"citations:    {len(citations)}")
    print(f"total_tokens: {total_tokens}")
    print(f"latency_ms:   {latency_ms:.1f}")
    print()

    check_labels = [("HTTP 200", resp.status_code == 200)]
    if body_parsed:
        check_labels += [
            ("Answer grounded (not fallback text)", not answer.startswith(FALLBACK_PREFIX)),
            (f"Citations present ({len(citations)})", bool(citations)),
            (f"total_tokens non-zero ({total_tokens})", bool(total_tokens)),
            (f"latency_ms >= 100 ({latency_ms:.1f}ms)", latency_ms >= 100),
        ]
    else:
        check_labels += [
            ("Answer grounded (not fallback text)", None),
            ("Citations present", None),
            ("total_tokens non-zero", None),
            ("latency_ms >= 100", None),
        ]
    for label, passed in check_labels:
        mark = "PASS" if passed is True else "SKIP" if passed is None else "FAIL"
        print(f"  [{mark}] {label}")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: PASS")


if __name__ == "__main__":
    main()
