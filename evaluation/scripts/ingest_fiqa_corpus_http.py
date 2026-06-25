"""
Ingest the FiQA corpus into RAGCore via HTTP POST to /ingest/text.

Reads evaluation/datasets/fiqa_corpus.json and POSTs each document to the
production API. Use this to restore the corpus after a persistent-disk migration
or other data-loss event.

Usage:
    python ingest_fiqa_corpus_http.py [--limit N] [--api-key KEY]

Environment:
    TARGET_URL  Base URL of the RAGCore API
                (default: https://kollipara-hema-ragcore.hf.space)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_TARGET_URL = "https://kollipara-hema-ragcore.hf.space"
SLEEP_SECONDS = 1.1      # keeps throughput at ~54 req/min, under the 60/min limit
PROGRESS_EVERY = 25


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest FiQA corpus into RAGCore via HTTP."
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop after N documents (for smoke-testing).",
    )
    parser.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="Value for X-API-Key header (auth currently off; future-proofing).",
    )
    args = parser.parse_args()

    target_url = os.environ.get("TARGET_URL", DEFAULT_TARGET_URL).rstrip("/")
    endpoint = f"{target_url}/ingest/text"

    corpus_path = REPO_ROOT / "evaluation" / "datasets" / "fiqa_corpus.json"
    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found. Run load_fiqa.py first.")
        sys.exit(1)

    with open(corpus_path) as f:
        corpus: list[dict] = json.load(f)

    if args.limit is not None:
        corpus = corpus[: args.limit]

    print(f"Target:         {endpoint}")
    print(f"Corpus loaded:  {len(corpus)} documents")
    if args.limit is not None:
        print(f"Limit:          {args.limit}")
    print()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    failures: list[dict] = []
    ingested = 0
    start = time.monotonic()

    for i, doc in enumerate(corpus, start=1):
        doc_id = doc.get("doc_id", f"idx-{i}")
        text = doc.get("text", "").strip()

        if not text:
            logger.warning("Skipping doc %s — empty text", doc_id)
            failures.append({"doc_id": doc_id, "status_code": None, "error": "empty text"})
            continue

        payload = {
            "text_content": text,
            "metadata": {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "source": "fiqa",
            },
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        except requests.RequestException as e:
            logger.error("Request failed for doc %s: %s", doc_id, e)
            failures.append({"doc_id": doc_id, "status_code": None, "error": str(e)})
            if i < len(corpus):
                time.sleep(SLEEP_SECONDS)
            continue

        # One-time retry on rate limit (in case 1.1s sleep + jitter pushed us over)
        if resp.status_code == 429:
            logger.warning("doc %s → 429 rate-limited, retrying after 2s", doc_id)
            time.sleep(2.0)
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
            except requests.RequestException as e:
                logger.error("Retry failed for doc %s: %s", doc_id, e)
                failures.append({"doc_id": doc_id, "status_code": None, "error": f"retry failed: {e}"})
                if i < len(corpus):
                    time.sleep(SLEEP_SECONDS)
                continue

        # One-time retry on gateway error (HF Space cold-start / restart window)
        if resp.status_code == 502:
            logger.warning("doc %s → 502 gateway error, retrying after 5s", doc_id)
            time.sleep(5.0)
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
            except requests.RequestException as e:
                logger.error("Retry failed for doc %s: %s", doc_id, e)
                failures.append({"doc_id": doc_id, "status_code": None, "error": f"retry failed: {e}"})
                if i < len(corpus):
                    time.sleep(SLEEP_SECONDS)
                continue

        if resp.status_code != 200:
            logger.error("doc %s → HTTP %d: %s", doc_id, resp.status_code, resp.text[:200])
            failures.append({
                "doc_id": doc_id,
                "status_code": resp.status_code,
                "error": resp.text[:200],
            })
        else:
            ingested += 1

        if i % PROGRESS_EVERY == 0 or i == len(corpus):
            elapsed = time.monotonic() - start
            print(
                f"  [{i:>3}/{len(corpus)}] ingested: {ingested}  "
                f"failures: {len(failures)}  ({elapsed:.1f}s)"
            )

        if i < len(corpus):
            time.sleep(SLEEP_SECONDS)

    elapsed = time.monotonic() - start

    print()
    print("=== Ingest complete ===")
    print(f"Documents attempted:   {len(corpus)}")
    print(f"Successfully ingested: {ingested}")
    print(f"Failed / skipped:      {len(failures)}")
    print(f"Time taken:            {elapsed:.1f}s")

    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print(
                f"  doc_id={failure['doc_id']}"
                f"  status={failure['status_code']}"
                f"  error={failure['error'][:120]}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
