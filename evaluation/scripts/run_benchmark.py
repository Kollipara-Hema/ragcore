"""
Stage C benchmark runner: baseline vs Self-RAG on FiQA-2018.

Usage:
    python evaluation/scripts/run_benchmark.py --strategy basic
    python evaluation/scripts/run_benchmark.py --strategy self_rag

GENERATION_STRATEGY is set via environment variable before any ragcore
imports, because config/settings.py reads it at module import time.

Why this runner exists instead of using run_against_orchestrator():
  - run_against_orchestrator() builds retrieved_doc_ids from citation UUIDs,
    which never match FiQA's integer-string corpus IDs. All retrieval metrics
    would silently be 0.0.
  - It returns only an aggregate EvalReport with no per-query breakdown.
  - Failed queries are silently dropped, not recorded.
  This runner handles all three issues directly.

Per-query results saved to evaluation/results/{strategy}_fiqa.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Parse strategy before importing anything from ragcore ────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy", choices=["basic", "self_rag"], required=True,
    help="Generation strategy to benchmark",
)
args = parser.parse_args()

os.environ["GENERATION_STRATEGY"] = args.strategy

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UUID → FiQA-ID translation (same workaround as stage_b_sanity.py)
# The ingestion pipeline assigns new UUIDs; FiQA corpus IDs are stored in
# metadata["custom"]["doc_id"]. Without this mapping every retrieval metric
# would be 0.0. See Commit 4 TODO for the right long-term fix.
# ─────────────────────────────────────────────────────────────────────────────

def build_uuid_to_fiqa_id(metadata_pkl: Path) -> dict[str, str]:
    with open(metadata_pkl, "rb") as f:
        meta_list = pickle.load(f)
    mapping: dict[str, str] = {}
    for m in meta_list:
        uuid_str = str(m.get("doc_id", ""))
        fiqa_id = str(m.get("custom", {}).get("doc_id", ""))
        if uuid_str and fiqa_id:
            mapping[uuid_str] = fiqa_id
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics (per-query)
# ─────────────────────────────────────────────────────────────────────────────

def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def retrieval_metrics(retrieved: list[str], relevant: list[str]) -> dict[str, float]:
    if not retrieved or not relevant:
        return {"hit_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "precision_at_5": 0.0, "recall_at_5": 0.0}

    # FiQA ground-truth labels are at the document level, but retrieval
    # operates at the chunk level. Multiple chunks from the same document
    # share a doc_id after UUID translation. Deduping by first-occurrence
    # rank aligns metric granularity with label granularity.
    seen: set[str] = set()
    deduped: list[str] = []
    for d in retrieved:
        if d not in seen:
            seen.add(d)
            deduped.append(d)

    rel_set = set(relevant)
    top5 = deduped[:5]
    hits = sum(1 for d in top5 if d in rel_set)

    rr = 0.0
    for rank, d in enumerate(deduped, start=1):
        if d in rel_set:
            rr = 1.0 / rank
            break

    return {
        "hit_at_5": 1.0 if any(d in rel_set for d in top5) else 0.0,
        "mrr": rr,
        "ndcg_at_5": _ndcg_at_k(deduped, rel_set),
        "precision_at_5": hits / len(top5) if top5 else 0.0,
        "recall_at_5": hits / len(rel_set) if rel_set else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic faithfulness (word-overlap proxy; same as GenerationEvaluator)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    answer_words = set(answer.lower().split())
    context_words: set[str] = set()
    for ctx in contexts:
        context_words |= set(ctx.lower().split())
    overlap = len(answer_words & context_words)
    return min(1.0, overlap / max(len(answer_words), 1))


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], strategy: str, total_sec: float) -> None:
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    metric_keys = ["hit_at_5", "mrr", "ndcg_at_5", "precision_at_5", "recall_at_5", "faithfulness"]
    latencies = [r["latency_ms"] for r in ok]
    tokens = [r["total_tokens"] for r in ok if r.get("total_tokens")]

    w = 68
    print(f"\n{'='*w}")
    print(f"  Strategy: {strategy}  |  Queries: {len(ok)} ok, {len(failed)} failed  |  Wall time: {total_sec:.1f}s")
    print(f"{'='*w}")
    print(f"  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*(w-4)}")
    for key in metric_keys:
        vals = [r[key] for r in ok if key in r]
        if vals:
            std = stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {key:<22}  {mean(vals):>8.4f}  {std:>8.4f}  {min(vals):>8.4f}  {max(vals):>8.4f}")
    print(f"  {'-'*(w-4)}")
    if latencies:
        std = stdev(latencies) if len(latencies) > 1 else 0.0
        print(f"  {'latency_ms':<22}  {mean(latencies):>8.1f}  {std:>8.1f}  {min(latencies):>8.1f}  {max(latencies):>8.1f}")
    if tokens:
        std = stdev(tokens) if len(tokens) > 1 else 0.0
        print(f"  {'total_tokens':<22}  {mean(tokens):>8.1f}  {std:>8.1f}  {min(tokens):>8.1f}  {max(tokens):>8.1f}")
    print(f"{'='*w}")
    if failed:
        print(f"\n  Failed queries ({len(failed)}):")
        for r in failed:
            print(f"    [{r.get('query_id')}] {r.get('query', '')[:60]}  →  {r['error']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(strategy: str) -> None:
    from orchestrator import RAGOrchestrator
    from utils.models import QueryRequest

    eval_path = REPO_ROOT / "evaluation" / "datasets" / "fiqa_eval.json"
    out_path = REPO_ROOT / "evaluation" / "results" / f"{strategy}_fiqa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_pkl = REPO_ROOT / "faiss_metadata.pkl"

    with open(eval_path) as f:
        golden: list[dict] = json.load(f)

    print(f"Building UUID→FiQA-ID map from {meta_pkl.name} …")
    uuid_to_fiqa = build_uuid_to_fiqa_id(meta_pkl)
    print(f"  {len(uuid_to_fiqa)} chunk entries indexed")

    print(f"Initialising orchestrator (strategy={strategy}) …")
    orch = RAGOrchestrator()
    print(f"Running {len(golden)} queries …\n")

    results: list[dict] = []
    wall_start = time.monotonic()

    for i, item in enumerate(golden, start=1):
        qid = item.get("query_id", str(i))
        query = item["query"]
        relevant_ids = item.get("relevant_doc_ids", [])

        try:
            t0 = time.monotonic()
            response = await orch.query(QueryRequest(query=query))
            latency_ms = (time.monotonic() - t0) * 1000

            # Translate UUIDs → FiQA corpus IDs
            cited_uuids = [c.get("doc_id", "") for c in response.citations]
            cited_fiqa = [uuid_to_fiqa.get(str(u), str(u)) for u in cited_uuids]
            contexts = [c.get("excerpt", "") for c in response.citations]

            ret = retrieval_metrics(cited_fiqa, relevant_ids)
            faith = heuristic_faithfulness(response.answer, contexts)

            record: dict = {
                "query_id": qid,
                "query": query,
                "generated_answer": response.answer,
                "relevant_doc_ids": relevant_ids,
                "cited_doc_ids": cited_fiqa,
                "latency_ms": round(latency_ms, 1),
                "total_tokens": response.total_tokens,
                "faithfulness": round(faith, 4),
                **{k: round(v, 4) for k, v in ret.items()},
            }
            if response.self_rag_stats:
                record["self_rag_stats"] = response.self_rag_stats
            results.append(record)

            hit = "✓" if ret["hit_at_5"] else "✗"
            print(
                f"[{i:>2}/{len(golden)}] {hit} "
                f"MRR={ret['mrr']:.3f}  faith={faith:.3f}  "
                f"{latency_ms:.0f}ms  '{query[:50]}'"
            )

        except Exception as e:
            logger.error("Query %s failed: %s", qid, e)
            results.append({
                "query_id": qid,
                "query": query,
                "error": str(e),
            })
            print(f"[{i:>2}/{len(golden)}] ERROR — {e}")

    total_sec = time.monotonic() - wall_start

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results, strategy, total_sec)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(run(args.strategy))
