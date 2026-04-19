"""
Stage B sanity check: run 3 randomly-selected FiQA queries through the
RAGCore orchestrator and verify citations overlap with relevant doc IDs.

Uses GENERATION_STRATEGY=basic for speed. Does NOT start uvicorn.
Results saved to evaluation/results/stage_b_sanity.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["GENERATION_STRATEGY"] = "basic"


def _build_uuid_to_fiqa_id(metadata_pkl: Path) -> dict[str, str]:
    """
    FAISS stores doc_id as UUID but the original FiQA corpus ID is in
    metadata['custom']['doc_id']. Build a lookup so we can translate
    citation UUIDs back to FiQA IDs for overlap checking.
    """
    import pickle
    with open(metadata_pkl, "rb") as f:
        meta_list = pickle.load(f)
    mapping: dict[str, str] = {}
    for m in meta_list:
        uuid_str = str(m.get("doc_id", ""))
        fiqa_id = str(m.get("custom", {}).get("doc_id", ""))
        if uuid_str and fiqa_id:
            mapping[uuid_str] = fiqa_id
    return mapping


async def main() -> None:
    from orchestrator import RAGOrchestrator
    from utils.models import QueryRequest

    eval_path = REPO_ROOT / "evaluation" / "datasets" / "fiqa_eval.json"
    out_path = REPO_ROOT / "evaluation" / "results" / "stage_b_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # UUID → FiQA corpus ID mapping (citations carry UUIDs; qrels use FiQA IDs)
    uuid_to_fiqa = _build_uuid_to_fiqa_id(REPO_ROOT / "faiss_metadata.pkl")

    with open(eval_path) as f:
        golden: list[dict] = json.load(f)

    random.seed(99)
    sampled = random.sample(golden, 3)

    print("Initialising orchestrator …")
    orch = RAGOrchestrator()

    results = []
    overlap_count = 0

    for i, item in enumerate(sampled, start=1):
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])
        print(f"\n[{i}/3] Query: {query!r}")

        request = QueryRequest(query=query)
        response = await orch.query(request)

        cited_doc_ids_uuid = [c.get("doc_id", "") for c in response.citations]
        # Translate UUIDs → FiQA corpus IDs for overlap check
        cited_fiqa_ids = [uuid_to_fiqa.get(str(uid), str(uid)) for uid in cited_doc_ids_uuid]
        overlap = [d for d in cited_fiqa_ids if d in relevant_ids]
        has_overlap = len(overlap) > 0

        if has_overlap:
            overlap_count += 1

        print(f"  Answer (first 200 chars): {response.answer[:200]!r}")
        print(f"  Citations: {len(response.citations)} | Cited FiQA IDs: {cited_fiqa_ids}")
        print(f"  Relevant doc_ids:         {list(relevant_ids)}")
        print(f"  Overlap: {overlap} {'✓' if has_overlap else '✗'}")

        results.append({
            "query_id": item.get("query_id"),
            "query": query,
            "answer": response.answer,
            "relevant_doc_ids": list(relevant_ids),
            "cited_doc_ids": cited_fiqa_ids,
            "overlap": overlap,
            "has_overlap": has_overlap,
            "num_citations": len(response.citations),
            "latency_ms": response.latency_ms,
            "total_tokens": response.total_tokens,
        })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Sanity check summary ===")
    print(f"Queries run:          3")
    print(f"Queries with ≥1 citation: {sum(1 for r in results if r['num_citations'] > 0)} / 3")
    print(f"Queries with overlap: {overlap_count} / 3  (need ≥2 to pass)")
    print(f"Pass: {'YES ✓' if overlap_count >= 2 else 'NO ✗'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
