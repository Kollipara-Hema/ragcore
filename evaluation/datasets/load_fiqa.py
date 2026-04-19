"""
Download and prepare the FiQA-2018 benchmark dataset for RAGCore evaluation.

Selects 50 queries by random.sample with seed=42 for reproducibility.
Corpus subset includes all relevant docs for those queries plus
100 random distractors (seed=43).

Outputs:
  evaluation/datasets/fiqa_eval.json   — 50 golden Q&A pairs
  evaluation/datasets/fiqa_corpus.json — corpus subset (~150-200 docs)

FiQA has no ground-truth answer text; the first relevant document's text
is used as the ground_truth proxy (standard BEIR practice).

golden_dataset dict schema (matches RAGEvaluator.run_against_orchestrator):
  {
    "query":            str,         # question text
    "ground_truth":     str,         # first relevant doc text (proxy)
    "relevant_doc_ids": list[str],   # all relevant doc IDs for this query
  }

corpus dict schema:
  {
    "doc_id": str,
    "title":  str,
    "text":   str,
  }
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from datasets import load_dataset  # type: ignore

    # ── 1. Load FiQA from HuggingFace ────────────────────────────────────────
    print("Loading FiQA corpus …")
    corpus_ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    print("Loading FiQA queries …")
    queries_ds = load_dataset("BeIR/fiqa", "queries", split="queries")
    print("Loading FiQA qrels (train) …")
    qrels_ds = load_dataset("BeIR/fiqa-qrels", split="train")

    # ── 2. Build lookup dicts ─────────────────────────────────────────────────
    # corpus: str(doc_id) -> {title, text}
    corpus_map: dict[str, dict] = {
        row["_id"]: {"doc_id": row["_id"], "title": row["title"], "text": row["text"]}
        for row in corpus_ds
    }

    # queries: str(query_id) -> query text
    query_map: dict[str, str] = {row["_id"]: row["text"] for row in queries_ds}

    # qrels: str(query_id) -> list[str(corpus_id)]
    qrels: dict[str, list[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        qrels.setdefault(qid, []).append(cid)

    # ── 3. Filter to queries that have ≥1 relevant doc in our corpus ──────────
    eligible_qids = [
        qid for qid in qrels
        if qid in query_map and any(cid in corpus_map for cid in qrels[qid])
    ]
    print(f"Eligible queries (have ≥1 relevant corpus doc): {len(eligible_qids)}")

    # ── 4. Sample 50 queries with fixed seed ─────────────────────────────────
    random.seed(42)
    selected_qids = random.sample(eligible_qids, 50)

    # ── 5. Collect relevant doc IDs for selected queries ──────────────────────
    relevant_doc_ids_by_query: dict[str, list[str]] = {}
    all_relevant_cids: set[str] = set()
    for qid in selected_qids:
        rel_cids = [cid for cid in qrels[qid] if cid in corpus_map]
        relevant_doc_ids_by_query[qid] = rel_cids
        all_relevant_cids.update(rel_cids)

    # ── 6. Sample 100 distractor docs with a different fixed seed ─────────────
    non_relevant_cids = [cid for cid in corpus_map if cid not in all_relevant_cids]
    random.seed(43)
    distractor_cids = random.sample(non_relevant_cids, min(100, len(non_relevant_cids)))

    corpus_subset_cids = list(all_relevant_cids) + distractor_cids

    # ── 7. Build golden dataset list ─────────────────────────────────────────
    golden_dataset: list[dict] = []
    for qid in selected_qids:
        rel_cids = relevant_doc_ids_by_query[qid]
        first_rel_doc = corpus_map[rel_cids[0]] if rel_cids else {}
        golden_dataset.append({
            "query_id": qid,
            "query": query_map[qid],
            "ground_truth": first_rel_doc.get("text", ""),
            "relevant_doc_ids": rel_cids,
        })

    # ── 8. Build corpus subset list ───────────────────────────────────────────
    corpus_subset: list[dict] = [corpus_map[cid] for cid in corpus_subset_cids]

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    eval_path = SCRIPT_DIR / "fiqa_eval.json"
    corpus_path = SCRIPT_DIR / "fiqa_corpus.json"

    with open(eval_path, "w") as f:
        json.dump(golden_dataset, f, indent=2)
    with open(corpus_path, "w") as f:
        json.dump(corpus_subset, f, indent=2)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    query_lengths = [len(item["query"].split()) for item in golden_dataset]
    doc_lengths = [len(doc["text"].split()) for doc in corpus_subset]
    total_relevant = sum(len(v) for v in relevant_doc_ids_by_query.values())

    print(f"\n=== Summary ===")
    print(f"Queries selected:          {len(golden_dataset)}")
    print(f"Relevant docs (unique):    {len(all_relevant_cids)}")
    print(f"Distractor docs:           {len(distractor_cids)}")
    print(f"Corpus subset total:       {len(corpus_subset)}")
    print(f"Avg relevant docs/query:   {total_relevant / len(golden_dataset):.2f}")
    print(f"Avg query length (words):  {sum(query_lengths) / len(query_lengths):.1f}")
    print(f"Avg doc length (words):    {sum(doc_lengths) / len(doc_lengths):.1f}")
    print(f"\nSaved: {eval_path}")
    print(f"Saved: {corpus_path}")
    print(f"fiqa_eval.json size:    {eval_path.stat().st_size / 1024:.1f} KB")
    print(f"fiqa_corpus.json size:  {corpus_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
