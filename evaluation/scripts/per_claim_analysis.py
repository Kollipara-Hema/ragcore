"""
Per-claim RAGAS faithfulness analysis helpers.

Provides PatchedFaithfulness — a Faithfulness subclass that side-channels
per-claim NLI verdicts before _ascore discards them — plus DataFrame builders
used by the analysis notebook.

Usage (notebook):
    import sys; sys.path.insert(0, str(Path("../scripts").resolve()))
    from per_claim_analysis import (
        PatchedFaithfulness, run_ragas_with_claims,
        build_claim_df, build_selfrag_internal_df, build_query_level_df,
    )
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

try:
    from ragas.metrics import Faithfulness
    from ragas import evaluate as _ragas_evaluate
    from datasets import Dataset as HFDataset
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    Faithfulness = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    import pandas as pd


class PatchedFaithfulness(Faithfulness):  # type: ignore[misc]
    """
    Subclass of ragas.metrics.Faithfulness that captures per-claim NLI verdicts
    into self.claim_cache before _ascore discards the NLIStatementOutput.

    The aggregate score and all other behaviour are unchanged.  One entry is
    appended to claim_cache per _ascore call (one per query row), so the list
    aligns index-to-index with the HFDataset rows passed to evaluate().

    Entry schema:
        {"user_input": str, "claims": [{"statement": str, "verdict": int, "reason": str}]}
    verdict is 1 (supported) or 0 (unsupported), matching RAGAS internals.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.claim_cache: list[dict] = []

    async def _ascore(self, row: dict, callbacks) -> float:
        assert self.llm is not None

        statements_obj = await self._create_statements(row, callbacks)
        statements = statements_obj.statements
        if not statements:
            self.claim_cache.append({
                "user_input": row.get("user_input", ""),
                "claims": [],
            })
            return np.nan

        verdicts = await self._create_verdicts(row, statements, callbacks)

        self.claim_cache.append({
            "user_input": row.get("user_input", ""),
            "claims": [
                {
                    "statement": v.statement,
                    "verdict": v.verdict,    # int: 1=supported, 0=unsupported
                    "reason": v.reason,
                }
                for v in verdicts.statements
            ],
        })

        return self._compute_score(verdicts)


def run_ragas_with_claims(
    records: list[dict],
    ragas_llm,
) -> tuple[list[float | None], list[dict]]:
    """
    Run PatchedFaithfulness on a list of benchmark result records.

    Requires each record to have "query", "generated_answer", and
    "retrieved_contexts" keys (retrieved_contexts is stored by the updated
    run_benchmark.py; older JSON files will get empty context lists).

    Returns:
        scores: per-query aggregate faithfulness (NaN for empty extractions)
        claim_cache: per-query claim dicts, aligned index-to-index with records
    """
    if not _RAGAS_AVAILABLE:
        raise ImportError("ragas and datasets are required for per-claim analysis")

    metric = PatchedFaithfulness()
    dataset = HFDataset.from_dict({
        "user_input": [r["query"] for r in records],
        "response": [r["generated_answer"] for r in records],
        "retrieved_contexts": [
            r.get("retrieved_contexts", r.get("contexts", []))
            for r in records
        ],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _ragas_evaluate(
            dataset,
            metrics=[metric],
            llm=ragas_llm,
            show_progress=True,
        )

    raw = result["faithfulness"]
    if isinstance(raw, (int, float)):
        scores: list[float | None] = [float(raw)]
    else:
        scores = [float(s) if s is not None else None for s in raw]

    return scores, metric.claim_cache


# ── DataFrame builders ────────────────────────────────────────────────────────

def build_claim_df(
    records: list[dict],
    claim_cache: list[dict],
    strategy: str,
) -> "pd.DataFrame":
    """
    Explode claim_cache into a per-claim DataFrame.

    Columns: query_id, query_idx, strategy, statement, verdict (0/1), reason
    """
    import pandas as pd
    rows = []
    for idx, (rec, entry) in enumerate(zip(records, claim_cache)):
        qid = rec.get("query_id", str(idx))
        for claim in entry.get("claims", []):
            rows.append({
                "query_id": qid,
                "query_idx": idx,
                "strategy": strategy,
                "statement": claim["statement"],
                "verdict": claim["verdict"],
                "reason": claim["reason"],
            })
    return pd.DataFrame(rows, columns=["query_id", "query_idx", "strategy",
                                       "statement", "verdict", "reason"])


def build_selfrag_internal_df(records: list[dict]) -> "pd.DataFrame":
    """
    Explode self_rag_stats.verified_claims / unsupported_claims into per-claim rows.

    Columns: query_id, query_idx, claim_text, self_rag_verdict (1=verified, 0=unsupported)
    """
    import pandas as pd
    rows = []
    for idx, rec in enumerate(records):
        qid = rec.get("query_id", str(idx))
        stats = rec.get("self_rag_stats", {})
        for claim in stats.get("verified_claims", []):
            rows.append({
                "query_id": qid,
                "query_idx": idx,
                "claim_text": claim,
                "self_rag_verdict": 1,
            })
        for claim in stats.get("unsupported_claims", []):
            rows.append({
                "query_id": qid,
                "query_idx": idx,
                "claim_text": claim,
                "self_rag_verdict": 0,
            })
    return pd.DataFrame(rows, columns=["query_id", "query_idx",
                                       "claim_text", "self_rag_verdict"])


def build_query_level_df(
    records_baseline: list[dict],
    records_selfrag: list[dict],
    claims_baseline: list[dict],
    claims_selfrag: list[dict],
) -> "pd.DataFrame":
    """
    Build per-query summary DataFrame joining baseline and self_rag.

    Columns:
      query_id,
      n_ragas_claims_basic, ragas_support_rate_basic, stored_ragas_basic,
      n_ragas_claims_selfrag, ragas_support_rate_selfrag, stored_ragas_selfrag,
      n_internal_claims, internal_support_rate, internal_faith_score
    """
    import pandas as pd

    def _per_query_ragas(records: list[dict], cache: list[dict]) -> pd.DataFrame:
        rows = []
        for rec, entry in zip(records, cache):
            claims = entry.get("claims", [])
            n = len(claims)
            supported = sum(1 for c in claims if c["verdict"] == 1)
            rows.append({
                "query_id": rec.get("query_id"),
                "n_claims": n,
                "support_rate": supported / n if n > 0 else None,
                "stored_ragas": rec.get("ragas_faithfulness"),
            })
        return pd.DataFrame(rows)

    def _per_query_internal(records: list[dict]) -> pd.DataFrame:
        rows = []
        for rec in records:
            stats = rec.get("self_rag_stats", {})
            v = len(stats.get("verified_claims", []))
            u = len(stats.get("unsupported_claims", []))
            n = v + u
            rows.append({
                "query_id": rec.get("query_id"),
                "n_internal_claims": n,
                "internal_support_rate": v / n if n > 0 else None,
                "internal_faith_score": stats.get("faithfulness_score"),
            })
        return pd.DataFrame(rows)

    df_b = _per_query_ragas(records_baseline, claims_baseline).rename(columns={
        "n_claims": "n_ragas_claims_basic",
        "support_rate": "ragas_support_rate_basic",
        "stored_ragas": "stored_ragas_basic",
    })
    df_sr = _per_query_ragas(records_selfrag, claims_selfrag).rename(columns={
        "n_claims": "n_ragas_claims_selfrag",
        "support_rate": "ragas_support_rate_selfrag",
        "stored_ragas": "stored_ragas_selfrag",
    })
    df_int = _per_query_internal(records_selfrag)

    return (
        df_b
        .merge(df_sr, on="query_id", how="inner")
        .merge(df_int, on="query_id", how="left")
    )
