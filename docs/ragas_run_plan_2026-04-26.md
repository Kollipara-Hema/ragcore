# RAGAS Re-run Analysis Plan
Date: 2026-04-26 (pre-registered before run)

## What I'm measuring
LLM-judged faithfulness (RAGAS) on the same 50 FiQA queries used in the
existing baseline-vs-Self-RAG benchmark. Same seed, same retrieval path,
same generation strategies. Only the eval metric changes.

## What I expect
The word-overlap benchmark showed faithfulness 0.36 (baseline) → 0.43 
(Self-RAG), +20% relative. I expect RAGAS to show:
[ fill in YOUR honest prior — e.g., "directionally similar but smaller 
delta, perhaps +5-10%" or "uncertain — word-overlap and LLM-judged 
faithfulness measure different things" ]

## Decision rules (committed before seeing results)

If RAGAS delta is statistically significant and ≥10%:
  → Replace word-overlap numbers in README with RAGAS numbers. 
  Headline holds.

If RAGAS delta is positive but smaller (5-10%) or not significant:
  → Report both metrics in README. Word-overlap shows X, RAGAS shows Y. 
  Frame Self-RAG as "modest faithfulness improvement, larger by some 
  metrics than others." No headline number.

If RAGAS delta is zero or negative:
  → Rewrite README. Self-RAG no longer headlined as faithfulness win.
  Reframe as "Self-RAG explored as an alternative; LLM-judged 
  faithfulness was not improved on this corpus. The infrastructure for 
  multi-strategy evaluation is the contribution."

## Statistical reporting (regardless of result)
- 95% confidence interval on the delta via paired bootstrap (n=50 paired
  observations, 1000 resamples). Stats PhD is the edge here.
- Per-query delta distribution, not just means.
- Note that n=50 is small and CIs will be wide — be honest about this.

## What I will NOT do
- Cherry-pick a subset of queries where Self-RAG wins.
- Run with a different seed if numbers look bad.
- Switch metrics post-hoc to find a favorable one.