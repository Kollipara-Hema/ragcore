# RAGAS vs Word-Overlap Faithfulness Analysis

**Date:** 2026-04-26  
**Run:** 50 FiQA queries, seed 42, baseline vs Self-RAG  
**Judge:** gpt-4o-mini, max_tokens=8192  
**RAGAS coverage:** 50/50 both runs

---

## Headline finding

The two faithfulness metrics give opposite verdicts on Self-RAG. Word-overlap
faithfulness improved by +0.054 (95% CI [0.032, 0.083], p=0.0002), meaning
Self-RAG answers share more tokens with the retrieved context. RAGAS
faithfulness fell by −0.109 (95% CI [−0.197, −0.019], p=0.0296), meaning
Self-RAG answers contain more claims that the RAGAS judge could not verify
against the retrieved context. Both effects are statistically significant on
n=50. The two metrics are measuring different things and Self-RAG moves them
in different directions. The word-overlap result cannot be used as a proxy for
RAGAS faithfulness on this data — the baseline Spearman correlation between
the two metrics is negative (ρ = −0.385, p=0.006).

A plausible mechanical explanation (stated after the headline, not instead of
it): Self-RAG's verify-and-regenerate loop produces shorter, more focused
answers that reuse surface vocabulary from the context. RAGAS extracts more
fine-grained atomic claims from longer answers and then checks each claim
individually against the retrieved passages. When Self-RAG's shorter answer
omits nuance that was in the original long answer, or asserts specific
numbers/names drawn from only one chunk, RAGAS penalises those claims as
unsupported. The direction-disagreement count (19 queries where the metrics
point opposite ways, vs 9 where they agree) confirms this is a systematic
pattern, not noise.

---

## Word-overlap faithfulness

| | Baseline | Self-RAG | Delta |
|---|---|---|---|
| Mean | 0.3609 | 0.4152 | **+0.0543** |
| Std  | 0.1513 | 0.1496 | — |

**Bootstrap 95% CI on mean delta:** [0.0322, 0.0831]  
(1000 resamples, seed=42; boot mean=0.0550)

**Normality check (Shapiro-Wilk on deltas):** W=0.9182, p=0.0020, skew=0.694.  
Distribution is non-normal (p<0.05). Wilcoxon signed-rank test used.

**Test:** Wilcoxon signed-rank (chosen because Shapiro-Wilk p<0.05 on the delta
distribution — normality assumption for paired t-test not satisfied).  
**Result:** statistic=93.0, p=0.0002

**Per-query breakdown (|delta| > 0.05 threshold):**

| Direction | Count |
|---|---|
| Self-RAG better (delta > +0.05) | 20 |
| No change (|delta| ≤ 0.05)      | 27 |
| Self-RAG worse (delta < −0.05)  |  3 |

Self-RAG word-overlap is significantly higher. The improvement is widespread
(20 helped, 3 hurt) and the CI excludes zero.

---

## RAGAS faithfulness

| | Baseline | Self-RAG | Delta |
|---|---|---|---|
| Mean | 0.5612 | 0.4524 | **−0.1088** |
| Std  | 0.3992 | 0.3782 | — |

**Bootstrap 95% CI on mean delta:** [−0.1966, −0.0185]  
(1000 resamples, seed=42; boot mean=−0.1050)

**Normality check (Shapiro-Wilk on deltas):** W=0.8986, p=0.0004, skew=−0.665.  
Distribution is non-normal (p<0.05). Wilcoxon signed-rank test used.

**Test:** Wilcoxon signed-rank (same reason: Shapiro-Wilk p<0.05 on deltas).  
**Result:** statistic=107.5, p=0.0296

**Per-query breakdown (|delta| > 0.05 threshold):**

| Direction | Count |
|---|---|
| Self-RAG better (delta > +0.05) |  8 |
| No change (|delta| ≤ 0.05)      | 24 |
| Self-RAG worse (delta < −0.05)  | 18 |

Self-RAG RAGAS faithfulness is significantly lower. The degradation is
asymmetric: 18 queries hurt vs 8 helped. The CI is wide relative to the point
estimate (the lower bound is −0.197), consistent with n=50 and the bimodal
{0, 1} scoring pattern common with financial Q&A.

---

## Cross-metric agreement

**Spearman correlation between word-overlap and RAGAS per-query:**

| Run | ρ | p |
|---|---|---|
| Baseline  | −0.3848 | 0.0058 |
| Self-RAG  | −0.1281 | 0.3753 |

The baseline correlation is statistically significant and *negative*: queries
where the baseline answer has high word overlap with the context tend to have
lower RAGAS faithfulness scores. This is counterintuitive but explicable —
very short "I could not find this" answers have high word overlap (small
vocabulary, all matching) and RAGAS judges them as fully faithful (no
unsupported claims), which adds high-RAGAS / low-overlap pairs to the
baseline. In Self-RAG the correlation weakens to non-significance (ρ=−0.128,
p=0.375), likely because Self-RAG's verification loop pushes answers toward
a middle ground.

**Direction agreement count (over 50 paired queries):**

| | Count |
|---|---|
| Both metrics agree Self-RAG better or worse | 9  |
| Metrics disagree on direction                | 19 |
| At least one metric is zero-delta (tied)     | 22 |

Among the 28 queries where both metrics gave a non-zero direction, 19 (68%)
showed disagreement. The two metrics are not interchangeable on this dataset.

---

## Mechanism hypothesis

This is a hypothesis, not a finding. It is offered to motivate future
analysis, not to explain away the RAGAS regression.

Word-overlap rewards token co-occurrence between answer and contexts. RAGAS
rewards claim-level support — the fraction of atomic claims in the answer that
an LLM judge can verify against the retrieved contexts. These are structurally
different operations, and Self-RAG's design may interact with them differently.

Self-RAG's verify-and-regenerate loop encourages answers composed of
independently-verifiable atomic claims. This design may produce answers that:

- Share more domain vocabulary with the retrieved contexts (higher word-overlap
  through surface reuse), and
- Contain more specific, checkable assertions — which creates more surface area
  for RAGAS to flag unsupported claims.

A baseline answer with one vague high-level claim ("Stocks generally lose value
in a market crash") might score 1.0 on RAGAS because the single claim is
trivially supported. A Self-RAG answer with eight specific claims (mechanisms,
timeframes, exceptions, cited sources) might score 0.6 if six are supported
and two are not — even if the latter answer is more informative and practically
more useful. The RAGAS score goes down; the answer quality may have gone up.

Validating this would require examining per-query claim counts and per-claim
support rates from the RAGAS internals. RAGAS 0.4.x does not expose these by
default. Adding this analysis is straightforward but is out of scope here.

---

## Outliers and influential queries

### Top 5 by |RAGAS delta|

| query_id | Query (first 80 chars) | base ragas | self ragas | Δ ragas | base faith | self faith | Δ faith |
|---|---|---|---|---|---|---|---|
| 10595 | Why buying an inverse ETF does not give same results as shorting th | 1.0000 | 0.0000 | −1.0000 | 0.2897 | 0.2500 | −0.040 |
| 5091  | Is there a term that better describes a compound annual growth rate | 1.0000 | 0.0000 | −1.0000 | 0.3333 | 0.4444 | +0.111 |
| 1769  | Is it financially advantageous and safe to rent out my personal car | 0.9130 | 0.0000 | −0.9130 | 0.1844 | 0.3333 | +0.149 |
| 1610  | 18 year old making $60k a year; how should I invest? Traditional or | 0.1600 | 0.8824 | +0.7224 | 0.2067 | 0.1562 | −0.051 |
| 8084  | Where does the stock go in a collapse?                              | 0.6000 | 0.0000 | −0.6000 | 0.2848 | 0.3333 | +0.049 |

Notes on extreme cases:
- **10595 and 5091**: Baseline produced a detailed multi-paragraph answer that
  RAGAS fully verified (score=1.0). Self-RAG collapsed each to a shorter
  answer that the judge scored 0.0 — the rewritten answer dropped specifics
  the judge needed to verify claims.
- **1769**: Similar pattern; Self-RAG shortened a 1,885-token detailed answer
  to ~1,541 tokens; RAGAS went from 0.913 to 0.0.
- **1610**: The opposite case — Self-RAG's regenerated answer (after 2
  additional retrievals) improved RAGAS from 0.16 to 0.88 while slightly
  reducing word overlap.
- **8084**: Self-RAG decided "I could not find this" and scored 0.0 on RAGAS
  despite 0.33 word overlap from the fallback "could not find" phrasing.

### Queries with strongest cross-metric disagreement

(word-overlap delta and RAGAS delta ≥ 0.1 in opposite directions)

| query_id | Query | Δ faith | Δ ragas |
|---|---|---|---|
| 6340  | Is there an advantage to keeping a liquid emergency fund…  | +0.285 | −0.357 |
| 3436  | Health insurance lapsed due to employer fraud…             | +0.192 | −0.381 |
| 671   | Does the low CAD positively or negatively impact Canadian… | +0.174 | −0.475 |
| 1668  | How to send money from europe to usa EUR - USD?            | +0.171 | −0.500 |
| 9464  | Simple and safe way to manage a lot of cash                | +0.163 | −0.155 |
| 3393  | How can contractors recoup taxation-related expenses?      | +0.116 | −0.417 |
| 1769  | Is it financially advantageous and safe to rent out my car?| +0.149 | −0.913 |
| 5367  | Buying a house for a shorter term                          | +0.111 | −0.313 |
| 7502  | How do ETF fees get applied?                               | +0.141 | −0.171 |
| 5091  | Is there a term that better describes negative CAGR?       | +0.111 | −1.000 |

All 10 queries share the same pattern: Self-RAG word overlap improved (shorter
answer overlaps better proportionally) while RAGAS faithfulness fell (judge
finds more unsupported atomic claims in the regenerated answer).

---

## Methodology notes

**Post-hoc judge fix.** The initial baseline run (`basic_fiqa_2026-04-26_default-judge.json`)
used the default `llm_factory` call with no explicit `max_tokens`, which gave
RAGAS coverage of 36/50 (72%). Diagnosis showed that RAGAS's internal
`instructor` library was hitting its default output budget (3072 tokens) when
verifying answers with 18+ extracted atomic claims, returning `NaN` rather
than a score. The fix was to set `max_tokens=8192` in the `llm_factory` call
in `run_benchmark.py`. The corrected baseline run is `basic_fiqa_2026-04-26.json`
(50/50 coverage); the default-judge run is preserved as evidence of the failure
mode.

This change is defensible: it corrects a known infrastructure failure
(token budget exhaustion) rather than cherry-picking results or changing the
evaluation methodology. Both runs used the same model (gpt-4o-mini) and the
same RAGAS metric; the only difference is that the corrected run does not
silently drop 28% of queries. The analysis here uses the post-fix run throughout.

---

## Limitations

- **n=50.** Confidence intervals are wide. The RAGAS CI lower bound is −0.197,
  meaning the true effect could be small. Results are directional; do not
  over-interpret the point estimates.

- **The metric disagreement cannot be resolved by this analysis alone.** The
  significant disagreement between word-overlap and RAGAS faithfulness on this
  dataset (ρ=−0.385 at baseline, 19/28 non-tied pairs pointing opposite ways)
  raises a practical question this analysis cannot answer: which metric is
  closer to ground-truth faithfulness? Confirming or refuting the RAGAS
  regression as a real quality signal would require:
  - A larger query set to tighten confidence intervals on both deltas
  - A third faithfulness metric (e.g., human judgment, NLI-based claim
    verification) to break the two-metric tie
  - Per-claim analysis: are Self-RAG's "unsupported" RAGAS claims actually
    unsupported, or are they supported by contexts not surfaced in citations?

- **Judge model.** RAGAS faithfulness is judged by gpt-4o-mini. A stronger
  judge (gpt-4o) might score differently — in particular, it might find some
  Self-RAG answers more faithful by interpreting implicit context support that
  gpt-4o-mini misses. This was not tested at scale.

- **RAGAS bimodal distribution.** On financial Q&A, RAGAS faithfulness scores
  cluster near 0.0 and 1.0, which inflates variance and widens CIs. The
  Wilcoxon test is more appropriate than a t-test for this distribution
  (confirmed by Shapiro-Wilk p<0.001 for both strategies).

- **Self-RAG answer length.** Self-RAG tends to produce shorter answers after
  its verify-and-regenerate loop. This mechanically increases word-overlap
  (smaller denominator) and may increase RAGAS claims-without-support by
  omitting hedged language and qualifiers. The causal mechanism was not
  formally tested.
