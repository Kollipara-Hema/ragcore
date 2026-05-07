# Per-Claim Faithfulness Analysis: Baseline vs Self-RAG

**Date:** 2026-05-05  
**Runs:** baseline_fiqa_2026-05-05_gpt4o-mini.json, self_rag_fiqa_2026-05-05_gpt4o-mini.json  
**Generator:** gpt-4o-mini (both strategies)  
**RAGAS judge:** gpt-4o-mini (in-run scores + patched per-claim pass)  
**Self-RAG verifier:** gpt-4o-mini (same model — judge-controlled comparison)  
**Queries:** 50 paired (baseline and Self-RAG)

---

## Question

The April 2026 analysis found that word-overlap faithfulness improved +0.054 while RAGAS
faithfulness fell −0.109 with Self-RAG, both statistically significant. The mechanism is unclear:
are RAGAS-flagged unsupported claims genuinely unsupported, or is RAGAS extracting different
claims than Self-RAG's internal verifier checked? This analysis tests that question directly.

---

## Method

Two re-runs with gpt-4o-mini as generator, verifier, and RAGAS judge:

1. **Baseline run** (`--strategy basic`): 50 FiQA queries, generation + RAGAS faithfulness.
2. **Self-RAG run** (`--strategy self_rag`): same queries, with claim extraction and internal
   verification during generation; `self_rag_stats` persisted per query.

After the runs, `PatchedFaithfulness` (a `Faithfulness` subclass that intercepts
`NLIStatementOutput` before `_ascore` discards it) re-evaluated both result sets to expose
per-claim verdicts. The sanity check confirmed the patched pass produced aggregate scores
within ≤0.1 mean absolute difference of the stored in-run scores.

**Claim-set mismatch note:** Self-RAG and RAGAS use different prompts to extract atomic claims.
Cross-referencing is at the query level (per-query support rates compared), not the claim level.

---

## Results

### Claim counts per answer

| | RAGAS claims (baseline) | RAGAS claims (Self-RAG) | Self-RAG internal claims |
|---|---|---|---|
| Mean | 10.72 | 5.82 | 4.76 |
| Std  | 8.27 | 5.03 | 2.50 |

### RAGAS support rates

| | Baseline | Self-RAG | Delta |
|---|---|---|---|
| Mean | 0.5602 | 0.4451 | **-0.1151** |
| Std  | 0.3572 | 0.3343 | — |

**Wilcoxon signed-rank:** statistic=140.5, p=0.0073

**Per-query breakdown (|Δ| > 0.05 threshold):**

| Direction | Count |
|---|---|
| Self-RAG better (delta > +0.05) | 7 |
| No change (|delta| ≤ 0.05)      | 21 |
| Self-RAG worse  (delta < −0.05) | 22 |

### Self-RAG internal support rate

| Mean | Std |
|---|---|
| 0.9319 | 0.1746 |

### Agreement between Self-RAG verifier and RAGAS judge

n=38 queries with both internal and RAGAS rates (queries with zero internal claims excluded).

**Pearson r:** -0.1175 (p=0.4822)  
**Spearman ρ:** -0.1170 (p=0.4841)

**Direction agreement (threshold=0.5):**

| | Count | % |
|---|---|---|
| Both judge supported (≥0.5)    | 24 | 63% |
| Both judge unsupported (<0.5)  | 0 | 0% |
| Internal accepts, RAGAS rejects              | 13 | 34% |
| Internal rejects, RAGAS accepts              | 1 | 3% |

---

## Mechanism Interpretation

Pearson r = -0.118, p = 0.4822 (weak or negative correlation).
Spearman ρ = -0.117, p = 0.4841.

Self-RAG internal support rate (mean): 0.9319
RAGAS support rate on Self-RAG (mean): 0.4451

The internal verifier and RAGAS are evaluating different notions of faithfulness, and their disagreement is 
systematic rather than random. The clearest evidence is the directional asymmetry: 34% of queries show 
the internal verifier accepting claims that RAGAS rejects, against only 3% in the reverse direction. If 
the disagreement were primarily stochastic judge noise, the mismatch would appear roughly symmetrically. 
Instead, one evaluator is consistently more permissive toward Self-RAG outputs than the other.

The asymmetry is reinforced by the support-rate gap. Self-RAG's internal verifier marks ~93% of extracted 
claims as supported, while the RAGAS judge marks only ~45% of claims it extracts from the 
same answers. Internally, Self-RAG appears highly grounded; externally, the same answers are only partially 
supported. The practical implication for RAG evaluation is that an internal self-verification signal ("the 
system believes its answers are grounded") and an external faithfulness metric ("an external judge agrees the 
answers are grounded") are measuring substantially different things on this dataset. Reporting only the internal 
signal would obscure systematic over-acceptance relative to an external evaluator.

The disagreement is not merely a thresholding difference over the same latent signal. The per-query Pearson r 
between the two support rates is -0.12 (p=0.48), with Spearman ρ of -0.12 (p=0.48). 
Despite using the same underlying LLM, the evaluators do not share a meaningful ranking signal over which 
answers are more faithful. This implicates prompt framing, claim decomposition, and verification procedure — 
not model capability — as the dominant drivers of evaluator behavior. Changing the judging procedure changes 
the measurement target itself.

A sharper observation: of the 38 queries where both verifiers produced support rates, 24 (63%) showed 
both judges agreeing the answer was supported (rate ≥ 0.5), 13 (34%) showed internal 
accepts / RAGAS rejects, 1 (3%) showed the reverse — and 0 queries 
had both judges agreeing the answer was unsupported. With only the small number of queries either verifier alone 
flags as unfaithful, the two procedures have effectively no agreement on the rejection side. If either evaluator 
were used as a production gate, it would block a different set of answers.

The asymmetry has a directional interpretation when paired with qualitative inspection. On query 2335 (gift tax),
the internal verifier marks a claim about a $28,000 annual exclusion as supported, while RAGAS rejects it citing 
that the retrieved context specifies $14,000, not $28,000. RAGAS catches a real factual error the internal verifier
misses. This single case does not generalize, but it constrains the interpretation: the asymmetry is not simply 
"RAGAS is over-strict." At least some internal-accept/RAGAS-reject disagreements reflect claims the internal 
verifier should have rejected. Determining whether the remaining disagreements reflect additional real errors 
or overzealous RAGAS rejection would require manual labeling.

The sanity-check rerun adds a separate methodological finding. The estimated noise floor of LLM-as-judge 
faithfulness is itself unstable across runs. Two PatchedFaithfulness passes on identical generations produce 
mean per-query differences of ~0.091 on baseline (max 1.000) and ~0.047 on Self-RAG 
(max 0.500); previous runs of this same notebook on the same data produced different values in the
same magnitude range. The metric exhibits measurable evaluator variance even when neither the answers nor the 
contexts change. Single-pass evaluations at n=50 should report multiple-pass averages or explicitly 
bracket point estimates with the observed measurement variance. Reporting faithfulness scores to three decimal 
places, as the April 2026 analysis did, overstates the reliability of the underlying instrument.

The instability extends to significance testing. The Wilcoxon p-value for the RAGAS support-rate regression is 
0.007 on this run; previous runs of the same notebook on the same data produced p-values straddling the 
conventional α = 0.05 threshold. The same data, run twice, can produce different binary conclusions about 
statistical significance. At this n, with this judge, single-pass accept/reject hypothesis-test decisions are 
unreliable; the same result published once would read as "marginal" and republished a day later would read as 
"significant."

Returning to the original question — why does Self-RAG produce a RAGAS regression while improving word-overlap 
faithfulness — the per-claim analysis suggests the regression is real but its magnitude is overstated by 
single-pass LLM judging. The April analysis reported a RAGAS faithfulness drop of −0.109 (p=0.029); this 
judge-controlled reproduction at gpt-4o-mini shows -0.115 with Wilcoxon p=0.007. The remaining 
gap is consistent with two interacting effects: Self-RAG produces more concentrated answers with fewer extracted 
claims per response (5.82 vs 10.72 RAGAS-extracted claims per answer on average), 
increasing the impact of each individual support decision; and its internal verifier applies a systematically 
more permissive notion of grounding than the external RAGAS evaluator. The headline is therefore not "Self-RAG 
hurts faithfulness," but that Self-RAG and RAGAS operationalize faithfulness differently, and the internal 
verifier's leniency obscures part of that disagreement from the production system itself.

---

## Limitations

- **n=50.** Confidence intervals on correlation coefficients are wide at this sample size.
  The Pearson r estimate should be treated as directional.

- **Claim-set mismatch is unresolved.** The analysis operates at the query level because Self-RAG
  and RAGAS extract different atomic claims from the same answer. A true claim-level comparison
  would require a shared claim-extraction pass feeding both verifiers, which was not implemented.

- **Single judge model.** Both verifiers use gpt-4o-mini. The agreement (or disagreement) pattern
  observed here reflects this model's consistency with itself across two different prompt framings,
  not cross-model agreement.

- **RAGAS bimodal distribution.** FiQA financial Q&A scores cluster near 0.0 and 1.0 on RAGAS,
  which inflates variance and compresses the signal in the middle range where disagreements
  would be most informative.

- **LLM-judge non-determinism.** Re-running RAGAS on the same answers produces different
  per-query scores. The patched per-claim pass diverged from the stored in-run scores by
  a mean of 0.0906 on the baseline (max 1.0000) and
  0.0473 on Self-RAG (max 0.5000). One single-claim query
  flipped supported→unsupported between the two passes (diff = 1.0). This noise floor
  is comparable in magnitude to the +0.054 / -0.109 deltas reported in the April 2026
  analysis, indicating that LLM-judged faithfulness measurements at n=50 should be
  treated with effective measurement uncertainty of ±0.05-0.07 even before accounting
  for sampling variance.
- **Self-RAG zero-extraction queries.** Self-RAG's internal claim extractor returned zero claims 
  on 12 of 50 queries (24%), while RAGAS extracted claims on all 50. These 12 queries are excluded 
  from the verifier-agreement comparison. Whether the zero-extraction reflects conservative claim 
  selection by Self-RAG, structural differences in answer shape, or silent extraction failures is 
  not diagnosed here.  

---

## Future Work

- Implement a shared claim-extraction pass to enable claim-level cross-reference rather than
  query-level rate comparison.
- Repeat with a stronger judge (gpt-4o) to test whether the correlation pattern holds.
- Evaluate on a harder retrieval subset (baseline hit@5 < 0.5) where Self-RAG's additional
  retrieval is actually needed and the mechanism question has higher practical stakes.
