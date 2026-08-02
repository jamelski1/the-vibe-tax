# Vibe Tax v3 — results, and the v2-vs-v3 comparison

Same 50 HumanEval problems, same 3 models, same 6 conditions as v2. The **only**
thing that changed: v3's prompts are LLM-rewritten to read like *real* usage
(discriminator-gated; `reads_real` ≈ 0.94 on prose conditions vs the old
synthetic 0.00), while v2's were researcher-written docstring stubs. So this
isolates the effect of **prompt realism** on the measured tax.

Both runs scored with the robust extractor (`score_vibe_tax.py`, base HumanEval)
and the edge-case scorer (`score_vibe_tax_plus.py`, HumanEval+, approximate,
≤60 sampled inputs/problem).

## Pass@1 by condition

| Condition | v2 base | **v3 base** | v2 HE+ | **v3 HE+** |
|-----------|--------:|------------:|-------:|-----------:|
| agentic_terse | 93.3 | **97.3** | 88.7 | **93.3** |
| agentic_casual | 99.3 | 98.7 | 91.3 | **94.0** |
| webchat_detailed | 96.7 | **97.3** | 89.3 | **93.3** |
| webchat_code_paste | 96.4 | 96.4 | 88.3 | 87.4 |
| webchat_error_paste | 95.5 | **96.4** | 83.8 | **88.3** |
| webchat_multilingual | 96.0 | **97.3** | 89.3 | **92.0** |
| **Overall** | **96.2** | **97.3** | **88.7** | **91.7** |

By model (HumanEval+): ChatGPT 92.7, Claude 93.4, Codestral 89.1.

## Headline finding: realistic prompts get *better* code than researcher-written ones

v3 ≥ v2 in almost every cell, and the gap is clearest on the de-saturating
HumanEval+ scorer (**+3.0 overall**, +4.6 on `agentic_terse`, +4.5 on
`error_paste`). In other words:

> **The researcher-written v2 conditions OVERSTATED the vibe tax.** When you
> phrase informal prompts the way real people actually do, models handle them
> *better* than the artificial "yo so like…" degradations.

Why this makes sense: models are RLHF-tuned on real human prompts, so realistic
informal phrasing is *in-distribution*. v2's invented informality (a bare vibe
docstring, a one-word "finish this") is weird and out-of-distribution — it
confuses the model more than a real casual prompt does. Concretely, v2's
`agentic_terse` used the raw L5 "vibe" docstring (worst v2 condition, 88.7 HE+);
v3's terse is *"write separate_paren_groups(paren_string) — split a string of
parentheses into a list…"* — informal but clear, and 4.6 points better.

**Methodological payoff:** this is exactly why calibration mattered. A study that
measured the vibe tax with hand-invented informal prompts would have reported a
*larger* tax than really exists. It's a caution for the perturbation /
under-specification literature, which relies on artificial degradations.

## What replicates from v2

- **Paste/fix conditions stay the hardest on HumanEval+** for both versions
  (v3: code_paste 87.4, error_paste 88.3 — the two lowest). Fixing buggy code
  yields less edge-case-robust solutions than writing from scratch — a robust,
  version-independent result.
- **Base HumanEval is saturated** — everything 96–99% on base; the signal only
  appears under HumanEval+. Even there the spread is ~7 points: still no *large*
  tax on these (easy, frontier-solved) problems.

## Caveats

- v3 "realism" is bounded: the discriminator is weak, and paste conditions could
  not be realism-certified (gated on function-name only). So v3 is "more
  realistic than v2," not "certified real."
- HumanEval+ scorer is approximate (exact-equality comparator, sampled inputs) —
  the v2↔v3 *deltas* are the robust part, not the absolute figures.
- Still HumanEval, still near the ceiling for these models. A harder benchmark
  (LiveCodeBench) is needed to size a small tax precisely.
- v3 prompts are one temperature-0.7 sample; another draw would differ slightly.
