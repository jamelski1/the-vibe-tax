# Partial credit across all three benchmarks (the saturation gradient, at the test level)

The same three models (GPT-5.4, Claude-opus-4-6, Codestral) × 4 framings = **12
attempts** per problem, scored the **same way** on every benchmark: the fraction
of a problem's tests each attempt passes (partial credit), then aggregated per
problem. This removes the binary "solved / not" cliff and shows capability as a
continuum — and shows that what changes across benchmarks is *headroom*, not the
models.

| benchmark | mean-attempt test-pass | best-attempt | fully solved | n |
|-----------|-----------------------:|-------------:|:-----------:|--:|
| HumanEval        | **96.6%** (93.1% raw) | ~100% | 40/50 | 50 |
| HumanEval+       | **96.1%** (91.1% raw) | ~100% | 38/50 | 50 |
| **LiveCodeBench**| **81.0%** | 98.5% | 35/167 | 167 |
| &nbsp;&nbsp;— LCB easy   | 95.0% | 100.0% | 26/43 | 43 |
| &nbsp;&nbsp;— LCB medium | 82.8% | 98.6% | 9/73 | 73 |
| &nbsp;&nbsp;— **LCB hard** | **66.7%** | 97.1% | 0/51 | 51 |

Sources: `data/vibe_tax_v2/he_partial_credit.json`,
`he_plus_partial_credit.json`, `data/vibe_tax_lcb/full_partial_credit.json`.

## Reading it

- **Mean-attempt test-pass rate tracks benchmark headroom exactly:**
  ~96% (HumanEval / HumanEval+) → 81% (LCB overall) → 67% (LCB hard). Same models,
  same method; only the benchmark changes. This is the saturation thesis shown at
  the *test* level, not just the problem level.
- **Capability is a continuum.** Binary pass@1 says "LCB hard = 0/51 solved," which
  reads as a wall. But the **best attempt passes 97% of a hard problem's tests on
  average**, and the average attempt 67%. The models solve nearly the whole test
  suite and miss an edge case or two — they are not incapable, they are imprecise.
- **HumanEval and HumanEval+ are both at the ceiling** once the approximate-scorer
  artifacts are set aside — the EvalPlus edge tests barely move the *partial* rate
  (they mostly convert a few near-100% attempts into 90-something), so neither
  benchmark leaves room to size a prompt- or model-level effect.

## Caveats

- HumanEval uses an **assert-based** approximate scorer, HumanEval+ an
  **output-equivalence** approximate scorer (sampled inputs vs the canonical
  solution). A handful of problems break those approximations (nested asserts in a
  loop: `HE/32`; and `HE/10`, `HE/163`), giving spuriously low partial rates; the
  starred means exclude them. **LiveCodeBench uses the real graded tests** with the
  fixed extractor + per-test timeout — no exclusions.
- All numbers are the three-model main study (GPT-5.4 / Claude / Codestral).
  GPT-5.6 is a separate ablation and is not included.
