# Vibe Tax v2 — first results

Run: 274 prompts × 3 models = 822 completions (ChatGPT `gpt-5.4`, Claude
`claude-opus-4-6`, Codestral `codestral-latest`), temperature 0. Scored against
the official HumanEval tests with `score_vibe_tax.py`.

## ⚠️ Read this first: scoring is the hard part

The raw run scored with the v1 tester gave **`webchat_error_paste = 0%` for all
three models** — which was a **scoring artifact, not a result**. Paste prompts
("here's my error, fix it") elicit *conversational* replies: prose, a snippet of
the buggy code, then "Fixed version:" + the corrected function. The v1
body-extractor grabbed the *first* code-like line (the buggy snippet in the
explanation) and mangled it. A second artifact — models mis-indenting only the
first line of a bare body — suppressed ~13% of Claude's completions.

`score_vibe_tax.py` fixes both by trying several extractions per completion
(last full `def`, fenced code block, first-line-repaired body, v1 normalizer)
and passing if any runs correctly. It **cannot inflate** correctness — every
candidate must still pass the real assertions; extraction only recovers
formatting. After the fix, of 31 residual failures **24 are genuine wrong
answers** and ~4 are long-tail extraction edge cases (~0.5%).

**Methodological takeaway:** for prompt-style studies, how you extract code from
the reply can swing a condition by 90+ points. Conversational/paste prompts are
especially fragile. Any "vibe tax" number is meaningless without a
paste-robust, multi-candidate scorer.

## Results (pass@1)

| Condition | ChatGPT | Claude | Codestral | **All** |
|-----------|--------:|-------:|----------:|--------:|
| agentic_terse (L5 vibe docstring, "finish this") | 90.0 | 98.0 | 92.0 | **93.3** |
| agentic_casual (L3 docstring) | 100.0 | 100.0 | 98.0 | **99.3** |
| webchat_detailed (formal spec, polite) | 100.0 | 100.0 | 90.0 | **96.7** |
| webchat_code_paste (buggy code) | 97.3 | 97.3 | 94.6 | **96.4** |
| webchat_error_paste (buggy code + real traceback) | 94.6 | 100.0 | 91.9 | **95.5** |
| webchat_multilingual (Chinese wrapper) | 100.0 | 100.0 | 88.0 | **96.0** |
| **by model** | **97.1** | **99.3** | **92.3** | 96.2 |

By medium: **agentic 96.3% vs web-chat 96.2%** — identical.

## What the numbers say (and don't)

- **The vibe tax is small on HumanEval** — every condition lands in a 93–99%
  band (~6-point spread). Frontier models are largely robust to prompt style.
- **H1 (spec tax): weakly supported, only at the extreme.** The single clearly
  low condition is `agentic_terse` (93.3%) — the *maximally* informal L5 "vibe"
  docstring with a one-word ask. The clearer `agentic_casual` (L3) hits 99.3%.
  So the tax is ~6 points and appears only at the very informal end.
- **H2 (medium interaction): not supported.** Agentic and web-chat averages are
  identical (96.3 vs 96.2).
- **H3 (paste premium): not supported.** Error-paste (95.5) ≈ code-paste (96.4).
  The traceback didn't help more — but both start from single-operator-buggy
  code that's easy to fix, so this is a ceiling.
- **H4 (language tax): not supported for strong models, present for the weak
  one.** Multilingual is 100% for ChatGPT/Claude but drops Codestral to 88% —
  its worst condition. Prompt-style sensitivity is concentrated in the weakest
  model.

## The dominant caveat: ceiling effect

With everything at 93–99%, **HumanEval is near-saturated for these models**, and
the doctests are visible in the docstring. That compresses any real tax toward
zero — this experiment can rule out a *large* tax on easy problems but cannot
measure a small one precisely. **Next step: re-run on a harder benchmark**
(MBPP+, LiveCodeBench, or the harder HumanEval+ tests) where headroom exists.

This matches the literature: prompt-technique effects on correctness are
[surprisingly small](https://arxiv.org/pdf/2412.20545), and under-specification
costs ~-11% on HumanEval specifically — i.e. real but modest, and easily
swamped by ceiling + scoring noise.

## Harder-benchmark re-score: HumanEval+ (no new API calls)

To test whether the flat base-HumanEval result was a *weak-test* artifact, we
re-scored the **same 822 completions** against HumanEval+ (EvalPlus), which adds
hundreds of edge-case inputs per problem. `score_vibe_tax_plus.py` computes
expected outputs from each canonical solution and requires the completion to
match on every sampled input. (Approximate harness — exact-equality comparator,
≤60 sampled inputs/problem, budget-capped — so treat absolute numbers as
directional, not the official EvalPlus figure.)

| Condition | base HumanEval | **HumanEval+** | Δ |
|-----------|---------------:|---------------:|--:|
| agentic_casual | 99.3 | 91.3 | −8.0 |
| agentic_terse | 93.3 | 88.7 | −4.6 |
| webchat_detailed | 96.7 | 89.3 | −7.4 |
| webchat_code_paste | 96.4 | 88.3 | −8.1 |
| **webchat_error_paste** | 95.5 | **83.8** | **−11.7** |
| webchat_multilingual | 96.0 | 89.3 | −6.7 |
| **Overall** | **96.2** | **88.7** | **−7.5** |

By model: ChatGPT 90.1, Claude 92.0, Codestral 83.9.

**Findings:**
- **The harder benchmark works and matters.** Rigorous tests knocked ~7.5 points
  off overall and, more importantly, **changed the condition ranking** — the
  base-test numbers were partly a weak-test artifact.
- **Error-paste is now the *lowest* condition** (−11.7, vs ~−7 elsewhere). The
  extra failures are genuine (15 base→plus flips over 7 problems, with **all
  three models failing the same Fibonacci-family problems**, HumanEval/63/39/55).
  Interpretation: "fix the bug my traceback shows" converges on code that passes
  the *visible* failing test but is **less robust to edge cases** (large-`n`
  inputs) than write-from-scratch code. A real, if modest, effect the weak base
  tests completely hid.
- Even de-saturated, the spread stays ~8 points — no *large* vibe tax on these
  problems. Full de-saturation likely needs harder *problems* (LiveCodeBench),
  not just harder tests.

**Takeaway:** this validates running the full v2 design on a harder benchmark
rather than trusting saturated base-HumanEval — the ranking is test-dependent.
