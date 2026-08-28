# Vibe Tax on LiveCodeBench — results (the framing effect is null; the artifact was real)

**Headline:** once code is extracted robustly, **prompt framing/register has no
measurable effect on correctness** on LiveCodeBench. The large "politeness tax"
we first reported (+9.3 pts, p=0.007) was a **code-extraction artifact**, not a
real effect. This is the honest, post-fix result.

**Setup:** 167 functional LiveCodeBench problems (43 easy, 73 medium, 51 hard),
all **contamination-free** (contest_date ≥ 2024-08-01). 4 framing conditions,
scored against LCB's own tests (`score_lcb.py`). The 4 conditions differ ONLY in
the *wrapper text* around an identical, full problem statement — so this isolates
**framing/register**, holding the task constant. All three models (ChatGPT +
Claude + Codestral, n=2,004) are included; the capable-model slice (ChatGPT +
Claude) is the cleaner read since Codestral floors on hard problems.

## The artifact (what happened, and why it mattered)

The system prompt asks for plain Python **without markdown fences**. Conversational
replies therefore come back as *code followed by a prose explanation* —
`class Solution: … return …` then "This checks every adjacent pair…". With no
fences to delimit the code, the original extractor handed the **whole thing**
(code + trailing English) to `exec()`, which threw `SyntaxError` on correct code.

The polite/detailed framing ("Hi! Could you help… Thank you!") elicits **more
explain-after-code**, so it was penalized hardest. Measured directly, the share
of completions whose extracted code even *compiled*:

| condition | old extractor | fixed extractor | penalty from the bug |
|-----------|--------------:|----------------:|---------------------:|
| terse | 88.0% | 100% | +12.0 |
| casual | 81.1% | 100% | +18.9 |
| multilingual | 77.2% | 100% | +22.8 |
| **detailed** | **73.4%** | 100% | **+26.6** |

Detailed lost **~15 points more than terse to the bug alone** — the same size as
the "effect." The fixed extractor (`score_lcb.py :: _trim_to_compilable`) trims
trailing lines until the source parses and still defines the target; it **cannot
inflate correctness** (extracted code must still pass the real tests).

## Post-fix result: no framing effect

Full 3-model set (ChatGPT + Claude + Codestral, n=2,004). Pass rate by condition:

| Condition (wrapper around identical problem) | overall (3 models) | capable (CG+Cl) | easy | medium | hard |
|----------------------------------------------|-------------------:|----------------:|-----:|-------:|-----:|
| agentic_terse | 58.7 | 74.9 | 88.4 | 60.3 | 31.4 |
| agentic_casual | 60.1 | 76.0 | 87.6 | 60.7 | 35.9 |
| webchat_detailed (*"Hi! Could you help… Thank you!"*) | 59.7 | 76.0 | 89.1 | 60.3 | 34.0 |
| webchat_multilingual (Chinese wrapper) | 62.5 | 78.7 | 93.0 | 63.5 | 35.3 |

Terse is if anything the **lowest** condition, not the highest — the pre-fix
"terse wins" ordering has fully inverted. Paired McNemar, `terse` vs each other
framing (reproduce: `python mcnemar_lcb.py --base agentic_terse`):

| terse vs … | slice | Δ (pts) | p | |
|------------|-------|--------:|--:|--|
| webchat_detailed | all | −1.0 | 0.511 | n.s. |
| | medium | 0.0 | 1.000 | n.s. |
| | capable models | −1.2 | 0.585 | n.s. |
| | capable × medium | 0.0 | 1.000 | n.s. |
| agentic_casual | all | −1.4 | 0.281 | n.s. |
| webchat_multilingual | all | −3.8 | **0.013** | *see note* |
| | medium | −3.2 | 0.210 | n.s. |
| | capable models | −3.9 | 0.060 | n.s. |

**The framing effect is null.** terse−detailed and terse−casual are n.s. in every
slice; before the fix terse−detailed read +5.9/+8.7/+9.3/+13.0 with p<0.01 — that
entire signal was the extractor. The one "significant" cell (terse < multilingual,
all −3.8, p=0.013, and easy −4.7, p=0.031) is **not robust**: it is n.s. in
medium, hard, capable-models, and capable×medium, and is carried by the pooled
"all" slice (which mixes in Codestral) plus a few near-ceiling easy problems. We
report it as a weak curiosity, not a finding — certainly not a *penalty* for
non-English, if anything the reverse.

## What this means

- **There is no framing/register tax.** Holding the problem identical, phrasing a
  request tersely, casually, politely, verbosely, or in another language makes no
  difference to correctness on LCB. **Difficulty dominates** (easy ~98%, medium
  ~81%, hard ~51%); framing does not move it.
- **The apparent tax was a measurement artifact** — the project's own thesis,
  turned on its own pipeline. We have now caught code extraction distorting a
  prompt-style result in **both** directions: it turned a *real* signal into a 0%
  null (v2 error-paste) and a *true null* into a false +9-pt effect (this).
- **Scope of the claim.** This tests *framing* (the wrapper around a complete
  problem). It does **not** test genuine **under-specification** (giving the model
  *less information*), nor the **paste/buggy-code** conditions — those are
  separate axes where a real cost could still exist. "No tax" means: *how you
  phrase a complete request doesn't tax correctness.*

## Robustness & caveats

- **Paired within-problem design** — same problem × model — so per-problem scorer
  quirks and difficulty cancel; the null is not a power artifact of pooling.
- **The fix cannot manufacture the null:** it only removes trailing prose so
  correct code runs; wrong code still fails every test. Pass rates rose across
  *all* conditions (terse least, detailed most) — the signature of removing an
  *asymmetric* artifact, not of leniency.
- **Codestral** is included (restored via the `seed_lcb_progress.py` merge);
  n=2,004. It floors on hard problems, so the capable-model slice is the cleaner
  read — and both give the same null.
- Framings are deterministic **mock** wrappers (clean minimal pairs); LLM-rewritten
  framings are a follow-up, but there is now no effect for them to reduce.
- **Under-specification and paste conditions on LCB are the open positive-result
  threads** — if a real "vibe tax" exists, that is where to look, not in register.
