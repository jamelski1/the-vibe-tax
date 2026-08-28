# Vibe Tax on LiveCodeBench — results (the framing effect is null; the artifact was real)

**Headline:** once code is extracted robustly, **prompt framing/register has no
measurable effect on correctness** on LiveCodeBench. The large "politeness tax"
we first reported (+9.3 pts, p=0.007) was a **code-extraction artifact**, not a
real effect. This is the honest, post-fix result.

**Setup:** 167 functional LiveCodeBench problems (43 easy, 73 medium, 51 hard),
all **contamination-free** (contest_date ≥ 2024-08-01). 4 framing conditions,
scored against LCB's own tests (`score_lcb.py`). The 4 conditions differ ONLY in
the *wrapper text* around an identical, full problem statement — so this isolates
**framing/register**, holding the task constant. Numbers below are the two
capable models (ChatGPT + Claude); Codestral was dropped in the easy re-run and
is optional to restore (it floors on hard and only adds noise — it will not move
a null).

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

| Condition (wrapper around identical problem) | overall | easy | medium | hard |
|----------------------------------------------|--------:|-----:|-------:|-----:|
| agentic_terse | 74.9 | 98.8 | 80.1 | 47.1 |
| agentic_casual | 76.0 | 95.3 | 80.1 | 53.9 |
| webchat_detailed (*"Hi! Could you help… Thank you!"*) | 76.0 | 100.0 | 80.1 | 50.0 |
| webchat_multilingual (Chinese wrapper) | 78.7 | 100.0 | 84.2 | 52.9 |

Detailed **ties** terse (76.0 vs 74.9); multilingual is if anything highest.
Paired McNemar, `terse` vs each other framing (reproduce:
`python mcnemar_lcb.py --base agentic_terse`):

| terse vs … | slice | Δ (pts) | p | |
|------------|-------|--------:|--:|--|
| webchat_detailed | all | −1.2 | 0.585 | n.s. |
| | medium | 0.0 | 1.000 | n.s. |
| | capable × medium | 0.0 | 1.000 | n.s. |
| agentic_casual | all | −1.2 | 0.541 | n.s. |
| webchat_multilingual | all | −3.9 | 0.060 | n.s. |

**Every comparison is null.** The discordant cells are near-symmetric
(terse>detailed 13, detailed>terse 17). Before the fix these same pairs read
+5.9/+8.7/+9.3/+13.0 with p<0.01 — that entire signal was the extractor.

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
- **Codestral** was dropped in the easy re-run (2-model numbers above). Restoring
  it (via `seed_lcb_progress.py` merge) is optional and won't change a null.
- Framings are deterministic **mock** wrappers (clean minimal pairs); LLM-rewritten
  framings are a follow-up, but there is now no effect for them to reduce.
- **Under-specification and paste conditions on LCB are the open positive-result
  threads** — if a real "vibe tax" exists, that is where to look, not in register.
