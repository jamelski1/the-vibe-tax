# Vibe Tax on LiveCodeBench — results (the non-saturated measurement)

**Setup:** 124 functional LiveCodeBench problems (73 medium, 51 hard), all
**contamination-free** (contest_date ≥ 2024-08-01). 4 framing conditions ×
3 models = 1,488 completions, scored against LCB's own tests
(`score_lcb.py`; ~37 tests/problem). The 4 conditions differ ONLY in the
*wrapper text* around an identical, full problem statement — so this isolates
the effect of **framing/register**, holding the task constant.

## Why LCB matters: headroom

| Benchmark | Overall pass |
|-----------|-------------:|
| HumanEval base | ~96% (ceiling) |
| HumanEval+ | ~89% (ceiling) |
| **LiveCodeBench** | **43.5%** |

By difficulty: **medium 52.2%**, **hard 31.0%**. By model: Claude 59.7,
ChatGPT 58.1, **Codestral 12.7** (near floor on hard). Finally, room for an
effect to show.

## Headline: verbose/polite framing carries a real penalty — vs terse

| Condition (wrapper around identical problem) | overall | medium | hard |
|----------------------------------------------|--------:|-------:|-----:|
| agentic_terse (*"need `Solution.m` for this: …"*) | **46.0** | **55.3** | 32.7 |
| agentic_casual | 44.1 | 53.4 | 30.7 |
| webchat_multilingual (Chinese wrapper) | 43.8 | 53.4 | 30.1 |
| webchat_detailed (*"Hi! Could you help… Thank you!"*) | **40.1** | **46.6** | 30.7 |

**terse vs detailed** (paired McNemar, same problem × model):

| Slice | Δ (terse − detailed) | p |
|-------|---------------------:|--:|
| All | **+5.9** | **0.012** |
| Medium only | **+8.7** | **0.009** |
| Capable models (ChatGPT+Claude) | **+9.3** | **0.007** |
| Hard only | +2.0 | 0.677 (n.s.) |

So the penalty is **real and significant**, and it lands where you'd expect an
effect to be *visible*: **medium-difficulty problems and capable models**. On
**hard** problems every condition collapses to ~30% — the difficulty dominates
and framing is washed out (the model either can or can't). On the ceiling of
HumanEval it was invisible; here, in the headroom, it's a clean ~9-point,
p<0.01 effect.

## Interpretation

- **The tax is on verbosity/politeness, not informality.** The *terse* "vibe"
  framing BEATS the polite, detailed one. That's counterintuitive but consistent
  with the tone literature ([Mind Your Tone](https://arxiv.org/abs/2510.04950):
  rude > polite on accuracy). Extra conversational scaffolding ("Hi!… Thank
  you!") appears to dilute focus on a problem that's already demanding.
- **No language tax:** the Chinese-wrapped condition matches English casual/terse
  (medium 53.4%) — these models handle a non-English wrapper around an English
  problem with no measurable cost.
- **A Goldilocks zone for measuring prompt effects:** you need enough difficulty
  to escape the ceiling, but not so much that the model is at the floor. Medium
  LCB problems are that zone; HumanEval (ceiling) and hard LCB (floor) both hide
  the effect.

## This recontextualizes the whole project

On saturated HumanEval, the realistic-vs-synthetic prompt difference was
+3.0 pts (significant but ceiling-crushed). On LCB with headroom, a pure
*framing* difference reaches **+9.3 pts (p=0.007)** in the right regime. The
tax is real; you simply cannot size it on a saturated benchmark — which was the
core methodological thesis, now demonstrated.

## Robustness & caveats

- **Paired within-problem design** — terse vs detailed on the *same* problem ×
  model — so per-problem scorer quirks and problem difficulty cancel out. Even
  if a few of the 16 all-fail problems are scorer edge cases, they can't bias
  the comparison (both conditions get 0 → not discordant).
- **Scorer validated:** correct solution passes, wrong fails, conversational
  extraction works; pass-rate distribution is smooth and all-fail problems
  concentrate in *hard* — the signature of real difficulty, not artifacts.
- Framings here are the deterministic **mock** wrappers (clean minimal pairs);
  LLM-rewritten framings are a follow-up (would add naturalness, likely same
  direction).
- **Codestral** sits near the floor (12.7%) on these hard problems and adds
  noise; the capable-model slice is the cleaner read.
- Paste conditions (code/error) not yet ported to LCB (need correct-then-mutate).
- n=124 problems; a larger LCB pull would tighten the hard-problem estimates.
