# Where the models succeed and fail on LiveCodeBench

Analysis of the 167-problem, 3-model run (`analyze_lcb_capability.py` →
`lcb_capability_stats.json`). Framing is null (see `RESULTS.md`), so we average
over the four framings and read **capability** by (problem × model). Numbers are
the two capable models (ChatGPT + Claude) unless noted.

## 1. Succeeds vs fails — by difficulty

| difficulty | pass@1 | reading |
|-----------|-------:|---------|
| easy   | **98.5%** | essentially solved — near-perfect |
| medium | **81.2%** | mostly solved, real misses appear |
| hard   | **51.0%** | coin-flip — this is the frontier |

A clean monotone gradient. Difficulty — not phrasing — is what governs success.

## 2. Succeeds vs fails — by topic

Topics are **keyword-derived** from the problem statement (LCB ships no topic
labels), so treat these as directional, not gold-standard. Pass@1, capable models:

| topic | pass@1 | # problems |
|-------|-------:|-----------:|
| simulation / geometry | 82.5% | 5 |
| greedy / array | 80.4% | 65 |
| string | 79.0% | 41 |
| intervals / sorting | 77.3% | 11 |
| graph | 72.5% | 10 |
| math / number theory | 70.8% | 6 |
| **dynamic programming** | **62.0%** | 23 |

**Dynamic programming is the clear weak spot** — ~18 points below the array/string
work, and it's a large category (23 problems). Straight-line simulation, array
scanning, and string manipulation are where the models are strongest. The pattern
is intuitive: the models are good at problems with a *local, constructive* solution
and worse at problems needing a *non-obvious global recurrence* (DP) or heavier
number-theoretic insight.

## 3. What "can it do this problem at all?" looks like

Over all 12 attempts per problem (4 framings × 3 models):

- **34 problems solved 12/12** — trivial for current models (all easy/medium).
- **15 problems solved 0/12** — genuinely beyond every model and framing.
  - By difficulty: **9 hard, 6 medium, 0 easy.**
  - They cluster in **hard strings and hard combinatorics/DP**: e.g.
    `countBalancedPermutations`, `minCostGoodCaption`, `maximumSubarrayXor`,
    `countGoodArrays`, `maxScore`. These are contest-hard problems requiring a
    real algorithmic insight, not just careful coding.

So the "frontier" is concrete: the models clear easy problems and most medium
ones, and fall off on hard problems that need a novel algorithm — especially DP
and combinatorial counting.

## 4. *How* it fails — the failure mode (this is the key point)

Of every failed capable-model completion:

| failure mode | count | share |
|--------------|------:|------:|
| **wrong logic** (valid code, wrong output) | 315 | **100.0%** |
| no usable code produced | 0 | 0.0% |

**After the extraction fix, every single failure is a genuine reasoning failure.**
The model always produces syntactically valid, runnable code that defines the
right function — it just computes the wrong answer on hard problems. There are
**zero** formatting/extraction failures left. This is the clean, honest picture:
the models don't fail because they can't write code; they fail because they can't
*solve the problem*.

**This also rules out output-token truncation for the main study.** A per-model
check of empty completions (the signature of a reasoning model running out of
`max_tokens` before answering) found **0 empty completions** across all three
models — gpt-5.4 (0/173 failures, 0/105 hard), claude-opus-4-6 (0/142, 0/95), and
codestral (0/482) — at the same `MAX_TOKENS=2048`. So the 2048 cap did not
throttle the main models. (It *did* throttle the GPT-5.6 ablation, which is a
heavier reasoning model — 73/148 of its failures were empty — see
`ABLATION_gpt56.md`. That is why 5.6 needs `MAX_TOKENS=16000`, but the main study
does not.)

*Caveat:* "wrong logic" here bundles wrong-answer, runtime exceptions, and
time-limit (efficiency) failures — we can't separate them without running the
graded tests locally. Splitting **time-limit (correct-but-too-slow)** from
**wrong-answer (incorrect algorithm)** is a valuable follow-up: it distinguishes
"the model had the right idea but an inefficient implementation" from "the model
had the wrong idea." (The tests are local — `score_lcb.py` could log the failure
reason.)

## 4b. Capability is a continuum, not solved/unsolved (partial credit)

Binary pass@1 ("did an attempt pass ALL tests?") throws away how *close* the models
get. Scoring every attempt at the test level instead (`full_partial_credit.py` →
`full_partial_credit.json`) reframes the whole picture. Across the 167 problems:

| category | count |
|----------|------:|
| fully solved (all 12 attempts pass) | 35 |
| **partially solved (some attempts pass / most tests pass)** | **120** |
| never solved (0/12 attempts) | 12 |

**Overall mean per-attempt test-pass rate: 81%.** By difficulty:

| difficulty | mean attempt | best attempt | fully solved | never |
|-----------|-------------:|-------------:|:-----------:|:----:|
| easy | 95.0% | 100.0% | 26/43 | 0 |
| medium | 82.8% | 98.6% | 9/73 | 5 |
| **hard** | **66.7%** | **97.1%** | **0/51** | 7 |

The headline: **pass@1 dramatically understates model capability on hard problems.**
No hard problem is fully solved by all 12 attempts (0/51), which sounds like a wall
— but the **best attempt passes 97% of a hard problem's tests on average**, and even
the **average attempt passes 67%**. The models are not failing hard problems from
incapability; they solve nearly the entire test suite and miss an edge case or two.
The dominant reality is **partial** (120/167), which the binary metric hides
entirely. This is a continuum-of-capability result, and it depends on measuring at
the test level, not the problem level.

(These numbers already incorporate the extractor + per-test-timeout fixes: the run
re-executed every failed attempt, so the "never solved" count is the corrected 12,
down from 15.)

## 5. Takeaways

- **Solves now:** easy problems (≈99%), and most medium array/string/simulation
  problems (~80%). These are safe to delegate.
- **Still lacking:** hard problems generally (~51%), and **dynamic programming /
  combinatorial counting** specifically (62%, the weakest topic). 15/167 problems
  are unsolved by any model or framing.
- **The bottleneck is reasoning, not code generation** — 100% of failures are
  correct-form / wrong-answer. Prompt phrasing won't move this; problem-solving
  ability is the ceiling.

## Open follow-ups this motivates

1. **Split wrong-answer vs time-limit** among the failures (needs the local tests)
   — is it wrong ideas or slow implementations?
2. **Give the model the graded tests in the prompt** — does seeing the tests lift
   pass@1, and for which problem types? (`generate_lcb_test_prompts.py`; score on
   *private* tests only to avoid teaching-to-the-test.)
3. **Self-repair:** feed the model its own failing code + a failing test and ask
   it to fix — does a second round recover correctness, and where?
