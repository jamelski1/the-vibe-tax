# Ablation: does a newer model (GPT-5.6) close the LiveCodeBench capability gap?

> ## ⚠️ THE NUMBERS BELOW ARE INVALID (token-truncation artifact) — RE-RUN REQUIRED
>
> The first GPT-5.6 run used `MAX_TOKENS=2048`. GPT-5.6 is a **reasoning model** that
> spends output budget on hidden reasoning before answering, so on hard problems the
> budget ran out and the API returned an **empty** completion → scored FAIL.
> Diagnosis (`check_gpt56_truncation.py`): **73 of 148 failures (49%) were empty
> completions**, including **65 of 101 hard failures (64%)** and 8 medium. These are
> not wrong answers — the model never got to answer. Manual re-tests confirm 5.6
> solves several of these reliably (e.g. lcb/3550, 3/3).
>
> **The 77.8% / +3.7pt result below is therefore a LOWER BOUND. Re-run with**
> **`MAX_TOKENS=16000`** (reasoning models need room to think *and* answer), then
> re-score and re-compare. This section will be rewritten with the valid numbers.
>
> This is a fourth measurement artifact the project has caught (after code-extraction,
> benchmark saturation, and a test-display cap): **an output-token cap can silently
> zero out a reasoning model on the hardest problems.**

**Exploratory capability probe**, not a controlled comparison — see caveats.
Re-run `python compare_models.py --new lcb_v3_gpt56_scored.json --model chatgpt`.

## Setup (stamped in the raw data for reproducibility)

| | baseline | new (ablation) |
|---|---|---|
| model_id | `gpt-5.4` | **`gpt-5.6`** |
| temperature | 0 (deterministic) | **default = 1** (5.6 rejects temp 0) |
| date run | (original study) | 2026-09-05 |
| items | 167 problems × 4 framings = 668 | same 668 |
| scored on | LCB public+private tests | same |

Every 5.6 record carries `model_id: "gpt-5.6"`, `temperature: null` (default), and a
timestamp; the run-stats file records the same. Files:
`lcb_v3_gpt56_responses.json`, `lcb_v3_gpt56_scored.json`, `lcb_v3_gpt56_run_stats.json`.

## Result: modest, significant gain — concentrated on medium

| difficulty | GPT-5.4 | GPT-5.6 | Δ |
|-----------|--------:|--------:|--:|
| easy | 97.7% | 98.8% | +1.1 (ceiling) |
| medium | 78.1% | **84.6%** | **+6.5** |
| hard | 48.5% | 50.5% | +2.0 |
| **all** | **74.1%** | **77.8%** | **+3.7** |

Paired McNemar (new vs baseline, per problem×framing): 74 newly-solved vs 49
regressions → **Δ +3.7 pts, p = 0.030 (significant)**.

## Reading it honestly

- **A real but modest improvement.** 5.6 is significantly better overall, and the
  gain is almost entirely on **medium** problems (+6.5). **Hard problems barely
  move** (48.5 → 50.5) — the contest-hard frontier (DP / combinatorial counting)
  holds even for the newer model. Easy is at ceiling for both.
- **Not uniformly better: 49 regressions.** The newer model *loses* 49 problem×framing
  cells that 5.4 solved, against 74 gains. A newer model is not a strict superset.
- **The temperature confound.** 5.6 could only run at its default **temperature 1**
  (stochastic, single sample), while 5.4 was **temperature 0** (deterministic). So
  this is *model + temperature*, not model alone, and much of the 74/49 churn is
  sampling noise, not capability. The net +3.7 is real; the per-cell flips are not
  all real.
## Per-problem view — the trustworthy signal

Aggregating the 4 framings per problem (167 problems) strips out single-framing
temperature noise. This is the honest capability delta:

| | count |
|---|--:|
| improved (5.6 solved more framings) | 34 |
| regressed (5.6 solved fewer) | 24 |
| unchanged | 109 |
| **robust gain** (5.4 0/4 → 5.6 4/4) | **5** |
| **robust loss** (5.4 4/4 → 5.6 0/4) | **4** |

So once noise is removed the real generational delta on these problems is **small**:
net +10 at the "improved" level, and only **+1 net at the robust level** (5 solid
gains vs 4 solid losses). The headline "74 newly solved" was mostly single-framing
churn from temperature 1.

**Robust gains** (problems 5.6 solidly cracked that 5.4 couldn't touch):

| task | difficulty | method |
|------|-----------|--------|
| lcb/3496 | medium | minNumberOfSeconds |
| lcb/3603 | hard | findAnswer |
| lcb/3680 | hard | countComponents |
| lcb/3687 | hard | longestSpecialPath |
| lcb/3776 | medium | minCost |

**Robust losses** (5.4 solved every framing, 5.6 none):

| task | difficulty | method |
|------|-----------|--------|
| lcb/3583 | hard | gcdValues |
| lcb/3715 | medium | maximumCoins |
| lcb/3739 | hard | distanceSum |
| lcb/3751 | medium | maxFrequency |

The gains skew slightly harder than the losses (3 hard vs 2), consistent with a
small real improvement on the frontier — but with n=5 vs 4 it is not a strong claim.

## Does it change the study's conclusions? No.

The core findings are unchanged and, if anything, reinforced:
- **Framing is still null** on 5.6 (agentic_terse 79.6 / casual 76.6 / detailed 77.2
  / multilingual 77.8 — flat, same as 5.4).
- **Difficulty still dominates** (easy ~99 / medium ~85 / hard ~50).
- **The hard frontier persists** across a model generation.

## To make it a clean (temperature-matched) comparison

Two honest options if this graduates from a probe to a claim:
1. **Match temperature:** re-run the 5.4 arm at temperature 1 too (both stochastic),
   so only the model differs. (`OPENAI_MODEL=gpt-5.4 API_TEMPERATURE=1 ONLY_MODELS=chatgpt`.)
2. **pass@k:** take k samples per problem for each model at temperature 1 and report
   pass@1/pass@k, which averages out the sampling noise.
