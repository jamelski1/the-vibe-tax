# Self-repair on LiveCodeBench — the positive result

**The move:** the framing study is a *null* (how you phrase a request doesn't change
correctness). Self-repair is its mirror image and should be a strong *positive*:
give the model a concrete signal — its own failing code plus one failing test — and
correctness recovers, a lot, in one or two rounds. Together they make a clean,
publishable story:

> **Prompt *wording* is not a lever on code correctness; execution *feedback* is.**
> The register/politeness/verbosity of a request moves pass@1 by ~0 points
> (paired, n.s.), but a single round of test-grounded self-repair recovers a large
> fraction of the *same* failed problems. The bottleneck isn't how you ask — it's
> whether the model gets to see what went wrong.

This reframes the null from a disappointment into the **control condition** that
makes the positive finding credible: same problems, same models, same paired
design — the only thing that changes is *information*, not *phrasing*.

## Design (`self_repair.py`)

- **Seed set:** every attempt the main run FAILED, deduped to one framing per
  (problem × model) so each starting point counts once (framing is null → the
  choice of framing is immaterial; default `agentic_terse`).
  Current counts (from `lcb_scored.json`): **184 seeds** — chatgpt 30, claude 31,
  codestral 123; by difficulty hard 92 / medium 77 / easy 15. Capable-only
  (chatgpt+claude) = **61 seeds**.
- **One repair round:** run the current code → find the FIRST failing test →
  show the model `{full problem, its current code, that test's input / expected /
  actual}` → ask for a corrected full solution → extract and **re-score on the
  full test set** (pass = ALL tests). Up to `--rounds` rounds; a fixed seed stops.
- **No teaching-to-the-test:** we *show* one failing test but *score* on all of
  them (add `--score-on-private` to score on private tests only — even stricter).
  A model can't pass by hardcoding the shown case; the other tests catch it.
- **Paired & apples-to-apples:** identical scorer (`score_lcb`: robust extractor +
  per-test timeout) and identical API plumbing (`run_vibe_tax`: same models,
  temperature, token cap) as the main study.

The output metric is a **recovery curve**: cumulative % of the failed seeds fixed
after round 1, 2, 3 — overall and sliced by difficulty, topic, and model.

## Why this is likely to be a real, sizable effect

- The main-study failure analysis already found **100% of capable-model failures
  are "valid code, wrong answer"** (CAPABILITY_ANALYSIS.md) — i.e. a *localized
  logic bug*, exactly what a failing test pinpoints. And partial credit showed the
  **best attempt already passes ~97% of a hard problem's tests** — the models are
  one edge-case away, which is the regime where a single counterexample helps most.
- Self-repair with execution feedback is a well-established lever; the novelty here
  is **contrasting it head-to-head with the framing null on the identical failed
  set**, so the paper's claim is a *comparison of levers*, not a bare capability demo.

**Predicted shape (to be confirmed by the run):** a steep round-1 jump
(a large share of wrong-answer bugs fixed from one counterexample), diminishing
returns by round 3, and a difficulty gradient — medium recovers more than hard
(hard failures are more likely genuine algorithmic gaps than off-by-one bugs).
The exact numbers come from the run; do not cite any until `self_repair_stats.json`
exists.

## Running it (local, needs `lcb_tests.jsonl` + API keys)

```
# 1. the effect: capable models, 3 rounds (183 calls max; fewer as fixed seeds stop)
python self_repair.py --rounds 3 --models chatgpt,claude

# 2. the CONTROL (run this too): same seeds, told "wrong" but shown NO test
python self_repair.py --rounds 3 --models chatgpt,claude --no-feedback

# full set, all three models (552 calls max each)
python self_repair.py --rounds 3
python self_repair.py --rounds 3 --no-feedback

# stricter: score repaired code on private tests only
python self_repair.py --rounds 3 --models chatgpt,claude --score-on-private
```

Resumable. The feedback run writes `self_repair_{trajectory,stats,progress}.json`;
the `--no-feedback` control writes the same with a `_nofb` suffix.
**The headline number is `stats.cumulative_fixed_by_round` (feedback) minus the
same in `_nofb` (control) — the feedback-specific recovery.**

## The paper framing (two levers, one figure)

A single figure carries the paper: the flat framing bars (terse ≈ casual ≈ polite
≈ multilingual, all n.s.) next to the self-repair recovery curve rising steeply
over rounds. One says "wording is not the lever," the other says "feedback is."
The methodological contribution (extraction robustness manufactured a phantom
framing tax) sits underneath as the reason the null is trustworthy in the first place.

## The built-in control (already in the harness)

**Feedback vs no-feedback** (`--no-feedback`): the control run tells the model its
code is wrong but shows *no* failing test, so it is a bare resample-with-"try-again."
`recovery(feedback) − recovery(no-feedback)` is the **feedback-specific** effect —
this is what makes the result a controlled *lever comparison* rather than a capability
demo, and it is the number to put next to the framing null. Run both (commands above).

## Natural extensions (if reviewers want more)

- **Split wrong-answer vs TLE** in the shown test (the harness already distinguishes
  `wrong` / `error` / `timeout`) — do slow-but-correct solutions get fixed differently
  than wrong-idea ones?
- **More rounds / oracle upper bound:** push `--rounds` higher to find where recovery
  plateaus (the asymptote is a "with-feedback ceiling" for these models).
- **Feedback richness:** one test vs multiple failing tests vs the full traceback —
  does more signal help, or is one counterexample enough?
