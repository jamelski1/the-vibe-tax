# The Vibe Tax: Prompt Style, Benchmark Saturation, and the Real Cost of How You Ask an LLM to Code

*Working draft. Numbers are reproducible from the scripts and `*_stats.json` files
in this repository; see `RESEARCH_LOG.md` for the full chronological record.*

## Abstract

"Vibe coding" — prompting an LLM in natural, informal language and iterating on
what runs — is now a dominant mode of software development. A natural question
follows: does *how* you ask cost correctness? We call any such penalty the
**vibe tax**. Answering it requires two things prior work lacks: (1) knowing what
real informal prompts actually look like, and (2) a benchmark with enough
headroom to measure a small effect. We contribute both. First, we characterize
**14,675 real coding prompts** across two media — agentic CLIs (committed Claude
Code transcripts) and web chatbots (WildChat-1M, LMSYS-Chat-1M) — along five axes,
and show that *the medium shapes the prompt*: agentic users are terse and
reference files (21%); web-chat users are verbose and paste code/errors (54%).
Second, we build a **realism-calibrated prompt generator** (an LLM rewriter gated
by a real-vs-synthetic discriminator) and run a controlled experiment across
three benchmarks of increasing difficulty. On HumanEval the effect is real but
ceiling-suppressed; a paired test of realistic vs researcher-written prompts
gives +3.0 pts (p=0.001) and shows that **hand-invented informal prompts
overstate the tax**. On **LiveCodeBench** (contamination-free, 43.5% pass — real
headroom), a purely *framing*-level manipulation yields a **significant +9.3-pt
penalty for verbose/polite framing vs terse** on medium-difficulty problems with
capable models (p=0.007). The direction is counterintuitive: the tax is on
*verbosity/politeness*, not informality. Our central methodological finding is
that **benchmark saturation and code-extraction bugs both silently hide
prompt-style effects** — the tax cannot be sized on a saturated benchmark.

## 1. Introduction

Developers increasingly delegate coding to LLMs through short, informal,
functionality-oriented prompts. Whether this informality carries a correctness
cost is unclear and consequential. Prior work on prompt perturbation and
under-specification (§2) reports mixed, generally small effects — but almost
entirely on HumanEval, where frontier models now score ~90%+, leaving little room
to measure anything, and using *researcher-invented* perturbations that may not
reflect real usage.

We make three contributions:

1. **An empirical map of real coding prompts (the "vibe spectrum").** A
   multi-medium corpus (agentic vs web-chat) labeled along five axes under one
   classifier, yielding a *medium effect* vs *population effect* distinction
   (§3).
2. **A realism-calibration method.** A logistic-regression discriminator that
   scores how distinguishable synthetic prompts are from real ones, plus a
   generator that rewrites benchmark tasks into realistic, still-gradeable
   prompts and gates them on the discriminator (§4).
3. **A controlled measurement across three benchmarks** showing that the tax is
   invisible under saturation, that synthetic informal prompts overstate it, and
   that on a non-saturated benchmark a framing manipulation produces a
   significant, counterintuitive penalty (§5–6).

## 2. Related work

Real-usage corpora **WildChat-1M** [Zhao et al., ICLR 2024] and
**LMSYS-Chat-1M** [Zheng et al., ICLR 2024] are the standard sources of authentic
user–LLM interactions; **WildCode** (2025) mines coding conversations from
WildChat, and **SWE-chat** (2026) collects real coding-*agent* sessions — but
none contrast media under a common lens or measure a correctness tax.
**ReCode** [Wang et al., ACL 2023] applies mechanical docstring/name/syntax
perturbations to HumanEval; the **under-specification** literature (2024–26)
finds informality costs roughly −11% to −15% pass@1 on HumanEval, *sometimes
reversing*. Tone work (**Mind Your Tone**, 2025) finds *rude* prompts can beat
polite ones on accuracy — consistent with our verbosity/politeness finding.
Our novelty: the two-medium characterization, the prevalence-calibrated
conditions, and the demonstration that benchmark saturation hides the effect.

## 3. The Vibe Spectrum: what real prompts look like

**Data.** We harvest genuine human turns from **committed Claude Code
transcripts** on public GitHub (agentic CLI; n=4,675 from 163 repositories, via
deep git-tree enumeration + ~31 search-query angles) and coding turns from
**WildChat-1M** and **LMSYS-Chat-1M** (web chat; n=5,000 each). All are labeled by
one heuristic classifier along five axes: specification level, intent, context
provision, language, and register. Raw corpora are not redistributed (they
contain third-party text, including — in one case — a real leaked API key that
GitHub push-protection caught); only aggregate statistics are released.

**The medium shapes the prompt.**

| Axis | Agentic CLI (n=4,675) | WildChat (n=5,000) | LMSYS (n=5,000) |
|------|--------------------:|------------------:|---------------:|
| Any paste (code/error/spec) | 8% | 54% | 36% |
| References files instead | **21%** | ~4% | ~4% |
| Pasted error | 2% | 12% | 3% |
| Casual/vague spec (L3–L5) | 87% | 55% | 53% |
| Conversational register | 38% | 58% | 54% |
| Multilingual | 23% | 42% | 16% |

The cleanest single signal is *how context is supplied*: agentic users
**reference files** (the agent can open them); web-chat users **paste** (the
model cannot reach their machine). Same need, opposite mechanism.

**Medium effects vs population effects.** Running the identical pipeline on a
second web-chat source (LMSYS) separates what replicates from what does not.
*Replicates* (medium effects): higher spec detail, conversational register, and
code-pasting on both web-chat datasets. *Does not replicate* (population
effects): error-paste rate (WildChat 12% vs LMSYS 3%) and multilingual share
(42% vs 16%) — these track *who uses the tool*, not the medium. Implication: an
error-paste study must name its population; "web-chat" is not monolithic.

## 4. Calibrating synthetic prompts to reality

To measure a tax we need gradeable tasks (with tests) phrased like real prompts.
We therefore hold *task content* fixed (a benchmark problem, with its tests) and
vary only *prompt style*.

**Realism discriminator.** A pure-Python logistic regression over 20
interpretable features (lengths, casing, punctuation, and flags like
`has_doctest`, `has_triplequote`, `has_signature`) trained to separate real
corpus prompts from synthetic ones; its test accuracy is a realism score.
Our initial synthetic prompts — both a hand-written 5-level formality ladder and
researcher-authored conditions — are **100% separable** from real prompts
(AUC 1.0). The top "tells" are triple-quotes, function signatures, and doctests:
our prompts were dressed-up code stubs, not messages.

**Calibrated generator.** An LLM rewrites each benchmark task into a realistic
message per condition, preserving the exact function/method name for
gradeability and stripping the scaffolding tells, gated by the discriminator.
On HumanEval this lifts realism from 0.00 to ~0.95 (prose conditions).

**A methodological hazard we hit: scoring.** Paste-style prompts elicit
*conversational* replies ("here's the fix: …"), which a naive
first-code-line extractor mangles — scoring one condition at **0% for all three
models**, a pure artifact. A robust multi-candidate extractor (last full
function / fenced block / indentation-repaired body) was required; it cannot
inflate correctness (every candidate still must pass the real tests). Any
prompt-style study without a paste-robust scorer will report noise.

## 5. Experiment

**Conditions** are grounded in the measured prevalences of §3: agentic terse,
agentic casual, web-chat detailed, web-chat code-paste, web-chat error-paste
(with *real captured tracebacks* from bug-injected canonical solutions), and
web-chat multilingual. **Models:** ChatGPT, Claude, Codestral (temperature 0).
**Benchmarks of increasing difficulty:** HumanEval (base + HumanEval+ edge-case
tests) and **LiveCodeBench** (functional, contamination-free problems dated after
model cutoffs). All comparisons are **paired within problem** and tested with
McNemar's test.

## 6. Results

**HumanEval is saturated.** All conditions score 93–99% on base tests; even under
HumanEval+ (edge-case tests) everything sits at ~84–94%. A realistic-vs-synthetic
comparison (calibrated v3 vs researcher-written v2) gives a paired
**+3.0 ± 1.7 pts (McNemar p=0.001)** — statistically real but small, because at
~90% there is no room for it to grow. Notably, **realistic prompts scored
*better* than researcher-written ones**: hand-invented informality ("yo so
like…") is out-of-distribution for RLHF'd models and *overstates* the tax.

**LiveCodeBench reveals the effect.** Overall pass is **43.5%** (medium 52.2%,
hard 31.0%; Claude 59.7 / ChatGPT 58.1 / Codestral 12.7) — real headroom. Four
conditions vary only the *framing* around an identical problem statement:

| Framing (wrapper around identical problem) | overall | medium | hard |
|--------------------------------------------|--------:|-------:|-----:|
| terse | 46.0 | 55.3 | 32.7 |
| casual | 44.1 | 53.4 | 30.7 |
| multilingual (Chinese wrapper) | 43.8 | 53.4 | 30.1 |
| detailed / polite | 40.1 | 46.6 | 30.7 |

Paired McNemar, **terse vs detailed**:

| Slice | Δ | p |
|-------|--:|--:|
| All | +5.9 | **0.012** |
| Medium | +8.7 | **0.009** |
| Capable models (ChatGPT+Claude) | +9.3 | **0.007** |
| Capable × medium | +13.0 | **0.007** |
| Hard | +2.0 | 0.677 (n.s.) |

**Verbose/polite framing significantly underperforms terse** — but only in a
*Goldilocks zone*: medium-difficulty problems and capable models. On **hard**
problems every framing collapses to ~30% (difficulty dominates); on **HumanEval**
the ceiling hid it entirely.

**Pairwise: it is politeness specifically.** Running terse against *each* other
framing (not just detailed) isolates the driver:

| terse vs … | Δ (all) | p | Δ (capable) | p |
|------------|--------:|--:|------------:|--:|
| detailed / polite | +5.9 | **0.012** | +9.3 | **0.007** |
| casual | +1.9 | 0.419 (n.s.) | +3.2 | 0.332 (n.s.) |
| multilingual (ZH) | +2.2 | 0.396 (n.s.) | +3.6 | 0.306 (n.s.) |

Terse vs **casual** and terse vs **multilingual** are **null in every slice**;
only the polite/verbose wrapper is significant. So the effect is not a penalty on
informality (casual ≈ terse) or on language (Chinese ≈ terse) — it is
specifically **politeness/verbosity**. This is the sharpest form of the central
claim.

## 7. Discussion

Three claims:

1. **The tax is on verbosity/politeness, not informality.** Terse "vibe" framing
   *beats* polite detailed framing (+9.3 pts, p=0.007, capable models),
   consistent with tone-effect literature. Extra conversational scaffolding
   appears to dilute focus on demanding problems.
2. **Saturation hides prompt-style effects.** The same underlying phenomenon is
   +3 pts (ceiling-crushed) on HumanEval and +9 pts (clear) on LiveCodeBench.
   Sizing a prompt-style effect on a saturated benchmark is not possible; a
   "Goldilocks" difficulty is required — enough headroom to escape the ceiling,
   not so much the model floors out.
3. **Method choices silently distort these studies.** Both benchmark saturation
   and code-extraction bugs can turn a real effect into a null (or a null into
   an artifact). Prompt-style research needs de-saturated benchmarks, paired
   within-problem designs, and paste-robust scoring.

## 8. Limitations

- **Heuristic classification** of the spectrum (regex/keyword/script-based);
  directional, not gold-standard.
- **Agentic corpus is ~163 committers**, not a random developer sample (its
  rates are stable across the scale-up, but style skews to power users). Web-chat
  corpora are 2023-era (GPT-3.5/4).
- **LiveCodeBench framings are deterministic mock wrappers** (clean minimal
  pairs); LLM-rewritten framings are future work (likely same direction). The
  *same* four wrappers are applied to HumanEval as a matched pair (the "HE4"
  set), holding the problem body constant so framing is isolated from
  specification detail — removing a confound present in the earlier
  six-condition HumanEval runs (where terse/casual/detailed also differed in
  spec level).
- **Difficulty coverage.** The headline LCB run is medium+hard; an easy slice is
  added for completeness (expected near-ceiling, like HumanEval), which completes
  the Goldilocks curve rather than changing the conclusion.
- **Paste conditions not yet ported to LCB** (they need correct-then-mutate, as
  LCB has no canonical solutions).
- **n=124 LCB problems** (medium+hard); a larger pull would tighten hard-problem
  estimates. Codestral sits near the floor on hard problems and adds noise; the
  capable-model slice is the cleaner read.
- Approximate scorers (HumanEval+ output-equivalence; sampled test cases). The
  **paired within-problem** design controls per-problem scorer quirks.

## 9. Conclusion

Whether informal prompting taxes LLM code correctness depends entirely on *where*
you measure it. On saturated benchmarks the effect is real but invisible, and
researcher-invented informal prompts overstate it. On a hard, contamination-free
benchmark with headroom, a controlled framing manipulation reveals a significant,
counterintuitive penalty: **verbose, polite framing costs ~9 points against terse
framing on medium-difficulty problems** — the vibe tax is a *politeness* tax, and
it only shows when the benchmark lets it.

## Reproducibility

All code, per-benchmark `RESULTS.md`, and aggregate stats are in this repository;
`RESEARCH_LOG.md` records the full 11-phase process, including the dead-ends
(the scoring-artifact iteration, the `datasets`-on-Windows saga). Raw prompt
corpora are withheld (third-party PII/secrets); everything else reproduces from
the scripts.
