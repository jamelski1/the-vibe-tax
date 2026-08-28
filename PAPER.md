# The Vibe Tax That Wasn't: How Benchmark Saturation and Code-Extraction Artifacts Manufacture Prompt-Style Effects

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
benchmarks of increasing difficulty. Our central — and cautionary — finding is
that **code-extraction robustness dominates the result**. Under a naive
extractor, a purely *framing*-level manipulation on **LiveCodeBench**
(contamination-free) appears to yield a large, significant penalty for
verbose/polite framing vs terse (+9.3 pts, p=0.007, capable models). **This
effect is an artifact.** Polite/verbose prompts elicit more explanation-after-code
prose; with no code fences, an unfenced extractor feeds that prose to the
interpreter and fails correct code — asymmetrically, hitting the polite condition
hardest (compile-rate terse 88% vs detailed 73%). With a robust extractor the
effect **vanishes** (paired Δ=0.0, p=1.0 on medium): **holding the problem
identical, how you phrase the request — terse, casual, polite, verbose, or in
another language — has no measurable effect on correctness; problem difficulty
does.** We further show the mirror failure — extraction turning a *real* signal
into a 0% null (a paste condition) — establishing that both **benchmark
saturation** and **code-extraction bugs** can manufacture *or* erase prompt-style
effects, in either direction. The methodological lesson is the contribution.

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
3. **A controlled measurement showing the "tax" is a measurement artifact.** On a
   non-saturated benchmark, a naive extractor makes verbose/polite framing look
   significantly worse than terse; a robust extractor erases the effect entirely.
   How you phrase a complete request does not tax correctness — and prompt-style
   studies are dominated by extraction robustness and benchmark saturation, both
   of which can manufacture or hide an effect (§5–6).

## 2. Related work

Real-usage corpora **WildChat-1M** [Zhao et al., ICLR 2024] and
**LMSYS-Chat-1M** [Zheng et al., ICLR 2024] are the standard sources of authentic
user–LLM interactions; **WildCode** (2025) mines coding conversations from
WildChat, and **SWE-chat** (2026) collects real coding-*agent* sessions — but
none contrast media under a common lens or measure a correctness tax.
**ReCode** [Wang et al., ACL 2023] applies mechanical docstring/name/syntax
perturbations to HumanEval; the **under-specification** literature (2024–26)
finds informality costs roughly −11% to −15% pass@1 on HumanEval, *sometimes
reversing*. Tone work (**Mind Your Tone**, 2025) reports *rude* prompts can beat
polite ones on accuracy — an effect in the same family we test; we find no such
framing effect once code is extracted robustly, which raises the question of how
much of the tone literature is scorer-dependent. Our novelty: the two-medium
characterization, the prevalence-calibrated conditions, and the demonstration
that both benchmark saturation and code-extraction robustness can manufacture or
erase a prompt-style effect.

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

**LiveCodeBench: the "effect" is an extraction artifact.** On LCB (functional,
contamination-free; 167 problems: 43 easy / 73 medium / 51 hard) four conditions
vary only the *framing* around an identical problem statement. Under a **naive**
extractor the result looked striking — a significant terse-beats-polite penalty:

| terse vs … (naive extractor) | all | capable models | capable × medium |
|------------------------------|----:|---------------:|-----------------:|
| detailed / polite | +5.9 (p=.012) | +9.3 (p=.007) | +13.0 (p=.007) |
| casual | +1.9 (n.s.) | +3.2 (n.s.) | — |
| multilingual | +2.2 (n.s.) | +3.6 (n.s.) | — |

But the system prompt requests plain Python **without fences**, so conversational
replies are *code followed by prose* ("…This checks every adjacent pair"). The
naive extractor fed that prose to the interpreter, failing correct code. The
polite/verbose framing produces more explain-after-code, so it was penalized
hardest — compile-rate **terse 88% vs detailed 73%**, a ~15-pt asymmetry that *is*
the reported effect. With a **robust** extractor (trim trailing lines until the
source parses, target still defined; cannot inflate correctness) the effect
disappears:

| Framing (robust extractor) | overall | easy | medium | hard |
|----------------------------|--------:|-----:|-------:|-----:|
| terse | 74.9 | 98.8 | 80.1 | 47.1 |
| casual | 76.0 | 95.3 | 80.1 | 53.9 |
| detailed / polite | 76.0 | 100.0 | 80.1 | 50.0 |
| multilingual (ZH) | 78.7 | 100.0 | 84.2 | 52.9 |

| terse vs … (robust extractor) | all | medium | capable × medium |
|-------------------------------|----:|-------:|-----------------:|
| detailed / polite | −1.2 (p=.585) | **0.0 (p=1.000)** | **0.0 (p=1.000)** |
| casual | −1.2 (p=.541) | 0.0 (n.s.) | 0.0 (n.s.) |
| multilingual | −3.9 (p=.060) | −4.1 (n.s.) | −4.1 (n.s.) |

**Detailed ties terse; every comparison is null** (discordant cells near-symmetric:
terse>detailed 13, detailed>terse 17). The entire +9.3-pt "politeness tax" was the
extractor. Correctness on LCB is governed by **difficulty** (easy ~98%, medium
~81%, hard ~51%), not by how the request is phrased.

## 7. Discussion

Three claims:

1. **There is no framing/register tax.** Holding the problem identical, phrasing a
   request tersely, casually, politely, verbosely, or in another language does not
   change correctness (paired Δ≈0, p≈1 on the measurable slice). Correctness is
   governed by problem difficulty. The widely assumed intuition that polite or
   verbose "vibe" prompting costs correctness is not supported once code is
   extracted correctly. (This is scoped to *framing* — varying the wrapper around
   a *complete* problem — and does not speak to genuine under-specification, where
   the model is given less information; see §8.)
2. **Code-extraction robustness dominates the measured result — in both
   directions.** A naive unfenced extractor manufactured a significant +9-pt
   "politeness tax" out of nothing, by asymmetrically failing the verbose
   condition's explanation-after-code; the same class of bug elsewhere erased a
   *real* signal, scoring a paste condition at 0%. Extraction can both create and
   destroy prompt-style effects. Any such study without a paste-/prose-robust,
   compile-checked extractor is reporting its scorer, not the model.
3. **Benchmark saturation compounds the hazard.** At ~90% (HumanEval) there is no
   room to measure anything; de-saturated benchmarks are necessary to test for an
   effect at all. But de-saturation is not sufficient — LCB has headroom and the
   framing effect is still null; the naive extractor nonetheless produced a
   confident false positive there. Prompt-style research needs de-saturated
   benchmarks, paired within-problem designs, **and** robust scoring; the last is
   what decided this result.

## 8. Limitations

- **Heuristic classification** of the spectrum (regex/keyword/script-based);
  directional, not gold-standard.
- **Agentic corpus is ~163 committers**, not a random developer sample (its
  rates are stable across the scale-up, but style skews to power users). Web-chat
  corpora are 2023-era (GPT-3.5/4).
- **Scope: framing, not under-specification.** We vary the *wrapper* around a
  *complete* problem and find no effect. We do **not** test genuine
  under-specification (giving the model less information) on a de-saturated
  benchmark, nor the paste/buggy-code conditions (which need correct-then-mutate,
  as LCB has no canonical solutions). A real correctness cost could still exist on
  those axes — that is the open positive-result thread. "No tax" means *how you
  phrase a complete request does not tax correctness.*
- **LiveCodeBench framings are deterministic mock wrappers** (clean minimal
  pairs). LLM-rewritten framings are a possible follow-up, but there is now no
  effect for added naturalness to reduce. The *same* four wrappers on HumanEval
  (the "HE4" matched set) hold the problem body constant, isolating framing from
  specification detail (a confound in the earlier six-condition HumanEval runs)
  — and land, as expected, flat at the ceiling (~97%).
- **n=167 LCB problems** (43 easy / 73 medium / 51 hard). The reported LCB numbers
  are the two capable models (ChatGPT + Claude); Codestral was dropped in the
  easy re-run and is optional to restore — it floors on hard, adds noise, and
  cannot change a null.
- Approximate scorers (HumanEval+ output-equivalence; sampled test cases). The
  **paired within-problem** design controls per-problem scorer quirks.

## 9. Conclusion

The intuition behind the "vibe tax" — that casual or polite phrasing costs
correctness — does not survive honest measurement. Holding the problem identical
and varying only how the request is framed produces **no** correctness difference
on a contamination-free benchmark with headroom; a confident, significant
+9-point "politeness tax" appeared only under a naive code extractor and vanished
once extraction was fixed. What determines whether an LLM solves a problem is the
problem's difficulty, not the register of the ask. The durable lesson is
methodological: prompt-style measurements are dominated by benchmark saturation
and code-extraction robustness, either of which can manufacture or erase an
effect — so the finding a study reports may be a property of its scorer. The open
question a real vibe tax might yet answer is not about *phrasing* but about
*information*: whether genuinely under-specified requests cost correctness. That
is where to look next.

## Reproducibility

All code, per-benchmark `RESULTS.md`, and aggregate stats are in this repository;
`RESEARCH_LOG.md` records the full 11-phase process, including the dead-ends
(the scoring-artifact iteration, the `datasets`-on-Windows saga). Raw prompt
corpora are withheld (third-party PII/secrets); everything else reproduces from
the scripts.
