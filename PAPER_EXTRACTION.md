# The Extractor Is Part of Your Benchmark: How Code-Extraction Choices Silently Distort LLM Correctness Results

*Standalone draft. Companion artifacts: `data/vibe_tax_lcb/score_lcb.py`,
`data/vibe_tax_v2/score_vibe_tax.py`, `EXTRACTION_PITFALLS.md` (failure catalogue),
`data/vibe_tax_lcb/test_class_shape.py` (prevalence probe).*

## Abstract

Functional code benchmarks (HumanEval, HumanEval+, LiveCodeBench, MBPP) report a
pass/fail verdict per problem, but between the model's raw reply and that verdict
sits an unreported step: code is *extracted* from free-form text and *assembled*
into a runnable program before any test runs. This step is code the evaluator
writes, and it has a bug surface. We show that reasonable-looking extraction
choices systematically misattribute *format* failures to *correctness* failures —
inflating measured error rates — and, worse, that when the volume of non-code text
in a reply co-varies with the experimental manipulation, an extraction bug becomes
a *confound* that fabricates a statistically significant effect. In our own study
of prompt framing, a naive extractor produced a confident **+9.3-point "politeness
tax"** (McNemar p=0.007) that **vanished to null** once extraction was made
prose-robust; the same class of bug had earlier driven a real signal to a **0%**
artifact in the opposite direction. We contribute (1) a taxonomy of eleven
extraction/scoring failure modes, each mapped to the specific wrong conclusion it
produces and to its measured prevalence in a 2,004-completion corpus; (2) a robust
multi-candidate extraction technique that provably cannot inflate correctness; and
(3) diagnostics and a checklist for catching these before publication. Our central
practical claim: **report your extractor, measure its rate, and reproduce
surprising results by hand — the extractor is part of your benchmark.**

## 1. Introduction

Evaluating an LLM on code looks deterministic: run the model, run the tests, report
pass@1. In practice the pipeline has four stages, and only the first and last are
usually described:

```
raw completion  ──►  EXTRACT code  ──►  ASSEMBLE program  ──►  run tests  ──►  pass/fail
                     (evaluator code)   (evaluator code)
```

The two middle stages are heuristics the evaluator writes to turn a chat reply into
something executable. Modern chat models rarely emit exactly the bare artifact a
harness expects: they add explanations, wrap code in markdown fences, define helper
classes above the answer, or deliver the solution in a slightly different container
than the harness looks for. Every one of those is a reason the extractor can hand
the interpreter something that fails to run — *even when the underlying algorithm is
correct.*

Our entry point into this problem was an anomaly we could not explain: a query that
**passed** when pasted into a web-chat UI **failed** when run through our API
harness — consistently, on the *same* returned code. If the model is the only thing
that varies, that is impossible. It was not the model; it was our extractor. Pulling
that thread turned a prompt-style study into a study of the measurement apparatus.

**Contributions.**
1. A **taxonomy** of extraction and scoring failure modes (§4), each tied to the
   specific wrong conclusion it yields and its measured prevalence (§6).
2. A **robust multi-candidate extraction technique** (§5) that accepts the first
   candidate program that both compiles and passes the tests — and therefore cannot
   inflate correctness.
3. A **case study** in which an extraction artifact manufactured a significant
   prompt-style effect (§6), plus the **diagnostics** that caught it and a
   **checklist** for practitioners and reviewers (§7–8).

## 2. Background and related work

HumanEval scores by appending a completion to a prompt stub and running assert-based
tests; HumanEval+ (EvalPlus) adds large edge-case test suites; LiveCodeBench uses
contest problems with a `class Solution` method and hidden tests. All three assume
the completion can be reduced to a runnable function of a known shape. The community
reports *which* benchmark and *which* tests were used; it rarely reports *how code
was extracted from the reply*, despite instruction-following variance across models
and the near-universal habit of conversational models to wrap code in prose. Our
work sits alongside the broader reproducibility literature that attributes surprising
results to measurement rather than to the system under test.

## 3. The extraction step, and a threat model

We separate two ground truths that a naive pipeline conflates:

- **Code correctness** — does the model's algorithm compute the right answers?
- **Format conformance / extractability** — can the harness recover a runnable,
  correctly-shaped program from the reply?

A "code smell" — a helper defined above the expected class, a nested or renamed
class, a bare function where a method was expected, a trailing prose paragraph — is
a *format* deviation, not a *correctness* failure. A harness that scores it FAIL
reports a false negative and **inflates the error rate**.

The threat becomes a *confound* when the trigger for an extraction failure
**co-varies with the experimental variable**. The amount of non-code prose a model
emits is not random: it scales with the model, with the model's release/training
era, and with the prompt wrapper (a terse imperative elicits less explanation than a
polite, verbose, or translated request). An extractor whose failure rate depends on
prose volume therefore has a failure rate that depends on the manipulation — and
will produce a spurious effect aligned with it. This is exactly what happened to us.

## 4. A taxonomy of failure modes

We group the modes by where they bite. The full catalogue with code locations is in
`EXTRACTION_PITFALLS.md`; the abridged version:

**(A) Prose / chatter contamination.** *Trailing prose* after unfenced code
(`return ans` then "This checks every adjacent pair…") is fed to `exec` and throws
`SyntaxError` on correct code. *Leading chatter* defeats a "first code-like line"
heuristic. In a "fix my error" reply, the buggy snippet is quoted *before* the fix,
so a first-`def` extractor grabs the wrong function.

**(B) Structure / shape mismatch.** A *helper class defined above* `class Solution`
is dropped when the extractor slices from the class anchor → `NameError`. Code
*split across multiple fenced blocks* loses a dependency if only one block is taken.
A *renamed or nested class*, or a *bare function* where a method was expected, is not
found where the harness looks.

**(C) Format / indentation.** *Fence-marker variants* (```` ```python ````, ```` ``` ````,
```` ```py ````, missing close) break brittle regexes. *Ragged first-line indentation*
(only the first body line mis-indented) breaks dedent-by-first-line assembly.

**(D) Harness / metric (not extraction, same symptom).** A *per-problem total timeout*
kills a correct-but-slow solution though each test would pass; an *output-token cap*
truncates a reasoning model to an empty completion; *weak base tests* pass
subtly-wrong code (the one false-*positive* direction); *binary pass@1* hides
near-misses.

## 5. A robust extraction technique

The extractor we converged on generates an **ordered list of candidate programs** and
accepts the **first that both compiles and passes the tests**
(`score_vibe_tax.py :: build_candidates`, `score_lcb.py :: extract_solution`):

1. fenced code blocks that mention the target;
2. **all** code blocks concatenated (keeps a helper split into a separate block);
3. a slice from the first top-level construct (keeps helpers defined above the class);
4. the **last** `def <entry>` (the fixed version in a "fix my error" reply);
5. the completion body appended to the prompt stub (HumanEval-style);
6. a bare body with first-line-indent repair.

Two properties make this safe. **It cannot inflate correctness:** a candidate must
still pass the real tests, so forgiving format never turns a wrong algorithm into a
pass. **It degrades gracefully:** each candidate is trimmed to its largest
compilable prefix (`_trim_to_compilable`), and a final raw fallback avoids spurious
`None`. To resolve shape, execution binds the entry point *regardless of container* —
bare function, method on `Solution`, renamed top-level class, or a class nested one
level in (`resolve_callable`).

## 6. Case studies and measured prevalence

**False positive — a fabricated politeness tax.** On LiveCodeBench (167 functional
problems, four framing conditions differing only in the wrapper around an identical
problem), a naive extractor produced a significant terse-beats-polite penalty:
**+9.3 pts, p=0.007** on the capable-model slice. The mechanism was mode (A): the
"no fences" instruction made replies end in prose, and polite/verbose/translated
framings elicited *more* trailing prose. Measured directly, the share of completions
whose extracted code even compiled tracked the framing:

| condition | naive extractor compile-rate | robust extractor |
|-----------|-----------------------------:|-----------------:|
| terse | 88.0% | 100% |
| casual | 81.1% | 100% |
| multilingual | 77.2% | 100% |
| **detailed / polite** | **73.4%** | 100% |

Detailed lost ~15 points of compile-rate to the bug alone — the same magnitude as
the "effect." With the robust extractor the framing effect is **null** in every
slice (terse−detailed +0.6, p=0.845, capable models).

**False negative erasing a real signal.** Earlier, on HumanEval, the
`webchat_error_paste` condition scored **0% for every model** — the v1 extractor
grabbed the *first* code-like line, which in a "here's my error, fix it" reply is the
buggy snippet quoted in the explanation. Extracting the *last* `def <entry>`
recovered the true rate. The same failure class thus distorted results in **both**
directions.

**Silent capability suppression.** Two structure/harness modes — dropped helpers
(mode B: **11** correct solutions scored FAIL) and a per-problem total timeout (mode
D) — jointly depressed the hard-problem numbers; fixing them cut "never solved" from
**15 → 10** and raised hard-problem pass@1 from **51% → 60.5%**. A token-cap
truncation made a reasoning model look far weaker (**65/101** hard "failures" were
empty completions).

**Prevalence is not uniform — measure it.** Not every catalogued mode matters in
every corpus. Probing our 2,004-completion corpus (`test_class_shape.py`), the
renamed/nested-class mode — real and reproducible in a unit test — affected **0 of
2,003** extractable completions: these models always emit `class Solution` or a bare
function, both already resolvable. Contrast the trailing-prose mode, which drove an
entire significant effect, and the helper-drop mode (11 cells). The lesson is not
that every mode is common; it is that **prevalence must be measured, not assumed**,
because a rare-looking mode can be the one correlated with your manipulation.

## 7. How to catch these (diagnostics)

The bugs were caught by four cheap habits, in rough order of power:

1. **Extraction/compile-rate by condition.** If extractability correlates with your
   manipulation, you have an artifact, not an effect. The 88%→73% compile gradient
   was the smoking gun.
2. **Manual reproduction of outliers.** Paste the exact query into a chat UI; if it
   passes there but the same completion fails via the harness, the harness is the
   variable. This is how the whole investigation started.
3. **Paired within-problem designs.** An artifact that hits all arms equally cancels
   in a paired test. This is why the framing *null* survived every fix while the
   absolute *capability* numbers moved — a useful signature for telling the two apart.
4. **Re-score, don't re-query.** Every fix was validated by re-scoring stored
   completions, so corrections cost no API calls and are exactly reproducible.

## 8. Prescription — a checklist for functional code evaluation

- Extract with **multiple candidates**; accept the first that compiles **and** passes.
- Be **prose-robust** (trim to the largest compilable prefix), **helper-robust**
  (keep top-level constructs preceding the target), **shape-robust** (resolve the
  entry point in any container), and **fence-robust**.
- Use a **per-test** timeout, not a per-problem budget.
- Give reasoning models an **adequate token budget**; log empty completions.
- Use a **de-saturated** benchmark and report **test-level partial credit**.
- Prefer **paired within-problem** comparisons.
- **Report the extractor and its extraction/compile rate**, per condition.
- **Reproduce surprising per-cell results by hand.**

## 9. Limitations

Robust extraction deliberately forgives *format* non-compliance; if the research
question is "does the model follow output-format instructions," that must be counted
separately (a robust extractor would mask it). Our HumanEval+ scorer is an
output-equivalence approximation with its own edge cases. Prevalence figures are
specific to our models/corpus; the taxonomy is general but the frequencies are not.
Finally, contamination affects absolute pass rates but not the artifact claims, which
are within-pipeline comparisons on fixed completions.

## 10. Conclusion

Most of the conclusions we nearly published were decided by measurement choices, not
by the models. An extractor that is too aggressive turns correct code into a syntax
error; one that is too naive grabs the wrong function; one that assumes a fixed shape
misses a renamed class. Because the amount of non-code text co-varies with the model
and the prompt, these are not just noise — they are confounds that can fabricate a
significant effect. The remedy is unglamorous and effective: robust multi-candidate
extraction, paired designs, reported extraction rates, and an adversarial habit of
reproducing outliers by hand. Report your extractor. It is part of your benchmark.
