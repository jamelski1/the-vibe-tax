# Code extraction is a hidden confound in LLM code evaluation — outline + failure catalogue

Working title candidates:
- *"The Extractor Is Part of Your Benchmark: How Code-Extraction Choices Silently Distort LLM Correctness Results"*
- *"Prose, Helpers, and Nested Classes: A Taxonomy of Code-Extraction Failures in LLM Evaluation"*
- *"You Measured Your Extractor, Not the Model: Extraction Artifacts in Prompt-Style and Correctness Benchmarks"*

**One-sentence thesis.** Between an LLM's raw reply and a pass/fail verdict sits an
*extraction + scoring* step that is almost never reported; we show that reasonable-looking
choices in that step manufacture false-negative correctness results — inflating measured
error rates and even fabricating a statistically significant "prompt-style tax" — and we
give a defensive extraction technique and a checklist for catching these before you publish.

Why this is the strongest paper: it is a **positive, generalizable methods contribution**
(a taxonomy + a technique + a diagnostic), it is grounded in a concrete case where an
artifact produced a *significant* wrong result we nearly reported, and it applies to every
functional code benchmark (HumanEval, HumanEval+, LiveCodeBench, MBPP, SWE-style), not just
this project.

---

## Paper outline

**1. Introduction.** The pipeline nobody reports: `raw completion → extract code → assemble
program → run tests → pass/fail`. The middle two steps are code you wrote, and they have a
bug surface. Motivating hook: the same query pasted into a web-chat UI *passed*, but our API
harness scored it *failed* — consistently — which is impossible if the model is the variable.
The difference was our extractor. Contributions: (a) a taxonomy of extraction/scoring failure
modes, each mapped to the *specific wrong conclusion* it produces; (b) a robust
multi-candidate extraction technique; (c) a case study where an extractor artifact created a
significant +9.3-pt "politeness tax" that vanished once fixed; (d) a reviewer/practitioner
checklist and diagnostics.

**2. Background & related.** HumanEval / HumanEval+ / LiveCodeBench scoring; the "no markdown
fences" instruction and why conversational models violate it; pass@1 conventions; the
reproducibility-crisis framing (measurement, not models). Position: prior work reports *what*
was scored, rarely *how the code was pulled out*.

**3. The extraction/scoring step, formally.** Define the stages and the two ground truths that
matter: *code correctness* (does the algorithm produce right answers) vs *extractability /
format conformance* (can the harness recover a runnable, correctly-shaped program). Central
claim: **most "failures" in naive pipelines are format/extraction failures misattributed to
correctness.** A "code smell" (helper above the class, a nested/renamed class, a bare function
where a method was expected) is *not* incorrectness, yet naive harnesses score it as such.

**4. A taxonomy of failure modes (the table below).** Grouped into: (A) prose/chatter
contamination, (B) code-shape/structure mismatch, (C) fence & format, (D) indentation, (E)
multi-block / dropped dependencies, (F) selection (which code) errors, (G) scoring-harness
issues (timeout, token cap, weak tests, binary metric). Each row: trigger → mechanism → wrong
conclusion → fix → evidence. Emphasize the **dependency structure of the trigger**: prose
volume scales with model, model release/training era, and prompt wrapper
(terse/polite/verbose/multilingual) — so an extraction bug that is sensitive to prose volume
produces a *spurious prompt-style effect*, which is exactly what bit us.

**5. A robust extraction technique.** The ordered multi-candidate method actually used
(`score_vibe_tax.py :: build_candidates`, `score_lcb.py :: extract_solution`): generate
several candidate programs (fenced blocks; concatenation of all code blocks to keep helpers;
anchor-to-construct slices; last-`def` for fix-replies; body-appended-to-prompt; bare-body
with indent repair), **trim each to its largest compilable prefix**, and accept the *first
candidate that compiles AND passes the tests*. Key properties: (i) it **cannot inflate**
correctness — a candidate still has to pass real tests; (ii) it is order-robust and
shape-robust; (iii) it degrades gracefully instead of returning `None`.

**6. Case studies (the payoff).**
- *False positive:* the phantom +9.3-pt politeness tax on LCB (naive extractor; compile-rate
  terse 88% vs detailed 73%), null after the fix.
- *False negative erasing a real signal:* `webchat_error_paste` scored 0% for every model on
  HumanEval v1 (first-code-line grabbed the buggy snippet), recovered by last-`def` extraction.
- *Silent capability suppression:* "15 never-solved" LCB problems → 10 after helper + timeout
  fixes; hard-problem pass 51% → 60.5%.

**7. How to catch these (methodology).** The diagnostics that actually found the bugs:
(i) **compile/extraction-rate by condition** — if extractability correlates with a
manipulation, you have an artifact, not an effect; (ii) **manual web-chat reproduction** of
surprising per-cell results; (iii) **paired within-problem designs**, which cancel artifacts
that hit all arms equally (this is why the framing *null* survived every fix while absolute
capability numbers moved); (iv) re-scoring old completions after each fix (no new API calls).

**8. Prescription / checklist.** A short, copyable checklist for functional code eval:
prose-robust, helper-robust, shape-robust extraction; per-*test* timeouts; adequate token
budgets (esp. reasoning models); de-saturated benchmark; test-level partial credit; paired
design; report extraction rate; manually reproduce outliers.

**9. Threats to validity / limitations.** Extraction robustness can *mask* genuine format
non-compliance (if you care about "did it follow instructions," count that separately);
approximate scorers (HumanEval+ output-equivalence) have their own edge cases; contamination
affects absolute rates (not the artifact claims). Distinguish *code correctness* from *format
compliance* — a robust extractor deliberately forgives the latter.

**10. Conclusion.** Report your extractor. Measure its rate. Reproduce outliers by hand. The
extractor is part of your benchmark.

---

## Failure-mode catalogue (grounded in the code)

Legend — **Effect:** FN = false negative (correct code scored FAIL, inflates error rate);
FP = false positive (wrong code scored PASS); NULL = extraction returns nothing usable.
**Layer:** EXT = extraction, ASM = program assembly, HARNESS = run/scoring.

| # | Failure mode | Layer | Where in the code (version) | Trigger — what makes it fire | Mechanism — why it fails | Effect | Fix (and where) |
|---|--------------|-------|------------------------------|------------------------------|--------------------------|:------:|-----------------|
| 1 | **Trailing prose after code** ("…return ans" then "This checks every adjacent pair…") | EXT | LCB naive `extract_solution` (pre-`2ae72a8`) | "output plain Python, no fences" system prompt + conversational/polite/verbose/multilingual wrapper; prose volume is **model- & wrapper-dependent** | With no fences to delimit it, the whole reply (code+prose) is `exec`'d → `SyntaxError` on otherwise-correct code | FN | Trim to largest compilable prefix — `_trim_to_compilable` (`score_lcb.py`) |
| 2 | **Leading prose / chatter before code** | EXT | HumanEval v1 `clean_completion` "first code-like line" (`run_tests.py`) | Any reply that opens with explanation; more common on polite/webchat framings | Heuristic starts at the first line containing `def/return/=/#…`; picks a sentence or an inline token, not the function | FN / NULL | Anchor to `def <entry>` / isolate fenced blocks (`extract_last_function`, `fenced_blocks`) |
| 3 | **Buggy-snippet-first in "fix my error" replies** | EXT | HumanEval v1 (first `def`) | `webchat_error_paste` condition: reply shows the buggy code, explanation, *then* the fix | Naive grabs the **first** `def <entry>` — the buggy one quoted in the explanation — not the corrected one | FN (systematic) | Extract the **last** `def <entry>` — `extract_last_function` (`score_vibe_tax.py`). *Recovered `error_paste` from 0% → real rate* |
| 4 | **Dropped helper defined above the class** (`Fenwick`, `DSU`, `BIT`, `SegTree`) | EXT | LCB robust-v1 (pre-`7751cde`) | Model defines a helper class/function *before* `class Solution` (common on hard DP/graph) | Extractor slices from the `class Solution` anchor → helper gone → `NameError` at runtime | FN | Try candidates that **keep preceding top-level constructs** first (`extract_solution`, `7751cde`). *Recovered 11/11 cells* |
| 5 | **Code split across multiple fenced blocks** (helper in block 1, `Solution` in block 2) | EXT | any single-block extractor | Model narrates: ```block1 helper``` then prose then ```block2 Solution``` | Picking one block misses the other → `NameError` / missing symbol | FN | Add an **"all code blocks concatenated"** candidate (`extract_solution` candidate B) |
| 6 | **Class-shape mismatch** (bare function vs method; nested or renamed class) | EXT/ASM | LCB `_worker` (`ns.get("Solution")` → `ns.get(entry)`); HE `check(entry)` at module scope | Model returns `def entry(...)` free-standing, or nests it, or renames the class — a *code smell*, not a bug | Harness looks for the method on `Solution` (LCB) or a module-level `entry` (HE); the mismatched shape isn't found where it looks | FN | Resolve **both** shapes: try method-on-class *and* bare function; try the starter's class name. (Partially handled; candidate for a documented fix) |
| 7 | **Body-vs-full-function ambiguity** | ASM | HumanEval body-append path (`clean_completion` appended after the prompt's `def` line) | Model returns a **complete** `def` (with signature) instead of just the body | Appending a full `def` under the prompt's `def` line → double signature / bad indent → `SyntaxError` | FN | Multi-candidate: try standalone full-function *and* body-appended (`build_candidates`) |
| 8 | **Ragged first-line indentation** | ASM | HumanEval `normalized_body` (the bug it fixes) | Model mis-indents only the **first** body line (e.g. 3 spaces then 4) | Anchoring dedent on the first line shifts every later line → `IndentationError` | FN | Raise the anomalous first line to match the rest, then dedent+reindent (`normalized_body`) |
| 9 | **Fence-marker variants** (```` ```python ```` vs ```` ``` ```` vs ```` ```py ````, missing close, stray prose fences) | EXT | all fence regexes / `strip_fences` | Different models/UI paste styles emit different fence syntaxes | A brittle regex misses the block or leaves stray backticks that break `exec` | FN / NULL | Tolerant fence regex + drop any line starting with ```` ``` ```` (`strip_fences`, `fenced_blocks`) |
| 10 | **Over-aggressive extraction → NULL** | EXT | any extractor that returns `None` on no-anchor | Very prose-heavy reply, or the model uses a different symbol name than `entry` | No candidate matches the anchor → extractor yields nothing → scored as no-code FAIL though runnable code exists | FN / NULL | Graceful last-resort fallback (raw fence-stripped text) so it never spuriously returns `None` (`extract_solution` final return) — balanced against #1 by requiring compile |
| 11 | **Per-problem total timeout** (correct-but-slow) | HARNESS | LCB `passes` (pre-`fb3fd62`: one 8 s budget for the whole suite) | A correct solution on a many-test problem whose *sum* of test times > budget | The suite is killed mid-run though **each** test would pass within limits | FN | **Per-test** timeout with streaming/early-exit (`passes`/`_worker`, `fb3fd62`) |
| 12 | **Output-token cap truncates a bigger/reasoning model** (two sub-states) | HARNESS | `run_vibe_tax.py` `MAX_TOKENS` | Larger/reasoning models spend the budget on hidden reasoning before the visible answer, and longer solutions need more tokens | At a low cap the completion is either (a) **empty** (budget spent before any visible code) or (b) **truncated mid-code** (non-empty but syntactically incomplete → won't parse/compile). Both are false fails that hit the *more capable* models hardest. | FN (misread as incapacity) | Raise budget (16k+); flag empties **and** parse-incomplete completions separately. *65/101 hard "failures" for GPT-5.6 were empty* |
| 13 | **Weak base tests pass subtly-wrong code** | HARNESS | HumanEval base (~7 tests) | Off-by-one / edge-case-wrong solutions | Too few tests to expose the bug | **FP** | Re-score on HumanEval+ edge tests (`score_vibe_tax_plus.py`) |
| 14 | **Binary pass@1 hides partial correctness** | HARNESS/metric | problem-level pass@1 everywhere | Near-miss solutions (one failing edge case) | All-or-nothing collapses a 97%-of-tests attempt to "fail" | misleads | Test-level partial credit (`full_partial_credit.py`) |

### The through-line for the paper
Rows **1–3** and **10** are all *prose/selection* errors whose trigger — how much explanatory
text the model emits — **co-varies with the model, its release/training era, and the prompt
wrapper**. That is precisely why a prose-sensitive extractor turns a *null* prompt-style
effect into a *significant* one (#1): the artifact is correlated with the independent variable.
Rows **4–6** are *structure* errors (a "code smell" is not incorrectness). Rows **11–14** are
harness/metric errors that don't touch extraction but produce the same symptom — a correct
model scored wrong — and belong in the same cautionary catalogue.

### Diagnostics that caught them (put these in §7)
1. **Extraction/compile-rate by condition** — if it correlates with your manipulation, it's an
   artifact (this exposed #1: terse 88% vs detailed 73% compile).
2. **Manual web-chat reproduction** — paste the exact query into the chat UI; if it passes
   there but fails via API on the *same* completion, the harness is the variable (how the whole
   extractor investigation started).
3. **Paired within-problem design** — cancels artifacts that hit all arms equally; the framing
   null survived every fix while absolute capability numbers moved.
4. **Re-score, don't re-query** — every fix was validated by re-scoring stored completions.
