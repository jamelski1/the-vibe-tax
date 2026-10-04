# Paper framing (concrete sketch) — "The Politeness Tax Is a Measurement Artifact"

The recommended spine, concrete enough to take to an advisor. Honest about what is novel vs.
built-on. Alternative framing (B) noted at the end. Supersedes nothing in PAPER_INGREDIENTS.md /
RELATED_WORK_NOTES.md — it selects from them.

---

## Title (pick one)

1. **The Politeness Tax Is a Measurement Artifact: Prompt-Register Effects as Code-Extraction Confounds** *(recommended)*
2. Mind Your Scorer, Not Your Tone: How Extraction Pipelines Manufacture Prompt-Style Effects
3. No Tax for Tone: A Controlled, Extraction-Robust Test of Prompt Register on Code Correctness

## One-sentence contribution (honest)

> Reported prompt-style/politeness effects on LLM *correctness* can be artifacts of the evaluation
> pipeline rather than model behavior: on a de-saturated code benchmark with a paired
> within-problem design, a standard-but-naive extractor yields a statistically significant
> **+9.3-point "politeness/verbosity tax" (p<0.01)** that **vanishes to null** under robust
> extraction — because the amount of explanatory prose, and thus extractor failure, co-varies
> with prompt register.

## Thesis (the spine)

The output-format/extraction confound is known on the **model axis** (SAFIM, Macedo). We show it on
the **prompt-register axis**, where it can **fabricate a significant causal finding** of exactly the
kind the politeness-prompting literature reports — and that literature is itself inconsistent
(Mind Your Tone finds effects; Cai et al. mostly null) and **does not control extraction**. We give
a controlled code-domain null, the mechanism, a reproducible pitfalls catalogue, and diagnostics.

## What is novel vs. built-on (state this explicitly in the paper)

**Built-on (cite, do NOT claim):**
- Extraction/output-format is a significant, conclusion-changing confound in code eval → **SAFIM**
  (model-training axis), **Macedo** (code translation; even names the prompt-phrasing→bias link).
- Robust/multi-candidate extraction as a technique → standard (EvalPlus `sanitize` tool, Macedo regex).
- Prompt surface choices confound evaluation; report a range → **FormatSpread**.
- Test insufficiency / buggy ground-truth flip rankings → **EvalPlus**.

**Novel (our contributions):**
1. A **controlled, paired within-problem** experiment isolating prompt **register** (problem held
   identical) on a **de-saturated code-generation** benchmark → **no register effect on correctness**.
2. Demonstration that a standard extractor **manufactures a significant false effect that vanishes**
   — extraction as a **directional confound on the prompt axis** (vs. Macedo/SAFIM's model axis;
   vs. FormatSpread's genuine-sensitivity-no-parsing).
3. **Reconciliation + critique of the politeness literature**: MYT (effects) vs Cai (null), neither
   controls extraction; our controlled null + mechanism explains the inconsistency.
4. A **reproducible catalogue** of evaluation pitfalls hit in one study, each with a minimal example
   + a **pre-publication diagnostics checklist**.
5. The **"closing-window"** argument: these artifacts get harder to detect as benchmarks saturate,
   provider pipelines turn opaque, and contamination grows.

## Target venue

Empirical-SE (MSR / ESEM / ICSE-SEIP) or an eval/reproducibility workshop; NeurIPS/ICLR
Datasets & Benchmarks is possible. **Not** a SOTA/leaderboard track.

## Section outline (with assets + citations per section)

1. **Introduction.** Hook: identical code **passed in web-chat, failed via API** → chased the
   apparatus. The live politeness debate (MYT vs Cai). Thesis + the 5 contributions.
2. **Background & related.** FormatSpread; EvalPlus (+sanitize tool caveat); **SAFIM + Macedo**
   (the mechanism — cite up front, position against); Ying survey (landscape); politeness lit
   (MYT, Cai). One paragraph: "what is prior, what is ours."
3. **Experimental design.** LCB, **167 contamination-window problems** (contest_date ≥ 2024-08-01;
   NOT contamination-free — model_provenance), 4 register framings (terse/casual/detailed/
   multilingual) wrapping an **identical** problem, 3 models, **paired within-problem**, McNemar.
   State the paired-design **contamination-immunity** argument.
4. **The artifact (core result).** Naive extractor → **+9.3, p<0.01** (capable models);
   **compile-rate-by-condition** (terse 88 / casual 81 / multilingual 77 / detailed 73 → 100% all);
   robust extractor → **null** (terse−detailed +0.6, p=0.845, all slices n.s.). Mechanism: prose
   volume ∝ register. *(This is THE result; Fig. 1 lives here.)*
5. **A reproducible pitfall catalogue.** The table (EXTRACTION_PITFALLS.md / PITFALL_COVERAGE.md):
   extraction sub-modes + saturation + contamination-window + per-test timeout + token-cap
   truncation (two sub-states) + binary-vs-partial. Each with a minimal reproducible example and a
   prior-art note (which are ours vs established).
6. **Reconciling the politeness literature.** MYT vs Cai vs ours; the extraction critique
   (**raise, don't assert** — MCQ parsing is robust); Cai's own "open-ended is noisier" admission.
7. **Diagnostics / checklist.** compile-rate-by-condition; manual reproduction; paired design;
   re-score-don't-re-query; model_provenance; empty/truncated-completion check.
8. **Discussion: the closing window.** Why detection gets harder over time (saturation, opaque
   pipelines, contamination, difficulty redefinition, LLM-mediated review). HumanEval→LCB is
   evidence (we already lived the saturation point).
9. **Limitations.** Register ≠ under-specification; framings are deterministic mock wrappers
   (LLM-rewritten is follow-up); MCQ critique of MYT/Cai is by analogy, not proof; approximate
   HE+ scorer; contamination affects absolute rates (not the paired framing result).
10. **Conclusion.** Report your extractor; measure its rate by condition; reproduce outliers. The
    extractor is part of your benchmark.

## Key figures / tables (what to build)

- **Fig. 1 (the money figure):** two panels — (left) compile-rate by condition, naive vs robust;
  (right) pass-rate by condition, naive (+9.3) vs robust (null). One picture tells the whole story.
- **Table 1:** pitfall catalogue × prior-art coverage (from PITFALL_COVERAGE.md).
- **Table 2:** politeness-literature comparison — MYT (effects) / Cai (mostly null) / ours
  (controlled null) × task, extraction-controlled?, conclusion.
- **Fig. 2:** cross-benchmark saturation gradient (HE ~96 / HE+ ~96 / LCB 81 / LCB-hard 67) →
  motivates de-saturation and seeds the closing-window argument.

## Honest contribution sentence (for the abstract / advisor email)

> "We don't discover that extraction biases code evaluation — SAFIM and Macedo established that.
> We show it can **fabricate a statistically significant prompt-*register* effect** (a '+9-pt
> politeness tax') that disappears under robust extraction, use that to **reconcile the conflicting
> politeness literature** (Mind Your Tone vs Cai), and package the evaluation pitfalls we hit into a
> reproducible catalogue with diagnostics — with a caution that these artifacts get harder to catch
> as benchmarks saturate."

## Reviewer objections to preempt (and the answer)

- *"Macedo/SAFIM already showed this."* → On the **model** axis, framed as measurement accuracy. We
  show the **prompt-register** axis + **fabricated significance** + the **politeness-lit correction**.
- *"Just fix your extractor — not a paper."* → The point is it's a **confound that manufactures a
  finding**; we quantify, catalogue, and give diagnostics so others catch it.
- *"Mock framings aren't realistic."* → Clean minimal pairs isolate register; LLM-rewritten framings
  are a stated follow-up, and there's now **no effect left to reduce**.
- *"Contamination."* → Paired within-problem design makes the framing result **contamination-immune**;
  absolute rates are explicitly caveated (model_provenance).
- *"Your MYT/Cai critique is speculative."* → We **raise** the confound (MCQ parsing is robust), not
  assert it; our demonstrated null is in **code**, where we control it.

## Extractor baseline — run three, don't claim one (IMPORTANT rigor add)

Our thesis puts our own extractor under suspicion, so **do not** rely on a bespoke extractor alone,
and **do not** claim "we reimplemented EvalPlus." Instead run **three extractors on the same stored
completions** and report all three:
- (a) **naive** (the one that produced +9.3),
- (b) **ours** (robust),
- (c) **EvalPlus `sanitize`** — an independent, widely-used reference implementation
  (tree-sitter AST; longest-valid-Python substring via `code_extract()`; helper preservation +
  reachability filtering from `entry_point` via `extract_target_code_or_empty()`; handles class
  methods & bare functions). NOTE: it is a **repo reference tool, not a formal standard** (the
  EvalPlus paper never describes it), and it already implements most "robust" features — so our
  extractor is **not** a novel contribution, just one of two independent robust extractors.

If the +9.3 appears under (a) and **vanishes under BOTH (b) and (c)**, the result is near-conclusive
(not a leniency quirk of our tool) — this answers the "is your extractor the real artifact?"
objection and costs no API calls (re-score only). `sanitize` is drop-in for the **HumanEval / HE+**
parts; it *likely* works on LCB's `class Solution` format via `entry_point` but **must be tested /
lightly adapted**. Frame `sanitize` as "the widely-used EvalPlus sanitizer," not "the standard."

## What's needed to finish (gaps)

- **Three-extractor robustness run — DONE** (see THREE_EXTRACTOR_RESULTS.md). naive terse−detailed
  = **+15.9 pts**; ours **+0.3**; sanitize **−2.4** → the tax is naive-extraction-specific and
  vanishes under both robust extractors. Remaining: attach McNemar p-values (run upgraded cell 7).
- Build Fig. 1 and Fig. 2 from existing data (have the numbers).
- Finalize Table 1 (catalogue) and Table 2 (politeness lit) — both drafted in repo docs.
- Decide whether to include the **self-repair** positive companion (optional; strengthens "feedback
  is the lever, not phrasing" but widens scope).
- Lock citations (all vetted except CodeTransBenchmark, which we can simply not cite).
- Optional: one LLM-rewritten-framing robustness run to preempt the "mock wrappers" objection.

---

## Alternative framing (B) — the reproducible-pitfalls SoK + closing-window

Same assets, different emphasis: lead on the **catalogue + closing-window**, with the politeness
result as the flagship case study rather than the headline. Pick B if the advisor wants a broader
"evaluation hygiene" contribution; pick A (above) if they want a sharper, more publishable single
claim. A is recommended: tighter, a live target to correct, and a clear money figure.
