# Paper ingredients — candidate contents to include (prune later)

A working inventory of everything we *could* put in the extraction/evaluation-pitfalls
paper. Nothing here is committed to the final structure yet. Tags:
**[HAVE]** = data/artifact already exists · **[CAND]** = candidate, not yet built ·
**[VERIFY]** = citation/number to confirm before use · **[FUTURE]** = extension.

---

## 1. Framing / thesis options

- **Core thesis.** Measurement & instrumentation choices in code-LLM evaluation can
  *manufacture false findings* — sign flips and fabricated statistical significance —
  not merely add noise. [HAVE: the +9.3→null demonstration]
- **The "closing window" thesis (distinctive hook).** These pitfalls get *harder to
  detect over time*; the conditions that let us catch them are eroding. Forces:
  1. **Rising saturation.** As models improve, more benchmarks hit the ceiling →
     less headroom → an artifact of a given size becomes invisible. (Exactly why
     HumanEval hid the effect and LiveCodeBench revealed it; eventually LCB saturates too.)
  2. **Opaque provider pipelines.** Hidden system prompts, server-side formatting,
     tool-use layers, and safety wrappers increasingly shape API output → extraction
     behavior shifts unpredictably and differs by provider/version, uncontrollably.
  3. **Growing contamination.** Larger training sets + aging benchmarks → absolute
     numbers inflate and a "clean, high-headroom" benchmark gets harder to find;
     contamination can both inflate and mask effects.
  4. **Difficulty re-definition.** As "hard" problems get solved, the frontier moves;
     the measurable-effect region shrinks/shifts and longitudinal comparability breaks.
  5. **LLM-mediated human understanding.** People increasingly read code/problems via
     LLM explanations → the "manually reproduce the surprising cell by hand, distrust
     the number" habit erodes → artifacts go uncaught.
  6. **Reasoning models / hidden tokens.** Hidden chain-of-thought spends the output
     budget → truncation/empty-completion artifacts become more common and more hidden.
  7. **Agentic / multi-turn harnesses.** More pipeline stages → more places for
     parsing/formatting/extraction confounds to hide.
  8. **Leaderboard pressure + benchmark proliferation** → less scrutiny per result.
  → One-line: *the window in which these confounds are visible is closing, so document
    the method for finding them now.*
- **Honest-narrative framing.** "We set out to measure a prompt-style tax and our own
  pipeline fooled us several different ways — here is the reproducible account,
  including the trap we ourselves fell into." (Ecological validity; self-critique.)
- **Systematization framing.** The pieces exist scattered (quiet commits, buried in
  other papers' appendices, inside company harnesses) but not assembled, grounded,
  and reproduced under one thesis. This paper is that reference.

## 2. The reproducible pitfall catalogue (the core deliverable)

Each should have: a one-line mechanism, a **minimal reproducible example**, the wrong
conclusion it produced, the fix, and a citation where one exists.

**Extraction (headline — it fabricated a *significant* effect):** [HAVE]
- trailing prose after unfenced code → SyntaxError
- leading prose / first-code-line grab
- buggy-snippet-first in "fix my error" replies (last-`def`)
- dropped helper defined above `class Solution` (NameError)
- code split across multiple fenced blocks
- class-shape mismatch (bare/renamed/nested) — *a code smell, not incorrectness* [HAVE: 0/2003 prevalence — honest "rare here" data point]
- body-vs-full-function assembly ambiguity
- ragged first-line indentation
- fence-marker variants
- over-aggressive extraction → spurious NULL
  (full 14-mode table already in EXTRACTION_PITFALLS.md)

**Benchmark / harness / metric pitfalls:** [HAVE]
- **Saturation** — no headroom hides small/real effects (the HumanEval→LCB narrative).
- **Contamination** — the filter dates the *contest*, not the model's *training*;
  "contamination-free" is wrong for current models. [HAVE: model_provenance + analysis]
- **Per-problem vs per-test timeout** — kills correct-but-slow (never-solved 15→10).
- **Output-token cap truncation** on reasoning models (GPT-5.6: 73/148 empty). [HAVE]
- **Binary pass@1 vs test-level partial credit** (best attempt ~97% of hard tests). [HAVE]
- **Weak base tests pass subtly-wrong code** (false positives → EvalPlus motivation). [HAVE]

**Candidate additions (if we want more coverage):** [CAND]
- sampling/temperature nondeterminism (and temp=0 rejected by reasoning models).
- prompt-template/format sensitivity itself (tie to FormatSpread).
- float/sequence equality subtleties (the `eq` tolerance choices).
- test *sampling* (max-tests cap) changing pass/fail.
- import/runtime-environment differences between harness and model assumptions.
- stop-token / max-length truncation of the *visible* answer.

## 3. Empirical assets we already have [HAVE]

- LCB framing experiment: 167 problems × 4 framings × 3 models = 2,004; paired McNemar;
  **+9.3 naive → null robust** (terse−detailed capable +0.6, p=0.845).
- Compile-rate-by-condition: terse 88 / casual 81 / multilingual 77 / **detailed 73** →
  100% after fix (the "smoking gun" that the artifact tracks the manipulation).
- HumanEval v2 vs v3: realistic beats researcher-written **+3.0, p=0.001** (invented
  informality *overstates* the tax; and ceiling-crushed → motivates LCB).
- Realism discriminator **AUC 1.0** (synthetic vs real prompts perfectly separable).
- The Vibe Spectrum (how people actually prompt; agentic vs web-chat).
- Capability continuum / partial credit; cross-benchmark saturation gradient
  (HE ~96 / HE+ ~96 / LCB 81 / LCB-hard 67).
- Model provenance (exact versions, release vs training-cutoff, contamination verdict).
- Post-fix capability shifts (hard 51→60.5%, medium 80→87.3%, codestral 12.7→27.8%).
- The 722 failed-attempts list (for slicing/inspection).

## 4. Methodological contributions [HAVE]

- **Robust multi-candidate extractor** that *cannot inflate* correctness (first candidate
  that compiles AND passes).
- **Shape-agnostic resolver** (class/bare/renamed/nested).
- **Per-test-timeout** streaming scorer.
- **Diagnostics suite** (the pre-publication checklist):
  - compile/extraction-rate **by condition** (if it tracks your manipulation → artifact)
  - **manual reproduction** of surprising per-cell results (web-chat vs API)
  - **paired within-problem** design (artifact-immune; separates null from capability)
  - **re-score, don't re-query** (fixes validated on stored completions, no API cost)
  - **model_provenance** (resolve exact versions + cutoffs)
  - **empty-completion check** (catch token-cap truncation)

## 5. Reproducibility artifacts [HAVE]

- Three Colab notebooks: standalone extractor, full pipeline, glass-box explainer
  (per-task `show_problem` / `show_tests` / `inspect_problem`).
- Scripts: `score_lcb.py`, `self_repair.py`, `model_provenance.py`, `test_class_shape.py`,
  `build_breakdown.py`, etc.
- Data: `lcb_v3_responses.json`, `lcb_scored.json`, `lcb_problems.jsonl`, stats JSONs, CSVs.

## 6. Related work to cite (status noted)

> Verbatim quotes + exact numbers for these live in **RELATED_WORK_NOTES.md**.

- **SAFIM** — Gong et al., ICML 2024 (2403.04814). **← CLOSEST NEIGHBOR.** [PDF-VERIFIED]
  Shows a *post-processing choice* is a **differential confound that changes comparative
  conclusions** (CodeLLaMa-13B 16.4%→41.4% vs InCoder-6B 21.8%→25.2% under truncation) —
  *structurally the same mechanism as our +9.3*, but on the **model-training axis**, not the
  **prompt axis**, and framed as *revealing truth* not *fabricating a significant false
  finding*. **Retires the "differential-vs-uniform confound" wedge; must cite & position
  against.** Our live wedge: prompt-register axis + fabricated significance that vanishes +
  politeness-lit critique + multi-pitfall reproducible compilation + closing-window.
- **FormatSpread** — Sclar et al., ICLR 2024 (2310.11324). [VERIFIED]
- **EvalPlus** — Liu et al., NeurIPS 2023 (`evalplus.sanitize`). [VERIFIED]
- **Macedo et al.** — Output Format Biases in Code Translation (2403.17214) — *unidirectional*
  underestimation framing (code translation). [VERIFIED paper; VERIFY exact stats 4.92/31.92, venue]
- **CodeTransBenchmark** (2609.20257) — "Flexible Extraction" ~53%. [VERIFY authors/venue — brand-new]
- **LLMs Are Biased Towards Output Formats** — Long et al., NAACL 2025. [VERIFY]
- **Mind Your Tone** — Dobariya & Kumar (2510.04950). [VERIFIED — a direct target]
- **Yin et al.** — Should We Respect LLMs? SICon 2024 (2402.14531). [VERIFIED — direct target]
- **LiveCodeBench** (the benchmark itself). [CAND — add citation]
- Contamination-audit literature (e.g., GSM8k careful-examination style). [CAND/VERIFY]
- Eval-pitfalls / "evaluation is broken" position papers, HELM methodology. [CAND — for positioning]

## 7. Narrative beats to use

- Origin: identical code **passed in web-chat, failed via API** → chased the apparatus.
- Started on **HumanEval → no headroom** → the mistakes couldn't surface → moved to LCB.
- Nearly reported a confident, significant **+9.3 politeness tax** → caught it.
- Caught the same artifact class in **both directions** (false +9.3; a real signal → 0%).
- Fell into the **contamination trap ourselves** and document it.

## 8. Future work / extensions

- **Self-repair** (positive-result companion; feedback vs no-feedback control). [HAVE harness]
- **Under-specification dose-response** (the real "vibe tax"). [CAND]
- **LLM-rewritten (natural) framings** vs deterministic wrappers. [CAND]
- **Longitudinal re-run** as models improve — *empirically demonstrate the window closing*. [FUTURE — ties to the thesis]
- De-contaminated / strictly-post-cutoff problem slice for the absolute numbers. [CAND]

## 9. Open decisions / risks

- **Venue:** NeurIPS/ICLR Datasets & Benchmarks, empirical-SE (MSR/ESEM/ICSE-SEIP),
  or an eval/reproducibility workshop. (Not a SOTA track.)
- **Novelty positioning:** claim "first to assemble/ground/reproduce under one thesis,"
  not "first to find." Position explicitly vs the crowded extraction-bias space.
- **Citation hygiene:** every reference/number verified (compilation papers live or die here).
- **Scope control:** a pitfalls paper is judged on being *the* reference — complete but
  not padded; every entry reproducible.
- **Keep "contamination-window," not "contamination-free."**
