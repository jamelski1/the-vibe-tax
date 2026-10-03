# Related-work notes — verified quotes & numbers (for accurate citation)

Exact, source-checked material to cite from. **[PDF-VERIFIED]** = quoted directly from
the paper PDF read in-session. **[SEARCH]** = from a web search result, re-confirm the
exact page/number against the source before it goes in the paper. Treat everything as
to-double-check at camera-ready.

---

## SAFIM — the closest neighbor (cite prominently; position against)

- **Citation:** Gong, Wang, Elhoushi, Cheung. *Evaluation of LLMs on Syntax-Aware Code
  Fill-in-the-Middle Tasks.* ICML 2024. arXiv:2403.04814. (proceedings.mlr.press/v235/gong24f.html)
- **Benchmark:** SAFIM, 17,720 FIM examples; sourced from code submitted **after April 2022**
  to reduce contamination. [SEARCH]
- **Why it's our closest neighbor:** it shows a *post-processing choice* acts as a
  **differential confound that changes comparative conclusions** — structurally the same
  mechanism as our +9.3 artifact, but on the **model-training axis** (FIM vs non-FIM)
  rather than the **prompt axis**.

**Verbatim quotes [PDF-VERIFIED]:**
- §4.2: *"Post-processing is vital for automatic evaluation of LLMs in code generation,
  yet its importance is often underestimated. The raw output from LLMs is not immediately
  suitable for evaluation due to potential inclusions of irrelevant natural language or
  extra code beyond the targeted structure."*
- §4.2 (Code Extraction for Chat Models): *"We use regex-based heuristics to extract code
  from outputs of chat models like GPT-4, which often mix natural language with code in the
  Markdown-formatted outputs."*
- §4.2 (Truncation): *"inconsistencies in truncation methods across different models have
  led to skewed comparisons in prior work."*
- §6.2: *"syntax-aware truncation benefits non-FIM models much more than FIM models. For
  example, CodeLLaMa-13B's Pass@1 rate jumps from 16.4% to 41.4% with truncation, changing
  its comparative performance against InCoder-6B, whose Pass@1 only increases marginally
  from 21.8% to 25.2%."*
- §6.2: *"The extra code, while removable by syntax-aware truncation, obscures
  CodeLLaMa-13B's true effectiveness when such truncation is not applied."*

**What SAFIM establishes (so we DON'T claim it):**
- Post-processing/extraction matters and is underestimated.
- A naive post-processing choice **differentially** affects groups and **changes
  comparative conclusions** (16.4→41.4 vs 21.8→25.2) — not uniform noise.
- Regex extraction of code from markdown-mixed chat outputs is a known, named step.

**What SAFIM does NOT do (our live wedge):**
- Manipulated axis is **model training type**, never **prompt register/politeness/verbosity/
  language**. It does not study prompt-style as the variable the artifact rides on.
- Frames truncation as **revealing truth / fairer ranking**, NOT as **manufacturing a
  statistically significant false finding that vanishes** (we report paired McNemar p<0.01
  → null).
- No connection to the **politeness/tone/vibe-prompting** literature.
- No **bidirectional** demonstration (artifact also erased a real signal → spurious 0%).
- Not a **reproducible multi-pitfall compilation** (extraction + contamination + saturation
  + timeout + token-cap + binary) with diagnostics, from one ecological study.
- Its truncation problem is about **non-instruction-tuned models not knowing when to stop**;
  ours is about **chat-model prose volume co-varying with prompt register**.

**Our one-line positioning vs SAFIM:**
> SAFIM shows post-processing choices skew *model* comparisons on FIM; we show the same
> class of artifact, operating on the **prompt** axis, can manufacture a statistically
> significant **prompt-style** effect where none exists — implicating the politeness/vibe
> literature — and we package it as a reproducible, diagnosable catalogue.

---

## Fachada et al. — practitioner exemplar (NOT a threat) [PDF-VERIFIED]

- Fachada, Fernandes, Fernandes, Ferreira-Saraiva, Matos-Carvalho. *GPT-4.1 Sets the Standard in
  Automated Experiment Design Using Novel Python Libraries.* MDPI, Sept 2025.
- **Type:** capability benchmark — LLMs writing Python with *unfamiliar* APIs (ParShift, pyclugen,
  scikit-learn), zero-shot, multi-run, functional correctness + prompt compliance + error analysis.
  GPT-4.1 100%; most <50%. (Adjacent to our *capability* work, not our pitfalls thesis.)
- **Useful as a cite, three ways:**
  1. **Ad-hoc-but-careful extraction exemplar:** *"extraction proceeds in up to two steps: (1) code
     from the first ` ```python ` block; (2) if that fails, the entire response is assessed"*; and
     they **score "No Python code could be extracted" as a failure category.** → practitioners DO
     build extraction fallbacks & score extraction failure (supports our premise); but they never
     validate extraction failure is uncorrelated with conditions (they don't vary conditions).
  2. Same *"single code block only… no explanation"* instruction as MYT/Macedo-Vanilla, yet still
     needs a fallback → more support for Macedo-Finding-3 (models ignore format instructions).
  3. **Contamination control via obscure libraries** (*"sparse online code examples"*) — a
     memorization-mitigation strategy different from date-filtering; one-line cite in our
     contamination discussion.
- **Not a mechanism/register/confound threat.**

## Ying et al. — landscape SURVEY (context for the compilation framing; not a mechanism threat) [PDF-VERIFIED]

- Ying, Towey, Zhang (U. Nottingham Ningbo China). *Evaluation of the Code Generated By Large
  Language Models: The State of the Art.* **IEEE COMPSAC 2025.**
- **Type:** systematic-review survey of LLM code evaluation — benchmarks (HumanEval, SWE-bench…),
  metrics (functional correctness, test coverage, code quality), error taxonomy (compile / runtime /
  TLE / style / maintainability / performance / security). Its contribution: a proposed **3-layer
  evaluation framework**.
- **Identified "Limitations of Current Benchmarks":** (1) **Language bias** (mostly Python/Java),
  (2) **evaluation-method fragmentation** (no unified framework). Neither is extraction/prompt.
- **Does NOT cover:** code-extraction-from-prose, fence handling, prompt-format/register
  sensitivity, or any reproducible pitfall analysis.
- **One adjacent, citable precedent:** *"Inaccurate Measurement"* category — **false positives
  where correct code scores low due to STRICT MATCHING**, incl. *"Code Style Difference:
  semantically identical… differences only in whitespace, line breaks… that do not affect
  semantics"* marked wrong. BUT scope = **text-similarity metrics (BLEU-style)** in code
  refinement, NOT extraction/execution. Cite as "measurement artifacts are known in code eval,"
  not the same mechanism. (Also cites LiveBench as a "contamination-free" benchmark.)
- **Implication for US:** a survey of this area exists → do NOT claim "first overview." Our
  compilation must be a **focused, reproducible, thesis-driven pitfalls artifact** (false
  *findings* + register application + closing-window), explicitly distinct from a landscape survey.

## Macedo et al. — STRONGEST overlap (cite prominently; peer-reviewed journal)

- **Citation [PDF-VERIFIED]:** Macedo, Tian, Cogo, Adams. *Output format biases in the
  evaluation of large language models for code translation.* **Empirical Software Engineering
  (2026) 31:41**, DOI 10.1007/s10664-025-10768-1 (accepted 3 Nov 2025). (preprint arXiv:2403.17214)
- **Scope:** code **translation**; 11 open + 5 closed LLMs; 3,820 pairs (C/C++/Go/Java/Python).
  Output formats = Direct / Wrapped (backticks) / Unbalanced, × with/without additional text.

**Verbatim / verified [PDF-VERIFIED]:**
- Finding 1: *"post-processing is required to extract the source code in up to **73.64%** of
  the outputs."* (abstract: between 26.4% and 73.7%.)
- Prompt+regex → **CSR 92.73%**.
- RQ3: lowest avg Computational Accuracy **4.92%** (VDE, direct compile) → highest **31.92%**
  (CRE, controlled prompt + regex extraction). *[CONFIRMED — safe to cite.]*
- Finding 6: *"The consideration of output formats can significantly alter the outcomes when
  benchmarking various LLMs"* (top-ranked model flips under extraction).
- Finding 7: filtering non-code tokens changes BLEU/CodeBLEU, *"statistically significant for
  the majority of the models"* (**α=0.05, Cliff's Delta**).
- **Finding 2: *"Different prompts produce different output format distributions"*** (Reference
  89.55% Direct Code vs Vanilla 52.5% Wrapped).
- Discussion (the dangerous one): *"Output format bias can also be influenced by the design of
  the prompt… **Different stakeholders may phrase their prompts in various ways, and this
  variation can introduce additional biases in model evaluation.**"*
- Directional: *"performance of open-source LLMs may be **underestimated** if their outputs,
  though accurate, contain additional text."*

**What Macedo establishes (so we DON'T claim):** the full chain — prompt design → output-format
distribution → extraction need → **significant** evaluation bias that **alters benchmarking
conclusions** — AND an explicit note that stakeholder prompt *phrasing* introduces biases.
This is the single biggest prior-art hit.

**What Macedo does NOT do (our remaining, narrower wedge):**
- "Different prompts" = **format-instruction templates** (Reference vs "output code only"
  Vanilla), **not register/politeness/verbosity/language**.
- Stops at *"could introduce biases"* (flagged implication); does **not** run a controlled
  experiment that **fabricates a significant *correctness* finding (terse>polite, p<0.01) that
  vanishes**.
- No engagement with the **politeness/vibe literature**; code **translation**, not generation;
  no **multi-pitfall compilation** / closing-window.

**Our one-line positioning vs Macedo:**
> Macedo shows output-format bias significantly skews code-*translation* metrics and flags that
> prompt phrasing can bias evaluation; we instantiate that flagged risk on the **register axis**
> and show it **fabricates a statistically significant politeness/correctness effect that is
> actually null** — correcting the politeness-prompting literature — and package multiple such
> pitfalls reproducibly.

**Honest note:** three uploaded papers (SAFIM, FormatSpread, Macedo) — the extraction-confound
MECHANISM is established prior art; Macedo even names the prompt-phrasing link. Lead on the
**application (politeness correction) + compilation + closing-window**, NOT the mechanism.

## CodeTransBenchmark — verify provenance

- arXiv:2609.20257 (2026). "Flexible Extraction" changes measured accuracy **~53% relative**;
  8 open LLMs, 3 datasets, 12 language pairs (67,071 translations). [SEARCH]
- **Status:** brand-new preprint; **authors/venue UNCONFIRMED** ("Kowalczuk et al." is
  unverified). Do not cite until the paper and authorship are confirmed.

## FormatSpread — prompt format sensitivity (foundational cite; NOT a novelty threat)

- Sclar, Choi, Tsvetkov, Suhr. *Quantifying Language Models' Sensitivity to Spurious Features
  in Prompt Design.* ICLR 2024. arXiv:2310.11324. [PDF-VERIFIED]
- Up to **76-point** accuracy swing (LLaMA-2-13B); **median ~7.5 pts** across 53 tasks;
  recommends reporting a **range**, not one format.

**Verbatim quotes [PDF-VERIFIED]:**
- *"we focus on LLM sensitivity to a quintessential class of meaning-preserving design
  choices: prompt formatting."* (separators, casing, spacing)
- *"fixing a formatting choice could introduce a significant confounding factor."*
- *"Results are reported using ranking accuracy unless specified otherwise."* (ranking =
  probability over valid options → **no free-form output, no parsing/extraction**)
- *"here we evaluate on classification tasks."* (53 SuperNaturalInstructions: 19 MCQ + 34 class.)

**Why it is NOT a threat (4 separations):**
1. **Source of effect:** genuine *model sensitivity* (real behavior) — the **opposite** of our
   claim (a *scoring/extraction artifact*, not behavior).
2. **Measurement:** ranking accuracy (probabilities), **no parsing** — they deliberately
   sidestep the extraction step; that excluded regime is exactly where we work.
3. **Axis:** typographic format (separators/casing/spacing) vs our register/politeness/
   verbosity/language.
4. **Task & conclusion:** classification/MCQ, "effect is real → report a range" vs our code
   generation, "apparent effect is null → the scorer fabricated it."

**Our one-line positioning vs FormatSpread:**
> FormatSpread shows models are genuinely sensitive to typographic format, measured via
> parsing-free ranking accuracy on classification; we show the complementary, more insidious
> case — in free-form code generation the *extraction step itself* can manufacture a
> significant prompt-register effect where the model's behavior is unchanged.

## EvalPlus — supporting cite (weak-tests + buggy ground-truth); NOT an extraction threat [PDF-VERIFIED]

- Liu, Xia, Wang, Zhang (UIUC / Nanjing). *Is Your Code Generated by ChatGPT Really Correct?
  Rigorous Evaluation of LLMs for Code Generation.* NeurIPS 2023.
- **Core thesis (= our row 13):** existing benchmark test-cases are insufficient in quantity &
  quality → wrong code passes. HumanEval+ adds **80× test-cases** (type-aware mutation + LLM-
  generated inputs + differential testing vs ground-truth). Catches previously-undetected wrong
  code, reducing **pass@k by 19.3–28.9%**.
- **Conclusion-changing (mis-ranking):** *"test insufficiency can lead to mis-ranking… both
  WizardCoder-CodeLlama and Phind-CodeLlama now outperform ChatGPT on HumanEval+, while none
  could on HumanEval."* (another flavor of meta-claim X.)
- **New pitfall it adds (row G):** **11% of HumanEval ground-truths are defective** — 18 defects
  (5 unhandled edge-cases, 10 bad logic, + input-format issues). Buggy *reference* solutions are
  their own eval gotcha.
- **IMPORTANT — the paper does NOT cover code extraction.** `evalplus.sanitize` (strip prose/
  markdown/extra code) exists only in the **repo/docs**, not as a paper contribution. So:
  - Cite the **tool/repo** for "extraction is a known, required step" (supports our premise that
    everyone hits extraction, so naive handling is a real risk).
  - Do **NOT** cite the paper's text for extraction sub-modes 1/2/9.
- **Not an extraction-novelty threat.** It's a foundational cite for test-quality/false-positive
  (row 13) and buggy ground-truth (row G), and a second example that eval choices flip rankings.
- Our row 14 (partial credit) is still ours: EvalPlus adds more tests but stays **pass@k/binary**.

## Mind Your Tone — DIRECT TARGET (not a novelty threat) [PDF-VERIFIED, full paper]

- Dobariya, Kumar (Penn State). *Mind Your Tone: Does Tone Alter LLM Performance?* **AMCIS 2026**
  (Americas Conf. on Information Systems). (Short version: arXiv:2510.04950.)
- **Task:** MCQ — 50 custom (5 tones) + **570-question MMLU** subset (7 tones); 4 models
  (ChatGPT-4o, ChatGPT-5-nano, Gemini 2.5 Flash / Flash Lite).
- **Method [PDF-VERIFIED]:** prompt instructs *"Respond with only the letter of the correct
  answer (A, B, C, or D). Do not explain."*; *"The response was parsed to extract the letter."*
- **Stats are solid:** 10 runs, within-subjects, repeated-measures ANOVA + Friedman + paired
  t-tests (Holm) + Cohen's dz + Wilcoxon. Significant tone effects, e.g. ChatGPT-5-nano spread
  **11.12 pp** (F(6,54)=60.46, p=3.4e-22); Gemini Flash Lite **12.46 pp**. **Model-dependent**
  (Neutral wins for ChatGPT-5-nano; Threatening for Gemini Flash; Polite for Flash Lite).
- **The gap our critique targets:** accuracy = parse a letter from a "do not explain" response,
  but they report **NO** parse-rate / instruction-following / exclusion policy **per tone**
  (grep for exclude/invalid/unparse/empty/refuse → nothing). If tone shifts output conformity
  or triggers hidden CoT, parsed-letter accuracy moves for **output-format** reasons, not
  reasoning — they can't separate these. Model-dependence is as consistent with a conformity
  artifact as with their "reasoning-mode routing" hypothesis.
- **Evidentiary bridge:** Macedo Finding 3 — models ignore "output only X" instructions ~59%
  of the time — undercuts the assumption that "do not explain" yields clean parseable output.

**HONESTY GUARDRAILS (do NOT overclaim):**
- MCQ letter-parsing is *more robust* than code extraction → **raise** the confound as an
  uncontrolled alternative, do **not** assert MYT's effect is an artifact.
- Their stats are reasonable; the gap is specifically *unreported extraction/conformity*, not sloppiness.
- Tone may genuinely change reasoning; our point is only that they didn't rule out the artifact.

**Framing:** *"tone→accuracy studies (MYT; Cai et al.) report significant effects from parsed
outputs without reporting extraction/instruction-following robustness per tone; our code-domain
results show such pipelines can manufacture significant prompt-style effects → re-audit needed."*

**TO VET NEXT:** Cai et al. (2025) *Does tone change the answer? Evaluating prompt [tone]* —
cited by MYT; same cluster. MYT also cites FormatSpread.

## Yin et al. — direct target (cross-lingual politeness)

- Yin, Wang, Horio, Kawahara, Sekine. *Should We Respect LLMs? A Cross-Lingual Study on the
  Influence of Prompt Politeness on LLM Performance.* SICon 2024 (ACL workshop).
  arXiv:2402.14531. [SEARCH] EN/ZH/JP; impolite often hurts; over-politeness no clear gain;
  best level is language-dependent. Our multilingual condition speaks to this; raw-accuracy,
  no extraction control.
