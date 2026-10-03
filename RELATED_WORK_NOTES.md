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

## Macedo et al. — unidirectional framing (code translation)

- **Citation:** Macedo, Tian, Cogo, Adams. *Output Format Biases in the Evaluation of LLMs
  for Code Translation.* arXiv:2403.17214. [SEARCH — verify venue/year; earlier external
  analysis claimed "EMSE 2026 / FORGE 2024" — DO NOT use until confirmed.]
- **Verified content [SEARCH]:** 11 instruct-tuned open LLMs, 3,820 translation pairs across
  C/C++/Go/Java/Python; output formats = Direct / Wrapped (backticks) / Unbalanced; prompt+
  regex reaches **Code extraction Success Rate (CSR) ~92.73%**.
- **Framing:** extraction bias as **unidirectional underestimation** of accuracy ("models
  look worse than they are"). **[VERIFY exact numbers "4.92% vs 31.92%" — unconfirmed.]**
- **Our wedge:** validity (fabricated significant effect on the prompt axis) vs their
  measurement-accuracy framing; different task (translation).

## CodeTransBenchmark — verify provenance

- arXiv:2609.20257 (2026). "Flexible Extraction" changes measured accuracy **~53% relative**;
  8 open LLMs, 3 datasets, 12 language pairs (67,071 translations). [SEARCH]
- **Status:** brand-new preprint; **authors/venue UNCONFIRMED** ("Kowalczuk et al." is
  unverified). Do not cite until the paper and authorship are confirmed.

## FormatSpread — prompt format sensitivity

- Sclar, Choi, Tsvetkov, Suhr. *Quantifying Language Models' Sensitivity to Spurious Features
  in Prompt Design.* ICLR 2024. arXiv:2310.11324. [SEARCH]
- Up to **76-point** accuracy swings from semantically-equivalent format changes; rankings
  flip; recommends reporting a **range** over formats. Backdrop for "surface form confounds."

## EvalPlus — the sanitize tool / weak-tests motivation

- Liu, Xia, Wang, Zhang. *Is Your Code Generated by ChatGPT Really Correct?* NeurIPS 2023.
  Ships **`evalplus.sanitize`** (post-processing to strip prose/extra code); HumanEval+ adds
  ~80× tests. [SEARCH] Use for: extraction is a known required step; weak base tests pass
  wrong code (our false-positive pitfall).

## Mind Your Tone — direct target (politeness)

- Dobariya, Kumar. *Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy.*
  arXiv:2510.04950 (2025). [SEARCH] Rude > polite (**84.8% vs 80.8%**) on 250 MCQ prompts,
  ChatGPT-4o, from **raw accuracy** (no extraction control). Our result is a methodological
  alternative explanation.

## Yin et al. — direct target (cross-lingual politeness)

- Yin, Wang, Horio, Kawahara, Sekine. *Should We Respect LLMs? A Cross-Lingual Study on the
  Influence of Prompt Politeness on LLM Performance.* SICon 2024 (ACL workshop).
  arXiv:2402.14531. [SEARCH] EN/ZH/JP; impolite often hurts; over-politeness no clear gain;
  best level is language-dependent. Our multilingual condition speaks to this; raw-accuracy,
  no extraction control.
