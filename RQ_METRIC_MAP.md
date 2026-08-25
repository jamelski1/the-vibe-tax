# Research Questions → Metrics Map

Which metric answers which research question. The RQs are as stated in the
Executive Summary; the metrics are the ones in the two workbooks
(**The_Vibe_Tax_Results**, **The_Vibe_Tax_Raw_Data**) and the committed
`*_stats.json` files they are transcribed from. Use this to label every figure
and table with the RQ it serves.

## The five research questions

| RQ | Question | Status |
|----|----------|--------|
| **RQ1** | How do people actually prompt LLMs for code, and does the medium (agentic CLI vs web chat) systematically shape prompt style? | Answered |
| **RQ2** | Does prompt framing impose a measurable correctness penalty — and is the driver *informality* or *verbosity/politeness*? | Answered |
| **RQ3** | Under what conditions is a prompt-style effect measurable at all — how does benchmark headroom govern whether the tax is visible? | Answered |
| **RQ4** | Do researcher-invented informal prompts estimate the tax faithfully, or do artificial degradations bias it? | Answered |
| **RQ5** | Can an automatic terseness correction (TaxCut) that strips politeness/verbosity recover the lost correctness (and tokens)? | Open (Abl. A8) |

## Metric → RQ

Legend for **Answers**: ● primary evidence · ○ supporting.

| # | Metric | Workbook tab / source file | RQ1 | RQ2 | RQ3 | RQ4 | RQ5 | What it establishes |
|---|--------|----------------------------|:---:|:---:|:---:|:---:|:---:|---------------------|
| M1 | Context-provision mix (any-paste %, references-files %, pasted-error %) | Raw Data · Spectrum → `vibe_spectrum_*_stats.json` `by_context_provision` | ● | | | | | Agentic users reference files (21%); web-chat users paste (54%). Same need, opposite mechanism. |
| M2 | Specification-level distribution (L1 formal … L5 pure-vibe) | Raw Data · Spectrum → `by_spec_level` | ● | | | | | 87% of agentic prompts are casual/vague (L3–L5); web-chat is more specified. |
| M3 | Register mix (terse-imperative / conversational / polite) | Raw Data · Spectrum → `by_register` | ● | | | | | Agentic skews terse-imperative; web-chat skews conversational. Defines the "politeness" axis M9 tests. |
| M4 | Multilingual share & error-paste share | Results · Vibe Spectrum → `multilingual_share`, `error_paste_share` | ● | | | | | **Medium vs population effect**: multilingual/error-paste differ WildChat(42/12%) vs LMSYS(16/3%) → track *who uses the tool*, not the medium. |
| M5 | Cross-source replication (WildChat vs LMSYS, same pipeline) | Results · Vibe Spectrum (two web-chat columns) | ● | | | | | Separates what replicates (spec detail, register, code-paste) from population effects (M4). |
| M6 | Realism discriminator AUC / separability | Raw Data · Discriminator → `realism_discriminator_stats.json` | | ○ | | ● | | Synthetic prompts are 100% separable from real ones (AUC 1.0) → hand-invented prompts are OOD. Motivates calibration. |
| M7 | Discriminator top "tells" (weights) | Raw Data · Discriminator → `top_tells` | | | | ● | | *Why* synthetic ≠ real: triple-quotes, signatures, doctests — dressed-up code stubs, not messages. |
| M8 | HumanEval v2 vs v3 pass@1 + Δ (base & HumanEval+) | Results · Experiment Results; Raw Data · HumanEval v2 v3 → `vibe_tax_*_scored_stats.json` | | | ○ | ● | | Realistic (v3) **beats** researcher-written (v2) by **+3.0 pts (p=0.001)** → invented informality **overstates** the tax. Small because ceiling-crushed (→RQ3). |
| M9 | LCB pass by framing × difficulty | Raw Data · LCB → `lcb_scored_stats.json` (recomputed level×difficulty) | | ● | ○ | | | The core measurement: terse 46% vs detailed/polite 40% overall; the gap is a *framing* effect on an identical problem. |
| M10 | **LCB pairwise McNemar** — terse vs detailed / casual / multilingual | Raw Data · LCB McNemar → `lcb_mcnemar_stats.json` (`mcnemar_lcb.py`) | | ● | | | | **Answers RQ2's "which driver":** terse−detailed **+9.3 pts, p=0.007** (capable) but terse−casual (+1.9, n.s.) and terse−multilingual (+2.2, n.s.) are **null** → the tax is *politeness/verbosity*, not informality or language. |
| M11 | Benchmark headroom (HumanEval 96.2% / HE+ 91.7% / LCB 43.5%) | Results · Dashboard; Raw Data · Fig 1 inputs | | | ● | | | The saturation story: you cannot size a small effect where models score ~90%. LCB provides the headroom. |
| M12 | Effect size by difficulty slice (easy* / medium / hard) | Raw Data · LCB + LCB McNemar (difficulty slices) | | ○ | ● | | | The **Goldilocks curve**: effect suppressed at ceiling, visible in headroom (medium), washed out at floor (hard). *(easy added via `ADD_EASY.md`.)* |
| M13 | **HE4** — 4-condition HumanEval, *same deterministic wrappers as LCB* | `vibe_tax_v2/he4_scored_stats.json` (`generate_he4_prompts.py`) | | ● | ● | | | Matched minimal pair to M9: same 4 wrappers on a *saturated* benchmark. Isolates framing from spec-detail (the old v2 confound) and shows the same manipulation shrinks toward null at the ceiling. |
| M14 | By-model breakdown (ChatGPT / Claude / Codestral) | Raw Data · LCB → `by_model` | | ○ | ○ | | | Capable-model slice is the clean read; Codestral floors on hard and adds noise. Governs where the effect is legible. |
| M15 | TaxCut efficacy: detailed_raw vs detailed_autocorrected vs terse | Results · Ablation Tracker (A8) — **not yet run** | | | | | ● | The intervention: does stripping politeness/verbosity recover pass@1 toward terse (and save tokens)? |

\* easy LCB slice is pending the `ADD_EASY.md` re-run.

## How the RQs are answered (one line each)

- **RQ1** ← M1–M5. The medium shapes the prompt; some axes are population effects.
- **RQ2** ← M9, M10 (primary), M8/M13 (framing at the ceiling), M3 (defines the axis). *Driver = politeness/verbosity, established by the pairwise nulls in M10.*
- **RQ3** ← M11, M12 (primary), M13/M8 (same effect ceiling-crushed vs visible), M14.
- **RQ4** ← M6, M7, M8 (primary). Invented prompts are separable and overstate the tax.
- **RQ5** ← M15 only (open; Ablation A8 / the TaxCut tool).

## Note on the second workbook

`vibe_coding_mapping_final` / `Vibe_Coding_Research_Type_Classification` are the
**systematic-mapping-study** (the 39-paper literature review, RQ1–RQ4 of *that*
study). Those are a separate project's RQs and are intentionally **not** mapped
here — this file covers the Vibe Tax experimental workbooks only.
