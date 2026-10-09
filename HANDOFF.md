# HANDOFF — The Vibe Tax / extraction-artifacts paper

Single entry point for continuing this project (e.g. in Claude Cowork). Everything is committed to
GitHub; start by reading this, then the files in **Key files** below.

- **Repo:** `jamelski1/the-vibe-tax`
- **Branch:** `claude/blissful-hopper-dy0rvr`
- **Read-first order:** this file → `PAPER_FRAMING.md` → `THREE_EXTRACTOR_RESULTS.md` →
  `RELATED_WORK_NOTES.md` → `PITFALL_COVERAGE.md`.

## What this project is

A code-evaluation study that became an **extraction/measurement-artifacts** paper. Core finding:
on LiveCodeBench, a standard-but-naive code extractor produces a statistically significant
"politeness/verbosity tax" that **vanishes under robust extraction** — the apparent prompt-style
effect is a scoring artifact, not model behavior.

## The headline result (done, solid)

Paired McNemar, terse vs detailed, capable models (ChatGPT+Claude), full LCB test suites:

| extractor | terse − detailed | McNemar p |
|---|---:|---:|
| naive | **+15.9 pts** | **1.6×10⁻⁷** |
| ours (`score_lcb.py`) | **0.0** | **1.0** |
| EvalPlus `sanitize` | **−2.4** | **0.27** |

The effect appears only under naive extraction and is null under **two independent robust
extractors** (ours + EvalPlus's published tool). All 167 problems have ≤44 tests → MAX_TESTS=60
runs every test (no sampling). Details + the compile-rate version in `THREE_EXTRACTOR_RESULTS.md`.

## Chosen framing (A)

"**The Politeness Tax Is a Measurement Artifact**" — a focused paper. See `PAPER_FRAMING.md` for the
full sketch (title, outline, novelty-vs-built-on, venue, reviewer objections). Alternative framing
(B, pitfalls-SoK + closing-window) is noted there.

## Honest guardrails (DO NOT violate — these were hard-won)

- **Do NOT claim we discovered extraction-is-a-confound.** It's established prior art — **SAFIM**
  (ICML'24, model axis), **Macedo** (EMSE'26, code translation; names the prompt-phrasing link),
  and **Ouédraogo** (EMSE'26, test generation; "prompt engineering strongly influences
  extractability", adapts Macedo's MSR/CSR). TWO peer-reviewed EMSE papers now establish it — cite
  BOTH and position against. Our novelty is the **register axis + fabricated-significance-that-
  vanishes-AND-reverses (four extractors, incl. official LCB) + politeness-lit correction +
  reproducible compilation + closing-window**.
- **Say "contamination-window," NOT "contamination-free."** Models' training cutoffs postdate the
  problems; the framing result is contamination-immune by the paired design, absolute rates are not.
- **On the politeness literature (Mind Your Tone, Cai):** RAISE the extraction confound as an
  uncontrolled alternative, do NOT assert their effects are artifacts (MCQ parsing is robust; their
  stats are fine; Cai is already mostly null).
- **Every citation/number must be verified against the primary source.** One fabricated-looking
  citation (CodeTransBenchmark) is flagged "do not cite until confirmed."

## Key files

- `PAPER_FRAMING.md` — the paper sketch (what to write, where each asset goes).
- `THREE_EXTRACTOR_RESULTS.md` — the headline result (compile-rate + pass-rate + McNemar).
- `RELATED_WORK_NOTES.md` — 8 vetted papers, verbatim quotes, positioning, verification status.
- `PITFALL_COVERAGE.md` — the 14+ pitfalls × which paper already covers each.
- `EXTRACTION_PITFALLS.md` — the failure-mode catalogue (per-mode code + fix).
- `PAPER_EXTRACTION.md` — an earlier full draft (predates the framing decision; mine for prose).
- `PAPER_INGREDIENTS.md` — the candidate-contents inventory.
- Notebooks: `The_Vibe_Tax_Three_Extractors.ipynb` (the robustness result),
  `The_Vibe_Tax_Extractor_Explained.ipynb` (glass-box walkthrough),
  `The_Vibe_Tax_Extractor_Standalone.ipynb`, `The_Vibe_Tax_Pipeline.ipynb`.
- Code: `data/vibe_tax_lcb/score_lcb.py` (the scorer/extractor), `self_repair.py`,
  `model_provenance.py`. Data: `lcb_v3_responses.json`, `lcb_scored.json`, `lcb_problems.jsonl`.

## Reproducibility notes

- `lcb_tests.jsonl` (LCB's official tests, ~436MB) is **HF-gated and NOT in the repo** — needed for
  any pass-rate scoring; regenerate with `extract_lcb_tests.py` (HF token) or Drive.
- Re-scoring is decoupled from the API: once responses exist, re-run `score_lcb.py` (no model calls).

## Next steps (in priority order)

1. **Build Fig. 1** — compile-rate + the three-extractor McNemar panel (numbers in
   `THREE_EXTRACTOR_RESULTS.md`). Publication-quality (matplotlib, SVG/PDF).
2. **Draft Results + Methods** around the three-extractor table and the pitfall catalogue.
3. **Draft Related Work** from `RELATED_WORK_NOTES.md` (verified cites only; position vs SAFIM/Macedo/Ouédraogo).
4. Optional: `MAX_TESTS=10000` confirmation (will be identical — max 44 tests); LLM-rewritten
   framings robustness run; the self-repair positive companion (`self_repair.py`).
5. Decide venue (empirical-SE: MSR/ESEM/ICSE-SEIP, or an eval/reproducibility workshop).

## How to continue in Cowork

Open a Cowork session connected to `jamelski1/the-vibe-tax` on branch
`claude/blissful-hopper-dy0rvr`, and start with: *"Read HANDOFF.md and PAPER_FRAMING.md, then help
me draft the Results section / build Fig. 1."* Keep committing to the same branch.
