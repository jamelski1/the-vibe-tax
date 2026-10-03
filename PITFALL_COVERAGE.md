# Pitfall × prior-art coverage matrix (living doc — update as papers are vetted)

For each evaluation pitfall we document, which published paper already covers it. Lets us see,
per pitfall, what is **established prior art** (cite, don't claim) vs **largely ours to
document**. Update this every time a new paper is vetted (append to the changelog at the bottom).

**Legend:** ✓ = clearly covers it · ~ = partial / adjacent / different cause · ✗ = not covered.
Columns are papers we've **PDF-verified** unless marked [s] (search-level only — reconfirm before
citing). Full quotes/citations in `RELATED_WORK_NOTES.md`.

| # | Pitfall | layer | SAFIM (ICML'24) | Macedo (EMSE'26) | FormatSpread (ICLR'24) | EvalPlus (NeurIPS'23)[s] | coverage verdict |
|---|---------|-------|:---:|:---:|:---:|:---:|---|
| 1 | Trailing prose after unfenced code | EXT | ~ (truncates extra content) | ✓ (text after code) | ✗ | ~ (sanitize) | **established** |
| 2 | Leading prose / chatter before code | EXT | ✗ | ✓ (text before/interspersed) | ✗ | ~ (sanitize) | **established** |
| 3 | Buggy-snippet-first in "fix my error" replies (take last `def`) | EXT | ✗ | ✗ | ✗ | ✗ | **largely ours** |
| 4 | Dropped helper defined above `class Solution` (NameError) | EXT | ✗ | ✗ | ✗ | ✗ | **largely ours** |
| 5 | Code split across multiple fenced blocks | EXT | ✗ | ~ (multi-format) | ✗ | ✗ | mostly ours |
| 6 | Class-shape mismatch (bare / renamed / nested) | EXT | ✗ | ✗ | ✗ | ✗ | **ours** (but 0/2003 prevalence here) |
| 7 | Body-vs-full-function assembly ambiguity (HumanEval) | ASM | ✗ | ✗ | ✗ | ~ (HE context) | mostly ours |
| 8 | Ragged first-line indentation | ASM | ✗ | ✗ | ✗ | ~ | mostly ours |
| 9 | Fence-marker variants (incl. unbalanced back-ticks) | EXT | ~ | ✓ ("Unbalanced Back-ticks" category) | ✗ | ~ | **established** |
| 10 | Over-aggressive extraction → spurious NULL | EXT | ~ (empty-after-truncation = compile err) | ~ (regex mismatch → 0%) | ✗ | ✗ | partial prior art |
| 11 | Per-problem total timeout fails correct-but-slow | HARNESS | ✗ | ✗ | ✗ | ✗ | **largely ours** |
| 12 | Output-token cap truncation — **empty** or **truncated-incomplete** (hits bigger/reasoning models) | HARNESS | ~ (notes empty-after-truncation; different cause: no-EOS) | ✗ | ✗ | ✗ | **largely ours** (reasoning-model cause under-documented) |
| 13 | Weak base tests pass subtly-wrong code (false positive) | HARNESS | ✗ | ✗ | ✗ | ✓✓ (EvalPlus's core thesis) | **established (EvalPlus)** |
| 14 | Binary pass@1 vs test-level partial credit | METRIC | ✗ | ✗ | ✗ | ✗ (EvalPlus adds tests, still binary) | mostly ours |
| S | Benchmark saturation hides small/real effects | BENCH | ~ (FIM has headroom; not framed as a pitfall) | ✗ | ✗ | ~ (HE+ harder) | partial prior art |
| C | Contamination: filter dates the *contest*, not *training* | BENCH | ~ (samples post-Apr-2022 to reduce it) | ✗ | ✗ | ✗ | partial prior art |
| X | **Confound rides the prompt axis → fabricates a *significant* result** (the meta-claim) | — | ~ (differential across *models*, changes ranking) | ✓ flags prompt-phrasing→bias; shows α=0.05, alters conclusions | ~ (format confounds comparison) | ✗ | **established mechanism** — don't claim; our wedge is the *register axis + politeness-correction + vanishing significance* |

## How to read this for the paper

- **Rows marked "established" / "established mechanism" → cite, don't claim.** Especially the
  core extraction cases (1, 2, 9), weak-tests (13, EvalPlus), and the meta-claim X (SAFIM + Macedo).
- **Rows marked "ours" / "largely ours" → the ones worth foregrounding as documented, reproduced
  contributions** (3, 4, 6, 11, 12, and partly 5, 7, 8, 14). None is a blockbuster alone, but as a
  *reproducible, unified catalogue* they carry the compilation paper.
- **The strongest individually-underexplored one is #12** (token-cap truncation on bigger/reasoning
  models, two sub-states) — closest to "not really covered elsewhere." Worth a clean reproducible
  example and its own subsection.
- **Reality check:** the *mechanism* (row X) is taken. Our novelty is the application (register/
  politeness), the fabricated-significance-that-vanishes demonstration, the compilation, and the
  closing-window thesis — not any single pitfall.

## Caveats

- EvalPlus column is search-level [s] — skim the paper to confirm cells (esp. whether `sanitize`
  enumerates sub-modes 1/2/9) before citing specifics.
- "~" is a judgment call; when a cell matters for a claim, pull the exact quote into
  `RELATED_WORK_NOTES.md` first.
- This matrix covers only papers vetted so far (SAFIM, Macedo, FormatSpread, EvalPlus). Add columns
  as new papers come in.

## Changelog
- Initial matrix: SAFIM, Macedo, FormatSpread (all PDF-verified), EvalPlus (search). Token-cap
  pitfall (#12) expanded to two sub-states (empty / truncated-incomplete).
