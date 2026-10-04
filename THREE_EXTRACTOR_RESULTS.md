# Three-extractor robustness result (naive / ours / EvalPlus sanitize)

The decisive robustness check: run three independent extractors on the **same** stored completions
and compare. If the "+9-pt politeness tax" is an extraction artifact, it appears under **naive** and
vanishes under both robust extractors. It does. Capable models (ChatGPT + Claude), LCB, the four
register framings wrapping an identical problem.

Extractors: **naive** = the pre-fix extractor (first fenced block, else whole fence-stripped reply
with trailing prose); **ours** = `score_lcb.py` robust extractor; **sanitize** = EvalPlus's
`sanitize` algorithm (vendored, byte-identical functions). Reproduce: `The_Vibe_Tax_Three_Extractors.ipynb`.

## Compile-rate by condition (needs only responses; no tests)

Share of completions whose extracted code parses AND defines the target. Capable models:

| condition | naive | ours | sanitize |
|-----------|------:|-----:|---------:|
| agentic_terse | 88.0 | 100.0 | 81.4 |
| agentic_casual | 81.1 | 100.0 | 84.1 |
| webchat_multilingual | 77.2 | 100.0 | 83.2 |
| webchat_detailed | **73.4** | 100.0 | 84.1 |
| **spread (max−min)** | **14.6** | **0.0** | **2.7** |

Naive: detailed is the *lowest* (the asymmetry that manufactures the tax). Ours + sanitize: flat.

## Pass-rate by condition (graded tests, MAX_TESTS=60)

| condition | naive | ours | sanitize |
|-----------|------:|-----:|---------:|
| agentic_terse | 73.1 | 81.7 | 66.8 |
| agentic_casual | 66.2 | 81.4 | 67.7 |
| webchat_detailed | **57.2** | 81.4 | 69.2 |
| webchat_multilingual | 65.9 | 84.4 | 69.8 |
| **spread (max−min)** | **15.9** | **3.0** | **3.0** |

**terse − detailed (the "politeness/verbosity tax"):**

| extractor | terse − detailed | reading |
|-----------|-----------------:|---------|
| **naive** | **+15.9 pts** | large apparent tax (terse beats polite; detailed worst) |
| **ours** | **+0.3 pts** | null |
| **sanitize** | **−2.4 pts** | null (if anything reversed) |

(terse − casual: naive +6.9 / ours +0.3 / sanitize −0.9. terse − multilingual: naive +7.2 / ours
−2.7 / sanitize −3.0. Under naive, terse beats *every* other framing; under both robust extractors
it does not.)

## Interpretation

- The apparent "terse > polite/detailed" effect is **naive-extraction-specific**: it appears only
  under the extractor that fails on trailing prose, and **vanishes under two independent robust
  extractors** — ours *and* the EvalPlus `sanitize` algorithm.
- This answers the key reviewer objection — *"is the null just your extractor being lenient?"* —
  No: an independent, differently-designed extractor (EvalPlus, HumanEval-oriented) reaches the same
  null asymmetry.
- **Absolute levels differ by design, read the ASYMMETRY.** sanitize sits ~67–70% (vs ours ~81%)
  because it is HumanEval-oriented — its reachability targets *top-level* functions, so on LCB's
  `class Solution` *method* format it mostly falls back to longest-valid-substring extraction.
  Lower absolute level, same flattening of the condition asymmetry.

## Status / next

- These are the **capable-model** numbers (ChatGPT + Claude), MAX_TESTS=60, full set (1,336 records
  per extractor). From a notebook run; recompute with the committed `score_lcb.py` for the paper's
  canonical table if desired.
- **McNemar p-values:** the deltas above are from aggregate pass-rates; run the upgraded cell 7
  (paired terse-vs-detailed McNemar per extractor) to attach exact p. Given +15.9 vs +0.3/−2.4 on
  ~1,336 paired observations, naive is clearly significant and both robust extractors clearly n.s.
- Feeds **Fig. 1** (compile-rate + pass-rate, naive vs robust) in `PAPER_FRAMING.md`.
