# Source for `Appendix_Worked_Examples.pdf`

`../Appendix_Worked_Examples.pdf` gives one worked example for each of the fifteen pitfalls in the
paper's Table II. Every example is a record from the study's stored data, and every statement about
what an extractor or the harness does with it comes from running that code on the record.

| File | What it is |
|---|---|
| `Appendix_Worked_Examples.md` | The text of the appendix (the PDF is built from this) |
| `appendix.css` | Page layout for the PDF |
| `build_pdf.py` | Markdown → HTML (pandoc) → PDF (headless Chromium/Chrome) |
| `verify_examples.py` | Re-runs every check quoted in the appendix (LiveCodeBench sections) |
| `he_phase.py` | The checks for pitfalls 3, 7, 8, 13, on the earlier HumanEval-phase stored replies |
| `historical.py` | Two pre-fix functions of the scorer, verbatim (pitfalls 4 and 11). Not used by any result |
| `evidence/lcb_scored_before_helper_and_timeout_fixes.json` | The stored scoring run made before the helper fix and the per-test timeout (pitfalls 4 and 11) |
| `verify_output.txt` | Captured output of `python verify_examples.py --slow` |

## Re-run the checks

```bash
cd artifact/appendix
python verify_examples.py            # all fifteen sections, about 2 minutes
python verify_examples.py 4 11       # only these pitfalls
python verify_examples.py --slow     # also re-runs the old whole-suite timeout check (pitfall 11)
```

Standard library only. The `sanitize` rows also need `tree_sitter` and `tree_sitter_python`
(`../requirements.txt`); without them those rows print "not run".

Some sections read files from the research tree of this branch, outside `artifact/`:
`data/politeness_tax_lcb/lcb_tests_sample.json` (the official tests for two problems),
`data/politeness_tax_lcb/lcb_v3_progress.json` (the checkpoint written while replies were collected),
and `data/politeness_tax_v2/` plus `data/HumanEval.jsonl/` (HumanEval-phase replies, tests and
scorers). If those are absent, the affected sections say so and are skipped.

Some replies in the output import `sortedcontainers`, which the scripts do not install; they are
reported as `missing_library`, as in the study.

## Rebuild the PDF

```bash
python build_pdf.py                  # writes ../Appendix_Worked_Examples.pdf
```

Needs `pandoc` and a Chromium-family browser (set `CHROME=/path/to/browser` if it is not found).
Fonts used if present: Liberation Serif, DejaVu Sans Mono, and a CJK font such as WenQuanYi Zen Hei
for the one section that shows Chinese text.

## What was and was not verified

| Pitfalls | Status |
|---|---|
| 1, 2, 4, 5, 9, 10, 12, 14 | Instance in the 1,336 replies; extractor or harness re-run on it |
| 3, 8, 13 | No instance in the 1,336 (HumanEval-phase pitfalls); instance from the HumanEval-phase stored replies, scorers re-run on it |
| 11 | Instance in the 1,336, shown by two stored scoring runs; timing not re-executed (official tests gated) |
| 15 | Garbled text verified in the stored file; the collection-time checkpoint shows the prompt was clean when sent |
| 6, 7 | No instance in the stored data; the sections say what the data contain and where the pitfall comes from |
| 12 (empty form) | No empty reply in the stored data; observed in an ablation whose replies are not stored |
