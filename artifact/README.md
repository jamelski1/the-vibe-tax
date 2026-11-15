# FORGE Politeness-Tax — LiveCodeBench reproduction artifact

This folder is the self-contained reproduction package. See the full guide in the repository-root
**[`../README.md`](../README.md)**: what the study is, the layout, exact reproduction steps, the
environment/requirements, and what is NOT included (the HF-gated `lcb_tests.jsonl` and model API
re-query). Paper names map to code names in `names.py` (table in the root README). Result-to-script mapping is in **`RESULTS_MAP.md`**; the register templates in
**`REGISTER_TEMPLATES.md`**; the disclosed encoding defect in **`ENCODING_DEFECT.md`**.

**Worked examples.** **[`Appendix_Worked_Examples.pdf`](Appendix_Worked_Examples.pdf)** — "Appendix A -
Worked Examples for the Pitfall Catalog" — gives one worked example for each of the fifteen pitfalls
in the paper's Table II, each a record from the stored data run through the relevant extractor or
harness. Its first section is the worked example the paper's Mechanism section points to (problem
3747). The source, the build script and the script that re-runs every check are in
**[`appendix/`](appendix/)**.

Quick start:
```bash
pip install -r requirements.txt
python extractors.py          # Naive / Sieve / sanitize / LCB-official on one real reply
python mechanism_table.py     # compile-rate + mechanism tables (no gated tests)
python mcnemar.py             # Sieve McNemar, offline
python encoding_robustness.py # encoding-defect audit + robustness
```
