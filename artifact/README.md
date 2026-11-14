# FORGE Politeness-Tax — LiveCodeBench reproduction artifact

This folder is the self-contained reproduction package. See the full guide in the repository-root
**[`../README.md`](../README.md)**: what the study is, the layout, exact reproduction steps, the
environment/requirements, and what is NOT included (the HF-gated `lcb_tests.jsonl` and model API
re-query). Result-to-script mapping is in **`RESULTS_MAP.md`**; the register templates in
**`REGISTER_TEMPLATES.md`**; the disclosed encoding defect in **`ENCODING_DEFECT.md`**.

Quick start:
```bash
pip install -r requirements.txt
python extractors.py          # the four extractors on one real reply
python mechanism_table.py     # compile-rate + mechanism tables (no gated tests)
python mcnemar.py             # robust-extractor McNemar, offline
python encoding_robustness.py # encoding-defect audit + robustness
```
