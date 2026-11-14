# FORGE Politeness-Tax — LiveCodeBench reproduction artifact

This branch (`artifact/forge-reproduction`) is a self-contained reproduction package for the
LiveCodeBench "politeness tax" study. Everything needed to reproduce the study end to end — from the
raw model replies to the paper's tables and figures — is under **[`artifact/`](artifact/)**.

## What the study is

On LiveCodeBench, we test whether the **register** of a coding prompt (terse / casual / polite-and-
detailed / a Chinese wrapper) changes a model's correctness. The same problem statement is wrapped
in four register templates; only the surrounding words differ. The finding:

- A **standard-but-naive** code extractor produces a large, highly significant "politeness tax"
  (terse beats detailed by **+15.9 points**, paired McNemar p ≈ 1.6×10⁻⁷).
- That effect is a **scoring artifact, not model behavior**: it **vanishes** under two independent
  robust extractors (Sieve 0.0, p=1.0; EvalPlus `sanitize` −2.4, p=0.27) and **reverses**
  under LiveCodeBench's own fence-only extractor (~−12 points). Same 1,336 replies, four extractors,
  three opposite conclusions.
- Mechanism: polite/detailed prompts elicit more trailing prose; the naive extractor feeds code +
  prose to the parser → `SyntaxError` on otherwise-correct code. Once extraction is controlled,
  register has no measurable effect on correctness; failures track **difficulty**, not phrasing.

Scope of this slice: **167 LiveCodeBench problems × 2 capable models (GPT-5.4 "chatgpt", Claude
Opus 4.6 "claude") × 4 register conditions = 1,336 replies.** (The raw file also contains a third
model, Codestral; the paper's slice is the 1,336 capable-model replies.)

## Layout (`artifact/`)

```
artifact/
  extractors.py                 The FOUR extractors as one runnable module:
                                  extract_naive, extract_sieve, extract_sanitize, extract_lcb_official
  names.py                      Paper name <-> code/data key (extractors, registers, models)
  score_lcb.py                  Scoring harness: Sieve extraction -> exec -> per-test run, 60-test cap,
                                  per-test timeout, pass/compile determination, failure-reason log
  extract_lcb_tests.py          Regenerates the (HF-gated) lcb_tests.jsonl from the LCB release
  generate_lcb_prompts.py       Prompt generation + the four register templates (_mock)
  mcnemar.py                    Paired terse-vs-detailed McNemar + 95% CI from a scored file (offline)
  score_all_extractors.py       Pass-rate + McNemar for all four extractors  [needs lcb_tests.jsonl]
  mechanism_table.py            Compile-rate table + mechanism table (no tests)
  encoding_robustness.py        CP437 encoding-defect audit + headline-robustness check (no tests)
  make_fig1.py / make_fig_mechanism.py / make_fig_encoding.py   Figure generators (-> pdf/svg/png)
  The_Politeness_Tax_Four_Extractors.ipynb    Notebook mechanism_table.py reads the extractors from
  REGISTER_TEMPLATES.md         The four templates, verbatim
  ENCODING_DEFECT.md            The disclosed CP437 defect + the 9 affected problems
  RESULTS_MAP.md                Paper result -> script that regenerates it
  requirements.txt              Python deps
  data/
    lcb_v3_responses.json              All 2,004 raw replies as stored (3 models x 167 x 4)
    lcb_v3_responses_capable.json      The 1,336 capable-model slice (chatgpt + claude)
    lcb_problems.jsonl                 Problem statements + metadata (entry_point, difficulty, date)
    lcb_scored_sieve.json              Per-reply Sieve verdict (test-verified pass/fail)
    lcb_scored_sieve_stats.json        Aggregate Sieve stats, all three models
    lcb_scored_sieve_stats_chatgpt-claude.json   Aggregate Sieve stats + failure-reason split, the paper's slice
```

A record in `lcb_v3_responses.json` has: `task_id`, `entry_point`, `level` (the condition key:
`agentic_terse` / `agentic_casual` / `webchat_detailed` / `webchat_multilingual`), `model`,
`prompt_text` (the wrapped prompt as sent), `completion` (the raw reply), and metadata.

## Paper name ↔ code name

Names in the code match the paper. The stored replies keep the keys they were collected under,
so the raw data is untouched; `artifact/names.py` holds the same mapping for scripts.

| In the paper | In the code / data |
|---|---|
| **Naive** scoring | `extract_naive` · `EXTRACTORS["naive"]` |
| **Sieve** (called "Ours" in earlier drafts) | `extract_sieve` · `EXTRACTORS["sieve"]` · the harness `score_lcb.py` applies it · outputs `data/lcb_scored_sieve*.json` |
| **sanitize** (EvalPlus) | `extract_sanitize` · `EXTRACTORS["sanitize"]` |
| **LCB-official** | `extract_lcb_official` · `EXTRACTORS["lcb_official"]` |
| register **terse** | reply field `level == "agentic_terse"` |
| register **casual** | `level == "agentic_casual"` |
| register **detailed** (polite) | `level == "webchat_detailed"` |
| register **multilingual** (Chinese template) | `level == "webchat_multilingual"` |
| **GPT-5.4** | reply field `model == "chatgpt"` |
| **Claude Opus 4.6** | `model == "claude"` |

## Reproduce

```bash
cd artifact
python -m pip install -r requirements.txt      # tree_sitter, scipy, matplotlib (see notes below)

# 1. Sanity: the four extractors on one real reply (Naive FAILs, Sieve/sanitize pass, LCB-official empty)
python extractors.py

# 2. Compile-rate + mechanism tables, all from stored replies (NO gated tests needed)
python mechanism_table.py

# 3. The Sieve McNemar result, offline from the shipped Sieve verdicts
python mcnemar.py                              # n=334, terse-detailed +0.6 pts, p=0.845 (null)

# 4. Encoding-defect audit + headline robustness excluding the 9 corrupted problems
python encoding_robustness.py

# 5. Figures -> pdf/svg/png
python make_fig1.py && python make_fig_mechanism.py && python make_fig_encoding.py

# 6. (needs lcb_tests.jsonl) pass-rate + McNemar for all four extractors
python extract_lcb_tests.py                    # -> lcb_tests.jsonl   [needs HF access; see below]
LCB_TESTS=lcb_tests.jsonl python score_all_extractors.py
```

`RESULTS_MAP.md` maps each paper number/table/figure to the exact command.

## Environment / requirements

- **Python 3.9+** (tested on 3.11 and 3.13).
- The **core** (extractors, scoring harness, `mcnemar.py`, `encoding_robustness.py`) needs only the
  **standard library**.
- `extract_sanitize` and `mechanism_table.py` need **`tree_sitter` + `tree_sitter_python`**;
  `mechanism_table.py` and the figure scripts also need **`scipy`** and **`matplotlib`**.
  `make_fig_encoding.py` renders Chinese and benefits from a CJK font (e.g. `fonts-wqy-zenhei`).
- See `requirements.txt`.

## What is NOT included (needed for full reproduction, named explicitly)

1. **`lcb_tests.jsonl` — LiveCodeBench's official test cases.** Required to recompute **pass-rates**
   (steps that run tests: `score_all_extractors.py`, re-scoring with `score_lcb.py`). It is **HF-
   gated** (hundreds of MB) and not redistributed here. Regenerate it with `extract_lcb_tests.py`,
   which reads the cached LiveCodeBench release files from Hugging Face
   (`livecodebench/code_generation_lite`) and needs Hugging Face access. **All compile-level results,
   the mechanism table, the encoding audit, the offline Sieve McNemar, and the figures reproduce
   WITHOUT it**; only the pass-rate table needs it. The frozen pass/fail verdicts for Sieve
   are shipped in `data/lcb_scored_sieve.json`, so downstream analysis is reproducible offline.
2. **Model API access to regenerate the raw replies.** The 1,336 replies are shipped as stored, so
   you do not need this to reproduce the analysis. You would only need it to re-query the models —
   e.g. to regenerate prompts with the CP437 encoding defect fixed (see `ENCODING_DEFECT.md`). The
   models used were GPT-5.4, Claude Opus 4.6 (and Codestral); re-querying requires the respective
   provider API keys, which are **not** in this artifact.
3. **LiveCodeBench version.** The problem set is pinned by the shipped `data/lcb_problems.jsonl`
   (167 problems, with `contest_date`); tests come from `livecodebench/code_generation_lite` via
   `extract_lcb_tests.py`.

## Disclosed data defect

A UTF-8 → CP437 encoding defect corrupted the Chinese multilingual framing and 9 problem statements
containing non-ASCII symbols. It is disclosed in the paper and documented in `ENCODING_DEFECT.md`,
with the audit script `encoding_robustness.py` and the check that the study's headline is unchanged
when the 9 affected problems are excluded.

## Provenance

Assembled from the authors' research repository. The upstream HumanEval harness
README that this repo was forked from is preserved as `README_humaneval_upstream.md`.
