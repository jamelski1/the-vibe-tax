# Adding EASY LiveCodeBench problems (comparison completeness)

The v1 LCB run used **medium + hard** only (124 problems). The advisor asked for
**easy** problems too, so the difficulty axis is complete (easy → medium → hard)
and we can show the effect appearing/vanishing across the full difficulty range.

Run **locally** (LCB is HF-gated + firewalled in the cloud sandbox; needs your
`HF_TOKEN` and the three model keys). The pipeline already supports this — the
only trick is **not** re-querying the 124 completions we already have. We do that
by seeding the progress file from the committed responses, so `run_vibe_tax.py`
only queries the NEW easy prompts.

## Steps (from `data/vibe_tax_lcb/`)

```powershell
# 1. Regenerate the problem set with easy INCLUDED (medium/hard task_ids are
#    stable, so nothing downstream changes for them). Keep the same contamination
#    filter so easy problems are also post-cutoff.
python prepare_lcb.py --min-date 2024-08-01 --difficulties easy,medium,hard

# 2. Regenerate the prompts (deterministic mock wrappers; medium/hard prompts are
#    byte-identical to before, easy prompts are added).
python generate_lcb_prompts.py --backend mock

# 3. Seed progress from the committed responses so the 124 done stay done.
python seed_lcb_progress.py            # -> lcb_progress.json (1,488 keys)

# 4. Run ONLY the new easy prompts (resume skips everything already in progress).
$env:PROMPTS_FILE  = "lcb_v3_prompts.json"
$env:OUTPUT_FILE   = "lcb_v3_responses.json"     # grows to include easy
$env:PROGRESS_FILE = "lcb_progress.json"
$env:STATS_FILE    = "lcb_v3_run_stats.json"
# LCB models (match the original run):
$env:OPENAI_MODEL = "gpt-5.4"; $env:ANTHROPIC_MODEL = "claude-opus-4-6"; $env:CODESTRAL_MODEL = "codestral-latest"
python ..\vibe_tax_v2\run_vibe_tax.py

# 5. Regenerate the test cases (now includes easy) and score everything.
python extract_lcb_tests.py
python score_lcb.py

# 6. Re-run the pairwise McNemar over the full set (now with an `easy` slice).
python mcnemar_lcb.py --base agentic_terse
```

## What you get

- `lcb_scored_stats.json` — pass rates by condition/model/difficulty, now with
  **easy** alongside medium/hard.
- `lcb_mcnemar_stats.json` — terse-vs-{casual,detailed,multilingual} with an
  added `easy` slice.

## What to expect (hypothesis)

Easy problems are near the **ceiling** (like HumanEval), so the framing effect
should be **small/invisible** there — the same reason HumanEval hid it. That's
the point of including easy: it completes the *Goldilocks* story — effect is
suppressed at the ceiling (easy), **visible** in the headroom (medium), and
washed out at the floor (hard). If easy shows a large effect, that's a
surprise worth investigating.

## Commit hygiene

Commit the aggregates and the tracked data files: `lcb_scored_stats.json`,
`lcb_mcnemar_stats.json`, the regenerated `lcb_problems.jsonl`/`lcb_sample.json`,
`lcb_scored.json`, and the grown `lcb_v3_responses.json` (already tracked — it's
the model outputs, safe to commit). The purely local artifacts —
`lcb_progress.json` and `lcb_tests.jsonl` — are gitignored.
