"""
Seed a run_vibe_tax progress file from an existing LCB responses file.

Purpose: when we ADD easy problems to the LCB set (re-running prepare_lcb.py with
--difficulties easy,medium,hard), we don't want to re-query the 124 medium/hard
completions we already have. run_vibe_tax.py resumes by the key
`task_id|condition|model`; this rebuilds that progress dict from the committed
lcb_v3_responses.json, so the next run only queries the NEW (easy) prompts.

Usage (from data/vibe_tax_lcb):
    python seed_lcb_progress.py            # lcb_v3_responses.json -> lcb_progress.json
    # then set PROGRESS_FILE to lcb_progress.json for run_vibe_tax.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES = os.getenv("SEED_FROM", os.path.join(SCRIPT_DIR, "lcb_v3_responses.json"))
PROGRESS = os.getenv("SEED_TO", os.path.join(SCRIPT_DIR, "lcb_progress.json"))


def run():
    if not os.path.exists(RESPONSES):
        sys.exit(f"no responses file at {RESPONSES}")
    responses = json.load(open(RESPONSES, encoding="utf-8"))
    progress, skipped = {}, 0
    for r in responses:
        # only seed successful completions; failed/None re-run
        if r.get("completion") is None:
            skipped += 1
            continue
        key = f"{r['task_id']}|{r['level']}|{r['model']}"
        progress[key] = r
    json.dump(progress, open(PROGRESS, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"seeded {len(progress)} completed keys -> {PROGRESS}"
          + (f"  ({skipped} null completions left to re-run)" if skipped else ""))
    print("run_vibe_tax.py with PROGRESS_FILE pointed here will skip these and "
          "only query the new (easy) prompts.")


if __name__ == "__main__":
    run()
