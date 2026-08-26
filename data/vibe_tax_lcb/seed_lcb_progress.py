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
# SEED_FROM may be a comma-separated list of response files; they are UNIONED by
# task_id|level|model. Use this to merge the current (2-model) responses with a
# git-history 3-model file to restore Codestral, e.g.:
#   git show ef8f741:data/vibe_tax_lcb/lcb_v3_responses.json > lcb_v3_responses_3model.json
#   $env:SEED_FROM = "lcb_v3_responses.json,lcb_v3_responses_3model.json"
#   python seed_lcb_progress.py
RESPONSES = os.getenv("SEED_FROM", os.path.join(SCRIPT_DIR, "lcb_v3_responses.json"))
PROGRESS = os.getenv("SEED_TO", os.path.join(SCRIPT_DIR, "lcb_progress.json"))


def run():
    files = [f.strip() for f in RESPONSES.split(",") if f.strip()]
    progress, skipped, per_file = {}, 0, {}
    for path in files:
        if not os.path.exists(path):
            sys.exit(f"no responses file at {path}")
        responses = json.load(open(path, encoding="utf-8"))
        n0 = len(progress)
        for r in responses:
            if r.get("completion") is None:      # only seed successes; failures re-run
                skipped += 1
                continue
            progress[f"{r['task_id']}|{r['level']}|{r['model']}"] = r
        per_file[path] = len(progress) - n0
    json.dump(progress, open(PROGRESS, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if len(files) > 1:
        for p, n in per_file.items():
            print(f"  +{n:5d} new keys from {os.path.basename(p)}")
    print(f"seeded {len(progress)} completed keys -> {PROGRESS}"
          + (f"  ({skipped} null completions left to re-run)" if skipped else ""))
    print("run_vibe_tax.py with PROGRESS_FILE pointed here will skip these and "
          "only query the still-missing prompts.")


if __name__ == "__main__":
    run()
