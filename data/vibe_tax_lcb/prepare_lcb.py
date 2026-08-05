"""
Prepare a LiveCodeBench (LCB) problem subset for the Vibe Tax port.  RUN LOCALLY
(LCB is HF-gated + firewalled in the cloud sandbox).

Loads LCB code-generation problems, filters to FUNCTIONAL problems (those with
starter_code — so our method-name-preservation + call-based scoring apply),
optionally by difficulty and contest date (contamination-free), parses the
method name/params from the starter code, and writes a normalized file our
generator can consume.

It is deliberately DEFENSIVE and SELF-REPORTING: LCB's exact field names and
version tags vary by release, so it prints the schema it actually sees and skips
fields it can't find rather than crashing. Push `lcb_sample.json` back so the
generator/scorer adapters can be finalized against the real schema.

Setup:
    pip install "datasets>=2.0" huggingface_hub
    export HF_TOKEN=hf_xxx            # and accept the dataset terms on HF once
    # dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite

Usage:
    python prepare_lcb.py --version release_v5 --min-date 2024-08-01 \
        --difficulties medium,hard --limit 300
"""

import argparse
import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")
SAMPLE = os.path.join(SCRIPT_DIR, "lcb_sample.json")

# LCB field names vary across releases; try these in order.
F_STARTER = ["starter_code", "starter", "code_stub"]
F_CONTENT = ["question_content", "problem_statement", "content", "question"]
F_DIFF = ["difficulty", "level"]
F_DATE = ["contest_date", "date", "release_date"]
F_ID = ["question_id", "task_id", "id", "question_title", "title"]


def first(d, names, default=None):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return default


def parse_starter(starter):
    """From a `class Solution:\\n def name(self, a, b):` stub, get (class, method, params)."""
    cls = None
    mcls = re.search(r"class\s+(\w+)", starter or "")
    if mcls:
        cls = mcls.group(1)
    mm = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", starter or "")
    if not mm:
        return cls, None, ""
    method = mm.group(1)
    params = [p.split(":")[0].split("=")[0].strip()
              for p in mm.group(2).split(",")]
    params = [p for p in params if p and p != "self"]
    return cls, method, ", ".join(params)


def load_lcb(version):
    """Return a list of raw problem dicts using LCB's official dataset."""
    from datasets import load_dataset
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs = dict(split="test", token=token, trust_remote_code=True)
    # newer releases require a version_tag kwarg; try with, then without.
    for attempt in ({"version_tag": version}, {}):
        try:
            ds = load_dataset("livecodebench/code_generation_lite", **attempt, **kwargs)
            return list(ds)
        except Exception as e:
            last = e
    raise SystemExit(f"Could not load LCB (accepted terms? token set?): {last}")


def run(version, min_date, difficulties, limit):
    rows = load_lcb(version)
    print(f"loaded {len(rows)} raw LCB problems")
    if rows:
        print("RAW FIELD NAMES (report these back if anything looks off):")
        print("  " + ", ".join(sorted(rows[0].keys())))

    diffs = {d.strip().lower() for d in difficulties.split(",")} if difficulties else None
    out = []
    skipped = {"no_starter": 0, "difficulty": 0, "date": 0, "no_method": 0}
    for r in rows:
        starter = first(r, F_STARTER, "")
        if not starter or "def " not in starter:      # functional-only
            skipped["no_starter"] += 1
            continue
        diff = str(first(r, F_DIFF, "")).lower()
        if diffs and diff not in diffs:
            skipped["difficulty"] += 1
            continue
        date = str(first(r, F_DATE, ""))
        if min_date and date and date[:10] < min_date:
            skipped["date"] += 1
            continue
        cls, method, params = parse_starter(starter)
        if not method:
            skipped["no_method"] += 1
            continue
        out.append({
            "task_id": f"lcb/{first(r, F_ID, len(out))}",
            "entry_point": method,
            "class_name": cls,
            "params": params,
            "starter_code": starter,
            "question_content": first(r, F_CONTENT, ""),
            "difficulty": diff,
            "contest_date": date[:10] if date else None,
        })
        if limit and len(out) >= limit:
            break

    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    json.dump(out[:10], open(SAMPLE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    from collections import Counter
    print("=" * 60)
    print(f"kept {len(out)} functional problems -> {OUT}")
    print(f"  by difficulty: {dict(Counter(o['difficulty'] for o in out))}")
    print(f"  skipped: {skipped}")
    print(f"sample (first 10) -> {SAMPLE}  <-- push this so the adapters can be built")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="release_v5", help="LCB version_tag")
    ap.add_argument("--min-date", default="", help="keep contest_date >= YYYY-MM-DD (contamination filter)")
    ap.add_argument("--difficulties", default="medium,hard", help="comma list, or '' for all")
    ap.add_argument("--limit", type=int, default=0, help="max problems (0 = all)")
    a = ap.parse_args()
    run(a.version, a.min_date, a.difficulties, a.limit)
