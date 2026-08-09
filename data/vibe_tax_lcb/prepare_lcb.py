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

Reads LCB's release JSONL files (test.jsonl .. test6.jsonl) DIRECTLY via
huggingface_hub — no `datasets` loading script, no Arrow cache, no streaming.
This sidesteps the Windows `datasets` failures (WinError 32 on the Arrow rename;
streaming hang). Files are cached, so a prior download is reused.

Setup:
    pip install huggingface_hub
    $env:HF_TOKEN = "hf_xxx"          # and accept the dataset terms on HF once
    # dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite

Usage:
    python prepare_lcb.py --min-date 2024-08-01 --difficulties medium,hard --limit 300
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


# LCB stores problems as plain JSONL, one file per release version, one problem
# per line. We read these DIRECTLY (via hf_hub_download) — no `datasets` loading
# script, no Arrow cache, no streaming. Dodges every Windows/datasets failure.
DATA_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl",
              "test4.jsonl", "test5.jsonl", "test6.jsonl"]


def iter_lcb_direct(files):
    """Yield raw problem dicts straight from the release JSONL files."""
    from huggingface_hub import hf_hub_download
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    for fn in files:
        print(f"  fetching {fn} (cached if already downloaded) ...", flush=True)
        try:
            path = hf_hub_download("livecodebench/code_generation_lite", fn,
                                   repo_type="dataset", token=token)
        except Exception as e:
            print(f"    skip {fn}: {type(e).__name__}: {str(e)[:80]}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def run(version, min_date, difficulties, limit, files):
    import itertools
    file_list = [f.strip() for f in files.split(",")] if files else DATA_FILES
    print(f"reading LCB release files: {file_list}")
    rows = iter_lcb_direct(file_list)
    it = iter(rows)
    try:
        first_row = next(it)
    except StopIteration:
        raise SystemExit("No problems parsed — check HF_TOKEN + dataset access.")
    print("RAW FIELD NAMES (report these back if anything looks off):")
    print("  " + ", ".join(sorted(first_row.keys())))
    rows = itertools.chain([first_row], it)

    diffs = {d.strip().lower() for d in difficulties.split(",")} if difficulties else None
    seen_ids = set()
    out = []
    skipped = {"no_starter": 0, "difficulty": 0, "date": 0, "no_method": 0}
    skipped["dup"] = 0
    for r in rows:
        rid = first(r, F_ID, None)
        if rid is not None and rid in seen_ids:        # releases overlap
            skipped["dup"] += 1
            continue
        if rid is not None:
            seen_ids.add(rid)
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
    ap.add_argument("--version", default="", help="(unused; kept for compatibility)")
    ap.add_argument("--min-date", default="", help="keep contest_date >= YYYY-MM-DD (contamination filter)")
    ap.add_argument("--difficulties", default="medium,hard", help="comma list, or '' for all")
    ap.add_argument("--limit", type=int, default=0, help="max problems (0 = all)")
    ap.add_argument("--files", default="", help="comma list of release JSONLs (default: all test*.jsonl)")
    a = ap.parse_args()
    run(a.version, a.min_date, a.difficulties, a.limit, a.files)
