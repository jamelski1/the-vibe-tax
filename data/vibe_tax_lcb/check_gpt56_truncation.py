"""
Diagnose whether the GPT-5.6 failures are a max_tokens TRUNCATION artifact rather
than real capability limits. Reasoning models spend output budget on hidden
reasoning; with max_completion_tokens=2048 the visible answer can be truncated or
empty on hard problems -> no code -> scored FAIL.

Reads the LOCAL run (responses + scored) and reports, among the FAILED cells, how
many produced empty / no-code / suspiciously-short completions.

    python check_gpt56_truncation.py
"""
import json
import os

SD = os.path.dirname(os.path.abspath(__file__))
RESP = os.path.join(SD, "lcb_v3_gpt56_responses.json")
SCORED = os.path.join(SD, "lcb_v3_gpt56_scored.json")

if not os.path.exists(RESP):
    raise SystemExit(f"need {RESP} locally (the 5.6 responses from your run)")

resp = {(x["task_id"], x["level"]): x for x in json.load(open(RESP, encoding="utf-8"))}
scored = json.load(open(SCORED, encoding="utf-8"))
fails = [(x["task_id"], x["level"], x["difficulty"]) for x in scored if not x["passed"]]
passes = [(x["task_id"], x["level"]) for x in scored if x["passed"]]


def has_code(c):
    return bool(c) and ("class Solution" in c or "def " in c)

empty = nocode = short = ok_code = 0
by_diff = {}
for tid, lvl, diff in fails:
    c = resp.get((tid, lvl), {}).get("completion") or ""
    d = by_diff.setdefault(diff, {"n": 0, "empty": 0, "nocode": 0})
    d["n"] += 1
    if not c.strip():
        empty += 1; d["empty"] += 1
    elif not has_code(c):
        nocode += 1; d["nocode"] += 1
    else:
        ok_code += 1
        if len(c) < 200:
            short += 1

pass_lens = [len(resp.get(k, {}).get("completion") or "") for k in passes]
fail_lens = [len(resp.get((t, l), {}).get("completion") or "") for t, l, _ in fails]
avg = lambda xs: round(sum(xs) / len(xs)) if xs else 0

print(f"FAILED cells: {len(fails)}")
print(f"  empty completion (no text)        : {empty}")
print(f"  has text but NO code              : {nocode}")
print(f"  has code (genuine wrong answer?)  : {ok_code}   (of which <200 chars: {short})")
print(f"\n  => empty+nocode = {empty + nocode} of {len(fails)} failures look like TRUNCATION,")
print(f"     not wrong answers. If this is a big fraction, re-run 5.6 with a higher")
print(f"     MAX_TOKENS (e.g. 16000) — the failures are a token-budget artifact.\n")
print("By difficulty (failures):")
for d, v in sorted(by_diff.items(), key=lambda kv: str(kv[0])):
    print(f"  {str(d):6}: {v['n']:3d} failed | empty {v['empty']:3d} | no-code {v['nocode']:3d}")
print(f"\nAvg completion length: passes {avg(pass_lens)} chars vs failures {avg(fail_lens)} chars")
print("(if failures are much SHORTER on average, that's the truncation signature)")
