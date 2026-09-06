"""
Per-ATTEMPT partial credit for EVERY problem — the full picture, including
partially-solved problems (some attempts pass, some fail), which the best-attempt
view hides.

For each problem it reports, over the 12 attempts (3 models x 4 framings):
  n_solved   how many attempts passed ALL tests (0 = never, 12 = all)
  best %     best single attempt's test-pass rate
  mean %     average test-pass rate across all 12 attempts  <- the new signal
  worst %    weakest attempt

Efficient: an attempt that the scorer marked PASS = 100% of tests (no re-run);
only FAILED attempts are executed for their partial test count. Still a longer run
than best-attempt (it runs every failed completion). RE-SCORE first so PASS is
based on the fixed scorer.

    python full_partial_credit.py --timeout 2
Writes full_partial_credit.json. Needs lcb_tests.jsonl (local).
"""

import argparse
import json
import multiprocessing
import os
import re
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from score_lcb import extract_solution      # noqa: E402
from test_breakdown import run_all_tests     # noqa: E402

TOPICS = [
    ("dynamic programming", r"\b(dynamic programming|dp\b|subsequence|number of ways|minimum cost|maximum (sum|score|value)|partitions?)\b"),
    ("graph", r"\b(graph|node|edge|adjacen|connected component|shortest path|cities|roads?|network)\b"),
    ("tree", r"\b(binary tree|tree|root|leaf|subtree|ancestor)\b"),
    ("string", r"\b(string|substring|palindrome|character|prefix|suffix|anagram|lexicograph)\b"),
    ("intervals/sorting", r"\b(interval|sort|sorted|merge|schedule|meeting|overlap)\b"),
    ("greedy/array", r"\b(array|subarray|greedy|adjacent|window|two pointers?)\b"),
    ("math/number", r"\b(prime|gcd|lcm|modulo|divisor|digits?|binary representation|factorial|arithmetic)\b"),
    ("bit manipulation", r"\b(bit|xor|bitwise|set bits)\b"),
    ("simulation/geometry", r"\b(simulat|grid|matrix|coordinate|move|direction|snake|robot|game)\b"),
]
def classify(t):
    t = t.lower()
    for n, p in TOPICS:
        if re.search(p, t):
            return n
    return "other"


def run(per, max_tests):
    L = lambda f: json.load(open(os.path.join(SCRIPT_DIR, f), encoding="utf-8"))
    scored = {(x["task_id"], x["level"], x["model"]): x["passed"] for x in L("lcb_scored.json")}
    probs = {json.loads(l)["task_id"]: json.loads(l)
             for l in open(os.path.join(SCRIPT_DIR, "lcb_problems.jsonl"), encoding="utf-8")}
    tf = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
    if not os.path.exists(tf):
        sys.exit("need lcb_tests.jsonl")
    tests = {json.loads(l)["task_id"]: json.loads(l) for l in open(tf, encoding="utf-8")}
    resp = L("lcb_v3_responses.json")
    resp_by_task = defaultdict(list)
    for r in resp:
        resp_by_task[r["task_id"]].append(r)

    ids = sorted(t for t in probs if t in tests and t in resp_by_task)
    n_run = sum(1 for r in resp if scored.get((r["task_id"], r["level"], r["model"])) is not True
                and r["task_id"] in tests)
    print(f"{len(ids)} problems | running {n_run} failed attempts for partial credit "
          f"(passed attempts = 100%, skipped)\n", flush=True)

    out = []
    done = 0
    for tid in ids:
        rec = tests[tid]; entry = rec["entry_point"]
        cases = (rec.get("public_tests", []) + rec.get("private_tests", []))[:max_tests]
        total = len(cases)
        rates = []
        for r in resp_by_task[tid]:
            if scored.get((tid, r["level"], r["model"])) is True:
                rates.append(100.0)
            else:
                code = extract_solution(r.get("completion"), entry)
                p, *_ = run_all_tests(code, entry, cases, per)
                rates.append(round(p / total * 100, 1) if total else 0.0)
                done += 1
                if done % 50 == 0:
                    print(f"  ...{done}/{n_run} failed attempts run", flush=True)
        pr = probs[tid]
        out.append({"task_id": tid, "difficulty": pr.get("difficulty"),
                    "topic": classify(pr.get("question_content", "") + " " + entry),
                    "n_attempts": len(rates),
                    "n_solved": sum(1 for x in rates if x == 100.0),
                    "best": max(rates) if rates else 0,
                    "mean": round(sum(rates) / len(rates), 1) if rates else 0,
                    "worst": min(rates) if rates else 0})

    json.dump(out, open(os.path.join(SCRIPT_DIR, "full_partial_credit.json"), "w"), indent=2)

    def med(xs): xs = sorted(xs); return xs[len(xs) // 2] if xs else 0
    print("\n" + "=" * 64)
    print("PARTIAL CREDIT across all problems (per-attempt test-pass rate):")
    fully = sum(1 for x in out if x["n_solved"] == x["n_attempts"])
    part = sum(1 for x in out if 0 < x["n_solved"] < x["n_attempts"])
    never = sum(1 for x in out if x["n_solved"] == 0)
    print(f"  fully solved (all attempts pass): {fully}")
    print(f"  partially solved (some pass)     : {part}")
    print(f"  never solved (0 attempts pass)   : {never}")
    print(f"  MEAN attempt test-pass rate: overall {round(sum(x['mean'] for x in out)/len(out),1)}%")
    bd = defaultdict(list)
    for x in out:
        bd[x["difficulty"]].append(x)
    print("  by difficulty:  (mean = avg over all 12 attempts; best = best attempt)")
    for d in ("easy", "medium", "hard"):
        g = bd[d]
        if g:
            print(f"    {d:6}: mean {round(sum(x['mean'] for x in g)/len(g),1):5}%  "
                  f"best {round(sum(x['best'] for x in g)/len(g),1):5}%  "
                  f"fully-solved {sum(1 for x in g if x['n_solved']==x['n_attempts'])}/{len(g)}  (n={len(g)})")
    print("-> full_partial_credit.json")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=2)
    ap.add_argument("--max-tests", type=int, default=60)
    a = ap.parse_args()
    run(a.timeout, a.max_tests)
