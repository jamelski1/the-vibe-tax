"""
For every "never solved" LCB problem (0/12 attempts passed the whole problem),
compute the BEST single attempt's test-pass rate — i.e., how CLOSE did the models
get? Distinguishes near-misses (e.g. 32/37) from genuinely-not-close (0/37).

    python never_solved_breakdown.py                 # all never-solved problems
    python never_solved_breakdown.py --timeout 2     # faster on TLE-heavy problems

Reads lcb_scored.json (to find never-solved), lcb_v3_responses.json (completions),
lcb_tests.jsonl (local; gitignored) and lcb_problems.jsonl (difficulty/topic).
Writes never_solved_breakdown.json (a committable aggregate). No API calls.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import multiprocessing
from score_lcb import extract_solution                     # noqa: E402
from test_breakdown import run_all_tests                    # noqa: E402

# keyword topics (same as analyze_lcb_capability.py) for labelling
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
    for name, pat in TOPICS:
        if re.search(pat, t):
            return name
    return "other"


def run(per, max_tests):
    L = lambda f: json.load(open(os.path.join(SCRIPT_DIR, f), encoding="utf-8"))
    scored = L("lcb_scored.json")
    probs = {json.loads(l)["task_id"]: json.loads(l)
             for l in open(os.path.join(SCRIPT_DIR, "lcb_problems.jsonl"), encoding="utf-8")}
    tf = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
    if not os.path.exists(tf):
        sys.exit("need lcb_tests.jsonl (run extract_lcb_tests.py first)")
    tests = {json.loads(l)["task_id"]: json.loads(l) for l in open(tf, encoding="utf-8")}
    resp_by_task = defaultdict(list)
    for r in L("lcb_v3_responses.json"):
        resp_by_task[r["task_id"]].append(r)

    # never-solved = every recorded attempt failed the whole problem
    passc = defaultdict(lambda: [0, 0])
    for x in scored:
        passc[x["task_id"]][0] += x["passed"]; passc[x["task_id"]][1] += 1
    never = [t for t, (p, n) in passc.items() if p == 0 and n > 0]
    print(f"{len(never)} never-solved problems. Best-attempt test-pass rate:\n")
    print(f"  {'task_id':10} {'diff':6} {'topic':20} {'best':>7} {'rate':>6}  best (model/framing)")

    out = []
    for tid in sorted(never, key=lambda t: probs.get(t, {}).get("difficulty", "")):
        rec = tests.get(tid)
        if not rec:
            continue
        cases = (rec.get("public_tests", []) + rec.get("private_tests", []))[:max_tests]
        total = len(cases)
        entry = rec["entry_point"]
        best = (-1, None, None)   # (passed, model, framing)
        for r in resp_by_task.get(tid, []):
            code = extract_solution(r.get("completion"), entry)
            p, f, to, _ = run_all_tests(code, entry, cases, per)
            if p > best[0]:
                best = (p, r["model"], r["level"])
        p = probs.get(tid, {})
        topic = classify(p.get("question_content", "") + " " + entry)
        rate = round(best[0] / total * 100, 1) if total else 0
        out.append({"task_id": tid, "difficulty": p.get("difficulty"), "topic": topic,
                    "entry_point": entry, "total_tests": total,
                    "best_passed": best[0], "best_rate": rate,
                    "best_model": best[1], "best_framing": best[2]})
        print(f"  {tid:10} {str(p.get('difficulty')):6} {topic:20} "
              f"{best[0]:>3}/{total:<3} {rate:>5}%  {best[1]}/{best[2]}")

    out.sort(key=lambda x: -x["best_rate"])
    json.dump(out, open(os.path.join(SCRIPT_DIR, "never_solved_breakdown.json"), "w"), indent=2)
    near = [x for x in out if x["best_rate"] >= 90]
    notclose = [x for x in out if x["best_rate"] < 20]
    print(f"\n{'='*60}")
    print(f"near-misses (best >= 90% of tests): {len(near)}  {[x['task_id'] for x in near]}")
    print(f"not close  (best < 20% of tests) : {len(notclose)}  {[x['task_id'] for x in notclose]}")
    print(f"median best-attempt rate: {sorted(x['best_rate'] for x in out)[len(out)//2] if out else 0}%")
    print("-> never_solved_breakdown.json")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3, help="seconds per test")
    ap.add_argument("--max-tests", type=int, default=60)
    a = ap.parse_args()
    run(a.timeout, a.max_tests)
