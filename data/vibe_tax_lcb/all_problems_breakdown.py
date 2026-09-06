"""
Best-attempt test-pass rate for EVERY LCB problem — a continuous view of the
frontier instead of binary solved/never. For each problem, of the 12 recorded
attempts (3 models x 4 framings), what fraction of the problem's tests did the
BEST attempt pass?

Efficient: a problem that ANY attempt fully solved is 100% by definition (read
from lcb_scored.json — so RE-SCORE first with the fixed scorer); only the
unsolved problems are actually re-run for partial credit.

    python all_problems_breakdown.py --timeout 2

Writes all_problems_breakdown.json (committable aggregate) + prints a distribution.
Needs lcb_tests.jsonl (local, gitignored).
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
from score_lcb import extract_solution          # noqa: E402
from test_breakdown import run_all_tests         # noqa: E402

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
    scored = L("lcb_scored.json")
    probs = {json.loads(l)["task_id"]: json.loads(l)
             for l in open(os.path.join(SCRIPT_DIR, "lcb_problems.jsonl"), encoding="utf-8")}
    tf = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
    if not os.path.exists(tf):
        sys.exit("need lcb_tests.jsonl")
    tests = {json.loads(l)["task_id"]: json.loads(l) for l in open(tf, encoding="utf-8")}
    resp_by_task = defaultdict(list)
    for r in L("lcb_v3_responses.json"):
        resp_by_task[r["task_id"]].append(r)

    solved = set(); attempts = defaultdict(int)
    for x in scored:
        attempts[x["task_id"]] += 1
        if x["passed"]:
            solved.add(x["task_id"])

    all_ids = sorted(t for t in probs if t in tests and t in resp_by_task)
    print(f"{len(all_ids)} problems | {len(solved)} solved (best=100%), "
          f"computing partial credit for {len(all_ids)-len(solved)} unsolved...\n")

    out = []
    for i, tid in enumerate(all_ids):
        rec = tests[tid]; entry = rec["entry_point"]
        cases = (rec.get("public_tests", []) + rec.get("private_tests", []))[:max_tests]
        total = len(cases)
        if tid in solved:
            best = total; who = "(solved)"
        else:
            best = -1; who = None
            for r in resp_by_task[tid]:
                code = extract_solution(r.get("completion"), entry)
                p, *_ = run_all_tests(code, entry, cases, per)
                if p > best:
                    best, who = p, f"{r['model']}/{r['level']}"
        rate = round(best / total * 100, 1) if total else 0
        pr = probs[tid]
        out.append({"task_id": tid, "difficulty": pr.get("difficulty"),
                    "topic": classify(pr.get("question_content", "") + " " + entry),
                    "total_tests": total, "best_passed": best, "best_rate": rate,
                    "best_attempt": who, "fully_solved": tid in solved})
        if not (tid in solved):
            print(f"  {tid:10} {str(pr.get('difficulty')):6} best {best}/{total} = {rate}%  ({who})")

    json.dump(out, open(os.path.join(SCRIPT_DIR, "all_problems_breakdown.json"), "w"), indent=2)

    def med(xs): xs = sorted(xs); return xs[len(xs)//2] if xs else 0
    rates = [x["best_rate"] for x in out]
    print("\n" + "=" * 60)
    print(f"BEST-ATTEMPT TEST-PASS RATE across {len(out)} problems:")
    print(f"  fully solved (100%): {sum(1 for x in out if x['fully_solved'])} "
          f"({round(sum(1 for x in out if x['fully_solved'])/len(out)*100)}%)")
    print(f"  median best-attempt rate: {med(rates)}%   mean: {round(sum(rates)/len(rates),1)}%")
    print("  distribution (best-attempt rate buckets):")
    buckets = defaultdict(int)
    for r in rates:
        buckets[min(int(r // 10) * 10, 90)] += 1
    for b in range(0, 100, 10):
        if buckets[b]:
            print(f"    {b:3d}-{b+9}% : {buckets[b]:3d}  " + "#" * buckets[b])
    print("  by difficulty (median best-attempt rate):")
    bd = defaultdict(list)
    for x in out:
        bd[x["difficulty"]].append(x["best_rate"])
    for d in ("easy", "medium", "hard"):
        if bd[d]:
            print(f"    {d:6}: median {med(bd[d])}%  mean {round(sum(bd[d])/len(bd[d]),1)}%  (n={len(bd[d])})")
    print("-> all_problems_breakdown.json")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3)
    ap.add_argument("--max-tests", type=int, default=60)
    a = ap.parse_args()
    run(a.timeout, a.max_tests)
