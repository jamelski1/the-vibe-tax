"""
Partial-credit view: for one LCB problem, show how many of its N tests EACH of the
12 recorded attempts (3 models x 4 framings) passed. A problem is "solved" only if
an attempt passes ALL N tests, so a "Never" (0/12) can still hide attempts that
passed most tests. This reveals that.

    python test_breakdown.py lcb/3527
    python test_breakdown.py lcb/3527 --timeout 3 --max-tests 40

Reads lcb_v3_responses.json (recorded completions) + lcb_tests.jsonl (local; falls
back to the committed sample for lcb/3517 & lcb/3527). Each test runs in a
subprocess with a per-test timeout so infinite loops / TLE are counted, not hung.
No API calls.
"""

import argparse
import ast
import json
import multiprocessing
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from score_lcb import extract_solution, parse_input, parse_lit, eq, _IMPORTS  # noqa: E402


def _stream_worker(code, entry, cases, start, q):
    ns = {}
    try:
        exec(_IMPORTS + code, ns)
    except Exception:
        return
    sol = ns.get("Solution")
    for idx in range(start, len(cases)):
        t = cases[idx]
        try:
            args = parse_input(t["input"]); expected = parse_lit(t["output"])
            fn = (getattr(sol(), entry, None) if sol else None) or ns.get(entry)
            ok = bool(eq(fn(*args), expected))
        except Exception:
            ok = False
        q.append((idx, ok))


def run_all_tests(code, entry, cases, per):
    """Return (passed, failed, timeouts). One worker runs the tests in order,
    streaming results; if it stalls past `per` seconds on a test, kill it, count
    that test as a timeout, and restart at the next index (efficient: 1 spawn + 1
    per hang, not one per test)."""
    if not code or not ("class Solution" in code or f"def {entry}" in code):
        return (0, 0, 0, len(cases))          # last field = "no runnable code" tests
    try:
        ast.parse(code)
    except SyntaxError:
        return (0, 0, 0, len(cases))
    mgr = multiprocessing.Manager()
    results = {}
    start = 0
    while start < len(cases):
        q = mgr.list()
        proc = multiprocessing.Process(target=_stream_worker, args=(code, entry, cases, start, q))
        proc.start()
        last_len, last_t, stuck = 0, time.time(), False
        while proc.is_alive():
            if len(q) > last_len:
                last_len = len(q); last_t = time.time()
            elif time.time() - last_t > per:
                stuck = True; break
            time.sleep(0.03)
        for idx, ok in list(q):
            results[idx] = ok
        if stuck:
            proc.kill(); proc.join(1)
            done = max(results.keys()) if results else start - 1
            results[done + 1] = "TO"
            start = done + 2
        else:
            proc.join(); break
    passed = sum(1 for v in results.values() if v is True)
    to = sum(1 for v in results.values() if v == "TO")
    failed = len(cases) - passed - to
    return (passed, failed, to, 0)


def load_tests():
    full = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
    if os.path.exists(full):
        return {json.loads(l)["task_id"]: json.loads(l) for l in open(full, encoding="utf-8")}
    sample = os.path.join(SCRIPT_DIR, "lcb_tests_sample.json")
    if os.path.exists(sample):
        print("(using lcb_tests_sample.json — only lcb/3517 & lcb/3527)\n")
        return {t["task_id"]: t for t in json.load(open(sample, encoding="utf-8"))}
    sys.exit("no lcb_tests.jsonl or lcb_tests_sample.json")


def run(task_id, per, max_tests):
    tests = load_tests()
    if task_id not in tests:
        sys.exit(f"{task_id} not in tests file")
    rec = tests[task_id]; entry = rec["entry_point"]
    cases = (rec.get("public_tests", []) + rec.get("private_tests", []))[:max_tests]
    total = len(cases)

    resp = [x for x in json.load(open(os.path.join(SCRIPT_DIR, "lcb_v3_responses.json"), encoding="utf-8"))
            if x["task_id"] == task_id]
    print(f"{task_id}  ({entry})  |  {total} tests  |  {len(resp)} recorded attempts\n")
    print(f"  {'model':10} {'framing':22} {'passed':>7} {'failed':>7} {'timeout':>8}  solved?")
    rows = []
    for r in sorted(resp, key=lambda r: (r["model"], r["level"])):
        code = extract_solution(r.get("completion"), entry)
        p, f, to, _ = run_all_tests(code, entry, cases, per)
        solved = (p == total and total > 0)
        rows.append((r["model"], r["level"], p, f, to, solved))
        print(f"  {r['model']:10} {r['level']:22} {p:>7} {f:>7} {to:>8}  {'YES' if solved else 'no'}")

    best = max(rows, key=lambda x: x[2]) if rows else None
    n_solved = sum(1 for x in rows if x[5])
    print("\n" + "=" * 60)
    print(f"solved the WHOLE problem (all {total} tests): {n_solved}/{len(rows)} attempts")
    if best:
        print(f"best single attempt: {best[2]}/{total} tests  "
              f"({best[0]} / {best[1]})  — still a FAIL unless it's {total}/{total}")
    print("Reminder: LCB scoring is all-or-nothing per problem; partial passes score 0.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("--timeout", type=int, default=4, help="seconds per test (default 4)")
    ap.add_argument("--max-tests", type=int, default=60, help="cap tests checked (default 60)")
    a = ap.parse_args()
    run(a.task_id, a.timeout, a.max_tests)
