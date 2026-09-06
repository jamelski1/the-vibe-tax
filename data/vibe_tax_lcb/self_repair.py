"""
SELF-REPAIR on LiveCodeBench — does showing a model its own failing code + ONE
failing test let it fix the problem, and by how much, by round and problem type?

This is the positive-result companion to the framing null: phrasing the *request*
differently does nothing (RESULTS.md), but feeding back a concrete *failure* should
recover correctness. We measure that recovery curve.

Design (paired, within-problem):
  * Seed set = every attempt the main run FAILED, deduped to ONE framing per
    (problem x model) so each starting point is counted once. Framing is null, so
    which framing we seed from doesn't matter; default is agentic_terse.
  * Each round: run the current code, find the FIRST failing test, show the model
    {full problem, its current code, that test's input / expected / actual}, ask
    for a corrected solution. Extract + RE-SCORE on the full sampled test set
    (pass = ALL tests). If it passes, record the round it was fixed; else feed the
    new (still-failing) code into the next round.
  * We SHOW one failing test but SCORE on all tests, so a model can't "pass" by
    hardcoding the one shown case (the other tests catch that). Use
    --score-on-private to score only on private tests for an even stricter read.

Reuses score_lcb (extractor + per-test-timeout scorer) and run_vibe_tax
(make_clients/query) verbatim — same models, temperature, token cap.

Run LOCALLY (needs lcb_tests.jsonl + API keys), same as the rest of the pipeline:
    # optional: ONLY the failing set for one model, 3 rounds
    $env:OPENAI_API_KEY=... ; $env:ANTHROPIC_API_KEY=... ; $env:CODESTRAL_API_KEY=...
    python self_repair.py --rounds 3 --condition agentic_terse
    python self_repair.py --rounds 3 --models chatgpt,claude --max-seeds 40

Writes self_repair_trajectory.json (per-seed, per-round) + self_repair_stats.json.
Resumable via self_repair_progress.json (keyed task_id|model).
"""

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "data", "vibe_tax_v2"))

from score_lcb import (extract_solution, passes, sample_tests,      # noqa: E402
                       parse_input, parse_lit, eq, _IMPORTS)

def out_paths(tag):
    sfx = f"_{tag}" if tag else ""
    return (os.path.join(SCRIPT_DIR, f"self_repair_trajectory{sfx}.json"),
            os.path.join(SCRIPT_DIR, f"self_repair_stats{sfx}.json"),
            os.path.join(SCRIPT_DIR, f"self_repair_progress{sfx}.json"))

SYSTEM_PROMPT = ("You are a helpful Python programming assistant. "
                 "When you provide code, output plain Python without markdown fences.")

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
    t = (t or "").lower()
    for n, p in TOPICS:
        if re.search(p, t):
            return n
    return "other"


def _one_test_worker(code, entry, case, q):
    """Run ONE test; append (status, got_repr). status in pass/wrong/error."""
    ns = {}
    try:
        exec(_IMPORTS + code, ns)
    except Exception as e:
        q.append(("error", f"import/exec failed: {type(e).__name__}: {e}")); return
    sol = ns.get("Solution")
    try:
        args = parse_input(case["input"])
        expected = parse_lit(case["output"])
        fn = (getattr(sol(), entry, None) if sol else None) or ns.get(entry)
        if fn is None:
            q.append(("error", f"no callable named {entry}")); return
        got = fn(*args)
        q.append(("pass" if eq(got, expected) else "wrong", repr(got)))
    except Exception as e:
        q.append(("error", f"raised {type(e).__name__}: {e}"))


def first_failing(code, entry, cases, per):
    """Return (case, status, got_repr) for the first non-passing test, or None if
    all pass. Each test runs in its own subprocess with a per-test timeout."""
    if not code:
        return None
    for case in cases:
        mgr = multiprocessing.Manager(); q = mgr.list()
        p = multiprocessing.Process(target=_one_test_worker, args=(code, entry, case, q))
        p.start(); p.join(per)
        if p.is_alive():
            p.kill(); p.join(1)
            return (case, "timeout", f"no result in {per}s (infinite loop / TLE)")
        if not q:
            return (case, "error", "worker crashed")
        status, got = q[0]
        if status != "pass":
            return (case, status, got)
    return None


def repair_prompt(prob, entry, cls, code, case, status, got, no_feedback=False):
    head = ("Your Python solution to the problem below is INCORRECT — it fails a test. "
            "Find the bug and return a corrected solution.\n\n"
            "PROBLEM:\n" + (prob.get("question_content") or "").strip() + "\n\n"
            f"The solution must be a method named exactly `{entry}` on a class named `{cls}`.\n\n"
            "YOUR CURRENT SOLUTION:\n" + (code or "").strip() + "\n\n")
    tail = ("Return the COMPLETE corrected solution as plain Python (the full `class "
            f"{cls}` with `{entry}`). Fix the underlying logic — do not special-case "
            "this one test. Output only code, no explanation.")
    if no_feedback:
        # CONTROL: told it's wrong, but shown NO failing test. The gap between this
        # and the fed-back run is the feedback-specific recovery (vs a mere resample).
        return head + "It is incorrect on at least one input. Return a corrected solution.\n\n" + tail
    inp = case["input"] if isinstance(case["input"], str) else "\n".join(map(str, case["input"]))
    if status == "timeout":
        actual = "(your code did not finish in time — likely too slow / an infinite loop)"
    elif status == "error":
        actual = f"(your code {got})"
    else:
        actual = got
    return (head + "IT FAILS THIS TEST:\n"
            f"Input (one argument per line):\n{inp}\n"
            f"Expected return: {case['output']}\n"
            f"Your solution returned: {actual}\n\n" + tail)


def load_seeds(condition, models, difficulties, tests_by_id):
    scored = json.load(open(os.path.join(SCRIPT_DIR, "lcb_scored.json"), encoding="utf-8"))
    resp = {(r["task_id"], r["level"], r["model"]): r
            for r in json.load(open(os.path.join(SCRIPT_DIR, "lcb_v3_responses.json"), encoding="utf-8"))}
    probs = {json.loads(l)["task_id"]: json.loads(l)
             for l in open(os.path.join(SCRIPT_DIR, "lcb_problems.jsonl"), encoding="utf-8")}
    seeds = []
    for x in scored:
        if x["passed"]:
            continue
        if condition and x["level"] != condition:
            continue
        if models and x["model"] not in models:
            continue
        if difficulties and (x.get("difficulty") or "") not in difficulties:
            continue
        tid = x["task_id"]
        if tid not in tests_by_id or tid not in probs:
            continue
        r = resp.get((tid, x["level"], x["model"]))
        if not r:
            continue
        seeds.append({"task_id": tid, "model": x["model"], "level": x["level"],
                      "entry_point": r["entry_point"], "difficulty": x.get("difficulty"),
                      "prob": probs[tid], "orig_completion": r.get("completion")})
    return seeds


def run(a):
    from run_vibe_tax import make_clients, query          # reuse exact API plumbing
    tag = a.tag or ("nofb" if a.no_feedback else "")
    TRAJ, STATS, PROGRESS = out_paths(tag)
    clients = make_clients()
    if not clients:
        sys.exit("No API keys configured.")
    models = {m.strip() for m in a.models.split(",") if m.strip()} if a.models else None
    diffs = {d.strip() for d in a.difficulties.split(",") if d.strip()} if a.difficulties else None

    tests_by_id = {r["task_id"]: r for r in
                   (json.loads(l) for l in open(os.path.join(SCRIPT_DIR, "lcb_tests.jsonl"), encoding="utf-8"))}
    seeds = load_seeds(a.condition, models, diffs, tests_by_id)
    seeds = [s for s in seeds if s["model"] in clients]           # only models we can call
    if a.max_seeds:
        seeds = seeds[:a.max_seeds]
    print(f"{len(seeds)} failed (problem x model) seeds to repair | rounds={a.rounds} "
          f"| condition={a.condition} | models={sorted({s['model'] for s in seeds})} "
          f"| mode={'NO-FEEDBACK control' if a.no_feedback else 'feedback'} | out tag='{tag}'", flush=True)

    progress = json.load(open(PROGRESS, encoding="utf-8")) if os.path.exists(PROGRESS) else {}
    traj = []
    for i, s in enumerate(seeds):
        key = f"{s['task_id']}|{s['model']}"
        if key in progress:
            traj.append(progress[key]); continue
        rec = tests_by_id[s["task_id"]]
        cases = sample_tests(rec, a.max_tests)
        score_cases = ([c for c in rec.get("private_tests", [])][:a.max_tests]
                       if a.score_on_private else cases)
        entry = s["entry_point"]; cls = s["prob"].get("class_name") or "Solution"
        code = extract_solution(s["orig_completion"], entry)

        rounds = []
        fixed_at = None
        for rnd in range(1, a.rounds + 1):
            fail = first_failing(code, entry, cases, a.timeout)
            if fail is None:            # already passes the shown-set; verify on score set
                if passes(code, entry, score_cases):
                    fixed_at = fixed_at or rnd - 1
                    break
                fail = first_failing(code, entry, score_cases, a.timeout) or (cases[0], "wrong", "?")
            case, status, got = fail
            prompt = repair_prompt(s["prob"], entry, cls, code, case, status, got,
                                   no_feedback=a.no_feedback)
            api_type = clients[s["model"]][0]
            try:
                t0 = time.time()
                completion = query(api_type, clients[s["model"]][1], prompt)
                dt = round(time.time() - t0, 1)
                err = None
            except Exception as e:
                completion, dt, err = None, None, f"{type(e).__name__}: {e}"
            new_code = extract_solution(completion, entry) if completion else None
            now_ok = bool(new_code) and passes(new_code, entry, score_cases)
            rounds.append({"round": rnd, "shown_test_status": status,
                           "passed_after": now_ok, "api_error": err, "secs": dt})
            print(f"  [{i+1}/{len(seeds)}] {s['task_id']:9s} {s['model']:9s} "
                  f"r{rnd} shown={status:7s} -> {'FIXED' if now_ok else 'still failing'}"
                  + (f" ({err})" if err else ""), flush=True)
            if now_ok:
                fixed_at = rnd
                code = new_code
                break
            if new_code:
                code = new_code                 # carry the revised code into next round

        entry_rec = {"task_id": s["task_id"], "model": s["model"], "level": s["level"],
                     "difficulty": s["difficulty"],
                     "topic": classify((s["prob"].get("question_content") or "") + " " + entry),
                     "n_rounds": len(rounds), "fixed_at_round": fixed_at,
                     "fixed": fixed_at is not None, "rounds": rounds}
        traj.append(entry_rec)
        progress[key] = entry_rec
        if (i + 1) % 5 == 0 or i + 1 == len(seeds):
            json.dump(progress, open(PROGRESS, "w"), indent=2)
            json.dump(traj, open(TRAJ, "w"), indent=2)

    json.dump(traj, open(TRAJ, "w"), indent=2)
    summarize(traj, a.rounds)


def summarize(traj, rounds):
    n = len(traj)
    if not n:
        print("no seeds."); return

    def curve(items):
        m = len(items)
        cum = []
        for r in range(1, rounds + 1):
            k = sum(1 for x in items if x["fixed"] and x["fixed_at_round"] <= r)
            cum.append((r, k, round(k / m * 100, 1) if m else 0.0))
        return m, cum

    m, cum = curve(traj)
    stats = {"n_seeds": n,
             "cumulative_fixed_by_round": [{"round": r, "fixed": k, "pct": p} for r, k, p in cum],
             "by_difficulty": {}, "by_topic": {}, "by_model": {}}
    print("=" * 64)
    print(f"SELF-REPAIR: {n} failed (problem x model) seeds, up to {rounds} rounds")
    print(f"  overall recovered: {cum[-1][1]}/{n} = {cum[-1][2]}%")
    print("  cumulative recovery by round:")
    for r, k, p in cum:
        print(f"    after round {r}: {k:4d}/{n}  {p}%")

    for label, key in (("difficulty", "difficulty"), ("topic", "topic"), ("model", "model")):
        print(f"  by {label}:")
        groups = defaultdict(list)
        for x in traj:
            groups[x.get(key)].append(x)
        bucket = stats["by_difficulty"] if key == "difficulty" else \
                 stats["by_topic"] if key == "topic" else stats["by_model"]
        for g, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
            gm, gcum = curve(items)
            bucket[str(g)] = {"n": gm, "final_pct": gcum[-1][2],
                              "by_round": [{"round": r, "pct": p} for r, k, p in gcum]}
            print(f"    {str(g):20s} n={gm:4d}  final {gcum[-1][2]:5}%  "
                  f"(r1 {gcum[0][2]}%)")
    json.dump(stats, open(STATS, "w"), indent=2)
    print("-> self_repair_trajectory.json + self_repair_stats.json")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3, help="max repair rounds")
    ap.add_argument("--condition", default="agentic_terse",
                    help="seed from this framing only (framing is null); '' = all framings")
    ap.add_argument("--models", default="", help="comma list e.g. chatgpt,claude (default: all with keys)")
    ap.add_argument("--difficulties", default="", help="comma list e.g. medium,hard (default: all)")
    ap.add_argument("--max-seeds", type=int, default=0, help="cap number of seeds (0 = all)")
    ap.add_argument("--max-tests", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=6, help="per-test seconds")
    ap.add_argument("--score-on-private", action="store_true",
                    help="score repaired code on PRIVATE tests only (stricter; shown test stays from full set)")
    ap.add_argument("--no-feedback", action="store_true",
                    help="CONTROL: tell the model it's wrong but show NO failing test "
                         "(writes *_nofb files). recovery_feedback - recovery_nofb = feedback effect")
    ap.add_argument("--tag", default="", help="output filename suffix (default: '' or 'nofb')")
    run(ap.parse_args())
