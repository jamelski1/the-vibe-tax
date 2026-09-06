"""
Per-assert partial credit for HumanEval (HE4 run: 3 models x 4 framings).

HumanEval grades with a check(candidate) function of ~7 asserts; normally one
failed assert = FAIL. Here we run EACH assert independently and count how many
pass, giving the same partial-credit / continuum view we built for LiveCodeBench,
so all three benchmarks are measured the same way. No API calls.

    python he_partial_credit.py
Writes he_partial_credit.json.
"""
import ast
import json
import multiprocessing
import os
import sys
from collections import defaultdict

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
from score_vibe_tax_plus import candidate_sources        # robust code extraction

HE = os.path.join(SD, "..", "HumanEval.jsonl", "human-eval-v2-20210705.jsonl")
RESP = os.path.join(SD, "he4_responses.json")
OUT = os.path.join(SD, "he_partial_credit.json")
TIMEOUT = 10


def _worker(prompt, completion, entry, test_src, q):
    fn = None
    for src in candidate_sources(prompt, completion, entry):
        ns = {}
        try:
            exec(src, ns)
        except Exception:
            continue
        if callable(ns.get(entry)):
            fn = ns[entry]; local = ns; break
    if fn is None:
        q.append((0, -1)); return                # -1 total = "no runnable code"
    try:
        check = next(n for n in ast.parse(test_src).body
                     if isinstance(n, ast.FunctionDef) and n.name == "check")
    except StopIteration:
        q.append((0, -1)); return
    local["candidate"] = fn
    passed = total = 0
    for st in check.body:
        mod = ast.Module(body=[st], type_ignores=[])
        try:
            code = compile(ast.fix_missing_locations(mod), "<c>", "exec")
        except Exception:
            continue
        if isinstance(st, ast.Assert):
            total += 1
            try:
                exec(code, local); passed += 1
            except Exception:
                pass
        else:                                     # setup lines (vars, helpers)
            try:
                exec(code, local)
            except Exception:
                pass
    q.append((passed, total))


def count(prompt, completion, entry, test_src):
    mgr = multiprocessing.Manager(); q = mgr.list()
    p = multiprocessing.Process(target=_worker, args=(prompt, completion, entry, test_src, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.kill(); p.join(2); return (0, None)     # None total = timed out
    return q[0] if q else (0, None)


def run():
    probs = {}
    for l in open(HE, encoding="utf-8"):
        d = json.loads(l); probs[d["task_id"]] = d
    resp = json.load(open(RESP, encoding="utf-8"))
    by_task = defaultdict(list)
    for r in resp:
        by_task[r["task_id"]].append(r)

    out = []
    for i, (tid, attempts) in enumerate(sorted(by_task.items())):
        pr = probs[tid]; entry = pr["entry_point"]
        n_asserts = sum(isinstance(s, ast.Assert) for s in
                        next(n for n in ast.parse(pr["test"]).body
                             if isinstance(n, ast.FunctionDef) and n.name == "check").body)
        rates = []
        for r in attempts:
            passed, total = count(pr["prompt"], r.get("completion"), entry, pr["test"])
            denom = total if (total and total > 0) else n_asserts
            rates.append(round(passed / denom * 100, 1) if denom else 0.0)
        out.append({"task_id": tid, "n_asserts": n_asserts, "n_attempts": len(rates),
                    "n_solved": sum(1 for x in rates if x == 100.0),
                    "best": max(rates), "mean": round(sum(rates)/len(rates), 1),
                    "worst": min(rates)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(by_task)} problems", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    fully = sum(1 for x in out if x["n_solved"] == x["n_attempts"])
    part = sum(1 for x in out if 0 < x["n_solved"] < x["n_attempts"])
    never = sum(1 for x in out if x["n_solved"] == 0)
    print("=" * 60)
    print(f"HumanEval (HE4, {len(out)} problems x {out[0]['n_attempts']} attempts) per-assert partial credit:")
    print(f"  fully solved (all attempts): {fully} | partially: {part} | never: {never}")
    print(f"  overall MEAN attempt assert-pass rate: {round(sum(x['mean'] for x in out)/len(out),1)}%")
    print(f"  mean BEST-attempt rate:                {round(sum(x['best'] for x in out)/len(out),1)}%")
    print(f"-> {OUT}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run()
