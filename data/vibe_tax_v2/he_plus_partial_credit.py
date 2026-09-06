"""
Per-input partial credit for HumanEval+ (HE4 completions, 3 models x 4 framings).

HumanEval+ adds hundreds of edge-case inputs per problem; normal scoring passes
only if ALL match. Here we COUNT the fraction of sampled inputs each attempt gets
right, giving the same partial-credit view as HumanEval and LiveCodeBench so all
three benchmarks are measured identically. No API calls.

Needs HumanEvalPlus.jsonl (gitignored). Get it once:
  python -c "import urllib.request,gzip; urllib.request.urlretrieve('https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz','HumanEvalPlus.jsonl.gz'); open('HumanEvalPlus.jsonl','wb').write(gzip.open('HumanEvalPlus.jsonl.gz').read())"
Then:
  python he_plus_partial_credit.py [--samples 60]
Writes he_plus_partial_credit.json.
"""
import argparse
import json
import multiprocessing
import os
import sys
from collections import defaultdict

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
from score_vibe_tax_plus import candidate_sources, outputs_equal, sample_inputs  # reuse

RESP = os.path.join(SD, "he4_responses.json")
PLUS = os.path.join(SD, "HumanEvalPlus.jsonl")
OUT = os.path.join(SD, "he_plus_partial_credit.json")
TIMEOUT = 12


def _worker(prompt, completion, entry, inputs, expected, atol, q):
    fn = None
    for src in candidate_sources(prompt, completion, entry):
        ns = {}
        try:
            exec(src, ns)
        except Exception:
            continue
        if callable(ns.get(entry)):
            fn = ns[entry]; break
    if fn is None:
        q.append(-1); return
    passed = 0
    for inp, exp in zip(inputs, expected):
        try:
            if outputs_equal(fn(*inp), exp, atol):
                passed += 1
        except Exception:
            pass
    q.append(passed)


def count(prompt, completion, entry, inputs, expected, atol):
    mgr = multiprocessing.Manager(); q = mgr.list()
    p = multiprocessing.Process(target=_worker,
                                args=(prompt, completion, entry, inputs, expected, atol, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.kill(); p.join(2); return None
    return q[0] if q else None


def run(samples):
    if not os.path.exists(PLUS):
        sys.exit(f"need {PLUS} (see the download line in this file's docstring)")
    plus = {json.loads(l)["task_id"]: json.loads(l) for l in open(PLUS, encoding="utf-8")}
    resp = json.load(open(RESP, encoding="utf-8"))
    by_task = defaultdict(list)
    for r in resp:
        by_task[r["task_id"]].append(r)

    # precompute expected outputs from the canonical solution (once per problem)
    print("precomputing expected outputs...", flush=True)
    cache = {}
    for tid in by_task:
        prob = plus[tid]; ns = {}
        exec(prob["prompt"] + prob["canonical_solution"], ns)
        ref = ns[prob["entry_point"]]
        inps, exps = [], []
        for inp in sample_inputs(prob, samples):
            try:
                exps.append(ref(*inp)); inps.append(inp)
            except Exception:
                pass
        cache[tid] = (inps, exps, prob.get("atol", 0))

    out = []
    for i, (tid, attempts) in enumerate(sorted(by_task.items())):
        prob = plus[tid]; entry = prob["entry_point"]
        inputs, expected, atol = cache[tid]
        total = len(inputs)
        rates = []
        for r in attempts:
            passed = count(prob["prompt"], r.get("completion"), entry, inputs, expected, atol)
            rates.append(round((passed if passed and passed > 0 else 0) / total * 100, 1) if total else 0.0)
        out.append({"task_id": tid, "n_inputs": total, "n_attempts": len(rates),
                    "n_solved": sum(1 for x in rates if x == 100.0),
                    "best": max(rates), "mean": round(sum(rates)/len(rates), 1), "worst": min(rates)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(by_task)}", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    print("=" * 60)
    print(f"HumanEval+ per-input partial credit ({len(out)} problems x {out[0]['n_attempts']} attempts):")
    print(f"  fully solved: {sum(1 for x in out if x['n_solved']==x['n_attempts'])} | "
          f"partial: {sum(1 for x in out if 0<x['n_solved']<x['n_attempts'])} | "
          f"never: {sum(1 for x in out if x['n_solved']==0)}")
    print(f"  overall MEAN attempt input-pass rate: {round(sum(x['mean'] for x in out)/len(out),1)}%")
    print(f"  mean BEST-attempt rate:               {round(sum(x['best'] for x in out)/len(out),1)}%")
    print(f"-> {OUT}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    a = ap.parse_args()
    run(a.samples)
