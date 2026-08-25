"""
Re-score the v2 completions against HumanEval+ (EvalPlus) rigorous tests.

Base HumanEval ships ~7 tests per problem and passes subtly-wrong code.
HumanEval+ adds hundreds of edge-case inputs per problem. Re-scoring the SAME
822 completions against these tells us whether the flat 93-99% v2 result was a
weak-test artifact (tax hidden) or a real difficulty ceiling — no new API calls.

Method (approximation of the EvalPlus harness; noted as such):
  - expected outputs = canonical_solution run on a deterministic sample of
    base+plus inputs (cached per problem);
  - each completion's function is extracted robustly (multi-candidate, same as
    score_vibe_tax.py) and compared to expected with float tolerance;
  - passes only if it matches on EVERY sampled input. Cannot inflate.

Timeouts use multiprocessing (Windows-safe — no SIGALRM), matching score_lcb.py,
so this runs on Windows and Linux alike.

Get the dataset (gitignored; ~1MB, public EvalPlus release) then run:
    curl -sSL https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz | gunzip > HumanEvalPlus.jsonl
    python score_vibe_tax_plus.py [--samples 60]
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from collections import defaultdict

PER_PROBLEM_BUDGET = 5.0   # wall-clock seconds to spend precomputing one problem
CANDIDATE_TIMEOUT = 8      # seconds to run one completion's checks (subprocess join)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from score_vibe_tax import (strip_fences, problem_imports, extract_last_function,  # noqa: E402
                            normalized_body, fenced_blocks, clean_completion)

RESPONSES = os.getenv("RESPONSES_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_responses.json"))
PLUS = os.path.join(SCRIPT_DIR, "HumanEvalPlus.jsonl")
OUT = os.getenv("PLUS_SCORED_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_plus_scored.json"))
STATS = os.getenv("PLUS_STATS_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_plus_scored_stats.json"))


def outputs_equal(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=max(atol or 0, 1e-6))
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(outputs_equal(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(outputs_equal(a[k], b[k], atol) for k in a)
    return a == b


def candidate_sources(prompt, completion, entry):
    srcs, imports = [], problem_imports(prompt)
    fn = extract_last_function(completion, entry)
    if fn:
        srcs.append(imports + "\n\n" + fn)
    for block in reversed(fenced_blocks(completion)):
        if f"def {entry}" in block:
            srcs.append(imports + "\n\n" + block)
    srcs.append(prompt + clean_completion(completion, entry))
    nb = normalized_body(completion)
    if nb:
        srcs.append(prompt + nb)
    srcs.append(prompt + strip_fences(completion))
    return srcs


def _passes_worker(srcs, entry, inputs, expected, atol, q):
    for src in srcs:
        ns = {}
        try:
            exec(src, ns)
            fn = ns.get(entry)
            if not callable(fn):
                continue
        except Exception:
            continue
        try:
            if all(outputs_equal(fn(*inp), e, atol) for inp, e in zip(inputs, expected)):
                q.append(True); return
        except Exception:
            continue
    q.append(False)


def passes(srcs, entry, inputs, expected, atol):
    """Run one completion's candidate sources in a subprocess (Windows-safe
    timeout). Passes only if some candidate matches on every sampled input."""
    mgr = multiprocessing.Manager(); q = mgr.list()
    p = multiprocessing.Process(target=_passes_worker,
                                args=(srcs, entry, inputs, expected, atol, q))
    p.start(); p.join(CANDIDATE_TIMEOUT)
    if p.is_alive():
        p.kill(); p.join(3); return False
    return bool(q) and q[0]


def sample_inputs(prob, k):
    base = prob.get("base_input", [])
    plus = prob.get("plus_input", [])
    need = max(0, k - len(base))
    stride = max(1, len(plus) // need) if need and plus else 1
    return base + (plus[::stride][:need] if need else [])


def run(samples):
    responses = json.load(open(RESPONSES, encoding="utf-8"))
    plus = {json.loads(l)["task_id"]: json.loads(l) for l in open(PLUS, encoding="utf-8")}

    task_ids = sorted({r["task_id"] for r in responses})
    print(f"Precomputing expected outputs for {len(task_ids)} problems "
          f"(<= {samples} inputs each)...", flush=True)
    cache = {}
    for tid in task_ids:
        prob = plus[tid]
        ns = {}
        exec(prob["prompt"] + prob["canonical_solution"], ns)
        ref = ns[prob["entry_point"]]
        good, exp = [], []
        t0 = time.time()
        for inp in sample_inputs(prob, samples):
            if time.time() - t0 > PER_PROBLEM_BUDGET:   # wall-clock guard (canonical is trusted+fast)
                break
            try:
                out = ref(*inp)
                good.append(inp); exp.append(out)
            except Exception:
                pass
        cache[tid] = (good, exp, prob.get("atol", 0))
    avg = sum(len(v[0]) for v in cache.values()) / len(cache)
    print(f"  avg {avg:.0f} usable inputs/problem", flush=True)

    scored = []
    for i, r in enumerate(responses):
        prob = plus[r["task_id"]]
        inputs, exp, atol = cache[r["task_id"]]
        ok = False
        if r.get("completion") and inputs:
            ok = passes(candidate_sources(prob["prompt"], r["completion"], r["entry_point"]),
                        r["entry_point"], inputs, exp, atol)
        scored.append({k: r[k] for k in ("task_id", "level", "medium", "model")} | {"passed": ok})
        if (i + 1) % 150 == 0:
            print(f"  scored {i+1}/{len(responses)}", flush=True)

    json.dump(scored, open(OUT, "w", encoding="utf-8"), indent=2)

    def rate(items):
        n = len(items); k = sum(x["passed"] for x in items)
        return {"passed": k, "total": n, "pass_rate": round(k / n * 100, 1) if n else None}

    def grp(key):
        d = defaultdict(list)
        for x in scored:
            d[x[key]].append(x)
        return {k: rate(v) for k, v in sorted(d.items())}

    cm = defaultdict(list)
    for x in scored:
        cm[f'{x["model"]}|{x["level"]}'].append(x)
    stats = {
        "note": f"HumanEval+ approx scorer, up to {samples} sampled inputs/problem",
        "overall": rate(scored),
        "by_condition": grp("level"),
        "by_model": grp("model"),
        "by_medium": grp("medium"),
        "by_condition_and_model": {k: rate(v) for k, v in sorted(cm.items())},
    }
    json.dump(stats, open(STATS, "w", encoding="utf-8"), indent=2)

    print("=" * 60)
    print(f"HumanEval+ overall: {stats['overall']['pass_rate']}%  (n={stats['overall']['total']})")
    print("\nby condition:")
    for k, v in stats["by_condition"].items():
        print(f"  {k:22s} {v['passed']:3d}/{v['total']:<3d} {v['pass_rate']}%")
    print("\nby model:")
    for k, v in stats["by_model"].items():
        print(f"  {k:12s} {v['pass_rate']}%")
    print(f"\nscored -> {OUT}\nstats  -> {STATS}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)   # Windows-safe / consistent
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60, help="max test inputs per problem")
    a = ap.parse_args()
    run(a.samples)
