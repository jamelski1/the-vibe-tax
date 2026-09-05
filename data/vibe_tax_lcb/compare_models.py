"""
Compare two model runs on LiveCodeBench (e.g. GPT-5.4 baseline vs GPT-5.6).

Pairs completions by (task_id, level) — same problem, same framing — for one
provider (default chatgpt), and reports pass rates by difficulty, a paired
McNemar test (did the new model improve?), and the concrete flips: problems the
old model failed that the new one solves, and any regressions.

    python compare_models.py --baseline lcb_scored.json --new lcb_v3_gpt56_scored.json
    python compare_models.py --new lcb_v3_gpt56_scored.json --model chatgpt

Pure stdlib. Reproducible: prints the model_id stamped in each file.
"""

import argparse
import json
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from mcnemar_lcb import exact_binom_two_sided
except Exception:  # inline fallback
    from math import comb
    def exact_binom_two_sided(k, n):
        if n == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def load(path, model):
    rows = json.load(open(path, encoding="utf-8"))
    rows = [r for r in rows if r.get("model") == model]
    ids = {r.get("model_id") for r in rows}
    return {(r["task_id"], r["level"]): r for r in rows}, ids


def rate(vals):
    n = len(vals); k = sum(vals)
    return f"{k}/{n} = {round(k/n*100,1) if n else 0}%"


def run(baseline, new, model):
    A, ida = load(baseline, model)
    B, idb = load(new, model)
    keys = sorted(A.keys() & B.keys())
    if not keys:
        raise SystemExit(f"no shared (task_id,level) for model={model} between the two files")

    print(f"model provider: {model}")
    print(f"  baseline model_id: {ida or '(unstamped — older run)'}  file={os.path.basename(baseline)}")
    print(f"  new      model_id: {idb or '(unstamped)'}  file={os.path.basename(new)}")
    print(f"  paired items (task x framing): {len(keys)}\n")

    # by difficulty
    diff_a = defaultdict(list); diff_b = defaultdict(list)
    b_only = []  # new passes, old fails  (improvement)
    c_only = []  # old passes, new fails  (regression)
    for k in keys:
        a, b = A[k]["passed"], B[k]["passed"]
        d = A[k].get("difficulty")
        diff_a[d].append(a); diff_b[d].append(b)
        if b and not a: b_only.append((k, d))
        elif a and not b: c_only.append((k, d))

    print(f"{'difficulty':10} {'baseline':>16} {'new':>16}")
    for d in ("easy", "medium", "hard", None):
        if d in diff_a:
            print(f"{str(d):10} {rate(diff_a[d]):>16} {rate(diff_b[d]):>16}")
    all_a = [v for vs in diff_a.values() for v in vs]
    all_b = [v for vs in diff_b.values() for v in vs]
    print(f"{'ALL':10} {rate(all_a):>16} {rate(all_b):>16}\n")

    b, c = len(b_only), len(c_only)
    p = exact_binom_two_sided(min(b, c), b + c)
    delta = (b - c) / len(keys) * 100
    print(f"Paired McNemar (new vs baseline): new-solves-only={b}  baseline-solves-only={c}")
    print(f"  Δ = {delta:+.1f} pts   p = {p:.4f}   {'SIGNIFICANT' if p < 0.05 else 'n.s.'}\n")

    def show(lbl, items):
        print(f"{lbl} ({len(items)}):")
        for (tid, lvl), d in sorted(items, key=lambda x: (str(x[1]), x[0][0]))[:40]:
            print(f"   {tid:10} {str(d):6} {lvl}")
        if len(items) > 40:
            print(f"   … and {len(items)-40} more")
    show("NEWLY SOLVED (baseline failed → new passes)", b_only)
    print()
    show("REGRESSIONS (baseline passed → new fails)", c_only)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=os.path.join(SCRIPT_DIR, "lcb_scored.json"),
                    help="scored file for the old model (default lcb_scored.json = GPT-5.4 run)")
    ap.add_argument("--new", required=True, help="scored file for the new model")
    ap.add_argument("--model", default="chatgpt", help="provider to compare (chatgpt/claude/codestral)")
    a = ap.parse_args()
    run(a.baseline, a.new, a.model)
