"""
Pairwise McNemar tests over LiveCodeBench framing conditions.

Reads lcb_scored.json (per-record: task_id, level, model, difficulty, passed) and
runs paired McNemar tests between framing conditions. Pairing is WITHIN
(task_id x model): for a given problem and model, condition A's pass/fail vs
condition B's pass/fail. The discordant cells are

    b = A passes & B fails      c = A fails & B passes

Delta = (b - c) / n_pairs (in points); p = two-sided exact binomial on
min(b,c) ~ Binomial(b+c, 0.5) (exact McNemar — correct for the small discordant
counts here, no chi-square approximation).

Reproduces the terse-vs-detailed numbers in RESULTS.md and adds the pairwise
comparisons the advisor asked for (terse vs casual, terse vs multilingual), for
every slice. Pure stdlib; no API keys, no network.

Usage:
    python mcnemar_lcb.py                # all pairs, all slices -> console + json
    python mcnemar_lcb.py --base agentic_terse    # only pairs vs a chosen base
"""

import argparse
import json
import os
from collections import defaultdict
from math import comb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORED = os.getenv("LCB_SCORED", os.path.join(SCRIPT_DIR, "lcb_scored.json"))
OUT = os.path.join(SCRIPT_DIR, "lcb_mcnemar_stats.json")

CAPABLE = {"chatgpt", "claude"}

# Slices: name -> predicate on a scored record
SLICES = {
    "all": lambda r: True,
    "medium": lambda r: r.get("difficulty") == "medium",
    "hard": lambda r: r.get("difficulty") == "hard",
    "easy": lambda r: r.get("difficulty") == "easy",
    "capable_models": lambda r: r.get("model") in CAPABLE,
    "capable_medium": lambda r: r.get("model") in CAPABLE and r.get("difficulty") == "medium",
}


def exact_binom_two_sided(k, n):
    """Two-sided exact binomial p at prob 0.5 (the exact McNemar p-value)."""
    if n == 0:
        return 1.0
    # P(X <= k) doubled, capped at 1 — standard two-sided exact McNemar
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pass_map(records, condition):
    """(task_id, model) -> bool passed, for one condition."""
    return {(r["task_id"], r["model"]): r["passed"]
            for r in records if r["level"] == condition}


def mcnemar(records, cond_a, cond_b, slice_pred):
    sel = [r for r in records if slice_pred(r)]
    a = pass_map(sel, cond_a)
    b = pass_map(sel, cond_b)
    keys = a.keys() & b.keys()
    b_cell = c_cell = 0        # b: A pass & B fail ; c: A fail & B pass
    a_pass = bpass = 0
    for k in keys:
        pa, pb = a[k], b[k]
        a_pass += pa
        bpass += pb
        if pa and not pb:
            b_cell += 1
        elif pb and not pa:
            c_cell += 1
    n = len(keys)
    if n == 0:
        return None
    disc = b_cell + c_cell
    p = exact_binom_two_sided(min(b_cell, c_cell), disc)
    return {
        "cond_a": cond_a, "cond_b": cond_b,
        "n_pairs": n,
        "a_pass_rate": round(a_pass / n * 100, 1),
        "b_pass_rate": round(bpass / n * 100, 1),
        "b_only": b_cell, "c_only": c_cell,
        "delta_pts": round((b_cell - c_cell) / n * 100, 1),
        "p": round(p, 4),
        "significant": p < 0.05,
    }


def run(base):
    records = json.load(open(SCORED, encoding="utf-8"))
    conditions = sorted({r["level"] for r in records})
    present_slices = {name: pred for name, pred in SLICES.items()
                      if any(pred(r) for r in records)}

    # Build the set of ordered pairs to test.
    if base:
        pairs = [(base, c) for c in conditions if c != base]
    else:
        pairs = [(a, b) for i, a in enumerate(conditions)
                 for b in conditions[i + 1:]]

    results = defaultdict(dict)
    for a, b in pairs:
        for sname, pred in present_slices.items():
            res = mcnemar(records, a, b, pred)
            if res:
                results[f"{a}__vs__{b}"][sname] = res

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)

    star = lambda r: "***" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "n.s.")
    print(f"LCB pairwise McNemar  (n_records={len(records)}, conditions={conditions})")
    print(f"delta = (a>b) - (b>a), in points; p = exact two-sided binomial\n")
    for pair, slices in results.items():
        a, b = pair.split("__vs__")
        print(f"=== {a}  vs  {b} ===")
        print(f"  {'slice':16s} {'n':>4s} {'A%':>6s} {'B%':>6s} {'a>b':>4s} {'b>a':>4s} {'Δpts':>6s} {'p':>7s}")
        for sname, r in slices.items():
            print(f"  {sname:16s} {r['n_pairs']:>4d} {r['a_pass_rate']:>6.1f} "
                  f"{r['b_pass_rate']:>6.1f} {r['b_only']:>4d} {r['c_only']:>4d} "
                  f"{r['delta_pts']:>+6.1f} {r['p']:>7.4f} {star(r)}")
        print()
    print(f"-> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="only test pairs against this condition (e.g. agentic_terse)")
    a = ap.parse_args()
    run(a.base)
