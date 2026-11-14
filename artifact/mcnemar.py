"""Paired terse-vs-detailed McNemar + Wald 95% CI from a per-record scored JSON.

Runs OFFLINE on the shipped data/lcb_scored.json (the "ours"/Sieve verdicts, which are
test-verified) with no gated tests needed, reproducing the paper's robust-extractor
McNemar result and discordant counts. For the naive / sanitize / lcb columns you need a
scored file produced by score_all_extractors.py (which needs lcb_tests.jsonl).

A scored JSON is a list of records with at least: task_id, level, model, passed (bool).

Usage:
  python mcnemar.py                         # uses data/lcb_scored.json, capable models
  python mcnemar.py path/to/scored.json
  python mcnemar.py --models chatgpt claude codestral
"""
import os, sys, json
from math import comb, sqrt

H = os.path.dirname(os.path.abspath(__file__))
A = "agentic_terse"       # "terse"
B = "webchat_detailed"    # "detailed"


def mcnemar_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def wald_ci(b, c, n, z=1.96):
    d = (b - c) / n
    se = sqrt((b + c) - (b - c) ** 2 / n) / n
    return 100 * d, 100 * (d - z * se), 100 * (d + z * se)


def main(argv):
    path = os.path.join(H, "data", "lcb_scored.json")
    models = {"chatgpt", "claude"}
    rest = [a for a in argv]
    if "--models" in rest:
        i = rest.index("--models"); models = set(rest[i + 1:]); rest = rest[:i]
    if rest:
        path = rest[0]
    rows = json.load(open(path))
    scored = {(r["task_id"], r["level"], r["model"]): bool(r["passed"])
              for r in rows if r["model"] in models}
    keys = {(t, m) for (t, l, m) in scored}
    b = c = n = 0
    for (t, m) in keys:
        pa, pb = scored.get((t, A, m)), scored.get((t, B, m))
        if pa is None or pb is None:
            continue
        n += 1
        if pa and not pb: b += 1
        elif pb and not pa: c += 1
    delta, lo, hi = wald_ci(b, c, n)
    print(f"file: {path}")
    print(f"models: {sorted(models)} | paired (problem x model) observations: n = {n}")
    print(f"terse-only (b) = {b}   detailed-only (c) = {c}")
    print(f"terse - detailed = {delta:+.1f} pts   95% Wald CI [{lo:.1f}, {hi:.1f}]")
    print(f"exact McNemar p = {mcnemar_p(b, c):.3g}")


if __name__ == "__main__":
    main(sys.argv[1:])
