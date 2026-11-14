"""Table 4 (mechanism) + Fig. 1a compile-rates, recomputed from stored completions.

Pulls three of the extractors (Naive / Sieve / EvalPlus sanitize) verbatim from
The_Politeness_Tax_Four_Extractors.ipynb so the numbers come from the same code as the notebook.
Needs only data/lcb_v3_responses.json (+ tree_sitter, tree_sitter_python).

Run:  python mechanism_table.py   (needs tree_sitter; uses this folder's data/ + notebook)
"""
import os, json, re, statistics as st
_H = os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
from scipy.stats import binomtest

nb = json.load(open(os.path.join(_H, "The_Politeness_Tax_Four_Extractors.ipynb")))
for i in (8, 10, 12, 14, 16):  # sieve, naive, sanitize, lcb_official, compile_ok
    exec("".join(nb["cells"][i]["source"]), globals())

RESP = json.load(open(os.path.join(_H, "data", "lcb_v3_responses.json")))
cap = [r for r in RESP if r["model"] in ("chatgpt", "claude")]
CONDS = ["agentic_terse", "agentic_casual", "webchat_multilingual", "webchat_detailed"]
EXT = {"naive": extract_naive, "sieve": extract_sieve, "sanitize": extract_sanitize}

agg = defaultdict(lambda: defaultdict(list))
ok = {k: {} for k in EXT}
for r in cap:
    c, e, L = r.get("completion") or "", r["entry_point"], r["level"]
    for k, fn in EXT.items():
        try:
            code = fn(c, e)
        except Exception:
            code = None
        ok[k][(r["task_id"], r["model"], L)] = compile_ok(code, e)
    sieve = extract_sieve(c, e) or ""
    agg[L]["len"].append(len(c))
    agg[L]["noncode"].append(max(0, len(c.strip()) - len(sieve.strip())))
    nv = extract_naive(c, e)
    agg[L]["syntax"].append(nv is not None and not compile_ok(nv, e))
    agg[L]["notarget"].append(nv is None)

print(f"{'condition':22s} {'medLen':>7} {'prose>20':>9} {'nvSyntax':>9} {'nvNoTgt':>8}"
      + "".join(f" {k:>9}" for k in EXT))
for L in CONDS:
    a, n = agg[L], len(agg[L]["len"])
    rates = [100 * sum(v for (t, m, l), v in ok[k].items() if l == L) / n for k in EXT]
    print(f"{L:22s} {st.median(a['len']):7.0f} {100*sum(p > 20 for p in a['noncode'])/n:8.1f}% "
          f"{sum(a['syntax']):9d} {sum(a['notarget']):8d}" + "".join(f" {x:8.1f}%" for x in rates))

print("\npaired compile, terse vs detailed (terse-only / detailed-only, exact McNemar p):")
keys = {(t, m) for t, m, l in ok["naive"]}
for k in EXT:
    b = sum(ok[k][(t, m, "agentic_terse")] and not ok[k][(t, m, "webchat_detailed")] for t, m in keys)
    c = sum(ok[k][(t, m, "webchat_detailed")] and not ok[k][(t, m, "agentic_terse")] for t, m in keys)
    print(f"  {k:9s} {b:3d} / {c:3d}  p = {binomtest(b, b + c).pvalue if b + c else 1.0:.2g}")
