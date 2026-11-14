"""
Verify the shape-agnostic resolver (score_lcb.resolve_callable) and estimate how
many real completions the OLD resolver (`ns['Solution']` only) would mis-score FAIL
because the correct code was delivered in a different container (renamed class,
nested class, or a bare function). A "code smell," not a wrong answer.

Part 1 (unit): four synthetic shapes — old resolver vs new resolver.
Part 2 (scan): over lcb_v3_responses.json, count completions whose extracted code
  is resolvable by the NEW resolver but NOT the OLD one, categorized by shape.
  Runs anywhere (no gated tests needed); it inspects code shape, not correctness.

    python test_class_shape.py
"""
import ast
import inspect
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from score_lcb import extract_solution, resolve_callable, _IMPORTS


def old_resolve(ns, entry):
    """The pre-fix resolver: hardcoded Solution + bare-function fallback."""
    sol = ns.get("Solution")
    return (getattr(sol(), entry, None) if sol else None) or ns.get(entry)


SHAPES = {
    "class Solution (expected)":
        "class Solution:\n    def foo(self, x):\n        return x + 1\n",
    "renamed class":
        "class Sol:\n    def foo(self, x):\n        return x + 1\n",
    "nested class":
        "class Outer:\n    class Solution:\n        def foo(self, x):\n            return x + 1\n",
    "bare function":
        "def foo(x):\n    return x + 1\n",
}


def _exec(code, quiet=False):
    ns = {}
    if quiet:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(_IMPORTS + code, ns)      # some completions print at top level
    else:
        exec(_IMPORTS + code, ns)
    return ns


def unit():
    print("PART 1 — resolver on four code shapes (entry='foo'):")
    print(f"  {'shape':28s} {'OLD resolver':14s} {'NEW resolver':14s}")
    for name, code in SHAPES.items():
        ns = _exec(code)
        old = old_resolve(ns, "foo")
        new = resolve_callable(ns, "foo")
        old_ok = callable(old) and _safe_call(old)
        new_ok = callable(new) and _safe_call(new)
        print(f"  {name:28s} {('works' if old_ok else 'FAILS'):14s} {('works' if new_ok else 'FAILS'):14s}")
    print()


def _safe_call(fn):
    try:
        return fn(1) == 2
    except Exception:
        return False


def scan():
    resp_path = os.path.join(SCRIPT_DIR, "lcb_v3_responses.json")
    if not os.path.exists(resp_path):
        print("(scan skipped: lcb_v3_responses.json not found)"); return
    resp = json.load(open(resp_path, encoding="utf-8"))
    total = 0
    exec_fail = 0
    old_none_new_ok = {"renamed class": 0, "nested class": 0, "bare function": 0, "other": 0}
    examples = {}
    for r in resp:
        entry = r["entry_point"]
        code = extract_solution(r.get("completion"), entry)
        if not code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        total += 1
        try:
            ns = _exec(code, quiet=True)
        except Exception:
            exec_fail += 1
            continue
        old = old_resolve(ns, entry)
        new = resolve_callable(ns, entry)
        if not callable(old) and callable(new):
            # categorize the shape that the OLD resolver missed
            has_bare = callable(ns.get(entry)) and not inspect.isclass(ns.get(entry))
            top_classes = {k: v for k, v in ns.items() if inspect.isclass(v)}
            has_named_solution = "Solution" in top_classes and callable(getattr(top_classes["Solution"], entry, None))
            nested = any(inspect.isclass(v) and callable(getattr(v, entry, None))
                         for c in top_classes.values() for v in vars(c).values())
            if has_bare and not has_named_solution:
                # bare function is actually handled by old resolver's fallback, so
                # reaching here means old failed for another reason; classify carefully
                cat = "bare function"
            elif nested:
                cat = "nested class"
            elif top_classes:
                cat = "renamed class"
            else:
                cat = "other"
            old_none_new_ok[cat] += 1
            examples.setdefault(cat, (r["task_id"], r["level"], r["model"]))

    print("PART 2 — scan of extracted completions (shape resolvability):")
    print(f"  extracted + parseable completions inspected: {total}")
    print(f"  (exec-time errors, skipped): {exec_fail}")
    affected = sum(old_none_new_ok.values())
    print(f"  OLD resolver miss, NEW resolver resolves: {affected}")
    for cat, n in old_none_new_ok.items():
        if n:
            print(f"      {cat:16s}: {n:4d}   e.g. {examples.get(cat)}")
    print("\n  NOTE: 'resolvable' is necessary, not sufficient — a resolved callable"
          " must still pass the graded tests. Re-run score_lcb.py locally for the"
          " actual PASS delta. This counts the population at risk.")


if __name__ == "__main__":
    unit()
    scan()
