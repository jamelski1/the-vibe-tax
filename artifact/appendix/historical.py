"""
Pre-fix versions of two pieces of the LiveCodeBench scorer, kept verbatim so the
worked examples for pitfalls 4 and 11 can be re-run.

Nothing here is used by the study's results. The current scorer is
`../score_lcb.py`; these are the function bodies as they stood in the study's
scorer before the corresponding fix, copied without edits (only the names
carry a suffix so they cannot be confused with the current ones).

  extract_solution_anchor_slice   the extractor before the "keep helper
                                  definitions" fix: it slices the reply from
                                  the `class Solution` anchor (pitfall 4).
  passes_whole_suite              the pass/fail check before the per-test
                                  timeout: one 8-second budget for the whole
                                  test suite of a problem (pitfall 11).
"""
import ast
import json
import multiprocessing
import re

TIMEOUT = 8

_IMPORTS = ("from typing import *\nimport collections, math, heapq, bisect, itertools, functools, re\n"
            "from collections import *\nfrom math import *\nfrom functools import *\n")


def parse_lit(s):
    if not isinstance(s, str):
        return s
    s = s.strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(s)
        except Exception:
            continue
    return s  # leave as raw string


def parse_input(inp):
    """LCB functional input = one literal per line -> list of args."""
    if isinstance(inp, list):
        return [parse_lit(x) for x in inp]
    return [parse_lit(line) for line in str(inp).split("\n") if line.strip() != ""]


def _trim_to_compilable_v1(code):
    """Drop trailing lines until the source parses."""
    lines = code.split("\n")
    while lines:
        src = "\n".join(lines)
        try:
            ast.parse(src)
            return src
        except SyntaxError:
            lines.pop()
    return None


def extract_solution_anchor_slice(completion, entry):
    """Pull runnable code that defines `class Solution` (or the method), robust to
    trailing prose and leading chatter. Returns the first candidate that compiles
    AND still defines the target (so we never hand exec() a prose paragraph)."""
    if not completion:
        return None
    cands = []
    # 1) fenced code blocks that mention the class/method (highest confidence)
    for b in re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL):
        if "class Solution" in b or f"def {entry}" in b:
            cands.append(b)
    # 2) unfenced: slice from the class/def anchor to the end (drops leading prose)
    text = "\n".join(l for l in completion.split("\n") if not l.strip().startswith("```"))
    for anchor in ("class Solution", f"def {entry}"):
        i = text.find(anchor)
        if i != -1:
            cands.append(text[i:])
    # return the first candidate that, after trimming trailing prose, compiles
    for c in cands:
        t = _trim_to_compilable_v1(c)
        if t and ("class Solution" in t or f"def {entry}" in t):
            return t
    # last resort: the raw text (old behaviour) so we never regress to None
    return cands[0] if cands else None


def eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


def _worker_whole_suite(code, entry, tests, q):
    ns = {}
    try:
        exec(_IMPORTS + code, ns)
    except Exception:
        q.append(False); return
    sol_cls = ns.get("Solution")
    for t in tests:
        try:
            args = parse_input(t["input"])
            expected = parse_lit(t["output"])
            inst = sol_cls() if sol_cls else None
            fn = getattr(inst, entry, None) or ns.get(entry)
            if fn is None:
                q.append(False); return
            got = fn(*args)
            if not eq(got, expected):
                q.append(False); return
        except Exception:
            q.append(False); return
    q.append(True)


def passes_whole_suite(code, entry, tests):
    if not code:
        return False
    mgr = multiprocessing.Manager(); q = mgr.list()
    p = multiprocessing.Process(target=_worker_whole_suite, args=(code, entry, tests, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.kill(); p.join(3); return False
    return bool(q) and q[0]
