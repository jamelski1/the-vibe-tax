"""
Score LiveCodeBench functional completions.

LCB functional test case: {"input": "<arg>\\n<arg>...", "output": "<literal>"}.
Each input line is one method argument (a JSON/Python literal); output is the
expected return. We extract the model's `class Solution`, run
`Solution().<entry>(*args)` per test, and compare to the parsed expected output.
A completion passes a problem only if it passes EVERY (sampled) test.

Windows-compatible: uses multiprocessing (spawn) with a timeout, not SIGALRM.

Inputs (env-overridable):
    LCB_TESTS      lcb_tests.jsonl       (from extract_lcb_tests.py)
    LCB_RESPONSES  lcb_v3_responses.json (from run_vibe_tax.py on lcb_v3_prompts)
Outputs:
    lcb_scored.json / lcb_scored_stats.json

Usage:  python score_lcb.py [--max-tests 60]
"""

import argparse
import ast
import json
import multiprocessing
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS = os.getenv("LCB_TESTS", os.path.join(SCRIPT_DIR, "lcb_tests.jsonl"))
RESPONSES = os.getenv("LCB_RESPONSES", os.path.join(SCRIPT_DIR, "lcb_v3_responses.json"))
OUT = os.path.join(SCRIPT_DIR, "lcb_scored.json")
STATS = os.path.join(SCRIPT_DIR, "lcb_scored_stats.json")
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


def _trim_to_compilable(code):
    """Drop trailing lines until the source parses. Conversational replies put a
    prose paragraph AFTER the code ('This checks every adjacent pair...'); with no
    fences to delimit it, that prose used to be exec'd and threw SyntaxError,
    failing correct code. The polite/detailed framing elicits more explain-after-
    code, so this bug penalized it hardest — a scoring artifact, not correctness.
    We keep the largest leading prefix that is valid Python and still has the
    class/method."""
    lines = code.split("\n")
    while lines:
        src = "\n".join(lines)
        try:
            ast.parse(src)
            return src
        except SyntaxError:
            lines.pop()
    return None


def extract_solution(completion, entry):
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
        t = _trim_to_compilable(c)
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


def _worker(code, entry, tests, q):
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


def passes(code, entry, tests):
    if not code:
        return False
    mgr = multiprocessing.Manager(); q = mgr.list()
    p = multiprocessing.Process(target=_worker, args=(code, entry, tests, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.kill(); p.join(3); return False
    return bool(q) and q[0]


def sample_tests(rec, k):
    pub = rec.get("public_tests", [])
    priv = rec.get("private_tests", [])
    if len(pub) + len(priv) <= k:
        return pub + priv
    need = max(0, k - len(pub))
    stride = max(1, len(priv) // need) if need else 1
    return pub + (priv[::stride][:need] if need else [])


def run(max_tests):
    tests_by_id = {r["task_id"]: r for r in (json.loads(l) for l in open(TESTS, encoding="utf-8"))}
    responses = json.load(open(RESPONSES, encoding="utf-8"))
    # difficulty isn't carried on the response records — pull it from the problem file
    diff_by_id = {}
    probs_path = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")
    if os.path.exists(probs_path):
        for l in open(probs_path, encoding="utf-8"):
            p = json.loads(l)
            diff_by_id[p["task_id"]] = p.get("difficulty")
    print(f"tests for {len(tests_by_id)} problems | {len(responses)} completions")

    scored = []
    for i, r in enumerate(responses):
        rec = tests_by_id.get(r["task_id"])
        ok = False
        if rec:
            code = extract_solution(r.get("completion"), r["entry_point"])
            ok = passes(code, r["entry_point"], sample_tests(rec, max_tests))
        scored.append({k: r.get(k) for k in ("task_id", "level", "medium", "model")}
                      | {"difficulty": diff_by_id.get(r["task_id"]), "passed": ok})
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(responses)}", flush=True)

    json.dump(scored, open(OUT, "w", encoding="utf-8"), indent=2)

    def rate(items):
        n = len(items); k = sum(x["passed"] for x in items)
        return {"passed": k, "total": n, "pass_rate": round(k / n * 100, 1) if n else None}

    def grp(key):
        d = defaultdict(list)
        for x in scored:
            d[x.get(key)].append(x)
        return {str(k): rate(v) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}

    stats = {"overall": rate(scored), "by_condition": grp("level"),
             "by_model": grp("model"), "by_difficulty": grp("difficulty"),
             "by_condition_and_model": {f"{x['model']}|{x['level']}": None for x in scored}}
    cm = defaultdict(list)
    for x in scored:
        cm[f"{x['model']}|{x['level']}"].append(x)
    stats["by_condition_and_model"] = {k: rate(v) for k, v in sorted(cm.items())}
    json.dump(stats, open(STATS, "w", encoding="utf-8"), indent=2)

    print("=" * 60)
    print(f"LCB overall: {stats['overall']['pass_rate']}%  (n={stats['overall']['total']})")
    print("by condition:")
    for k, v in stats["by_condition"].items():
        print(f"  {k:22s} {v['passed']:3d}/{v['total']:<3d} {v['pass_rate']}%")
    print("by difficulty:")
    for k, v in stats["by_difficulty"].items():
        print(f"  {k:8s} {v['pass_rate']}%")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tests", type=int, default=60, help="max test cases per problem")
    a = ap.parse_args()
    run(a.max_tests)
