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
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS = os.getenv("LCB_TESTS", os.path.join(SCRIPT_DIR, "lcb_tests.jsonl"))
RESPONSES = os.getenv("LCB_RESPONSES", os.path.join(SCRIPT_DIR, "lcb_v3_responses.json"))
OUT = os.getenv("LCB_SCORED_OUT", os.path.join(SCRIPT_DIR, "lcb_scored.json"))
STATS = os.getenv("LCB_SCORED_STATS", os.path.join(SCRIPT_DIR, "lcb_scored_stats.json"))
PER_TEST_TIMEOUT = int(os.getenv("LCB_PER_TEST_TIMEOUT", "6"))  # seconds per test (not per problem)
# For the `with_tests` experiment: score on PRIVATE tests only, so a model that
# was shown the public tests can't pass by hardcoding their outputs.
PRIVATE_ONLY = os.getenv("LCB_PRIVATE_ONLY", "").lower() in ("1", "true", "yes")

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
    trailing prose, leading chatter, AND helper definitions (Fenwick/DSU/Segments…)
    the model defines BEFORE `class Solution`. Returns the first candidate that
    compiles and still defines the target. Candidates that KEEP the helpers are
    tried first, so we don't hand exec() a Solution that references an undefined
    helper we chopped off."""
    if not completion:
        return None
    text = "\n".join(l for l in completion.split("\n") if not l.strip().startswith("```"))
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL)
    code_blocks = [b for b in blocks if re.search(r"(?m)^\s*(?:class|def|import|from|@)\s", b)]

    cands = []
    # (B) from the FIRST top-level code construct to the end — keeps helper
    #     classes/functions defined above `class Solution` (fixes the drop bug).
    m = re.search(r"(?m)^(?:class|def|import|from|@)\s", text)
    if m:
        cands.append(text[m.start():])
    # (A) all fenced code blocks concatenated — helper and Solution in separate blocks.
    if code_blocks:
        cands.append("\n\n".join(code_blocks))
    # (C) a single block that names the target (old high-confidence path).
    for b in blocks:
        if "class Solution" in b or f"def {entry}" in b:
            cands.append(b)
    # (D) old anchor slice, last resort.
    for anchor in ("class Solution", f"def {entry}"):
        i = text.find(anchor)
        if i != -1:
            cands.append(text[i:])

    for c in cands:
        t = _trim_to_compilable(c)
        if t and ("class Solution" in t or f"def {entry}" in t):
            return t
    return cands[0] if cands else None


def resolve_callable(ns, entry):
    """Find the `entry` implementation regardless of how the model SHAPED its code.

    LCB expects `Solution().<entry>(*args)`. But a correct solution is sometimes
    delivered in a different container — a bare module-level function, a method on a
    RENAMED top-level class, or a method on a class nested one level inside another.
    Those are code *smells*, not wrong answers, yet the old resolver (hardcoded
    `ns['Solution']`) scored them FAIL. We look past the shape and bind to the actual
    implementation. Returns a callable (bound, if a method) or None.

    Order: (1) a bare module-level function named `entry`; else (2) a class exposing
    a callable `entry`, preferring one literally named `Solution`, searching both
    top-level classes and classes nested one level inside another class."""
    import inspect
    f = ns.get(entry)
    if callable(f) and not inspect.isclass(f):
        return f                                   # bare function shape
    classes = [v for v in ns.values() if inspect.isclass(v)]
    nested = [v for c in classes for v in vars(c).values() if inspect.isclass(v)]
    cands = [c for c in (classes + nested) if callable(getattr(c, entry, None))]
    if not cands:
        return None
    cands.sort(key=lambda c: c.__name__ != "Solution")   # prefer the expected name
    try:
        inst = cands[0]()
    except Exception:
        return None
    return getattr(inst, entry, None)


def eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


# Failure-reason taxonomy (the `reason` field on every scored record).
# "pass" on success; otherwise exactly one of these, resolved at the FIRST failing
# stage so a completion is attributed to how it first fails:
#   no_code        extraction returned nothing (no usable code in the reply)
#   compile_error  extracted code does not exec (SyntaxError / import-time error)
#   missing_library an import failed because the package is absent from the scoring
#                  env (e.g. sortedcontainers, which the real LCB/LeetCode judge
#                  provides) -> ENVIRONMENT, not a model failure; install & re-score
#   no_target      code execs but no `entry` callable is resolvable (shape mismatch)
#   runtime_error  a test call raised an exception (not a timeout)   -> SEMANTIC
#   wrong_answer   a test call returned a value != expected          -> SEMANTIC
#   timeout        a test exceeded PER_TEST_TIMEOUT (TLE / hang)     -> SEMANTIC (efficiency)
#   no_tests       the problem carried no usable tests (scoring gap, not the model)
# The three SEMANTIC reasons are the "code ran, output wrong/slow" bucket that
# CAPABILITY_ANALYSIS.md previously lumped as "wrong logic"; splitting them
# separates "wrong idea" (wrong_answer) from "right idea, too slow" (timeout).


def _worker(code, entry, tests, q):
    """Run tests in order, STREAMING one (status, detail) tuple per test into q, and
    stop at the first failure (early exit). `status` is "pass" or a failure reason
    from the taxonomy above; `detail` carries the exception type where relevant.
    Streaming lets grade() enforce a PER-TEST timeout instead of one budget for the
    whole suite — a correct-but-slow solution on a many-test problem must not be
    failed just because the SUM of test times exceeds a single cap (LCB, like any
    judge, limits each test, not the total)."""
    ns = {}
    try:
        exec(_IMPORTS + code, ns)
    except (ImportError, ModuleNotFoundError) as e:
        q.append(("missing_library", getattr(e, "name", None) or str(e))); return
    except Exception as e:
        q.append(("compile_error", type(e).__name__)); return
    fn = resolve_callable(ns, entry)               # shape-agnostic (class/bare/renamed/nested)
    if fn is None:
        q.append(("no_target", "")); return
    for t in tests:
        try:
            args = parse_input(t["input"])
            expected = parse_lit(t["output"])
            ok = eq(fn(*args), expected)
        except (ImportError, ModuleNotFoundError) as e:                 # import inside a method
            q.append(("missing_library", getattr(e, "name", None) or str(e))); return
        except Exception as e:
            q.append(("runtime_error", type(e).__name__)); return   # semantic: exception
        if not ok:
            q.append(("wrong_answer", "")); return                  # semantic: wrong output
        q.append(("pass", ""))


def grade(code, entry, tests):
    """Return (passed, reason, detail). passed iff EVERY test passes within
    PER_TEST_TIMEOUT seconds each; reason is "pass" or the first failing stage (see
    taxonomy above); detail carries the exception type / missing module name where
    relevant, else "". A wrong answer fails fast; a test that hangs (infinite loop /
    TLE) fails on its own timeout without killing the tests that already passed."""
    if not code:
        return (False, "no_code", "")
    if not tests:
        return (False, "no_tests", "")
    mgr = multiprocessing.Manager(); q = mgr.list()
    p = multiprocessing.Process(target=_worker, args=(code, entry, tests, q))
    p.start()
    seen = 0; last_progress = time.time()
    while p.is_alive():
        n = len(q)
        if n > seen:
            if q[n - 1][0] != "pass":             # newest test failed -> stop
                reason, detail = q[n - 1]
                p.kill(); p.join(2); return (False, reason, detail)
            seen = n; last_progress = time.time()
        elif time.time() - last_progress > PER_TEST_TIMEOUT:   # hung on current test
            p.kill(); p.join(2); return (False, "timeout", "")  # semantic: efficiency/hang
        time.sleep(0.02)
    res = list(q)
    if res and res[-1][0] != "pass":              # process ended on a recorded failure
        return (False, res[-1][0], res[-1][1])
    passed = len(res) == len(tests) and all(x[0] == "pass" for x in res)
    return (passed, "pass" if passed else "incomplete", "")


def passes(code, entry, tests):
    """Backward-compatible boolean wrapper around grade()."""
    return grade(code, entry, tests)[0]


def sample_tests(rec, k):
    pub = [] if PRIVATE_ONLY else rec.get("public_tests", [])
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
        if not rec:
            ok, reason, detail = False, "no_problem_tests", ""
        else:
            code = extract_solution(r.get("completion"), r["entry_point"])
            tl = sample_tests(rec, max_tests)
            ok, reason, detail = grade(code, r["entry_point"], tl) if tl else (False, "no_tests", "")
        scored.append({k: r.get(k) for k in ("task_id", "level", "medium", "model", "model_id")}
                      | {"difficulty": diff_by_id.get(r["task_id"]), "passed": ok,
                         "reason": reason, "reason_detail": detail})
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

    def reason_breakdown(items):
        d = defaultdict(int)
        for x in items:
            d[x["reason"]] += 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    stats = {"overall": rate(scored), "by_condition": grp("level"),
             "by_model": grp("model"), "by_difficulty": grp("difficulty"),
             "by_condition_and_model": {f"{x['model']}|{x['level']}": None for x in scored}}
    cm = defaultdict(list)
    for x in scored:
        cm[f"{x['model']}|{x['level']}"].append(x)
    stats["by_condition_and_model"] = {k: rate(v) for k, v in sorted(cm.items())}

    # Failure-reason taxonomy: overall, and (failures only) by condition — capable
    # models — so the syntax-vs-semantic split can be read per register.
    SEMANTIC = {"wrong_answer", "runtime_error", "timeout"}
    EXTRACTION = {"no_code", "compile_error", "no_target"}
    ENVIRONMENT = {"missing_library", "no_tests", "no_problem_tests"}
    stats["reason_counts"] = reason_breakdown(scored)
    fails = [x for x in scored if not x["passed"]]
    stats["failure_reasons"] = reason_breakdown(fails)
    stats["semantic_vs_other"] = {
        "semantic (wrong_answer/runtime_error/timeout)": sum(x["reason"] in SEMANTIC for x in fails),
        "extraction (no_code/compile_error/no_target)": sum(x["reason"] in EXTRACTION for x in fails),
        "environment (missing_library/no_tests)": sum(x["reason"] in ENVIRONMENT for x in fails),
    }
    cap_fail = defaultdict(list)
    for x in fails:
        if x["model"] in ("chatgpt", "claude"):
            cap_fail[x["level"]].append(x)
    stats["failure_reasons_by_condition_capable"] = {
        k: reason_breakdown(v) for k, v in sorted(cap_fail.items())}
    json.dump(stats, open(STATS, "w", encoding="utf-8"), indent=2)

    print("=" * 60)
    print(f"LCB overall: {stats['overall']['pass_rate']}%  (n={stats['overall']['total']})")
    print("by condition:")
    for k, v in stats["by_condition"].items():
        print(f"  {k:22s} {v['passed']:3d}/{v['total']:<3d} {v['pass_rate']}%")
    print("by difficulty:")
    for k, v in stats["by_difficulty"].items():
        print(f"  {k:8s} {v['pass_rate']}%")
    print("failure reasons (all failures):")
    for k, v in stats["failure_reasons"].items():
        print(f"  {k:16s} {v}")
    sv = stats["semantic_vs_other"]
    print(f"  -> semantic (ran, wrong/slow): {sv['semantic (wrong_answer/runtime_error/timeout)']}"
          f" | extraction: {sv['extraction (no_code/compile_error/no_target)']}"
          f" | environment: {sv['environment (missing_library/no_tests)']}")
    ml = [x for x in scored if x["reason"] == "missing_library"]
    if ml:
        mods = sorted({x.get("reason_detail", "") for x in ml})
        print("!" * 60)
        print(f"WARNING: {len(ml)} completions failed on a MISSING LIBRARY, not on correctness.")
        print("  These are environment failures the real LCB/LeetCode judge would not hit.")
        print("  Install the package(s) and re-score, e.g.:  pip install sortedcontainers")
        print("!" * 60)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tests", type=int, default=60, help="max test cases per problem")
    a = ap.parse_args()
    run(a.max_tests)
