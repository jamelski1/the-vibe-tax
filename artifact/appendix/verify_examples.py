"""
Re-run every check quoted in Appendix_Worked_Examples.pdf.

Each section takes ONE stored record, runs the relevant extractor or harness on
it, and prints what happens. Nothing is constructed: replies, prompts, problem
statements, tests and verdicts are all read from files on this branch.

    cd artifact/appendix
    python verify_examples.py            # all sections   (~1 minute)
    python verify_examples.py 4 11       # only these pitfalls
    python verify_examples.py --slow     # also re-run the whole-suite-timeout
                                         # check on the two problems with local tests

Needs only the standard library. `sanitize` additionally needs tree_sitter and
tree_sitter_python; without them its rows print "not run".

Inputs
  ../data/lcb_v3_responses.json          the stored replies (2,004; the paper's slice is
                                         the 1,336 with model chatgpt or claude)
  ../data/lcb_problems.jsonl             stored problem statements
  ../data/lcb_scored_sieve.json          stored Sieve verdicts (official tests)
  evidence/lcb_scored_before_helper_and_timeout_fixes.json
                                         the stored scoring run made BEFORE the helper fix
                                         and the per-test timeout (pitfalls 4 and 11)
  historical.py                          the two pre-fix functions, verbatim
  <repo>/data/politeness_tax_lcb/lcb_tests_sample.json
                                         official tests for two problems (the full set is gated)
  <repo>/data/politeness_tax_lcb/lcb_v3_progress.json
                                         the checkpoint written while replies were collected
  <repo>/data/politeness_tax_v2/, <repo>/data/HumanEval.jsonl/
                                         stored replies, tests and scorers of the earlier
                                         HumanEval phase (pitfalls 3, 7, 8, 13)
Sections that need the <repo>/data tree say so and are skipped if it is absent.
"""
import ast
import contextlib
import io
import json
import multiprocessing
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
ROOT = os.path.dirname(ART)
sys.path.insert(0, ART)
sys.path.insert(0, HERE)

import score_lcb as S                                                  # noqa: E402
from extractors import (extract_naive, extract_sieve,                  # noqa: E402
                        extract_sanitize, extract_lcb_official)
import historical as H                                                 # noqa: E402

MODEL = {"chatgpt": "GPT-5.4", "claude": "Claude Opus 4.6", "codestral": "Codestral"}
LEVEL = {"agentic_terse": "terse", "agentic_casual": "casual",
         "webchat_detailed": "detailed", "webchat_multilingual": "multilingual"}

ALL = json.load(open(os.path.join(ART, "data", "lcb_v3_responses.json"), encoding="utf-8"))
CAPABLE = [r for r in ALL if r["model"] in ("chatgpt", "claude")]
PROBLEMS = {}
for _l in open(os.path.join(ART, "data", "lcb_problems.jsonl"), encoding="utf-8"):
    _p = json.loads(_l); PROBLEMS[_p["task_id"]] = _p
NOW = {(s["task_id"], s["model"], s["level"]): s["passed"]
       for s in json.load(open(os.path.join(ART, "data", "lcb_scored_sieve.json"), encoding="utf-8"))}
BEFORE = {(s["task_id"], s["model"], s["level"]): s["passed"]
          for s in json.load(open(os.path.join(HERE, "evidence",
                                               "lcb_scored_before_helper_and_timeout_fixes.json"), encoding="utf-8"))}
SAMPLE_TESTS_PATH = os.path.join(ROOT, "data", "politeness_tax_lcb", "lcb_tests_sample.json")
PROGRESS_PATH = os.path.join(ROOT, "data", "politeness_tax_lcb", "lcb_v3_progress.json")


# ----------------------------------------------------------------------------- helpers
def reply(task, model, level):
    return next(r for r in ALL if r["task_id"] == task and r["model"] == model and r["level"] == level)


def pf(b):
    return "PASS" if b else "FAIL"


def head(n, title):
    print("\n" + "=" * 100)
    print(f"PITFALL {n}. {title}")
    print("=" * 100)


def who(r):
    return (f"{r['task_id']} | {MODEL[r['model']]} | {LEVEL[r['level']]} register | "
            f"{PROBLEMS[r['task_id']]['difficulty']} | reply: {len(r['completion'])} chars, "
            f"{len(r['completion'].split(chr(10)))} lines")


def compile_status(code):
    if not code:
        return "nothing extracted"
    try:
        compile(code, "<extracted>", "exec")
        return "compiles"
    except SyntaxError as ex:
        return f"SyntaxError: {ex.msg} (line {ex.lineno} of the extracted text)"


def examples(task):
    """(args, expected) pairs from the 'Input: ... / Output: ...' lines of the stored statement."""
    p = PROBLEMS[task]; q = p["question_content"]
    names = [x.strip() for x in p["params"].split(",")]
    out = []
    for m in re.finditer(r"Input:\s*(.*?)\nOutput:\s*(.*?)\n", q, re.DOTALL):
        inp, exp = m.group(1).strip(), m.group(2).strip()
        args = []
        for i, n in enumerate(names):
            nxt = names[i + 1] if i + 1 < len(names) else None
            pat = rf"\b{re.escape(n)}\s*=\s*(.*?)" + (rf",\s*{re.escape(nxt)}\s*=" if nxt else r"$")
            args.append(S.parse_lit(re.search(pat, inp, re.DOTALL).group(1).strip()))
        out.append((args, S.parse_lit(exp)))
    return out


def _call(code, entry, args, q):
    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(S._IMPORTS + code, ns)
            fn = S.resolve_callable(ns, entry)
            if fn is None:
                q.put(("no callable named %s is defined" % entry, None)); return
            q.put(("ok", fn(*args)))
    except BaseException as ex:                      # noqa: BLE001  (report whatever the code raises)
        q.put((f"{type(ex).__name__}: {str(ex)[:90]}", None))


def call(code, entry, args, timeout=20):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_call, args=(code, entry, args, q)); p.start()
    try:
        r = q.get(timeout=timeout)
    except Exception:
        r = ("no result within %ds" % timeout, None)
    p.kill(); p.join()
    return r


def on_statement_examples(code, entry, task):
    """Run extracted code on the examples printed in the stored problem statement."""
    out = []
    for i, (args, exp) in enumerate(examples(task), 1):
        st, got = call(code, entry, args)
        if st == "ok":
            out.append(f"example {i}: returns {got!r}, statement says {exp!r} -> "
                       + ("matches" if S.eq(got, exp) else "DOES NOT MATCH"))
        else:
            out.append(f"example {i}: {st}")
    return out


def sanitize_or_none(c, e):
    try:
        return extract_sanitize(c, e)
    except RuntimeError:
        return None                                   # tree_sitter not installed


def four_extractors(r, run_examples=True):
    c, e, t = r["completion"], r["entry_point"], r["task_id"]
    rows = [("Naive", extract_naive(c, e)), ("Sieve", extract_sieve(c, e)),
            ("sanitize", sanitize_or_none(c, e)), ("LCB-official", extract_lcb_official(c, e))]
    for name, code in rows:
        if name == "sanitize" and code is None:
            print(f"  {name:13s} not run (tree_sitter not installed)"); continue
        st = compile_status(code)
        n = len(code) if code else 0
        print(f"  {name:13s} extracts {n} chars; {st}")
        if code and st == "compiles":
            ns_has = ("class Solution" in code) or (f"def {e}" in code)
            if not ns_has:
                print(f"  {'':13s} the extracted text does not define Solution or {e}")
            elif run_examples:
                for line in on_statement_examples(code, e, t):
                    print(f"  {'':13s} {line}")
    return dict(rows)


def load_sample_tests():
    if not os.path.exists(SAMPLE_TESTS_PATH):
        return None
    return {r["task_id"]: r for r in json.load(open(SAMPLE_TESTS_PATH, encoding="utf-8"))}


def _per_test(code, entry, tests, q):
    ns = {}
    try:
        exec(S._IMPORTS + code, ns)
    except BaseException as ex:                      # noqa: BLE001
        q.put(("exec", type(ex).__name__)); q.put(None); return
    for i, t in enumerate(tests):
        try:
            fn = S.resolve_callable(ns, entry)
            ok = S.eq(fn(*S.parse_input(t["input"])), S.parse_lit(t["output"]))
            q.put((i, "pass" if ok else "wrong answer"))
        except BaseException as ex:                  # noqa: BLE001
            q.put((i, type(ex).__name__))
    q.put(None)


def per_test(code, entry, tests, each=10):
    """Run EVERY test (no early exit) and return one outcome per test."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_per_test, args=(code, entry, tests, q)); p.start()
    res = []
    while True:
        try:
            x = q.get(timeout=each)
        except Exception:
            res.append((len(res), "no result within %ds" % each)); break
        if x is None:
            break
        res.append(x)
    p.kill(); p.join()
    return res


# ----------------------------------------------------------------------------- sections
def p1():
    head(1, "Trailing prose after unfenced code")
    r = reply("lcb/3747", "chatgpt", "webchat_detailed"); print(who(r))
    print("last 4 lines of the reply:")
    for l in [x for x in r["completion"].split("\n") if x.strip()][-4:]:
        print("   | " + l)
    four_extractors(r)
    print("stored Sieve verdict (official tests):", pf(NOW[("lcb/3747", "chatgpt", "webchat_detailed")]))


def p2():
    head(2, "Leading prose before code")
    r = reply("lcb/3535", "claude", "agentic_casual"); print(who(r))
    lines = r["completion"].split("\n")
    first_code = next(i for i, l in enumerate(lines) if l.startswith("class Solution"))
    print(f"lines 1-{first_code} are prose; the code starts on line {first_code + 1} and runs to the last line")
    print("line 1:", lines[0])
    print("fence lines in the reply:", sum(1 for l in lines if "```" in l))
    four_extractors(r)
    print("stored Sieve verdict (official tests):", pf(NOW[("lcb/3535", "claude", "agentic_casual")]))


def p4():
    head(4, "Helper defined above Solution dropped")
    k = ("lcb/3680", "chatgpt", "agentic_terse"); r = reply(*k); print(who(r))
    c, e = r["completion"], r["entry_point"]
    tops = [f"{type(x).__name__} {x.name}" for x in ast.parse(extract_sieve(c, e)).body
            if isinstance(x, (ast.ClassDef, ast.FunctionDef))]
    print("top-level definitions in the reply, in order:", tops)
    old = H.extract_solution_anchor_slice(c, e)
    print("pre-fix extractor (slice from the `class Solution` anchor):")
    print(f"  extracts {len(old)} chars; first line: {old.splitlines()[0]!r}; contains 'class DSU': {'class DSU' in old}; "
          f"{compile_status(old)}")
    for line in on_statement_examples(old, e, k[0]):
        print("  " + line)
    new = extract_sieve(c, e)
    print("Sieve (current extractor):")
    print(f"  extracts {len(new)} chars; first line: {new.splitlines()[0]!r}; contains 'class DSU': {'class DSU' in new}; "
          f"{compile_status(new)}")
    for line in on_statement_examples(new, e, k[0]):
        print("  " + line)
    print("stored verdict, scoring run made with the pre-fix extractor:", pf(BEFORE[k]))
    print("stored verdict, current Sieve run (official tests):         ", pf(NOW[k]))
    n = 0
    for r2 in CAPABLE:
        a = H.extract_solution_anchor_slice(r2["completion"], r2["entry_point"]) or ""
        names = [x.name for x in ast.parse(extract_sieve(r2["completion"], r2["entry_point"])).body
                 if isinstance(x, (ast.ClassDef, ast.FunctionDef))]
        above = names[:names.index("Solution")] if "Solution" in names else []
        if any(f"class {h}" not in a and f"def {h}" not in a for h in above):
            n += 1
    print(f"replies in the 1,336 where the pre-fix extractor drops a definition placed above Solution: {n}")


def fence_blocks(c):
    return re.findall(r"```(\w*)[^\n]*\n(.*?)```", c, re.DOTALL)


def p5():
    head(5, "Code split across several fenced blocks")
    k = ("lcb/3789", "claude", "agentic_terse"); r = reply(*k); print(who(r))
    c, e = r["completion"], r["entry_point"]
    for i, (tag, b) in enumerate(fence_blocks(c), 1):
        print(f"fenced block {i}: tag {tag!r}, {len(b.splitlines())} lines, defines Solution: {'class Solution' in b}")
    print("fenced block 2, in full:", repr(fence_blocks(c)[1][1]))
    got = four_extractors(r)
    blocks = [b for _, b in fence_blocks(c)]
    for name, code in got.items():
        if code is None:
            continue
        which = [i for i, b in enumerate(blocks, 1) if code.strip() == b.strip()]
        print(f"  {name:13s} output is fenced block {which[0] if which else '(none exactly)'}")
    print("stored Sieve verdict (official tests):", pf(NOW[k]))
    two = [r2 for r2 in CAPABLE if len(fence_blocks(r2["completion"])) >= 2]
    diff = [r2 for r2 in two if (extract_naive(r2["completion"], r2["entry_point"]) or "").strip()
            != extract_lcb_official(r2["completion"], r2["entry_point"]).strip()]
    print(f"replies in the 1,336 with two or more fenced blocks: {len(two)}; "
          f"of these, Naive and LCB-official extract different text in {len(diff)}")
    split = 0
    for r2 in two:
        tops = []
        for _, b in fence_blocks(r2["completion"]):
            try:
                tops.append({x.name for x in ast.parse(b).body if isinstance(x, (ast.ClassDef, ast.FunctionDef))})
            except SyntaxError:
                tops.append(set())
        has_sol = [i for i, t in enumerate(tops) if "Solution" in t]
        helper_elsewhere = any(t - {"Solution"} for i, t in enumerate(tops) if i not in has_sol)
        split += bool(has_sol) and helper_elsewhere
    print(f"  replies with a helper definition in one block and Solution in another: {split}")


def p6():
    head(6, "Shape mismatch: bare, nested or renamed")
    counts = {}
    for r in ALL:
        code = extract_sieve(r["completion"], r["entry_point"])
        try:
            body = ast.parse(code).body if code else None
        except SyntaxError:
            body = None
        if body is None:
            shape = "no code extracted"
        else:
            top = {x.name: x for x in body if isinstance(x, (ast.ClassDef, ast.FunctionDef))}
            if "Solution" in top and any(isinstance(m, ast.FunctionDef) and m.name == r["entry_point"]
                                         for m in top["Solution"].body):
                shape = "method on a top-level class Solution"
            elif r["entry_point"] in top:
                shape = "bare top-level function"
            else:
                shape = "other (renamed or nested class)"
        counts.setdefault(r["model"], {}).setdefault(shape, 0)
        counts[r["model"]][shape] += 1
    for m in ("chatgpt", "claude", "codestral"):
        print(f"  {MODEL[m]:16s} {counts[m]}")
    k = ("lcb/3517", "codestral", "agentic_terse"); r = reply(*k); print("instance OUTSIDE the 1,336:", who(r))
    print("the reply, in full:")
    for l in r["completion"].split("\n"):
        print("   | " + l)
    code = extract_sieve(r["completion"], r["entry_point"])
    ns = {}; exec(S._IMPORTS + code, ns)
    print("after exec: 'Solution' defined:", "Solution" in ns, "| resolve_callable binds:",
          getattr(S.resolve_callable(ns, r["entry_point"]), "__name__", None))
    tests = load_sample_tests()
    if tests:
        rec = tests[k[0]]; tl = rec["public_tests"] + rec["private_tests"]
        print(f"score_lcb.grade on the {len(tl)} official tests stored for this problem:", S.grade(code, r["entry_point"], tl))
    print("stored verdict (official tests):", pf(NOW[k]))


def p9():
    head(9, "Fence-marker variants, unbalanced backticks")
    k = ("lcb/3785", "claude", "webchat_multilingual"); r = reply(*k); print(who(r))
    c, e = r["completion"], r["entry_point"]; lines = c.split("\n")
    print("lines containing a fence marker:", [(i + 1, l) for i, l in enumerate(lines) if "```" in l])
    for i, (tag, b) in enumerate(fence_blocks(c), 1):
        print(f"fenced block {i}: {b.strip()!r}")
    print("line on which `class Solution` starts:", next(i + 1 for i, l in enumerate(lines) if l.startswith("class Solution")),
          "(outside every fence)")
    got = four_extractors(r)
    print("LCB-official output, in full:", repr(got["LCB-official"]))
    print("stored Sieve verdict (official tests):", pf(NOW[k]))
    print("second form, an opening fence that is never closed:")
    k2 = ("lcb/3733", "claude", "webchat_detailed"); r2 = reply(*k2); print(" ", who(r2))
    l2 = r2["completion"].split("\n")
    print("  lines containing a fence marker:", [(i + 1, l) for i, l in enumerate(l2) if "```" in l])
    o = extract_lcb_official(r2["completion"], r2["entry_point"])
    print(f"  LCB-official takes the text between the last two markers: {len(o)} chars; {compile_status(o)}")
    print("  its first line:", repr(o.strip().splitlines()[0][:110]))
    odd = [r3 for r3 in CAPABLE if sum(1 for l in r3["completion"].split("\n") if l.strip().startswith("```")) % 2]
    print("replies in the 1,336 with an odd number of fence markers:",
          [(r3["task_id"], MODEL[r3["model"]], LEVEL[r3["level"]]) for r3 in odd])


def p10():
    head(10, "Over-aggressive extraction returns nothing")
    k = ("lcb/3517", "chatgpt", "agentic_terse"); r = reply(*k); print(who(r))
    c, e = r["completion"], r["entry_point"]
    print("first line of the reply:", repr(c.split("\n")[0]), "| last line:", repr(c.rstrip().split("\n")[-1]))
    print("fence lines in the reply:", sum(1 for l in c.split("\n") if "```" in l))
    print("whole reply compiles as Python:", compile_status(c))
    o = extract_lcb_official(c, e); s = extract_sieve(c, e)
    print(f"LCB-official returns {o!r}")
    tests = load_sample_tests()
    if tests:
        rec = tests[k[0]]; tl = rec["public_tests"] + rec["private_tests"]
        print(f"score_lcb.grade(LCB-official output) on the {len(tl)} official tests stored for this problem:", S.grade(o, e, tl))
        print(f"score_lcb.grade(Sieve output)        on the same tests:", S.grade(s, e, tl))
    print("stored Sieve verdict (official tests):", pf(NOW[k]))
    empty = sum(1 for r2 in CAPABLE if not extract_lcb_official(r2["completion"], r2["entry_point"]))
    print(f"replies in the 1,336 for which LCB-official returns the empty string: {empty}")


def p11(slow=False):
    head(11, "Whole-suite timeout instead of per test")
    task = "lcb/3551"
    print(f"{task} | {PROBLEMS[task]['difficulty']} | entry point {PROBLEMS[task]['entry_point']}")
    print("stored verdicts, same stored replies, two scoring runs:")
    print(f"  {'reply':34s} {'8 s for the whole suite':26s} {'6 s per test (current)':24s} extracted code identical")
    for m in ("chatgpt", "claude"):
        for lv in LEVEL:
            r = reply(task, m, lv)
            same = H.extract_solution_anchor_slice(r["completion"], r["entry_point"]) == extract_sieve(r["completion"], r["entry_point"])
            print(f"  {MODEL[m] + ', ' + LEVEL[lv]:34s} {pf(BEFORE[(task, m, lv)]):26s} {pf(NOW[(task, m, lv)]):24s} {same}")
    k = (task, "chatgpt", "agentic_terse"); r = reply(*k)
    code = extract_sieve(r["completion"], r["entry_point"])
    print("instance:", who(r))
    for line in on_statement_examples(code, r["entry_point"], task):
        print("  " + line)
    flips = [k2 for k2 in BEFORE if k2[1] in ("chatgpt", "claude") and not BEFORE[k2] and NOW[k2]]
    back = [k2 for k2 in BEFORE if k2[1] in ("chatgpt", "claude") and BEFORE[k2] and not NOW[k2]]
    same_code = [k2 for k2 in flips if H.extract_solution_anchor_slice(reply(*k2)["completion"], reply(*k2)["entry_point"])
                 == extract_sieve(reply(*k2)["completion"], reply(*k2)["entry_point"])]
    print(f"across the 1,336: FAIL->PASS between the two runs: {len(flips)}; PASS->FAIL: {len(back)}; "
          f"FAIL->PASS with byte-identical extracted code: {len(same_code)}")
    print("NOT re-executed here: the official tests for this problem are gated and not on the branch.")
    tests = load_sample_tests()
    if slow and tests:
        print("re-run on the two problems whose official tests ARE on the branch (old whole-suite check vs current per-test check):")
        differ = 0
        for r2 in CAPABLE:
            if r2["task_id"] not in tests:
                continue
            rec = tests[r2["task_id"]]; tl = rec["public_tests"] + rec["private_tests"]
            code2 = extract_sieve(r2["completion"], r2["entry_point"])
            a = H.passes_whole_suite(code2, r2["entry_point"], tl); b = S.grade(code2, r2["entry_point"], tl)
            differ += (a != b[0])
            print(f"  {r2['task_id']} {MODEL[r2['model']]:16s} {LEVEL[r2['level']]:13s} whole-suite: {pf(a)}  per-test: {pf(b[0])} ({b[1]})")
        print(f"  replies where the two checks disagree: {differ}")


def stops_mid_reply(r):
    """A reply that was cut off: it leaves a code fence open (odd number of fence markers),
    or its last non-blank line is a comment inside an indented block."""
    lines = [l for l in r["completion"].split("\n") if l.strip()]
    odd = sum(1 for l in lines if l.strip().startswith("```")) % 2 == 1
    last = lines[-1]
    return odd or (last.startswith((" ", "\t")) and last.strip().startswith("#"))


def returns_in_method(code, entry):
    """Number of `return` statements that belong to the entry method itself (not to nested helpers)."""
    cls = next(x for x in ast.parse(code).body if isinstance(x, ast.ClassDef) and x.name == "Solution")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == entry)
    n = 0; stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        n += isinstance(node, ast.Return)
        stack.extend(ast.iter_child_nodes(node))
    return n


def p12():
    head(12, "Token cap: empty or truncated mid-code")
    k = ("lcb/3700", "claude", "agentic_casual"); r = reply(*k); print(who(r))
    c, e = r["completion"], r["entry_point"]; lines = c.split("\n")
    print("fence markers:", [(i + 1, l) for i, l in enumerate(lines) if "```" in l], "(opened, never closed)")
    print("last 3 lines of the reply:")
    for l in lines[-3:]:
        print("   | " + l)
    s = extract_sieve(c, e)
    print(f"Sieve extracts {len(s.splitlines())} lines (everything from `class Solution` to the cut); {compile_status(s)}")
    print(f"  `return` statements in the method {e} itself: {returns_in_method(s, e)}")
    for line in on_statement_examples(s, e, k[0]):
        print("  " + line)
    print("stored Sieve verdict (official tests):", pf(NOW[k]))
    cut = [r2 for r2 in CAPABLE if stops_mid_reply(r2)]
    print("replies in the 1,336 that were cut off (open code fence, or last line a comment inside a block):")
    for r2 in cut:
        print(f"  {r2['task_id']} {MODEL[r2['model']]:16s} {LEVEL[r2['level']]:13s} {len(r2['completion'])} chars, "
              f"{PROBLEMS[r2['task_id']]['difficulty']}, stored verdict {pf(NOW[(r2['task_id'], r2['model'], r2['level'])])}; "
              f"ends: {r2['completion'].rstrip().splitlines()[-1].strip()[-48:]!r}")
    print("empty replies in the 1,336:", sum(1 for r2 in CAPABLE if not (r2["completion"] or "").strip()))
    for m in ("chatgpt", "claude"):
        L = sorted(len(r2["completion"]) for r2 in CAPABLE if r2["model"] == m)
        print(f"  longest replies, {MODEL[m]}: {L[-6:]} chars")


def p13_14():
    tests = load_sample_tests()
    k = ("lcb/3527", "chatgpt", "webchat_detailed"); r = reply(*k)
    if not tests:
        print("\n(pitfall 14 needs <repo>/data/politeness_tax_lcb/lcb_tests_sample.json; skipped)"); return
    rec = tests[k[0]]; pub, priv = rec["public_tests"], rec["private_tests"]
    code = extract_sieve(r["completion"], r["entry_point"])
    head(14, "Binary pass@1 hides near-misses")
    print(who(r))
    res = per_test(code, r["entry_point"], pub + priv)
    okp = sum(1 for i, st in res if i < len(pub) and st == "pass")
    okq = sum(1 for i, st in res if i >= len(pub) and st == "pass")
    print(f"official tests stored for this problem: {len(pub)} public + {len(priv)} private")
    print(f"per-test outcomes, every test run: public {okp}/{len(pub)} pass, private {okq}/{len(priv)} pass, "
          f"total {okp + okq}/{len(pub) + len(priv)} = {100 * (okp + okq) / (len(pub) + len(priv)):.1f}%")
    print("failing tests (0-based index, outcome):", [(i, st) for i, st in res if st != "pass"])
    print("score_lcb.grade on all tests:        ", S.grade(code, r["entry_point"], pub + priv))
    print("score_lcb.grade on public tests only:", S.grade(code, r["entry_point"], pub))
    print("stored Sieve verdict (official tests):", pf(NOW[k]))
    print("same problem, same model, other registers (per-test outcomes):")
    for lv in LEVEL:
        if lv == k[2]:
            continue
        r2 = reply(k[0], k[1], lv)
        res2 = per_test(extract_sieve(r2["completion"], r2["entry_point"]), r2["entry_point"], pub + priv)
        n = sum(1 for _, st in res2 if st == "pass")
        print(f"  {LEVEL[lv]:13s} {n}/{len(pub) + len(priv)} pass; first non-pass: "
              f"{next(((i, st) for i, st in res2 if st != 'pass'), None)}; stored verdict {pf(NOW[(k[0], k[1], lv)])}")


def cp437_state(s):
    if all(ord(ch) < 128 for ch in s):
        return "ascii"
    try:
        s.encode("cp437").decode("utf-8")
        return "garbled"
    except (UnicodeEncodeError, UnicodeDecodeError):
        return "clean non-ascii"


def p15():
    head(15, "Non-ASCII prompt corrupted by code page")
    k = ("lcb/3540", "chatgpt", "webchat_multilingual"); r = reply(*k); print(who(r))
    pt = r["prompt_text"]; rec = pt.encode("cp437").decode("utf-8")
    print("stored prompt_text, first line:   ", pt.split("\n")[0])
    print("after cp437->utf-8 recovery:      ", rec.split("\n")[0])
    i = pt.find("ΓåÆ")
    print("stored statement fragment:        ", pt[i - 44:i + 22].replace("\n", " "))
    j = rec.find("→")
    print("after recovery:                   ", rec[j - 44:j + 16].replace("\n", " "))
    print("the model's reply defines the method and Sieve extracts it:", compile_status(extract_sieve(r["completion"], r["entry_point"])))
    for line in on_statement_examples(extract_sieve(r["completion"], r["entry_point"]), r["entry_point"], k[0]):
        print("  " + line)
    print("stored Sieve verdict (official tests):", pf(NOW[k]))
    tab = {}
    for r2 in CAPABLE:
        key = (cp437_state(r2["prompt_text"]), cp437_state(r2["completion"]))
        tab[key] = tab.get(key, 0) + 1
    print("the 1,336 stored records, (prompt_text state, completion state) -> count:")
    for key in sorted(tab):
        print(f"  {key}: {tab[key]}")
    if not os.path.exists(PROGRESS_PATH):
        print("(<repo>/data/politeness_tax_lcb/lcb_v3_progress.json not present; comparison with the collection-time checkpoint skipped)")
        return
    prog = {(v["task_id"], v["model"], v["level"]): v for v in json.load(open(PROGRESS_PATH, encoding="utf-8")).values()}
    o = prog[k]
    print("collection-time checkpoint (lcb_v3_progress.json), same record, same timestamp:", o["timestamp"] == r["timestamp"])
    print("  checkpoint prompt_text, first line:", o["prompt_text"].split("\n")[0])
    print("  checkpoint prompt_text == recovered stored prompt_text:", o["prompt_text"] == rec)
    n = same = recovered = other = 0; by = {}
    for r2 in ALL:
        o2 = prog.get((r2["task_id"], r2["model"], r2["level"]))
        if o2 is None:
            continue
        n += 1
        for f in ("prompt_text", "completion"):
            if r2[f] == o2[f]:
                same += 1
            elif r2[f].encode("cp437", "replace").decode("utf-8", "replace") == o2[f]:
                recovered += 1; by[f] = by.get(f, 0) + 1
            else:
                other += 1
    print(f"  records present in both files: {n} (same timestamps: "
          f"{sum(1 for r2 in ALL if (kk := (r2['task_id'], r2['model'], r2['level'])) in prog and prog[kk]['timestamp'] == r2['timestamp'])})")
    print(f"  fields identical: {same}; fields that differ and become identical after cp437->utf-8 recovery: {recovered} {by}; "
          f"fields that differ otherwise: {other}")
    print("  garbled fields in the checkpoint:", sum(1 for v in prog.values() for f in ("prompt_text", "completion") if cp437_state(v[f]) == "garbled"))
    a = sum(1 for r2 in CAPABLE if cp437_state(r2["prompt_text"]) == "ascii" and cp437_state(r2["completion"]) == "garbled")
    print(f"  stored replies in the 1,336 that are garbled although their prompt is pure ASCII: {a}")


SECTIONS = {1: p1, 2: p2, 4: p4, 5: p5, 6: p6, 9: p9, 10: p10, 11: p11, 12: p12, 14: p13_14, 15: p15}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    slow = "--slow" in sys.argv
    try:
        import he_phase                                        # pitfalls 3, 7, 8, 13 (HumanEval-phase stored data)
        SECTIONS.update(he_phase.SECTIONS)
    except Exception as ex:                                    # noqa: BLE001
        print(f"(HumanEval-phase sections 3, 7, 8, 13 unavailable: {type(ex).__name__}: {ex})")
    want = [int(a) for a in args] or sorted(SECTIONS)
    print("Stored replies:", len(ALL), "| capable-model slice:", len(CAPABLE), "| python", sys.version.split()[0])
    for n in want:
        if n not in SECTIONS:
            print(f"\n(no runnable section for pitfall {n})"); continue
        if n == 11:
            p11(slow)
        else:
            SECTIONS[n]()
