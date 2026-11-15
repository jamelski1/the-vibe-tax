"""
Worked-example checks for pitfalls 3, 7, 8 and 13.

These four pitfalls belong to the study's earlier HumanEval phase. They have no
instance among the 1,336 LiveCodeBench replies, so the instances are taken from
the HumanEval-phase replies stored in the research tree of this branch:

  <repo>/data/politeness_tax_v2/politeness_tax_responses.json   822 replies, six conditions
  <repo>/data/politeness_tax_v2/he4_responses.json              600 replies, four registers
  <repo>/data/politeness_tax_v2/politeness_tax_scored.json      stored verdicts, base tests
  <repo>/data/politeness_tax_v2/politeness_tax_plus_scored.json stored verdicts, HumanEval+ inputs
  <repo>/data/politeness_tax_v2/he4_scored.json
  <repo>/data/HumanEval.jsonl/human-eval-v2-20210705.jsonl      problems, tests, reference solutions

and are run through the two scorers of that phase, imported unmodified:

  <repo>/data/HumanEval.jsonl/run_tests.py                 the first scorer ("v1"): clean_completion,
                                                           then prompt + body, then standalone
  <repo>/data/politeness_tax_v2/score_politeness_tax.py    the current multi-candidate scorer

Imported by verify_examples.py; can also be run on its own:  python he_phase.py [3 7 8 13]
"""
import contextlib
import io
import json
import multiprocessing
import os
import re
import signal
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HE_DIR = os.path.join(ROOT, "data", "HumanEval.jsonl")
V2_DIR = os.path.join(ROOT, "data", "politeness_tax_v2")
if not (os.path.isdir(HE_DIR) and os.path.isdir(V2_DIR)):
    raise ImportError("the <repo>/data research tree is not present next to artifact/")
sys.path.insert(0, ROOT); sys.path.insert(0, HE_DIR); sys.path.insert(0, V2_DIR)

import run_tests as V1                      # noqa: E402
import score_politeness_tax as V2           # noqa: E402

MODEL = {"chatgpt": "GPT-5.4", "claude": "Claude Opus 4.6", "codestral": "Codestral"}
LEVEL = {"agentic_terse": "terse", "agentic_casual": "casual", "webchat_detailed": "detailed",
         "webchat_multilingual": "multilingual", "webchat_code_paste": "code-paste",
         "webchat_error_paste": "error-paste (\"fix my error\")"}

PROBLEMS = {}
for _l in open(os.path.join(HE_DIR, "human-eval-v2-20210705.jsonl"), encoding="utf-8"):
    _p = json.loads(_l); PROBLEMS[_p["task_id"]] = _p


def _load(name):
    return json.load(open(os.path.join(V2_DIR, name), encoding="utf-8"))


def _key(r):
    return (r["task_id"], r["level"], r["model"])


MAIN = {_key(r): r for r in _load("politeness_tax_responses.json")}
HE4 = {_key(r): r for r in _load("he4_responses.json")}
MAIN_BASE = {_key(r): r["passed"] for r in _load("politeness_tax_scored.json")}
MAIN_PLUS = {_key(r): r["passed"] for r in _load("politeness_tax_plus_scored.json")}
HE4_BASE = {_key(r): r["passed"] for r in _load("he4_scored.json")}


def pf(b):
    return "PASS" if b else "FAIL"


def head(n, title):
    print("\n" + "=" * 100)
    print(f"PITFALL {n}. {title}")
    print("=" * 100)


def who(r, src):
    return (f"{r['task_id']} ({r['entry_point']}) | {MODEL[r['model']]} | {LEVEL[r['level']]} | "
            f"reply: {len(r['completion'])} chars, {len(r['completion'].split(chr(10)))} lines | stored in {src}")


def _child(prog, q):
    signal.alarm(10)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(prog, {})
        q.put("passes the tests")
    except BaseException as ex:                                  # noqa: BLE001
        msg = str(ex).split(" (<string>")[0][:80]
        where = f", line {ex.lineno} of the assembled program" if isinstance(ex, SyntaxError) else ""
        q.put(f"{type(ex).__name__}{': ' + msg if msg else ''}{where}")


def run_program(prog):
    """exec a full program (code + the problem's official `check`) in a child process."""
    q = multiprocessing.Queue(); p = multiprocessing.Process(target=_child, args=(prog, q)); p.start()
    try:
        r = q.get(timeout=12)
    except Exception:
        r = "no result within 12s"
    p.join(2)
    if p.is_alive():
        p.kill()
    return r


def tail(prob):
    return "\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"


def v1_outcomes(prob, completion):
    """The first scorer, as in run_tests.py: clean_completion, then its two strategies."""
    ep = prob["entry_point"]; comp = V1.clean_completion(completion, ep)
    imports = "\n".join(l for l in prob["prompt"].split("\n") if l.startswith("from ") or l.startswith("import "))
    s1 = run_program(prob["prompt"] + comp + "\n" + prob["test"] + "\n" + f"check({ep})")
    s2 = run_program(imports + "\n\n" + comp + "\n" + prob["test"] + "\n" + f"check({ep})")
    return comp, s1, s2


def nth_function(text, entry, which):
    """The block-walk of score_politeness_tax.extract_last_function, for the which-th `def <entry>`
    (0 = first, -1 = last). Used only to show what a first-`def` extractor would have taken."""
    lines = V2.strip_fences(text).split("\n")
    starts = [i for i, l in enumerate(lines) if re.match(rf"\s*def\s+{re.escape(entry)}\s*\(", l)]
    start = starts[which]; base = len(lines[start]) - len(lines[start].lstrip()); body = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip():
            body.append(l); continue
        if len(l) - len(l.lstrip()) <= base:
            break
        body.append(l)
    return textwrap.dedent("\n".join(body)), start + 1


def p3():
    head(3, "Buggy snippet quoted first in \"fix my error\"")
    k = ("HumanEval/52", "webchat_error_paste", "chatgpt"); r = MAIN[k]; prob = PROBLEMS[k[0]]; ep = prob["entry_point"]
    print(who(r, "politeness_tax_responses.json"))
    c = r["completion"]; lines = c.split("\n")
    defs = [i + 1 for i, l in enumerate(lines) if re.match(rf"\s*def\s+{ep}\s*\(", l)]
    print(f"lines of the reply that start a `def {ep}`:", defs)
    imports = V2.problem_imports(prob["prompt"])
    for label, which in (("first", 0), ("last", -1)):
        fn, line = nth_function(c, ep, which)
        print(f"  the {label} def (reply line {line}), run standalone against the official tests: "
              f"{run_program(imports + chr(10) * 2 + fn + tail(prob))}")
    comp, s1, s2 = v1_outcomes(prob, c)
    print("first scorer (run_tests.py):")
    print("  clean_completion keeps, as the function body, text starting:", repr(comp.split("\n")[0]), "/", repr(comp.split("\n")[1]))
    print("  strategy 1, prompt + body:", s1)
    print("  strategy 2, standalone:   ", s2)
    print("current scorer (score_politeness_tax.run_one, last `def` first):", pf(V2.run_one(prob["prompt"], c, prob["test"], ep)))
    print("stored verdict, current scorer:", pf(MAIN_BASE[k]))
    n = a = b = quoted = 0
    for k2, r2 in MAIN.items():
        if k2[1] != "webchat_error_paste" or k2[2] == "codestral":
            continue
        p2 = PROBLEMS[k2[0]]; n += 1
        _, t1, t2 = v1_outcomes(p2, r2["completion"])
        a += ("passes" in t1) or ("passes" in t2)
        b += MAIN_BASE[k2]
        if len(re.findall(rf"(?m)^\s*def\s+{re.escape(p2['entry_point'])}\s*\(", r2["completion"])) >= 2:
            f0, _ = nth_function(r2["completion"], p2["entry_point"], 0)
            f1, _ = nth_function(r2["completion"], p2["entry_point"], -1)
            im = V2.problem_imports(p2["prompt"])
            if "passes" not in run_program(im + "\n\n" + f0 + tail(p2)) and "passes" in run_program(im + "\n\n" + f1 + tail(p2)):
                quoted += 1
    print(f"all {n} error-paste replies of the two capable models: first scorer passes {a}; current scorer (stored) passes {b}")
    print(f"  of these, replies whose first `def` of the target fails the tests while the last one passes: {quoted}")


def p7():
    head(7, "Full def appended under the prompt's def")
    k = ("HumanEval/71", "agentic_casual", "chatgpt"); r = HE4[k]; prob = PROBLEMS[k[0]]; ep = prob["entry_point"]
    print(who(r, "he4_responses.json"))
    c = r["completion"]
    print("first 3 lines of the reply:", c.split("\n")[:3])
    comp, s1, s2 = v1_outcomes(prob, c)
    print("first scorer (run_tests.py): clean_completion turns the full function into a body:")
    for l in comp.rstrip("\n").split("\n"):
        print("   | " + l)
    print("  `import math` kept in the body:", "import math" in comp)
    print("  strategy 1, prompt's def + that body:", s1)
    print("  strategy 2, that body standalone:    ", s2)
    print("the reply appended under the prompt unchanged:", run_program(prob["prompt"] + V2.strip_fences(c) + tail(prob)))
    print("the reply run standalone:                     ", run_program(V2.problem_imports(prob["prompt"]) + "\n\n" + V2.strip_fences(c) + tail(prob)))
    print("current scorer (score_politeness_tax.run_one):", pf(V2.run_one(prob["prompt"], c, prob["test"], ep)))
    print("stored verdict, current scorer:", pf(HE4_BASE[k]))
    clean = both = raw_bad = v1_bad = 0
    for src in (MAIN, HE4):
        for k2, r2 in src.items():
            if k2[2] == "codestral":
                continue
            p2 = PROBLEMS[k2[0]]; body = V2.strip_fences(r2["completion"] or "")
            if not re.search(rf"(?m)^def\s+{re.escape(p2['entry_point'])}\s*\(", body):
                continue
            try:
                compile(body, "<reply>", "exec")
            except SyntaxError:
                continue                      # prose or broken code: not a clean full function
            if "passes" not in run_program(V2.problem_imports(p2["prompt"]) + "\n\n" + body + tail(p2)):
                continue
            clean += 1
            raw_ok = "passes" in run_program(p2["prompt"] + body + tail(p2))
            _, t1, t2 = v1_outcomes(p2, r2["completion"])
            raw_bad += not raw_ok
            v1_bad += not ("passes" in t1 or "passes" in t2)
    print(f"capable-model replies in the two files that are a clean, complete, correct function: {clean}")
    print(f"  fail when appended under the prompt unchanged (double signature): {raw_bad}")
    print(f"  fail under the first scorer (both strategies):                     {v1_bad}")


def p8():
    head(8, "Ragged first-line indentation")
    k = ("HumanEval/35", "agentic_casual", "claude"); r = MAIN[k]; prob = PROBLEMS[k[0]]; ep = prob["entry_point"]
    print(who(r, "politeness_tax_responses.json"))
    c = r["completion"]
    print("the reply, in full, with leading spaces counted:")
    for l in c.split("\n"):
        print(f"   {len(l) - len(l.lstrip()):2d} spaces | {l.lstrip()}")
    comp, s1, s2 = v1_outcomes(prob, c)
    print("first scorer (run_tests.py): clean_completion re-indents relative to the FIRST line:")
    for l in comp.rstrip("\n").split("\n"):
        print(f"   {len(l) - len(l.lstrip()):2d} spaces | {l.lstrip()}")
    print("  strategy 1, prompt + body:", s1)
    print("  strategy 2, standalone:   ", s2)
    nb = V2.normalized_body(c)
    print("current scorer, normalized_body lifts the short first line, then re-indents:")
    for l in nb.rstrip("\n").split("\n"):
        print(f"   {len(l) - len(l.lstrip()):2d} spaces | {l.lstrip()}")
    print("  prompt + normalized body:", run_program(prob["prompt"] + nb + tail(prob)))
    print("current scorer (score_politeness_tax.run_one):", pf(V2.run_one(prob["prompt"], c, prob["test"], ep)))
    print("stored verdict, current scorer:", pf(MAIN_BASE[k]))
    n = bad = 0; who_ = {}
    for src in (MAIN, HE4):
        for k2, r2 in src.items():
            ls = [l for l in V2.strip_fences(r2["completion"] or "").split("\n") if l.strip()]
            if len(ls) < 2 or f"def {PROBLEMS[k2[0]]['entry_point']}" in (r2["completion"] or ""):
                continue
            first = len(ls[0]) - len(ls[0].lstrip()); rest = min(len(l) - len(l.lstrip()) for l in ls[1:])
            if 0 < first < rest:
                n += 1; who_[k2[2]] = who_.get(k2[2], 0) + 1
                _, t1, t2 = v1_outcomes(PROBLEMS[k2[0]], r2["completion"])
                bad += not ("passes" in t1 or "passes" in t2)
    print(f"bare-body replies in the two files whose first line is indented less than every later line: {n} {who_}; "
          f"fail under the first scorer: {bad}")


def _value(src, entry, arg, q):
    try:
        ns = {}; exec(src, ns); q.put(repr(ns[entry](arg)))
    except BaseException as ex:                                  # noqa: BLE001
        q.put(f"raises {type(ex).__name__}: {ex}")


def value(src, entry, arg):
    q = multiprocessing.Queue(); p = multiprocessing.Process(target=_value, args=(src, entry, arg, q)); p.start()
    try:
        r = q.get(timeout=10)
    except Exception:
        r = "no result within 10s"
    p.kill(); p.join(); return r


def p13():
    head(13, "Weak tests pass subtly wrong code")
    k = ("HumanEval/55", "webchat_detailed", "chatgpt"); r = MAIN[k]; prob = PROBLEMS[k[0]]; ep = prob["entry_point"]
    print(who(r, "politeness_tax_responses.json"))
    c = r["completion"]
    print("lines 10-11 of the reply:", [l.strip() for l in c.split("\n")[9:11]])
    asserts = [l.strip() for l in prob["test"].split("\n") if l.strip().startswith("assert")]
    print(f"the official base test has {len(asserts)} assertions:", asserts)
    print("reply run against the base test:", run_program(V2.strip_fences(c) + tail(prob)))
    ref = prob["prompt"] + prob["canonical_solution"]
    for n in (0, 1, 2, 10):
        print(f"  {ep}({n}): reference solution -> {value(ref, ep, n)} | reply -> {value(V2.strip_fences(c), ep, n)}")
    print("stored verdict, base tests:      ", pf(MAIN_BASE[k]))
    print("stored verdict, HumanEval+ inputs:", pf(MAIN_PLUS[k]))
    both = [k2 for k2 in MAIN_BASE if MAIN_BASE[k2] and not MAIN_PLUS[k2]]
    cap = [k2 for k2 in both if k2[2] != "codestral"]
    print(f"stored cells that pass the base tests and fail the HumanEval+ inputs: {len(both)} of {len(MAIN_BASE)} "
          f"({len(cap)} of them from the two capable models)")


SECTIONS = {3: p3, 7: p7, 8: p8, 13: p13}

if __name__ == "__main__":
    for n in ([int(a) for a in sys.argv[1:]] or sorted(SECTIONS)):
        SECTIONS[n]()
