"""
Manually test one LLM reply against a LiveCodeBench problem's real tests.

Paste whatever the model gave you (code, or code + explanation, fences or not)
into a text file, then:

    python manual_test.py lcb/3517 reply.txt

It extracts the code exactly like the real scorer (score_lcb.py), runs it against
that problem's public + private tests, and prints PASS/FAIL per test with the
expected vs actual output so you can see WHY it failed.

Tests come from lcb_tests.jsonl (from extract_lcb_tests.py); if that's missing it
falls back to the committed lcb_tests_sample.json (only lcb/3517 and lcb/3527).
No API keys, no network.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from score_lcb import extract_solution, parse_input, parse_lit, eq, _IMPORTS  # reuse scorer logic

MAX_SHOW = 25   # cap tests shown


def load_tests():
    full = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
    if os.path.exists(full):
        return {json.loads(l)["task_id"]: json.loads(l) for l in open(full, encoding="utf-8")}
    sample = os.path.join(SCRIPT_DIR, "lcb_tests_sample.json")
    if os.path.exists(sample):
        print("(using lcb_tests_sample.json — only lcb/3517 and lcb/3527)\n")
        return {t["task_id"]: t for t in json.load(open(sample, encoding="utf-8"))}
    sys.exit("no lcb_tests.jsonl or lcb_tests_sample.json found")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python manual_test.py <task_id> [reply.txt]")
    task_id = sys.argv[1]
    reply_file = sys.argv[2] if len(sys.argv) > 2 else "reply.txt"
    if not os.path.exists(reply_file):
        sys.exit(f"paste the model's answer into '{reply_file}' first")

    tests = load_tests()
    if task_id not in tests:
        sys.exit(f"{task_id} not in the tests file (have: {list(tests)[:5]}...)")
    rec = tests[task_id]
    entry = rec["entry_point"]
    reply = open(reply_file, encoding="utf-8").read()

    code = extract_solution(reply, entry)
    if not code:
        print("!! No runnable code found in the reply (no `class Solution` / "
              f"`def {entry}`).")
        return
    print("=" * 64)
    print("EXTRACTED CODE (this is what gets run):")
    print("-" * 64)
    print(code)
    print("=" * 64)

    ns = {}
    try:
        exec(_IMPORTS + code, ns)
    except (SyntaxError, IndentationError) as e:
        print(f"!! Code does not even run (parse error): {e}")
        print("   HINT: this usually means a COPY-PASTE artifact, not a model error —")
        print("   e.g. the model put some code OUTSIDE the ``` block and the web UI")
        print("   flattened its indentation when you copied it. Ask the model to put")
        print("   ALL the code in ONE Python code block, or use the block's copy button,")
        print("   then re-paste. (In the automated pipeline the API text is exact, so")
        print("   this doesn't happen there.)")
        return
    except Exception as e:
        print(f"!! Code does not even run (runtime error at import): {e}")
        return
    sol_cls = ns.get("Solution")

    # guard: does the code actually define the method THIS problem needs?
    inst_probe = sol_cls() if sol_cls else None
    if getattr(inst_probe, entry, None) is None and ns.get(entry) is None:
        defined = [m for m in dir(inst_probe) if not m.startswith("_")] if inst_probe else \
                  [k for k, v in ns.items() if callable(v) and not k.startswith("_")]
        print(f"!! MISMATCH: {task_id} needs a method named `{entry}`, but the pasted")
        print(f"   code defines: {defined}")
        print(f"   -> You're probably testing the wrong problem for this answer, or the")
        print(f"      wrong answer for this problem. Make the task_id match the solution")
        print(f"      (tip: `python print_prompt.py {task_id} terse` shows the right problem).")
        return

    cases = rec.get("public_tests", []) + rec.get("private_tests", [])
    npass = 0
    for i, t in enumerate(cases[:MAX_SHOW]):
        args = parse_input(t["input"])
        expected = parse_lit(t["output"])
        try:
            inst = sol_cls() if sol_cls else None
            fn = getattr(inst, entry, None) or ns.get(entry)
            got = fn(*args)
            ok = eq(got, expected)
        except Exception as e:
            got = f"EXCEPTION: {type(e).__name__}: {e}"
            ok = False
        npass += ok
        print(f"test {i:2d}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"         args     = {args}")
            print(f"         expected = {expected}")
            print(f"         got      = {got}")
    shown = min(len(cases), MAX_SHOW)
    print("=" * 64)
    print(f"RESULT: {npass}/{shown} shown tests passed"
          + ("  -> PASSES (all shown)" if npass == shown else "  -> FAILS")
          + (f"   ({len(cases)} total tests; showing first {MAX_SHOW})" if len(cases) > MAX_SHOW else ""))


if __name__ == "__main__":
    main()
