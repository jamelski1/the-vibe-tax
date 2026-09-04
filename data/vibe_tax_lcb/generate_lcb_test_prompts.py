"""
Build a `with_tests` condition: the terse framing PLUS the problem's public test
cases embedded in the prompt. Answers the advisor's question — "if you give it the
unit tests, does it generate correct code, and does that depend on problem type?"

IMPORTANT (anti-cheating): the model sees only the PUBLIC tests here; score this
condition on PRIVATE tests only (`LCB_PRIVATE_ONLY=1 python score_lcb.py`) so we
measure genuine generalization, not the model hardcoding the shown outputs.

Reads lcb_problems.jsonl + lcb_tests.jsonl (both local; tests are gitignored).
Output: lcb_test_prompts.json (schema matches run_vibe_tax.py). Run LOCALLY.

Usage:
    python generate_lcb_test_prompts.py [--max-shown 6]
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")
TESTS = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
OUT = os.path.join(SCRIPT_DIR, "lcb_test_prompts.json")


def format_tests(cases, cls, entry, max_shown):
    lines = []
    for t in cases[:max_shown]:
        args = str(t.get("input", "")).replace("\n", ", ")
        lines.append(f"  {cls}().{entry}({args})  ->  {t.get('output')}")
    return "\n".join(lines)


def run(max_shown):
    problems = [json.loads(l) for l in open(PROBLEMS, encoding="utf-8")]
    if not os.path.exists(TESTS):
        raise SystemExit(f"need {TESTS} (run extract_lcb_tests.py first)")
    tests = {json.loads(l)["task_id"]: json.loads(l) for l in open(TESTS, encoding="utf-8")}

    out, skipped = [], 0
    for i, p in enumerate(problems):
        rec = tests.get(p["task_id"])
        pub = (rec or {}).get("public_tests", [])
        if not pub:
            skipped += 1
            continue
        cls = p.get("class_name") or "Solution"
        entry, params = p["entry_point"], p["params"]
        shown = format_tests(pub, cls, entry, max_shown)
        prompt = (
            f"need `{cls}.{entry}({params})` for this:\n\n"
            f"{p['question_content']}\n\n"
            f"It must pass these tests (inputs -> expected output):\n{shown}"
        )
        out.append({
            "task_id": p["task_id"], "problem_number": i,
            "entry_point": entry, "class_name": cls,
            "condition": "with_tests", "medium": "agentic",
            "difficulty": p["difficulty"], "prompt": prompt,
        })

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {len(out)} `with_tests` prompts (skipped {skipped} w/o public tests) -> {OUT}")
    print("Run these through run_vibe_tax.py, then score with LCB_PRIVATE_ONLY=1 "
          "(private tests only — the model already saw the public ones).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shown", type=int, default=6, help="max public tests to embed")
    a = ap.parse_args()
    run(a.max_shown)
