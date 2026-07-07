"""
Generate the Vibe Tax v2 experiment prompt set.

Turns the 50 HumanEval problems + the 5-level docstring rewrites
(data/HumanEval.jsonl/vibe_spectrum_data.json) into SIX experiment conditions
that match how real users actually prompt, per the empirical Vibe Spectrum
(data/real_prompts/VIBE_SPECTRUM.md):

  Agentic-CLI arm (terse, no pasting, context lives in the "file"):
    A1 agentic_terse      minimal ask over the vibe-level docstring   (models the 87% casual/vague mode)
    A2 agentic_casual     lowercase casual one-liner + L3 docstring   (modal agentic prompt)

  Web-chat arm (verbose, conversational, paste-heavy):
    W1 webchat_detailed   polite conversational chat + L1 formal spec (45% detailed/formal mode)
    W2 webchat_code_paste "I wrote this, it's not working" + BUGGY code        (28% of WildChat)
    W3 webchat_error_paste buggy code + REAL captured traceback                 (12% of WildChat)
    W4 webchat_multilingual Chinese instruction wrapper, English code (code-switching; 42% of
                            WildChat is non-English, Chinese the largest group)

W2/W3 use genuinely broken code: we mutate the HumanEval canonical solution
with small realistic bug operators (flipped comparison, off-by-one, and/or swap,
index shift), execute it against the official tests in a subprocess, and keep
the mutation only if it actually fails — capturing the REAL error output for
the prompt. No LLM is needed to build any of this.

Output: vibe_tax_prompts.json — one entry per (problem x available condition),
with the exact user prompt to send, plus generation stats printed at the end.

Usage:
    python generate_experiment_prompts.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRUM_FILE = os.path.join(SCRIPT_DIR, "..", "HumanEval.jsonl", "vibe_spectrum_data.json")
HUMANEVAL_FILE = os.path.join(SCRIPT_DIR, "..", "HumanEval.jsonl", "human-eval-v2-20210705.jsonl")
OUT_FILE = os.path.join(SCRIPT_DIR, "vibe_tax_prompts.json")

EXEC_TIMEOUT = 10  # seconds per mutated-solution test run
MAX_ERROR_LINES = 14  # tail of the traceback included in error-paste prompts


# ---------------------------------------------------------------------------
# Bug injection
# ---------------------------------------------------------------------------
# Each operator is (name, pattern, replacement, count). Applied to the SOLUTION
# BODY only (never the signature/docstring), first match only, in order —
# the first mutation that (a) changes the code, (b) still parses, and
# (c) FAILS the official tests is kept.

MUTATIONS = [
    ("comparison_flip_lt", r"(?<![<>=!])<(?![<=])", ">=", 1),
    ("comparison_flip_gt", r"(?<![<>=!])>(?![>=])", "<=", 1),
    ("comparison_flip_le", r"<=", ">", 1),
    ("comparison_flip_ge", r">=", "<", 1),
    ("equality_flip", r"==", "!=", 1),
    ("off_by_one_plus", r"\+ 1\b", "- 1", 1),
    ("off_by_one_minus", r"- 1\b", "+ 1", 1),
    ("bool_and_or", r"\band\b", "or", 1),
    ("bool_or_and", r"\bor\b", "and", 1),
    ("index_zero_one", r"\[0\]", "[1]", 1),
    ("range_start", r"range\(len\(", "range(1, len(", 1),
    ("true_false", r"\bTrue\b", "False", 1),
]


def run_program(program: str):
    """Run a candidate program in a subprocess; return (returncode, stderr)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(program)
        path = tf.name
    try:
        r = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=EXEC_TIMEOUT
        )
        return r.returncode, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "TimeoutExpired: execution exceeded {}s".format(EXEC_TIMEOUT)
    finally:
        os.unlink(path)


def scrub_paths(stderr: str) -> str:
    """Replace the tempfile path with a plausible user filename, like a real paste."""
    return re.sub(r'File "[^"]*"', 'File "solution.py"', stderr)


def make_buggy(problem):
    """Mutate the canonical solution until the official tests fail.

    Returns (buggy_full_function, error_tail, mutation_name) or None.
    """
    prompt, solution = problem["prompt"], problem["canonical_solution"]
    test, entry = problem["test"], problem["entry_point"]

    # sanity: canonical must pass
    ok_code, _ = run_program(prompt + solution + "\n" + test + f"\ncheck({entry})\n")
    if ok_code != 0:
        return None

    for name, pat, repl, count in MUTATIONS:
        mutated = re.sub(pat, repl, solution, count=count)
        if mutated == solution:
            continue
        program = prompt + mutated + "\n" + test + f"\ncheck({entry})\n"
        try:
            compile(program, "<candidate>", "exec")
        except SyntaxError:
            continue
        rc, stderr = run_program(program)
        if rc == 0 or not stderr.strip():
            continue  # mutation didn't break it (or broke silently)
        error_tail = "\n".join(scrub_paths(stderr).strip().splitlines()[-MAX_ERROR_LINES:])
        return prompt + mutated, error_tail, name
    return None


# ---------------------------------------------------------------------------
# Prompt framings — phrasing patterned on real corpus prompts
# ---------------------------------------------------------------------------

def signature_block(entry, level_text):
    """level_1_formal already contains imports+signature+docstring; other levels
    are bare docstrings, so wrap them under the real signature."""
    return level_text


def extract_sig(level_1_formal):
    lines = level_1_formal.split("\n")
    imports = [l for l in lines if l.startswith(("from ", "import "))]
    sig = next((l.rstrip() for l in lines if l.strip().startswith("def ")), "")
    return ("\n".join(imports) + "\n\n" if imports else "") + sig


def build_conditions(entry_spec, problem, buggy):
    """Return {condition: user_prompt} for one problem."""
    entry = entry_spec["entry_point"]
    sig = extract_sig(entry_spec["level_1_formal"])
    l1 = entry_spec["level_1_formal"]
    l3 = entry_spec["level_3"]
    l5 = entry_spec["level_5_vibe"]

    conds = {}

    # A1 — agentic terse: context (signature + vibe docstring) is "the file";
    # the human ask is minimal, like "finish this" / "implement this".
    conds["agentic_terse"] = (
        f"{sig}\n    {l5}\n\nfinish this"
    )

    # A2 — agentic casual: lowercase, no punctuation ceremony, references the file.
    conds["agentic_casual"] = (
        f"can you implement {entry}? this is whats in the file\n\n"
        f"{sig}\n    {l3}\n\n"
        f"just write the body"
    )

    # W1 — web-chat detailed: polite, conversational, full formal spec.
    conds["webchat_detailed"] = (
        f"Hello! I'm working on a Python project and I need help implementing a "
        f"function. Here is the full specification:\n\n{l1}\n\n"
        f"Could you please write the implementation for me? Thank you!"
    )

    # W2/W3 need a working bug.
    if buggy is not None:
        buggy_code, error_tail, _ = buggy
        conds["webchat_code_paste"] = (
            f"I wrote this function but it's not working correctly and I can't "
            f"figure out why. Can you fix it?\n\n{buggy_code}\n"
            f"Please give me the corrected version."
        )
        conds["webchat_error_paste"] = (
            f"my code keeps failing and i don't understand this error:\n\n"
            f"{error_tail}\n\n"
            f"here's my code:\n\n{buggy_code}\n"
            f"how do i fix this??"
        )

    # W4 — multilingual code-switching: Chinese instructions, English code/spec
    # (the dominant observed pattern: non-English wrapper around English code).
    conds["webchat_multilingual"] = (
        f"帮我实现这个 Python 函数：\n\n{sig}\n    {l3}\n\n"
        f"只需要写函数体，不要解释，谢谢。"
    )

    return conds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    with open(SPECTRUM_FILE, encoding="utf-8") as f:
        spectrum = json.load(f)
    problems = {}
    with open(HUMANEVAL_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            problems[p["task_id"]] = p

    out = []
    bug_ok = 0
    mutation_used = {}
    for i, entry in enumerate(spectrum):
        task_id = entry["task_id"]
        problem = problems[task_id]
        buggy = make_buggy(problem)
        if buggy:
            bug_ok += 1
            mutation_used[buggy[2]] = mutation_used.get(buggy[2], 0) + 1
        conds = build_conditions(entry, problem, buggy)
        for cond, prompt_text in conds.items():
            out.append({
                "task_id": task_id,
                "problem_number": entry["problem_number"],
                "entry_point": entry["entry_point"],
                "condition": cond,
                "medium": "agentic" if cond.startswith("agentic") else "webchat",
                "prompt": prompt_text,
            })
        print(f"[{i+1}/{len(spectrum)}] {task_id}: {len(conds)} conditions"
              + ("" if buggy else "  (no viable bug — paste conditions skipped)"))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    from collections import Counter
    by_cond = Counter(x["condition"] for x in out)
    print("=" * 60)
    print(f"problems             : {len(spectrum)}")
    print(f"with injected bug    : {bug_ok}")
    print(f"total prompts        : {len(out)}")
    for c, n in sorted(by_cond.items()):
        print(f"  {c:22s} {n}")
    print(f"mutations used       : {dict(mutation_used)}")
    print(f"output -> {OUT_FILE}")


if __name__ == "__main__":
    run()
