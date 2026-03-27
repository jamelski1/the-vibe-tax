"""
Validate model responses against HumanEval unit tests.

Loads completions from all_model_responses.json, combines each with the
original function signature and HumanEval test harness, executes in a
subprocess, and reports pass/fail.

Usage:
    python validate_responses.py
"""

import json
import os
import re
import subprocess
import sys
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_FILE = os.path.join(SCRIPT_DIR, "all_model_responses.json")
HUMANEVAL_FILE = os.path.join(SCRIPT_DIR, "human-eval-v2-20210705.jsonl")
VIBE_DATA_FILE = os.path.join(SCRIPT_DIR, "vibe_spectrum_data.json")

TIMEOUT_SECONDS = 10


def load_humaneval_tests():
    """Load HumanEval problems into a dict keyed by task_id."""
    problems = {}
    with open(HUMANEVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            problems[p["task_id"]] = p
    return problems


def load_vibe_data():
    """Load vibe spectrum data for function signatures at each level."""
    with open(VIBE_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["task_id"]: entry for entry in data}


def strip_markdown_fences(code):
    """Remove markdown code fences if the model wrapped its response."""
    code = code.strip()
    # Remove opening ```python or ``` and closing ```
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code


def build_test_script(prompt, completion, test_code, entry_point):
    """Build a complete Python script that defines the function and runs tests.

    The HumanEval prompt contains the function signature + docstring.
    The completion contains the function body.
    The test_code contains check(candidate) assertions.
    """
    completion = strip_markdown_fences(completion)

    # The prompt already has the function signature and docstring.
    # The completion should be the body. We need to indent it properly
    # if it isn't already indented.
    lines = completion.split("\n")

    # Check if the completion includes the full function def (some models do this)
    has_def = any(line.strip().startswith(f"def {entry_point}") for line in lines)

    if has_def:
        # Model returned full function — use it directly
        # But we still need any helper functions from the prompt (like is_palindrome)
        # Extract everything before the main function def from the prompt
        prompt_lines = prompt.split("\n")
        preamble_lines = []
        for pl in prompt_lines:
            if pl.strip().startswith(f"def {entry_point}"):
                break
            preamble_lines.append(pl)
        preamble = "\n".join(preamble_lines)
        function_code = preamble + "\n" + completion
    else:
        # Completion is just the body — needs to be indented under the prompt
        # Check if lines are already indented
        non_empty = [l for l in lines if l.strip()]
        if non_empty and not non_empty[0].startswith(("    ", "\t")):
            # Indent the body
            completion = textwrap.indent(completion, "    ")
        function_code = prompt + completion

    script = f"""{function_code}

{test_code}

check({entry_point})
print("PASSED")
"""
    return script


def run_test(script):
    """Execute a test script in a subprocess. Returns (passed, error_msg)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and "PASSED" in result.stdout:
            return True, None
        error = result.stderr.strip() or result.stdout.strip()
        # Truncate long errors
        if len(error) > 500:
            error = error[:500] + "..."
        return False, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    # Load data
    if not os.path.exists(RESPONSES_FILE):
        print(f"ERROR: {RESPONSES_FILE} not found. Run query_all_models.py first.")
        sys.exit(1)

    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)

    humaneval = load_humaneval_tests()
    vibe_data = load_vibe_data()

    print(f"Loaded {len(responses)} responses to validate")
    print(f"Loaded {len(humaneval)} HumanEval problems")
    print("=" * 70)

    # Track results
    results = []
    summary = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})

    for i, resp in enumerate(responses):
        task_id = resp["task_id"]
        level = resp["level"]
        model = resp["model"]
        completion = resp.get("completion")
        label = f"{task_id} | {level} | {model}"

        # Skip entries with errors (no completion)
        if completion is None:
            summary[(model, level)]["skipped"] += 1
            results.append({**resp, "test_passed": None, "test_error": resp.get("error")})
            print(f"  SKIP  {label} (query error)")
            continue

        # Get the HumanEval test
        if task_id not in humaneval:
            print(f"  SKIP  {label} (no HumanEval entry)")
            continue

        he = humaneval[task_id]
        vibe = vibe_data.get(task_id, {})
        entry_point = he["entry_point"]
        test_code = he["test"]

        # Use the prompt from the vibe data (which has the right formality level)
        # For level_1_formal, this includes the full function signature
        prompt = vibe.get(level, he["prompt"])

        # If the prompt doesn't contain a def line, use the original HumanEval prompt
        if f"def {entry_point}" not in prompt:
            prompt = he["prompt"]

        script = build_test_script(prompt, completion, test_code, entry_point)
        passed, error = run_test(script)

        results.append({**resp, "test_passed": passed, "test_error": error})

        if passed:
            summary[(model, level)]["passed"] += 1
            status = "PASS"
        else:
            summary[(model, level)]["failed"] += 1
            status = "FAIL"

        print(f"  {status}  {label}" + (f"  -- {error[:80]}" if error else ""))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    levels = ["level_1_formal", "level_2", "level_3", "level_4", "level_5_vibe"]
    models = sorted(set(r["model"] for r in responses))

    # Per model x level
    print(f"\n{'Model':<12} {'Level':<18} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Rate':>8}")
    print("-" * 60)
    for model in models:
        for level in levels:
            s = summary[(model, level)]
            total = s["passed"] + s["failed"]
            rate = f"{s['passed']/total*100:.1f}%" if total > 0 else "N/A"
            print(f"{model:<12} {level:<18} {s['passed']:>6} {s['failed']:>6} {s['skipped']:>6} {rate:>8}")
        print()

    # Per model overall
    print(f"\n{'Model':<12} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Rate':>8}")
    print("-" * 45)
    for model in models:
        p = sum(summary[(model, l)]["passed"] for l in levels)
        f_ = sum(summary[(model, l)]["failed"] for l in levels)
        sk = sum(summary[(model, l)]["skipped"] for l in levels)
        total = p + f_
        rate = f"{p/total*100:.1f}%" if total > 0 else "N/A"
        print(f"{model:<12} {p:>6} {f_:>6} {sk:>6} {rate:>8}")

    # Save detailed results
    output_file = os.path.join(SCRIPT_DIR, "validation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
