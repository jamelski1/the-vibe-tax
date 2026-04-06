"""
Run HumanEval unit tests against all 750 model completions.

Reads all_model_responses.json, pairs each completion with the original
HumanEval problem's prompt and test suite, executes in a sandboxed
subprocess, and saves pass/fail results.

Works on both Windows and Linux (no signal.SIGALRM dependency).

Usage:
    python run_tests.py

Output:
    test_results.json       — per-completion pass/fail results
    test_results_stats.json — aggregate stats by model, level, category
"""

import json
import logging
import multiprocessing
import os
import sys
import textwrap
import time
from datetime import datetime

# Add project root to path so we can import human_eval
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from human_eval.data import read_problems


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_FILE = os.path.join(SCRIPT_DIR, "all_model_responses.json")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "test_results.json")
STATS_FILE = os.path.join(SCRIPT_DIR, "test_results_stats.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "run_tests.log")

TIMEOUT = 10.0  # seconds per test
N_WORKERS = 4   # parallel workers


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    logger = logging.getLogger("test_runner")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# Execution (Windows-compatible)
# ---------------------------------------------------------------------------

def _worker(prompt, comp, test, ep, result_list):
    """Run a completion against its test suite (runs in child process).

    Tries multiple strategies:
    1. prompt + cleaned body (standard HumanEval approach)
    2. completion as standalone code + test (if model returned full function)
    """
    strategies = []

    # Strategy 1: prompt + completion as body
    strategies.append(prompt + comp + "\n" + test + "\n" + f"check({ep})")

    # Strategy 2: completion as standalone (model may have returned full function)
    # Extract any imports from the prompt and prepend them
    prompt_lines = prompt.split("\n")
    imports = "\n".join(
        line for line in prompt_lines
        if line.startswith("from ") or line.startswith("import ")
    )
    standalone = imports + "\n\n" + comp + "\n" + test + "\n" + f"check({ep})"
    strategies.append(standalone)

    for check_program in strategies:
        try:
            exec_globals = {}
            exec(check_program, exec_globals)
            result_list.append("passed")
            return
        except Exception:
            continue

    # All strategies failed — run the first one again to capture the error
    try:
        exec_globals = {}
        exec(strategies[0], exec_globals)
        result_list.append("passed")
    except Exception as e:
        result_list.append(f"failed: {e}")


def _execute_completion(problem_prompt, completion, test_code, entry_point, timeout):
    """
    Execute a completion against its test suite in a separate process.
    Returns (passed: bool, result: str).
    """
    manager = multiprocessing.Manager()
    result_list = manager.list()

    p = multiprocessing.Process(
        target=_worker,
        args=(problem_prompt, completion, test_code, entry_point, result_list),
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=5)
        return False, "timed out"

    if not result_list:
        return False, "timed out (no result)"

    outcome = result_list[0]
    return outcome == "passed", outcome


# ---------------------------------------------------------------------------
# Completion cleaning
# ---------------------------------------------------------------------------

def _remove_markdown_fences(text):
    """Strip markdown code fences (```python ... ```) from text."""
    lines = text.split("\n")
    result = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                continue
            else:
                in_fence = True
                continue
        if not in_fence or True:
            # Keep lines whether inside or outside fences
            # (we just skip the fence markers themselves)
            result.append(line)
    # Actually simpler: just strip leading/trailing fences
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()
        text = text[:text.rfind("```")]
    return text.strip()


def _find_function_body(text, entry_point):
    """
    If the text contains a function definition for entry_point,
    extract just the function body (after signature and docstring).
    Returns (body, full_text_is_function).
    """
    lines = text.split("\n")

    # Find the def line for our target function
    def_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"def {entry_point}(") or stripped.startswith(f"def {entry_point} ("):
            def_idx = i
            break

    if def_idx is None:
        return text, False

    # Find where the body starts (after def line + optional docstring)
    body_start = def_idx + 1

    # Skip docstring if present
    if body_start < len(lines):
        stripped = lines[body_start].strip()
        for quote in ['"""', "'''"]:
            if stripped.startswith(quote):
                if stripped.count(quote) >= 2 and stripped.endswith(quote) and len(stripped) > 6:
                    # Single-line docstring
                    body_start += 1
                else:
                    # Multi-line docstring — find the closing quotes
                    for j in range(body_start + 1, len(lines)):
                        if quote in lines[j]:
                            body_start = j + 1
                            break
                    else:
                        # No closing quotes found, give up
                        return text, False
                break

    body = "\n".join(lines[body_start:])
    return body, True


def clean_completion(completion_text, entry_point=""):
    """
    Clean up a model completion so it can be appended after the HumanEval prompt.

    Models return varying formats:
    - Just the function body (ideal)
    - Full function with def + docstring
    - Markdown-wrapped code
    - Imports + full function
    - Explanatory text mixed in

    This function does its best to extract just the function body.
    The _worker function also has a fallback strategy that tries
    running the completion as standalone code.
    """
    if completion_text is None:
        return "    pass\n"

    text = completion_text.strip()
    if not text:
        return "    pass\n"

    # Step 1: Remove markdown code fences
    text = _remove_markdown_fences(text)

    # Step 2: Remove any leading explanatory text before code
    # Look for the first line that looks like code
    lines = text.split("\n")
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("def ")
                or stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("return ")
                or stripped.startswith("if ")
                or stripped.startswith("for ")
                or stripped.startswith("while ")
                or stripped.startswith("try:")
                or stripped.startswith("#")
                or "=" in stripped
                or stripped == ""
                or stripped[0:1].isspace()):
            code_start = i
            break
    text = "\n".join(lines[code_start:])

    # Step 3: Try to extract function body if it contains a def for entry_point
    if entry_point:
        body, was_function = _find_function_body(text, entry_point)
        if was_function:
            text = body

    # Step 4: Normalize indentation to 4 spaces
    # The HumanEval prompt ends with the docstring inside a function def.
    # The completion body must be indented with exactly 4 spaces.
    lines = text.split("\n")

    # Find the indentation of the first non-empty line
    first_indent = None
    for line in lines:
        if line.strip():
            first_indent = len(line) - len(line.lstrip())
            break

    if first_indent is not None:
        if first_indent == 0:
            # Not indented at all — add 4 spaces
            text = textwrap.indent(text, "    ")
        elif first_indent != 4:
            # Wrong indentation — re-indent to 4 spaces
            reindented = []
            for line in lines:
                if not line.strip():
                    reindented.append("")
                else:
                    current_indent = len(line) - len(line.lstrip())
                    # Calculate relative indent from first line, then base at 4
                    relative = current_indent - first_indent
                    new_indent = 4 + max(0, relative)
                    reindented.append(" " * new_indent + line.lstrip())
            text = "\n".join(reindented)

    # Step 5: Ensure it ends with newline
    if not text.endswith("\n"):
        text += "\n"

    return text


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

CATEGORIES = {
    "String Manipulation & Pattern Matching": [1, 6, 10, 22, 28, 41, 43, 58, 67, 153],
    "List Processing & Transformations": [0, 4, 9, 18, 25, 33, 34, 35, 54, 68],
    "Mathematical & Numerical": [13, 30, 32, 37, 48, 49, 60, 63, 66, 157],
    "Logic & Conditionals": [24, 36, 39, 40, 42, 55, 64, 75, 77, 80],
    "Data Structures & Comparison": [52, 59, 65, 71, 78, 96, 152, 155, 158, 163],
}

PROBLEM_TO_CATEGORY = {}
for cat, nums in CATEGORIES.items():
    for num in nums:
        PROBLEM_TO_CATEGORY[num] = cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("=" * 60)
    log.info("VIBE TAX — Test Runner Started")
    log.info("=" * 60)

    # Load model responses
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        responses = json.load(f)
    log.info("Loaded %d completions from %s", len(responses), RESPONSES_FILE)

    # Load original HumanEval problems (for prompts and tests)
    problems = read_problems()
    log.info("Loaded %d HumanEval problems", len(problems))

    results = []
    total = len(responses)
    passed_count = 0
    failed_count = 0
    error_count = 0
    start_time = datetime.now()

    for i, resp in enumerate(responses):
        task_id = resp["task_id"]
        model = resp["model"]
        level = resp["level"]
        entry_point = resp["entry_point"]

        # Get original problem
        if task_id not in problems:
            log.error("Problem %s not found in HumanEval dataset!", task_id)
            continue

        problem = problems[task_id]
        prompt = problem["prompt"]
        test_code = problem["test"]

        # Clean the completion
        completion = clean_completion(resp.get("completion"), entry_point)

        # Run the test
        test_start = time.time()
        passed, outcome = _execute_completion(prompt, completion, test_code, entry_point, TIMEOUT)
        elapsed = time.time() - test_start

        result_entry = {
            "task_id": task_id,
            "problem_number": resp["problem_number"],
            "entry_point": entry_point,
            "level": level,
            "model": model,
            "passed": passed,
            "result": outcome,
            "elapsed_seconds": round(elapsed, 2),
            "category": PROBLEM_TO_CATEGORY.get(resp["problem_number"], "Unknown"),
        }
        results.append(result_entry)

        if passed:
            passed_count += 1
            status = "PASS"
        elif outcome.startswith("timed out"):
            error_count += 1
            status = "TIMEOUT"
        else:
            failed_count += 1
            status = "FAIL"

        log.info(
            "[%d/%d] %-7s %s | %-18s | %-10s (%.1fs) %s",
            i + 1, total, status, task_id, level, model, elapsed,
            "" if passed else f"-- {outcome[:80]}"
        )

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            log.info(">>> CHECKPOINT: saving %d results...", len(results))
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Save final results
    # ---------------------------------------------------------------------------
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed_total = datetime.now() - start_time

    # ---------------------------------------------------------------------------
    # Build stats
    # ---------------------------------------------------------------------------
    stats_by_model = {}
    stats_by_level = {}
    stats_by_category = {}
    stats_by_model_level = {}

    for r in results:
        model = r["model"]
        level = r["level"]
        category = r["category"]
        p = 1 if r["passed"] else 0

        # By model
        if model not in stats_by_model:
            stats_by_model[model] = {"passed": 0, "failed": 0, "total": 0}
        stats_by_model[model]["total"] += 1
        stats_by_model[model]["passed"] += p
        stats_by_model[model]["failed"] += 1 - p

        # By level
        if level not in stats_by_level:
            stats_by_level[level] = {"passed": 0, "failed": 0, "total": 0}
        stats_by_level[level]["total"] += 1
        stats_by_level[level]["passed"] += p
        stats_by_level[level]["failed"] += 1 - p

        # By category
        if category not in stats_by_category:
            stats_by_category[category] = {"passed": 0, "failed": 0, "total": 0}
        stats_by_category[category]["total"] += 1
        stats_by_category[category]["passed"] += p
        stats_by_category[category]["failed"] += 1 - p

        # By model + level combo
        key = f"{model}|{level}"
        if key not in stats_by_model_level:
            stats_by_model_level[key] = {"model": model, "level": level, "passed": 0, "failed": 0, "total": 0}
        stats_by_model_level[key]["total"] += 1
        stats_by_model_level[key]["passed"] += p
        stats_by_model_level[key]["failed"] += 1 - p

    # Add pass rates
    for d in [stats_by_model, stats_by_level, stats_by_category]:
        for v in d.values():
            v["pass_rate"] = f"{v['passed'] / v['total'] * 100:.1f}%" if v["total"] else "N/A"

    for v in stats_by_model_level.values():
        v["pass_rate"] = f"{v['passed'] / v['total'] * 100:.1f}%" if v["total"] else "N/A"

    stats = {
        "run_timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed_total.total_seconds(),
        "elapsed_readable": str(elapsed_total),
        "total_tests": len(results),
        "total_passed": passed_count,
        "total_failed": failed_count,
        "total_timed_out": error_count,
        "overall_pass_rate": f"{passed_count / len(results) * 100:.1f}%" if results else "N/A",
        "by_model": stats_by_model,
        "by_level": stats_by_level,
        "by_category": stats_by_category,
        "by_model_and_level": list(stats_by_model_level.values()),
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    log.info("=" * 60)
    log.info("TEST RUN COMPLETE")
    log.info("=" * 60)
    log.info("Elapsed: %s", elapsed_total)
    log.info("Total: %d tests | %d passed | %d failed | %d timed out",
             len(results), passed_count, failed_count, error_count)
    log.info("Overall pass rate: %s", stats["overall_pass_rate"])
    log.info("")

    log.info("By model:")
    for m, s in stats_by_model.items():
        log.info("  %-12s %d/%d passed (%s)", m, s["passed"], s["total"], s["pass_rate"])
    log.info("")

    log.info("By level:")
    for lv in ["level_1_formal", "level_2", "level_3", "level_4", "level_5_vibe"]:
        if lv in stats_by_level:
            s = stats_by_level[lv]
            log.info("  %-18s %d/%d passed (%s)", lv, s["passed"], s["total"], s["pass_rate"])
    log.info("")

    log.info("By category:")
    for cat, s in stats_by_category.items():
        log.info("  %-45s %d/%d passed (%s)", cat, s["passed"], s["total"], s["pass_rate"])
    log.info("")

    log.info("By model + level:")
    for entry in sorted(stats_by_model_level.values(), key=lambda x: (x["model"], x["level"])):
        log.info("  %-12s %-18s %d/%d passed (%s)",
                 entry["model"], entry["level"], entry["passed"], entry["total"], entry["pass_rate"])
    log.info("")

    log.info("Results saved to: %s", RESULTS_FILE)
    log.info("Stats saved to:   %s", STATS_FILE)
    log.info("Log saved to:     %s", LOG_FILE)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run()
