"""
Generate the 4-condition HumanEval prompt set ("HE4") using the SAME
deterministic framing wrappers as the LiveCodeBench port
(data/vibe_tax_lcb/generate_lcb_prompts.py :: _mock).

WHY THIS EXISTS (advisor request): the LCB experiment varies only the *wrapper*
around an identical problem statement — a clean minimal pair. The original v2/v3
HumanEval conditions did NOT: terse used the L5 "vibe" docstring, casual used L3,
detailed used the L1 formal spec, so framing AND specification detail both moved
at once (confounded). To compare HumanEval against LCB apples-to-apples, we run
HumanEval with the *same four wrappers* around an identical, held-constant
problem body — so the only thing that changes between conditions is the framing.

The four conditions mirror LCB exactly:
  agentic_terse         "need `f(params)` for this: <problem>"
  agentic_casual        "hey can you solve this? need a `f` function <problem>"
  webchat_detailed      "Hi! Could you help… I need a function `f(params)`. <problem> Thank you!"
  webchat_multilingual  Chinese framing, English problem + function name

The held-constant "problem" is the canonical HumanEval prompt (imports +
signature + docstring) — the irreducible spec, exactly like LCB holds the full
problem statement constant. The function name is always named explicitly so the
completion stays gradeable by the existing scorer (score_vibe_tax.py).

Output schema matches run_vibe_tax.py, so the SHARED runner + scorer just work:
    python generate_he4_prompts.py
    # then, locally with keys (PowerShell):
    $env:PROMPTS_FILE  = "he4_prompts.json"
    $env:OUTPUT_FILE   = "he4_responses.json"
    $env:PROGRESS_FILE = "he4_progress.json"
    $env:STATS_FILE    = "he4_run_stats.json"
    python run_vibe_tax.py
    # then score (base HumanEval tests):
    $env:RESPONSES_FILE = "he4_responses.json"
    $env:SCORED_FILE    = "he4_scored.json"
    $env:STATS_FILE     = "he4_scored_stats.json"
    python score_vibe_tax.py
    # optional edge-case tests:
    $env:RESPONSES_FILE = "he4_responses.json"; python score_vibe_tax_plus.py
"""

import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRUM_FILE = os.path.join(SCRIPT_DIR, "..", "HumanEval.jsonl", "vibe_spectrum_data.json")
HUMANEVAL_FILE = os.path.join(SCRIPT_DIR, "..", "HumanEval.jsonl", "human-eval-v2-20210705.jsonl")
OUT_FILE = os.path.join(SCRIPT_DIR, "he4_prompts.json")


def parse_params(prompt, entry):
    """Pull the parameter list from the `def entry(...)` signature."""
    m = re.search(r"def\s+" + re.escape(entry) + r"\s*\(([^)]*)\)", prompt)
    if not m:
        return ""
    parts = [p.split(":")[0].split("=")[0].strip() for p in m.group(1).split(",")]
    return ", ".join(p for p in parts if p and p != "self")


def wrappers(entry, params, problem):
    """The four deterministic wrappers — identical in construction to LCB's
    _mock(), adapted from `Solution.method` to a free function. Only the wrapper
    text differs between conditions; `problem` is byte-for-byte identical."""
    return {
        "agentic_terse":
            f"need `{entry}({params})` for this:\n\n{problem}",
        "agentic_casual":
            f"hey can you solve this? need a `{entry}` function\n\n{problem}",
        "webchat_detailed":
            f"Hi! Could you help me solve this problem? I need a function "
            f"`{entry}({params})`.\n\n{problem}\n\nThank you!",
        "webchat_multilingual":
            f"帮我解决这个问题，请实现 `{entry}({params})` 函数：\n\n{problem}",
    }


def run():
    with open(SPECTRUM_FILE, encoding="utf-8") as f:
        spectrum = json.load(f)          # the 50-problem subset (task_id, problem_number, entry_point)
    problems = {}
    with open(HUMANEVAL_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            problems[p["task_id"]] = p

    out = []
    for entry in spectrum:
        task_id = entry["task_id"]
        prob = problems[task_id]
        ep = entry["entry_point"]
        problem_body = prob["prompt"].rstrip()     # imports + signature + docstring, held constant
        params = parse_params(problem_body, ep)
        for cond, text in wrappers(ep, params, problem_body).items():
            assert ep in text, f"gradeability: {task_id}/{cond} lost the function name"
            out.append({
                "task_id": task_id,
                "problem_number": entry["problem_number"],
                "entry_point": ep,
                "condition": cond,
                "medium": "agentic" if cond.startswith("agentic") else "webchat",
                "prompt": text,
            })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    from collections import Counter
    print(f"problems : {len(spectrum)}")
    print(f"prompts  : {len(out)}  ({len(spectrum)} problems x 4 conditions)")
    for c, n in sorted(Counter(x['condition'] for x in out).items()):
        print(f"  {c:22s} {n}")
    print(f"output -> {OUT_FILE}")
    print("\nEvery condition wraps the SAME problem body; only the framing differs "
          "(matched minimal pair with the LCB experiment).")


if __name__ == "__main__":
    run()
