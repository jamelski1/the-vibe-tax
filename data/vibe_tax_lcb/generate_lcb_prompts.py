"""
Generate vibe-spectrum prompts for LiveCodeBench (LCB) functional problems.

Unlike HumanEval, an LCB problem statement is IRREDUCIBLE — you can't make it
"terse" by dropping the spec (it's a real competitive problem; the statement IS
the task). So the conditions here vary the FRAMING / register / language around
the (full) problem, while always preserving the required `Solution.<method>`
signature so completions stay gradeable by LCB's tests.

v1 scope: the 4 non-paste conditions (terse, casual, detailed, multilingual).
Paste conditions (code/error) need a known-correct solution to mutate — LCB has
none — so they come later via correct-then-mutate.

Backends: mock (keyless, deterministic) / anthropic / openai. The realism
discriminator score is recorded as ADVISORY only (it was trained on web-chat vs
HumanEval, so it doesn't judge leetcode-style prompts well); the hard gate is
just "method name preserved" (gradeability).

Output: lcb_v3_prompts.json (schema matches run_vibe_tax.py).

Usage:
  python generate_lcb_prompts.py --backend mock
  python generate_lcb_prompts.py --backend anthropic     # local, with keys
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "data", "real_prompts"))
from realism_discriminator import RealismDiscriminator, load_real, load_synthetic  # noqa: E402

PROBLEMS = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")
OUT = os.path.join(SCRIPT_DIR, "lcb_v3_prompts.json")

CONDITIONS = {
    "agentic_terse": ("agentic",
        "Extremely terse and lowercase, like a quick note-to-self — minimal framing, just the problem and what you need."),
    "agentic_casual": ("agentic",
        "Casual and lowercase, like messaging a teammate for help."),
    "webchat_detailed": ("webchat",
        "A polite, clear chat message asking for help."),
    "webchat_multilingual": ("webchat",
        "Write ALL of your own framing text in Chinese (keep the problem statement and the method name in English)."),
}


def build_instruction(prob, cond):
    style = CONDITIONS[cond][1]
    entry, params, cls = prob["entry_point"], prob["params"], prob.get("class_name") or "Solution"
    return (
        "Rewrite the following competitive-programming task as a REALISTIC user "
        f"message asking an assistant to solve it. Style: {style}\n\n"
        f"The solution must be a method named exactly `{entry}({params})` on a class "
        f"named `{cls}` (say so, so it can be auto-graded). Include the FULL problem "
        "statement below verbatim (it's the spec — do not summarize or drop the "
        "examples/constraints). Do NOT solve it yourself.\n\n"
        f"--- PROBLEM ---\n{prob['question_content']}\n--- END ---\n\n"
        "Return ONLY the user message."
    )


# --------------------------------------------------------------------------- backends

def make_backend(name):
    if name == "auto":
        name = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else ("openai" if os.getenv("OPENAI_API_KEY") else "mock")
    if name == "anthropic":
        from anthropic import Anthropic
        c = Anthropic(); m = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        return name, lambda instr: c.messages.create(model=m, max_tokens=2000, temperature=0.7,
                                messages=[{"role": "user", "content": instr}]).content[0].text.strip()
    if name == "openai":
        from openai import OpenAI
        c = OpenAI(); m = os.getenv("OPENAI_MODEL", "gpt-5.4")
        return name, lambda instr: c.chat.completions.create(model=m, temperature=0.7,
                                max_completion_tokens=2000, messages=[{"role": "user", "content": instr}]).choices[0].message.content.strip()
    return "mock", _mock


def _mock(instr):
    entry = re.search(r"named exactly `(\w+)", instr).group(1)
    params = re.search(r"`\w+\(([^)]*)\)`", instr).group(1)
    cls = re.search(r"class named `(\w+)`", instr).group(1)
    problem = instr.split("--- PROBLEM ---\n", 1)[1].split("\n--- END ---", 1)[0].strip()
    if "framing text in Chinese" in instr:
        return f"帮我解决这个问题，请实现 `{cls}` 类的 `{entry}({params})` 方法：\n\n{problem}"
    if "Extremely terse" in instr:
        return f"need `{cls}.{entry}({params})` for this:\n\n{problem}"
    if "Casual and lowercase" in instr:
        return f"hey can you solve this? need a `{entry}` method on a `{cls}` class\n\n{problem}"
    return (f"Hi! Could you help me solve this problem? I need a method "
            f"`{entry}({params})` on a class `{cls}`.\n\n{problem}\n\nThank you!")


# --------------------------------------------------------------------------- run

def run(backend_name):
    problems = [json.loads(l) for l in open(PROBLEMS, encoding="utf-8")]
    real = load_real("all")
    disc = RealismDiscriminator().fit(real, load_synthetic("all")) if real else None
    print(f"discriminator: {'active (advisory)' if disc else 'disabled (no real corpus)'}")
    bname, call = make_backend(backend_name)
    print(f"backend: {bname} | problems: {len(problems)}\n")

    out, dropped = [], 0
    for i, prob in enumerate(problems):
        for cond, (medium, _) in CONDITIONS.items():
            try:
                prompt = call(build_instruction(prob, cond))
            except Exception as e:
                print(f"  {prob['task_id']}/{cond}: backend error: {e}"); continue
            if prob["entry_point"] not in prompt:      # hard gate: gradeability
                dropped += 1; continue
            out.append({
                "task_id": prob["task_id"], "problem_number": i,
                "entry_point": prob["entry_point"], "class_name": prob.get("class_name") or "Solution",
                "condition": cond, "medium": medium, "difficulty": prob["difficulty"],
                "generator": "lcb_calibrated", "backend": bname,
                "reads_real": round(disc.predict_proba(prompt), 3) if disc else None,
                "prompt": prompt,
            })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(problems)} problems")

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    from collections import defaultdict
    rr = defaultdict(list)
    for x in out:
        if x["reads_real"] is not None:
            rr[x["condition"]].append(x["reads_real"])
    print("=" * 60)
    print(f"prompts: {len(out)}  (dropped {dropped} for missing method name)  -> {OUT}")
    for c, v in rr.items():
        print(f"  {c:22s} avg reads_real {sum(v)/len(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "anthropic", "openai", "mock"], default="auto")
    a = ap.parse_args()
    run(a.backend)
