"""
Print the ready-to-paste prompt(s) for a LiveCodeBench problem — the exact system
+ user text sent to the model — so you can copy them straight into ChatGPT/Claude.

    python print_prompt.py lcb/3517            # all 4 framings
    python print_prompt.py lcb/3517 terse      # just one (terse/casual/detailed/multilingual)
    python print_prompt.py --list              # list available task_ids

The user text is regenerated deterministically from lcb_problems.jsonl using the
same wrappers as generate_lcb_prompts.py — byte-identical to what was sent. Pipe
to the clipboard on Windows:  python print_prompt.py lcb/3517 terse | clip
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")

# The system prompt prepended to every call (run_vibe_tax.py :: SYSTEM_PROMPT).
SYSTEM = ("You are a helpful Python programming assistant. "
          "When you provide code, output plain Python without markdown fences.")

# Same wrappers as generate_lcb_prompts.py :: _mock (class-method form).
WRAPPERS = {
    "terse":        lambda c, e, p, prob: f"need `{c}.{e}({p})` for this:\n\n{prob}",
    "casual":       lambda c, e, p, prob: f"hey can you solve this? need a `{e}` method on a `{c}` class\n\n{prob}",
    "detailed":     lambda c, e, p, prob: f"Hi! Could you help me solve this problem? I need a method `{e}({p})` on a class `{c}`.\n\n{prob}\n\nThank you!",
    "multilingual": lambda c, e, p, prob: f"帮我解决这个问题，请实现 `{c}` 类的 `{e}({p})` 方法：\n\n{prob}",
}
ALIASES = {  # accept the full condition names too
    "agentic_terse": "terse", "agentic_casual": "casual",
    "webchat_detailed": "detailed", "webchat_multilingual": "multilingual",
}


def load():
    return {json.loads(l)["task_id"]: json.loads(l)
            for l in open(PROBLEMS, encoding="utf-8")}


def main():
    args = sys.argv[1:]
    probs = load()
    if not args or args[0] == "--list":
        print(f"{len(probs)} problems. task_ids like:")
        for t in list(probs)[:20]:
            p = probs[t]
            print(f"  {t:10s} {p['difficulty']:6s} {p['entry_point']}")
        print("  ...")
        return

    task_id = args[0]
    which = args[1].lower() if len(args) > 1 else "all"
    which = ALIASES.get(which, which)
    if task_id not in probs:
        sys.exit(f"{task_id} not found (try --list)")
    p = probs[task_id]
    cls = p.get("class_name") or "Solution"
    entry, params, prob = p["entry_point"], p["params"], p["question_content"]

    conds = list(WRAPPERS) if which == "all" else [which]
    for i, cond in enumerate(conds):
        if cond not in WRAPPERS:
            sys.exit(f"unknown condition '{cond}' (choose from {list(WRAPPERS)} or 'all')")
        user = WRAPPERS[cond](cls, entry, params, prob)
        bar = "#" * 70
        print(f"\n{bar}\n# {task_id}  |  {p['difficulty']}  |  condition: {cond}\n{bar}")
        print("----- SYSTEM PROMPT -----")
        print(SYSTEM)
        print("\n----- USER PROMPT -----")
        print(user)
        if i < len(conds) - 1:
            print()


if __name__ == "__main__":
    main()
