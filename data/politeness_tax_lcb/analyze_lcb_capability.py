"""
Where do the models succeed and fail on LiveCodeBench? Characterizes the 167
problems by difficulty and by (keyword-derived) topic, and splits failures into
"produced no usable code" vs "wrong logic", using the committed scored data +
completions. No API keys, no tests file needed (compile-check only for the
failure-mode split).

Framing is null (see RESULTS.md), so we average over the 4 framings and read
capability by (problem × model).

Outputs: lcb_capability_stats.json  + prints tables.
Usage:  python analyze_lcb_capability.py
"""

import ast
import json
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORED = os.path.join(SCRIPT_DIR, "lcb_scored.json")
PROBLEMS = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")
RESPONSES = os.path.join(SCRIPT_DIR, "lcb_v3_responses.json")
OUT = os.path.join(SCRIPT_DIR, "lcb_capability_stats.json")

# Keyword -> topic. First match wins (ordered); checked against the lowercased
# problem statement + method name. Deliberately transparent, not ML.
TOPICS = [
    ("dynamic_programming", r"\b(dynamic programming|dp\b|subsequence|number of ways|minimum cost|maximum (sum|score|value)|partitions?)\b"),
    ("graph",               r"\b(graph|node|edge|adjacen|connected component|shortest path|cities|roads?|network)\b"),
    ("tree",                r"\b(binary tree|tree|root|leaf|subtree|ancestor|parent node)\b"),
    ("string",              r"\b(string|substring|palindrome|character|prefix|suffix|anagram|lexicograph)\b"),
    ("intervals_sorting",   r"\b(interval|sort|sorted|merge|schedule|meeting|overlap)\b"),
    ("greedy_array",        r"\b(array|subarray|greedy|adjacent|window|two pointers?)\b"),
    ("math_number",         r"\b(prime|gcd|lcm|modulo|divisor|digits?|binary representation|factorial|combinatinatorics|arithmetic)\b"),
    ("bit_manipulation",    r"\b(bit|xor|bitwise|binary and|set bits|and of|or of)\b"),
    ("simulation_geometry", r"\b(simulat|grid|matrix|coordinate|move|direction|snake|robot|game)\b"),
]


def classify(text):
    t = text.lower()
    for name, pat in TOPICS:
        if re.search(pat, t):
            return name
    return "other"


def extract(completion, entry):
    """Same robust extraction as score_lcb: does the reply yield compilable code
    that defines the target? (Distinguishes 'no usable code' from 'wrong logic'.)"""
    if not completion:
        return None
    cands = []
    for b in re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL):
        if "class Solution" in b or f"def {entry}" in b:
            cands.append(b)
    text = "\n".join(l for l in completion.split("\n") if not l.strip().startswith("```"))
    for anchor in ("class Solution", f"def {entry}"):
        i = text.find(anchor)
        if i != -1:
            cands.append(text[i:])
    for c in cands:
        lines = c.split("\n")
        while lines:
            src = "\n".join(lines)
            try:
                ast.parse(src)
                if "class Solution" in src or f"def {entry}" in src:
                    return src
                break
            except SyntaxError:
                lines.pop()
    return None


def rate(items):
    n = len(items); k = sum(items)
    return {"passed": k, "total": n, "pass_rate": round(k / n * 100, 1) if n else None}


def run():
    scored = json.load(open(SCORED, encoding="utf-8"))
    probs = {json.loads(l)["task_id"]: json.loads(l)
             for l in open(PROBLEMS, encoding="utf-8")}
    for tid, p in probs.items():
        p["topic"] = classify(p["question_content"] + " " + p["entry_point"])

    # attach topic to each scored record
    for x in scored:
        x["topic"] = probs.get(x["task_id"], {}).get("topic", "other")

    CAP = {"chatgpt", "claude"}  # capable-model read
    cap = [x for x in scored if x["model"] in CAP]

    # ---- pass rate by difficulty and topic (capable models) ----
    by_diff = defaultdict(list); by_topic = defaultdict(list); by_topic_diff = defaultdict(list)
    for x in cap:
        by_diff[x["difficulty"]].append(x["passed"])
        by_topic[x["topic"]].append(x["passed"])
        by_topic_diff[(x["topic"], x["difficulty"])].append(x["passed"])

    # ---- per-problem solve rate (over all 12 attempts: 4 framings x 3 models) ----
    per_prob = defaultdict(list)
    for x in scored:
        per_prob[x["task_id"]].append(x["passed"])
    solve_frac = {t: sum(v) / len(v) for t, v in per_prob.items()}
    always = [t for t, f in solve_frac.items() if f == 1.0]
    never = [t for t, f in solve_frac.items() if f == 0.0]

    # ---- failure-mode split: of FAILED capable completions, did code compile? ----
    resp = {(x["task_id"], x["level"], x["model"]): x
            for x in json.load(open(RESPONSES, encoding="utf-8"))}
    no_code = wrong_logic = 0
    for x in cap:
        if x["passed"]:
            continue
        r = resp.get((x["task_id"], x["level"], x["model"]))
        code = extract(r.get("completion") if r else None, probs[x["task_id"]]["entry_point"])
        if code:
            wrong_logic += 1
        else:
            no_code += 1

    stats = {
        "by_difficulty_capable": {k: rate(v) for k, v in sorted(by_diff.items())},
        "by_topic_capable": {k: rate(v) for k, v in sorted(by_topic.items(),
                              key=lambda kv: rate(kv[1])["pass_rate"], reverse=True)},
        "topic_counts": {t: sum(1 for p in probs.values() if p["topic"] == t)
                         for t in sorted({p["topic"] for p in probs.values()})},
        "always_solved_n": len(always), "never_solved_n": len(never),
        "never_solved_by_difficulty": dict(__import__("collections").Counter(
            probs[t]["difficulty"] for t in never)),
        "failure_modes_capable": {
            "wrong_logic": wrong_logic, "no_usable_code": no_code,
            "wrong_logic_pct": round(wrong_logic / (wrong_logic + no_code) * 100, 1) if (wrong_logic+no_code) else None,
        },
    }
    json.dump(stats, open(OUT, "w", encoding="utf-8"), indent=2)

    print("=" * 64)
    print("PASS RATE BY DIFFICULTY (capable models: ChatGPT + Claude)")
    for k in ("easy", "medium", "hard"):
        v = stats["by_difficulty_capable"].get(k)
        if v: print(f"  {k:8s} {v['passed']:3d}/{v['total']:<3d}  {v['pass_rate']}%")
    print("\nPASS RATE BY TOPIC (capable models), high -> low")
    for k, v in stats["by_topic_capable"].items():
        print(f"  {k:22s} {v['passed']:3d}/{v['total']:<3d}  {v['pass_rate']:5.1f}%   "
              f"({stats['topic_counts'][k]} problems)")
    print(f"\nPER-PROBLEM (all 12 attempts each):")
    print(f"  always solved (12/12): {len(always)} problems")
    print(f"  never solved (0/12)  : {len(never)} problems  by difficulty "
          f"{stats['never_solved_by_difficulty']}")
    fm = stats["failure_modes_capable"]
    print(f"\nFAILURE MODES (failed capable completions):")
    print(f"  wrong logic (code compiles, tests fail): {fm['wrong_logic']}  ({fm['wrong_logic_pct']}%)")
    print(f"  no usable code produced                : {fm['no_usable_code']}")
    print(f"\n-> {OUT}")

    # a few concrete never-solved problems to eyeball
    print("\nSample NEVER-solved problems (task_id | difficulty | topic | method):")
    for t in sorted(never, key=lambda t: probs[t]["difficulty"])[:10]:
        p = probs[t]
        print(f"  {t:10s} {p['difficulty']:6s} {p['topic']:20s} {p['entry_point']}")


if __name__ == "__main__":
    run()
