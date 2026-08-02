"""
Robust scorer for the Vibe Tax v2 experiment.

The v1 scorer (run_tests.py) assumes completions are function *bodies* or a
single clean function. That breaks on the paste conditions: "here's my error,
fix it" elicits a CONVERSATIONAL reply — prose, a snippet of the buggy code,
then "Fixed version:" + the real corrected function. v1's body-extractor grabs
the first code-like line (the buggy snippet in the explanation) and mangles it,
scoring correct fixes as syntax errors. That made webchat_error_paste read 0%
for every model — a scoring artifact, not a correctness result.

This scorer extracts the LAST complete `def <entry_point>` in the reply (the
fixed version) and runs it standalone against the official tests, with the v1
prompt+body approach as a fallback. Use it for every condition so all arms are
scored on equal footing.

Usage:
    python score_vibe_tax.py
Outputs:
    vibe_tax_scored.json        per-completion pass/fail
    vibe_tax_scored_stats.json  pass@1 by condition, model, medium, condition x model
"""

import json
import multiprocessing
import os
import re
import sys
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES = os.path.join(SCRIPT_DIR, "vibe_tax_responses.json")
HE_DIR = os.path.join(SCRIPT_DIR, "..", "HumanEval.jsonl")
HUMANEVAL = os.path.join(HE_DIR, "human-eval-v2-20210705.jsonl")

# Reuse v1's body-normalizer (fixes ragged indentation on bare-body replies,
# which the paste-oriented extractor below does not handle).
sys.path.insert(0, HE_DIR)
from run_tests import clean_completion  # noqa: E402
OUT = os.path.join(SCRIPT_DIR, "vibe_tax_scored.json")
STATS = os.path.join(SCRIPT_DIR, "vibe_tax_scored_stats.json")
TIMEOUT = 10


def strip_fences(text):
    """Drop ``` fence markers but keep the code between them."""
    return "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))


def problem_imports(prompt):
    return "\n".join(l for l in prompt.split("\n")
                     if l.startswith(("import ", "from ")))


def extract_last_function(text, entry):
    """Return the LAST complete `def <entry>(...)` block, dedented, or None.

    Robust to leading prose, buggy-code snippets shown earlier in the reply,
    and markdown fences — the fixed version is (almost) always last.
    """
    text = strip_fences(text)
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines)
              if re.match(rf"\s*def\s+{re.escape(entry)}\s*\(", l)]
    if not starts:
        return None
    start = starts[-1]
    base = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip():
            body.append(l)
            continue
        indent = len(l) - len(l.lstrip())
        if indent <= base:      # dedented back to/under def level -> function ended
            break
        body.append(l)
    # pull along any decorators / imports that sat just above the def
    head = []
    for l in reversed(lines[:start]):
        if l.startswith(("import ", "from ")) or l.strip().startswith("@"):
            head.insert(0, l)
        elif l.strip() == "":
            continue
        else:
            break
    return textwrap.dedent("\n".join(head + body))


def normalized_body(completion):
    """Reconstruct a bare function body robustly.

    Models often mis-indent only the FIRST line (e.g. 3 spaces then 4). Anchoring
    on the first line (as v1 does) then shifts every later line and breaks the
    structure. Here we raise an anomalously-shallow first line to match the rest,
    then dedent+reindent to a clean 4 spaces.
    """
    body = strip_fences(completion)
    lines = body.split("\n")
    nb = [(i, l) for i, l in enumerate(lines) if l.strip()]
    if len(nb) < 2:
        return None
    first_indent = len(nb[0][1]) - len(nb[0][1].lstrip())
    rest_min = min(len(l) - len(l.lstrip()) for _, l in nb[1:])
    if first_indent < rest_min:
        i0 = nb[0][0]
        lines[i0] = " " * rest_min + lines[i0].lstrip()
    txt = textwrap.indent(textwrap.dedent("\n".join(lines)), "    ")
    return txt


def fenced_blocks(text):
    """Return the contents of ```...``` code blocks (isolates code from prose)."""
    return re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)


def build_candidates(prompt, completion, test, entry):
    """Ordered list of full programs to try; first that passes wins."""
    cands = []
    imports = problem_imports(prompt)
    # 1. conversational "here's the fixed version" -> last full function standalone
    fn = extract_last_function(completion, entry)
    if fn:
        cands.append(imports + "\n\n" + fn + "\n" + test + f"\ncheck({entry})\n")
    # 1b. isolate a fenced code block (drops surrounding prose / stray arrows)
    for block in reversed(fenced_blocks(completion)):
        if f"def {entry}" in block:
            cands.append(imports + "\n\n" + block + "\n" + test + f"\ncheck({entry})\n")
        else:
            nb = normalized_body(block)
            if nb:
                cands.append(prompt + nb + "\n" + test + f"\ncheck({entry})\n")
    # 2. bare body appended to prompt, with v1 indentation normalization
    cands.append(prompt + clean_completion(completion, entry) + "\n" + test + f"\ncheck({entry})\n")
    # 2b. bare body with first-line-indent repair (handles ragged first line)
    nb = normalized_body(completion)
    if nb:
        cands.append(prompt + nb + "\n" + test + f"\ncheck({entry})\n")
    # 3. raw (fence-stripped) completion appended, and standalone, as last resorts
    body = strip_fences(completion)
    cands.append(prompt + body + "\n" + test + f"\ncheck({entry})\n")
    cands.append(imports + "\n\n" + body + "\n" + test + f"\ncheck({entry})\n")
    return cands


def _worker(cands, q):
    for prog in cands:
        try:
            compile(prog, "<c>", "exec")
        except SyntaxError:
            continue
        try:
            exec(prog, {})
            q.append("passed")
            return
        except Exception:
            continue
    q.append("failed")


def run_one(prompt, completion, test, entry):
    if not completion:
        return False
    cands = build_candidates(prompt, completion, test, entry)
    mgr = multiprocessing.Manager()
    q = mgr.list()
    p = multiprocessing.Process(target=_worker, args=(cands, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.kill(); p.join(5); return False
    return bool(q) and q[0] == "passed"


def run():
    responses = json.load(open(RESPONSES, encoding="utf-8"))
    problems = {}
    for line in open(HUMANEVAL, encoding="utf-8"):
        p = json.loads(line); problems[p["task_id"]] = p

    scored = []
    for i, r in enumerate(responses):
        prob = problems[r["task_id"]]
        ok = run_one(prob["prompt"], r.get("completion"), prob["test"], r["entry_point"])
        scored.append({**{k: r[k] for k in
                          ("task_id", "problem_number", "entry_point", "level", "medium", "model")},
                       "passed": ok})
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(responses)}", flush=True)

    json.dump(scored, open(OUT, "w", encoding="utf-8"), indent=2)

    def rate(items):
        n = len(items); k = sum(x["passed"] for x in items)
        return {"passed": k, "total": n, "pass_rate": round(k / n * 100, 1) if n else None}

    by_cond = defaultdict(list); by_model = defaultdict(list)
    by_medium = defaultdict(list); by_cm = defaultdict(list)
    for x in scored:
        by_cond[x["level"]].append(x); by_model[x["model"]].append(x)
        by_medium[x["medium"]].append(x); by_cm[f'{x["model"]}|{x["level"]}'].append(x)

    stats = {
        "overall": rate(scored),
        "by_condition": {k: rate(v) for k, v in sorted(by_cond.items())},
        "by_model": {k: rate(v) for k, v in sorted(by_model.items())},
        "by_medium": {k: rate(v) for k, v in sorted(by_medium.items())},
        "by_condition_and_model": {k: rate(v) for k, v in sorted(by_cm.items())},
    }
    json.dump(stats, open(STATS, "w", encoding="utf-8"), indent=2)

    print("=" * 60)
    print(f"overall: {stats['overall']['pass_rate']}%  (n={stats['overall']['total']})")
    print("\nby condition:")
    for k, v in stats["by_condition"].items():
        print(f"  {k:22s} {v['passed']:3d}/{v['total']:<3d} {v['pass_rate']}%")
    print("\nby model:")
    for k, v in stats["by_model"].items():
        print(f"  {k:12s} {v['pass_rate']}%")
    print(f"\nscored -> {OUT}\nstats  -> {STATS}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run()
