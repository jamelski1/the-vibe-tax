"""
Calibrated prompt generator (v3).

The v2 conditions were researcher-written and are 100% distinguishable from real
prompts (see realism_discriminator). v3 instead *rewrites* each HumanEval task
into a prompt styled like real usage, and gates each rewrite through the
RealismDiscriminator so obviously-synthetic scaffolding is rejected.

Pipeline (generate → gate → keep):
  1. For each problem × condition, build a rewrite instruction that:
       - preserves the EXACT function name + parameter order (so it stays
         gradeable by the HumanEval tests),
       - preserves the behavioural spec (from the docstring, doctests stripped),
       - targets the condition's real style (terse / detailed / paste / zh),
       - forbids the scaffolding tells (no `def` line, no triple-quote
         docstring, no `>>>` doctests) — phrase it as a chat MESSAGE.
  2. Call an LLM backend to produce the message.
  3. GATE it: reject if it still contains scaffolding, if it drops the function
     name (ungradeable), or if the discriminator still reads it as synthetic.
     Retry up to --retries times.
  4. Save vibe_tax_v3_prompts.json (same schema as v2 -> run_vibe_tax.py works).

Backends (auto-selected, or force with --backend):
  anthropic  ANTHROPIC_API_KEY   openai  OPENAI_API_KEY   mock  (no keys)
The `mock` backend does a deterministic scaffolding-free rewrite so the whole
harness runs and is testable without any API access.

Usage:
  python generate_calibrated_prompts.py                 # auto backend
  python generate_calibrated_prompts.py --backend mock --limit 5
"""

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
V2_DIR = os.path.join(ROOT, "data", "vibe_tax_v2")
RP_DIR = os.path.join(ROOT, "data", "real_prompts")
HE_DIR = os.path.join(ROOT, "data", "HumanEval.jsonl")
sys.path.insert(0, V2_DIR)
sys.path.insert(0, RP_DIR)

from generate_experiment_prompts import make_buggy, extract_sig  # noqa: E402
from realism_discriminator import (RealismDiscriminator, load_real,  # noqa: E402
                                   load_synthetic, featurize)

SPECTRUM = os.path.join(HE_DIR, "vibe_spectrum_data.json")
HUMANEVAL = os.path.join(HE_DIR, "human-eval-v2-20210705.jsonl")
OUT = os.path.join(SCRIPT_DIR, "vibe_tax_v3_prompts.json")

SCAFFOLD = re.compile(r'"""|\'\'\'|>>>|\bdef\s+\w+\s*\(')


# ---------------------------------------------------------------------------
# Task text helpers
# ---------------------------------------------------------------------------

def params_of(signature):
    """'def f(a: int, b=3) -> x:' -> 'a, b'."""
    m = re.search(r"\(([^)]*)\)", signature)
    if not m:
        return ""
    parts = []
    for p in m.group(1).split(","):
        name = p.split(":")[0].split("=")[0].strip()
        if name and name not in ("self",):
            parts.append(name)
    return ", ".join(parts)


def spec_prose(entry_1_formal):
    """The docstring description with signature/imports/doctests stripped."""
    lines = entry_1_formal.split("\n")
    keep = []
    for l in lines:
        s = l.strip()
        if s.startswith((">>>",)) or s.startswith(("from ", "import ", "def ")):
            continue
        if s in ('"""', "'''"):
            continue
        keep.append(re.sub(r'^\s*("""|\'\'\')', "", l))
    text = "\n".join(keep)
    # drop a trailing example/output block after the first blank following prose
    text = re.sub(r'"""|\'\'\'', "", text).strip()
    return text


# ---------------------------------------------------------------------------
# Conditions (same six as v2, now with realistic-rewrite style directives)
# ---------------------------------------------------------------------------

CONDITIONS = {
    "agentic_terse": dict(medium="agentic", needs_bug=False,
        style="Extremely terse and lowercase, like a quick CLI ask — a handful of words, no pleasantries."),
    "agentic_casual": dict(medium="agentic", needs_bug=False,
        style="Casual and lowercase, one or two sentences, like messaging a teammate."),
    "webchat_detailed": dict(medium="webchat", needs_bug=False,
        style="A polite, clear chat message of a few sentences describing what you need."),
    "webchat_code_paste": dict(medium="webchat", needs_bug=True,
        style="You wrote code that doesn't work. Paste the broken code and ask for a fix."),
    "webchat_error_paste": dict(medium="webchat", needs_bug=True,
        style="Your code errors out. Paste the error message AND the broken code, and ask how to fix it."),
    "webchat_multilingual": dict(medium="webchat", needs_bug=False,
        style="Write the ENTIRE message in Chinese (the function name stays in English)."),
}


def build_instruction(entry, params, prose, cond, buggy):
    style = CONDITIONS[cond]["style"]
    paste = ""
    if CONDITIONS[cond]["needs_bug"] and buggy:
        code, err, _ = buggy
        if cond == "webchat_error_paste":
            paste = f"\n\nInclude this exact error and this exact broken code in your message:\nERROR:\n{err}\n\nCODE:\n{code}"
        else:
            paste = f"\n\nInclude this exact broken code in your message:\n{code}"
    return (
        "Rewrite the following coding task as a REALISTIC user message — the way a "
        f"real developer would actually type it. Style: {style}\n\n"
        f"The message MUST ask for a function named exactly `{entry}` taking parameters "
        f"({params}) in that order (so it can be auto-graded). Convey what it should do:\n\n"
        f"{prose}\n"
        "\nHARD RULES: do NOT write a `def` line, a triple-quoted docstring, or `>>>` "
        "doctest examples. It is a chat message, not a code stub. Keep every behavioural "
        f"constraint so it's still solvable.{paste}\n\nReturn ONLY the message text."
    )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def make_backend(name):
    if name == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "mock"
    if name == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        def call(instr):
            r = client.messages.create(model=model, max_tokens=1200, temperature=0.7,
                                       messages=[{"role": "user", "content": instr}])
            return r.content[0].text.strip()
        return name, call
    if name == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-5.4")
        def call(instr):
            r = client.chat.completions.create(model=model, temperature=0.7,
                max_completion_tokens=1200, messages=[{"role": "user", "content": instr}])
            return r.choices[0].message.content.strip()
        return name, call
    # deterministic mock: scaffolding-free rewrite good enough to exercise the harness
    return "mock", _mock_call


def _one_line(prose):
    s = re.sub(r"\s+", " ", prose).strip()
    s = s.split(". ")[0]
    return s[:200].rstrip(". ")


def _mock_call(instr):
    # parse back the pieces the instruction embedded
    entry = re.search(r"named exactly `(\w+)`", instr).group(1)
    params = re.search(r"taking parameters \(([^)]*)\)", instr).group(1)
    prose = instr.split("Convey what it should do:\n\n", 1)[1].split("\nHARD RULES", 1)[0].strip()
    desc = _one_line(prose)
    code_m = re.search(r"CODE:\n(.*?)\n\nReturn ONLY", instr, re.DOTALL) or \
             re.search(r"broken code in your message:\n(.*?)\n\nReturn ONLY", instr, re.DOTALL)
    err_m = re.search(r"ERROR:\n(.*?)\n\nCODE:", instr, re.DOTALL)
    if "ENTIRE message in Chinese" in instr:
        return f"帮我写一个叫 {entry}({params}) 的函数，功能是：{desc}。谢谢！"
    if err_m and code_m:
        return (f"my code keeps failing and i don't get this error:\n\n{err_m.group(1).strip()}\n\n"
                f"here's the code for {entry}:\n\n{code_m.group(1).strip()}\n\nhow do i fix it?")
    if code_m:
        return (f"i wrote {entry}({params}) but it's not working, can you fix it?\n\n"
                f"{code_m.group(1).strip()}")
    if "Extremely terse" in instr:
        return f"write {entry}({params}) - {desc.lower()}"
    if "Casual and lowercase" in instr:
        return f"hey can you write {entry}({params})? it should {desc[0].lower()+desc[1:]}"
    return (f"Hi! Could you help me write a function {entry}({params})? "
            f"It needs to {desc[0].lower()+desc[1:]}. Thanks so much!")


# ---------------------------------------------------------------------------
# Generate + gate
# ---------------------------------------------------------------------------

def gate(prompt, entry, cond, disc, threshold):
    """Condition-aware gate.

    Paste conditions legitimately CONTAIN code (a `def` line, maybe a docstring)
    and a traceback — that's the whole point. The weak feature discriminator
    mis-reads that as synthetic, and the scaffolding regex would reject it. So
    for paste conditions we only require the function name to survive (so it
    stays gradeable) and record the discriminator score as ADVISORY. Prose
    conditions get the full gate (must be scaffold-free and read real).
    """
    is_paste = CONDITIONS[cond]["needs_bug"]
    reasons = []
    if not is_paste and SCAFFOLD.search(prompt):
        reasons.append("scaffolding")
    if entry not in prompt:
        reasons.append("dropped_function_name")
    p_real = disc.predict_proba(prompt) if disc is not None else None
    if not is_paste and p_real is not None and p_real < threshold:
        reasons.append(f"reads_synthetic({p_real:.2f})")
    return reasons, p_real


def run(limit, backend_name, threshold, retries):
    spectrum = json.load(open(SPECTRUM, encoding="utf-8"))
    problems = {json.loads(l)["task_id"]: json.loads(l) for l in open(HUMANEVAL, encoding="utf-8")}
    if limit:
        spectrum = spectrum[:limit]

    # Fit the discriminator gate (needs a real corpus to be present).
    real = load_real("all")
    disc = None
    if real:
        disc = RealismDiscriminator().fit(real, load_synthetic("all"))
        print(f"Discriminator gate active (fit on {len(real)} real prompts).")
    else:
        print("WARNING: no real corpus found -> discriminator gate disabled "
              "(scaffolding + name checks still apply).")

    bname, call = make_backend(backend_name)
    print(f"Backend: {bname}\n")

    out, stats = [], {"kept": 0, "regenerated": 0, "failed": 0, "reads_real_avg": []}
    for i, entry_spec in enumerate(spectrum):
        tid = entry_spec["task_id"]
        entry = entry_spec["entry_point"]
        sig = extract_sig(entry_spec["level_1_formal"])
        params = params_of(sig)
        prose = spec_prose(entry_spec["level_1_formal"])
        buggy = make_buggy(problems[tid])

        for cond, cfg in CONDITIONS.items():
            if cfg["needs_bug"] and not buggy:
                continue
            instr = build_instruction(entry, params, prose, cond, buggy)
            best = None
            for attempt in range(retries + 1):
                try:
                    prompt = call(instr)
                except Exception as e:
                    print(f"  [{tid}/{cond}] backend error: {e}")
                    break
                reasons, p_real = gate(prompt, entry, cond, disc, threshold)
                if not reasons:
                    best = (prompt, p_real)
                    break
                best = best or (prompt, p_real)
                if attempt < retries:
                    stats["regenerated"] += 1
            if best is None:
                stats["failed"] += 1
                continue
            prompt, p_real = best
            reasons, _ = gate(prompt, entry, cond, disc, threshold)
            if reasons:
                stats["failed"] += 1
            else:
                stats["kept"] += 1
            if p_real is not None:
                stats["reads_real_avg"].append(p_real)
            out.append({
                "task_id": tid, "problem_number": entry_spec["problem_number"],
                "entry_point": entry, "condition": cond, "medium": cfg["medium"],
                "generator": "calibrated", "backend": bname,
                "reads_real": round(p_real, 3) if p_real is not None else None,
                "gate_ok": not reasons, "prompt": prompt,
            })
        print(f"[{i+1}/{len(spectrum)}] {tid}: generated")

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    avg = (sum(stats["reads_real_avg"]) / len(stats["reads_real_avg"])) if stats["reads_real_avg"] else None
    print("=" * 60)
    print(f"prompts written : {len(out)}  -> {OUT}")
    print(f"passed gate     : {sum(1 for x in out if x['gate_ok'])}/{len(out)}")
    print(f"regenerations   : {stats['regenerated']}")
    if avg is not None:
        print(f"avg reads_real  : {avg:.2f}  (discriminator P(real); higher = more realistic)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "anthropic", "openai", "mock"], default="auto")
    ap.add_argument("--limit", type=int, default=0, help="max problems (0 = all 50)")
    ap.add_argument("--threshold", type=float, default=0.5, help="min discriminator P(real) to pass")
    ap.add_argument("--retries", type=int, default=2, help="regeneration attempts on gate failure")
    a = ap.parse_args()
    run(a.limit, a.backend, a.threshold, a.retries)
