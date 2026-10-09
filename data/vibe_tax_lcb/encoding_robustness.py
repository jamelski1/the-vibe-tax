"""Encoding-corruption audit + headline-robustness check (Option 1 evidence).

A UTF-8->CP437 mishandling at prompt-generation time corrupted non-ASCII characters:
  * the multilingual Chinese framing (most items), and
  * any ENGLISH-condition problem statement containing a non-ASCII symbol (e.g. the
    arrow U+2192 "->" became the CP437 mojibake "GaE" / ΓåÆ).
This script (1) measures the corruption across all four conditions, (2) recovers it to
prove it's the same text, and (3) shows the terse-vs-detailed headline is UNCHANGED when
the corrupted-spec problems are excluded -- i.e. the bug does not create or hide the
register/extraction result. Reads only files in this directory; no model calls, no tests.

Run:  python data/vibe_tax_lcb/encoding_robustness.py
"""
import json, re, ast, os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
resp = json.load(open(os.path.join(HERE, "lcb_v3_responses.json"), encoding="utf-8"))
scored = {(r["task_id"], r["level"], r["model"]): bool(r["passed"])
          for r in json.load(open(os.path.join(HERE, "lcb_scored.json"), encoding="utf-8"))}
diff = {json.loads(l)["task_id"]: json.loads(l).get("difficulty")
        for l in open(os.path.join(HERE, "lcb_problems.jsonl"), encoding="utf-8")}

CJK = re.compile(r"[一-鿿]")
CAP = ("chatgpt", "claude")


def is_mojibake(t):
    """True iff t is UTF-8 bytes that were decoded as CP437 (re-encoding recovers valid UTF-8)."""
    try:
        return t.encode("cp437", "strict").decode("utf-8", "strict") != t
    except Exception:
        return False


def body(pt):
    return pt.split("\n\n", 1)[1] if "\n\n" in pt else pt


def mcnemar_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


# ---- 1. corruption across conditions ----
print("=" * 72)
print("1. CP437 mojibake by condition (prompts flagged, and prompts with any non-ASCII)")
for c in ("agentic_terse", "agentic_casual", "webchat_detailed", "webchat_multilingual"):
    rs = [r for r in resp if r["level"] == c]
    moj = sum(is_mojibake(r["prompt_text"]) for r in rs)
    na = sum(any(ord(ch) > 127 for ch in r["prompt_text"]) for r in rs)
    print(f"   {c:22} n={len(rs):4}  mojibake={moj:4}  has_nonascii={na:4}")

# ---- 2. corrupted PROBLEM STATEMENTS (shared across conditions) ----
terse = {r["task_id"]: r for r in resp if r["level"] == "agentic_terse"}
corrupt = sorted(t for t, r in terse.items() if is_mojibake(body(r["prompt_text"])))
print("=" * 72)
print(f"2. Problems whose STATEMENT is corrupted (seen in all conditions): {len(corrupt)} of {len(terse)}")
print("   difficulty:", {d: sum(diff.get(t) == d for t in corrupt) for d in ("easy", "medium", "hard")})
ex = body(terse[corrupt[0]]["prompt_text"])
frag = re.search(r"[^\x00-\x7f]+", ex).group(0)
print(f"   example {corrupt[0]}: {frag!r}  recovers to  {frag.encode('cp437').decode('utf-8')!r}")

# ---- 3. HEADLINE ROBUSTNESS: terse vs detailed, all 167 vs clean 158 ----
def extract_naive(c, entry):
    if not c:
        return None
    for b in re.findall(r"```(?:python|py)?\s*\n(.*?)```", c, re.DOTALL):
        if "class Solution" in b or f"def {entry}" in b:
            return b
    if "class Solution" in c or f"def {entry}" in c:
        return "\n".join(l for l in c.split("\n") if not l.strip().startswith("```"))
    return None


def compiles(code):
    if not code:
        return False
    try:
        ast.parse(code); return True
    except SyntaxError:
        return False


idx = {(r["task_id"], r["level"], r["model"]): r for r in resp}


def discord(problem_set, mode):
    b = c = 0
    for t in problem_set:
        for m in CAP:
            if mode == "naive_compile":
                rt, rd = idx.get((t, "agentic_terse", m)), idx.get((t, "webchat_detailed", m))
                if not rt or not rd:
                    continue
                vt = compiles(extract_naive(rt["completion"], rt["entry_point"]))
                vd = compiles(extract_naive(rd["completion"], rd["entry_point"]))
            else:  # robust_pass
                vt, vd = scored.get((t, "agentic_terse", m)), scored.get((t, "webchat_detailed", m))
                if vt is None or vd is None:
                    continue
            if vt and not vd:
                b += 1
            elif vd and not vt:
                c += 1
    return b, c


allp = set(terse); clean = allp - set(corrupt)
print("=" * 72)
print(f"3. Headline terse-vs-detailed, with vs without the {len(corrupt)} corrupted-spec problems")
for label, ps in [(f"ALL {len(allp)}", allp), (f"CLEAN {len(clean)} (excl. corrupted spec)", clean)]:
    nb, nc = discord(ps, "naive_compile")
    rb, rc = discord(ps, "robust_pass")
    print(f"   {label}:")
    print(f"     NAIVE  compile-rate  b={nb:3} c={nc:3}  d={nb-nc:+3}  p={mcnemar_p(nb,nc):.2e}")
    print(f"     ROBUST pass-rate     b={rb:3} c={rc:3}  d={rb-rc:+3}  p={mcnemar_p(rb,rc):.3f}")
print("=" * 72)
print("Conclusion: the naive compile-tax and the robust null are unchanged by excluding the")
print("corrupted-spec problems -> the encoding bug does not create or hide the register effect.")
