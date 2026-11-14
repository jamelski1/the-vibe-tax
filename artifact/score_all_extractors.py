"""Pass-rate + paired terse-vs-detailed McNemar for ALL FOUR extractors.

Regenerates the paper's headline pass-rate table and the terse-vs-detailed McNemar for
Naive / Sieve / sanitize / LCB-official on the paper's slice (GPT-5.4 + Claude Opus 4.6,
1,336 completions). Scores each extractor's output with the same harness the paper uses
(score_lcb.passes / sample_tests, per-test timeout, 60-test cap).

REQUIRES lcb_tests.jsonl, which is HF-gated and NOT shipped in this artifact. Obtain it with
extract_lcb_tests.py (needs the cached LiveCodeBench release JSONLs + Hugging Face access),
then either set LCB_TESTS=/path/to/lcb_tests.jsonl or place it at data/lcb_tests.jsonl.
sanitize needs tree_sitter + tree_sitter_python.

Run:  LCB_TESTS=/path/to/lcb_tests.jsonl python score_all_extractors.py
"""
import os, sys, json
from math import comb
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import extractors          # noqa: E402  (artifact-local)
import score_lcb           # noqa: E402  (artifact-local scoring harness)
from names import REGISTERS, REGISTER_NAME, MODELS  # noqa: E402  (paper name -> stored key)

TESTS = os.getenv("LCB_TESTS", os.path.join(H, "data", "lcb_tests.jsonl"))
MAX_TESTS = int(os.getenv("MAX_TESTS", "60"))
CONDS = [REGISTERS[r] for r in ("terse", "casual", "multilingual", "detailed")]
TERSE, DETAILED = REGISTERS["terse"], REGISTERS["detailed"]

if not os.path.exists(TESTS):
    sys.exit(
        f"lcb_tests.jsonl not found at {TESTS}.\n"
        "It is HF-gated and not shipped. Regenerate it with extract_lcb_tests.py "
        "(needs the LiveCodeBench release JSONLs + Hugging Face access), then set "
        "LCB_TESTS=/path/to/lcb_tests.jsonl or place it at data/lcb_tests.jsonl.")


def mcnemar_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    tests_by_id = {json.loads(l)["task_id"]: json.loads(l) for l in open(TESTS, encoding="utf-8")}
    resp = [r for r in json.load(open(os.path.join(H, "data", "lcb_v3_responses.json")))
            if r["model"] in set(MODELS.values())]
    print(f"capable completions: {len(resp)} | tests for {len(tests_by_id)} problems | MAX_TESTS={MAX_TESTS}\n")
    print(f"{'extractor':13}" + "".join(f"{REGISTER_NAME[c][:8]:>9}" for c in CONDS)
          + f"{'t-d':>7}{'t-only':>8}{'d-only':>8}{'McNemar p':>12}")
    for name, extract in extractors.EXTRACTORS.items():
        scored = {}
        for r in resp:
            rec = tests_by_id.get(r["task_id"])
            if not rec:
                continue
            code = extract(r["completion"], r["entry_point"])
            scored[(r["task_id"], r["level"], r["model"])] = bool(
                score_lcb.passes(code, r["entry_point"], score_lcb.sample_tests(rec, MAX_TESTS)))
        agg = defaultdict(lambda: [0, 0])
        for (t, l, m), p in scored.items():
            agg[l][0] += p; agg[l][1] += 1
        rate = {l: (100 * agg[l][0] / agg[l][1] if agg[l][1] else float("nan")) for l in CONDS}
        keys = {(t, m) for (t, l, m) in scored}
        b = c = 0
        for (t, m) in keys:
            pt, pd = scored.get((t, TERSE, m)), scored.get((t, DETAILED, m))
            if pt is None or pd is None:
                continue
            if pt and not pd: b += 1
            elif pd and not pt: c += 1
        d = rate[TERSE] - rate[DETAILED]
        print(f"{name:13}" + "".join(f"{rate[c]:>8.1f}%" for c in CONDS)
              + f"{d:>+7.1f}{b:>8}{c:>8}{mcnemar_p(b, c):>12.2g}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
