"""
Extract LiveCodeBench test cases for the problems in lcb_problems.jsonl, so LCB
completions can be graded locally.  RUN LOCALLY (reads the cached release JSONLs).

LCB stores tests as:
  public_test_cases  : a JSON string -> list of {input, output, testtype}
  private_test_cases : base64 -> zlib.decompress -> pickle.loads -> json.loads
                       (falls back to plain json if not compressed)
Each case: {"input": <str>, "output": <str>, "testtype": "functional"|...}.

Writes:
  lcb_tests.jsonl        (local, gitignored — one line/problem: task_id + tests)
  lcb_tests_sample.json  (2 problems — PUSH THIS so the scorer can be built)

It self-reports the decoded structure so we build the scorer against reality.

Usage:  python extract_lcb_tests.py
"""

import base64
import json
import os
import pickle
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(SCRIPT_DIR, "lcb_problems.jsonl")
OUT = os.path.join(SCRIPT_DIR, "lcb_tests.jsonl")
SAMPLE = os.path.join(SCRIPT_DIR, "lcb_tests_sample.json")
DATA_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl",
              "test4.jsonl", "test5.jsonl", "test6.jsonl"]


def decode_private(blob):
    if not blob:
        return []
    # plain json first
    try:
        return json.loads(blob)
    except Exception:
        pass
    # base64 -> zlib -> pickle -> json  (LCB's compressed format)
    for stage in ("pickle", "json"):
        try:
            raw = zlib.decompress(base64.b64decode(blob.encode("utf-8")))
            obj = pickle.loads(raw) if stage == "pickle" else raw.decode("utf-8")
            return json.loads(obj) if isinstance(obj, str) else obj
        except Exception:
            continue
    return []


def decode_public(blob):
    if not blob:
        return []
    if isinstance(blob, list):
        return blob
    try:
        return json.loads(blob)
    except Exception:
        return []


def run():
    want = {}
    for l in open(PROBLEMS, encoding="utf-8"):
        p = json.loads(l)
        # task_id is "lcb/<question_id>"
        want[p["task_id"].split("/", 1)[1]] = p
    print(f"want tests for {len(want)} problems")

    from huggingface_hub import hf_hub_download
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    found = {}
    reported = False
    for fn in DATA_FILES:
        print(f"  scanning {fn} ...", flush=True)
        try:
            path = hf_hub_download("livecodebench/code_generation_lite", fn,
                                   repo_type="dataset", token=token)
        except Exception as e:
            print(f"    skip {fn}: {e}"); continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            qid = str(r.get("question_id"))
            if qid not in want or qid in found:
                continue
            pub = decode_public(r.get("public_test_cases"))
            priv = decode_private(r.get("private_test_cases"))
            if not reported and (pub or priv):
                print("  DECODED STRUCTURE (first problem) — report this back:")
                ex = (pub or priv)[0]
                print("   public:", len(pub), "| private:", len(priv))
                print("   example case keys:", list(ex.keys()) if isinstance(ex, dict) else type(ex).__name__)
                print("   example input :", repr(str(ex.get('input'))[:120]) if isinstance(ex, dict) else "")
                print("   example output:", repr(str(ex.get('output'))[:120]) if isinstance(ex, dict) else "")
                reported = True
            found[qid] = {
                "task_id": want[qid]["task_id"],
                "entry_point": want[qid]["entry_point"],
                "class_name": want[qid].get("class_name") or "Solution",
                "public_tests": pub,
                "private_tests": priv,
            }

    with open(OUT, "w", encoding="utf-8") as f:
        for v in found.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    json.dump(list(found.values())[:2], open(SAMPLE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    npub = sum(len(v["public_tests"]) for v in found.values())
    npriv = sum(len(v["private_tests"]) for v in found.values())
    print("=" * 60)
    print(f"extracted tests for {len(found)}/{len(want)} problems -> {OUT}")
    print(f"  total public cases: {npub} | total private cases: {npriv}")
    print(f"sample (2 problems) -> {SAMPLE}  <-- push this so the scorer can be built")
    missing = set(want) - set(found)
    if missing:
        print(f"  WARNING: {len(missing)} problems had no matching tests")


if __name__ == "__main__":
    run()
