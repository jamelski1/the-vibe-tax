"""
Pull a real WEB-CHAT prompt corpus (WildChat / ShareGPT-style conversations) to
cover the prompting patterns that agentic-CLI transcripts can't — chiefly the
"copy-paste error / code" pattern, where a user pastes a stack trace into a chat
box and says "fix this".

HuggingFace is firewalled in this environment, so instead of pulling WildChat
from HF directly we harvest conversation dumps that people have committed to
public GitHub (reachable via the same allowed GitHub access the transcript
crawler uses). The richest clean source found is WildChat conversations mirrored
as individual JSON files.

Pipeline mirrors crawl_real_prompts.py:
  discover conversation files -> fetch raw -> extract human turns ->
  keep coding turns (web-chat is broad-topic) -> dedupe -> save corpus.

Output schema matches real_prompts_corpus.json so classify_prompts.py runs on it
unchanged (just point --in at webchat_corpus.json).

Usage:
    python pull_webchat_corpus.py --limit 300
"""

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(SCRIPT_DIR, "webchat_corpus.json")
SAMPLE_FILE = os.path.join(SCRIPT_DIR, "webchat_sample.txt")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""

# Known mirror repos that store individual real web-chat conversations.
SEED_REPO_DIRS = [
    ("MiPlayer123/turing-test-data-pipeline", "data/raw"),
]
# Code-search phrase that identifies the WildChat mirror format on other repos.
WILDCHAT_SIGNATURE = '"Real user conversation (WildChat)"'

MAX_FILE_BYTES = 2_000_000

# Coding-relevance gate — web-chat is broad-topic (essays, roleplay, etc.), so
# unlike the agentic transcripts we MUST keep only coding-related turns.
CODING_HINTS = re.compile(
    r"\b(fix|bug|error|function|code|python|javascript|java\b|c\+\+|rust|golang|"
    r"compile|traceback|exception|class|method|api|script|implement|debug|"
    r"syntax|variable|array|loop|import|npm|pip|sql|regex|html|css|react|django|"
    r"def |return |for \(|while \(|console\.log|print\()\b", re.IGNORECASE)

NON_ASCII = re.compile(r"[^\x00-\x7f]")
ERROR_PASTE = re.compile(
    r"(Traceback \(most recent call last\)|^\s*File \".*\", line \d+|"
    r"\b\w*(Error|Exception)\b\s*:|SyntaxError|TypeError|NameError|ValueError|"
    r"undefined reference|segmentation fault|panic:|unhandled|"
    r"\bat .+\(.+:\d+:\d+\)|\bline \d+\b.*\b(error|failed)\b)",
    re.IGNORECASE | re.MULTILINE,
)
CODE_PASTE = re.compile(
    r"```|\bdef \w+\s*\(|\bclass \w+|\bfunction \w+\s*\(|=>|;\s*\n|\{\s*\n|import \w+",
)


def _get(url, raw=False, timeout=30):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "vibe-tax-webchat-puller")
    if GITHUB_TOKEN and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data if raw else json.loads(data)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_repo_dir(repo, path):
    """Return [(download_url, repo, path)] for human web-chat files in a dir."""
    out = []
    try:
        items = _get(f"https://api.github.com/repos/{repo}/contents/{path}?per_page=100")
    except Exception as e:
        print(f"  list error {repo}/{path}: {e}")
        return out
    for it in items:
        name = it.get("name", "")
        # WildChat human conversations in the mirror format.
        if "wildchat" in name.lower() and it.get("download_url"):
            out.append((it["download_url"], repo, f"{path}/{name}"))
    return out


def discover_via_search(limit):
    """Find other repos carrying the WildChat mirror format."""
    out = []
    q = urllib.parse.quote(WILDCHAT_SIGNATURE + " extension:json")
    for page in (1, 2):
        try:
            res = _get(f"https://api.github.com/search/code?q={q}&per_page=100&page={page}")
        except Exception as e:
            print(f"  search error: {e}")
            break
        items = res.get("items", [])
        for it in items:
            repo = it.get("repository", {})
            full = repo.get("full_name") if isinstance(repo, dict) else repo
            path = it.get("path")
            ref = "HEAD"
            if "ref=" in it.get("url", ""):
                ref = it["url"].split("ref=")[-1]
            if full and path:
                url = f"https://raw.githubusercontent.com/{full}/{ref}/{urllib.parse.quote(path)}"
                out.append((url, full, path))
            if len(out) >= limit:
                return out
        if len(items) < 100:
            break
        time.sleep(2)
    return out


# ---------------------------------------------------------------------------
# Extraction — handle both the turing-test/WildChat schema and ShareGPT schema
# ---------------------------------------------------------------------------

def extract_human_turns(doc):
    turns = []
    if isinstance(doc, dict) and "turns" in doc:                 # WildChat mirror
        for t in doc["turns"]:
            if t.get("model") == "human" or t.get("speaker", "").startswith("model_a"):
                c = t.get("content")
                if isinstance(c, str):
                    turns.append(c)
    elif isinstance(doc, dict) and "conversations" in doc:        # ShareGPT
        for t in doc["conversations"]:
            if t.get("from") in ("human", "user"):
                v = t.get("value")
                if isinstance(v, str):
                    turns.append(v)
    elif isinstance(doc, list):                                   # list of msgs
        for t in doc:
            if isinstance(t, dict) and t.get("role") == "user":
                c = t.get("content")
                if isinstance(c, str):
                    turns.append(c)
    return turns


def normalize(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def run(limit, max_chars):
    print("Discovering web-chat conversation files ...")
    files = []
    for repo, path in SEED_REPO_DIRS:
        files += list_repo_dir(repo, path)
    print(f"  seed repos: {len(files)} files")
    if len(files) < limit:
        extra = discover_via_search(limit - len(files))
        # de-dup by url
        seen_urls = {u for u, _, _ in files}
        files += [(u, r, p) for (u, r, p) in extra if u not in seen_urls]
    files = files[:limit]
    print(f"  total candidate files: {len(files)}")

    corpus = []
    seen = set()
    convs_used = 0
    for i, (url, repo, path) in enumerate(files):
        try:
            raw = _get(url, raw=True)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] fetch fail {repo}: {e}")
            continue
        if len(raw) > MAX_FILE_BYTES or raw[:40].startswith(b"version https://git-lfs"):
            continue
        try:
            doc = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue

        kept = 0
        for p in extract_human_turns(doc):
            p = p.strip()
            if not (8 <= len(p) <= max_chars):
                continue
            is_coding = bool(CODING_HINTS.search(p))
            has_error = bool(ERROR_PASTE.search(p))
            has_code = bool(CODE_PASTE.search(p))
            # web-chat is broad-topic: keep only coding-relevant turns
            if not (is_coding or has_error or has_code):
                continue
            norm = normalize(p)
            if norm in seen:
                continue
            seen.add(norm)
            corpus.append({
                "prompt": p,
                "repo": repo,
                "path": path,
                "source": "webchat",
                "n_words": len(p.split()),
                "n_chars": len(p),
                "looks_coding": is_coding,
                "non_english": bool(NON_ASCII.search(p)),
                "has_error_paste": has_error,
                "has_code_paste": has_code,
            })
            kept += 1
        if kept:
            convs_used += 1
        time.sleep(0.2)

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    err = sum(1 for c in corpus if c["has_error_paste"])
    code = sum(1 for c in corpus if c["has_code_paste"])
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        f.write("REAL WEB-CHAT CODING PROMPTS (WildChat) — sample\n" + "=" * 60 + "\n\n")
        for c in sorted(corpus, key=lambda x: -x["has_error_paste"])[:80]:
            tag = "ERR" if c["has_error_paste"] else ("CODE" if c["has_code_paste"] else "NL")
            f.write(f"[{tag} | {c['n_words']}w | {c['repo']}]\n{c['prompt'][:600]}\n" + "-" * 40 + "\n")

    print("=" * 60)
    print(f"conversations used : {convs_used}")
    print(f"coding prompts     : {len(corpus)}")
    print(f"  with pasted error: {err} ({err/len(corpus)*100:.0f}%)" if corpus else "  none")
    print(f"  with pasted code : {code} ({code/len(corpus)*100:.0f}%)" if corpus else "")
    print(f"corpus -> {CORPUS_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="max conversation files")
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()
    run(args.limit, args.max_chars)
