"""
Crawl GitHub for committed Claude Code session transcripts and harvest the
REAL human prompts people typed while coding with AI.

Pipeline
--------
1. DISCOVER  — GitHub code search (paginated) for committed *.jsonl transcripts
               that carry Claude Code's distinctive fields.
2. FETCH     — download each raw file from raw.githubusercontent.com.
3. EXTRACT   — parse JSONL, keep genuine human `type:"user"` turns, drop tool
               results / interrupts / system-injected / slash-command noise.
4. DEDUPE    — drop exact + near-duplicate prompts.
5. SCORE     — rough informality ("vibe") heuristic, 0.0 (formal) .. 1.0 (vibe).
6. SAVE      — corpus JSON + readable sample + stats.

This reads ONLY public GitHub data. Each prompt keeps its source repo + path so
provenance is auditable.

Usage:
    python crawl_real_prompts.py --limit 80          # pilot
    python crawl_real_prompts.py --limit 1000         # bigger crawl

Env:
    GITHUB_TOKEN / GH_TOKEN  — optional, raises search rate limits.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(SCRIPT_DIR, "real_prompts_corpus.json")
SAMPLE_FILE = os.path.join(SCRIPT_DIR, "real_prompts_sample.txt")
STATS_FILE = os.path.join(SCRIPT_DIR, "real_prompts_stats.json")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""

# Search queries targeting Claude Code transcript files. GitHub code search caps
# each query at 1000 results, so we use several complementary angles.
#
# `"userType":"external"` is the strongest signal for a MAIN session that holds
# genuine human-typed turns — sub-agent sidechain transcripts (agent-*.jsonl)
# don't carry external user turns. It leads the list so the crawl budget is
# spent on files that actually contain prompts.
SEARCH_QUERIES = [
    '"userType":"external" extension:jsonl',
    '"userType" "external" "parentUuid" extension:jsonl',
    '"sessionId" "toolUseResult" extension:jsonl',
    'path:.claude "parentUuid" extension:jsonl',
]

# Sub-agent sidechain transcripts follow this naming convention and never hold
# human-typed prompts — skip them at discovery time.
SIDECHAIN_FILENAME = re.compile(r"(^|/)agent-[0-9a-f]+\.jsonl$", re.IGNORECASE)

# Max bytes to download per transcript (skip giant logs).
MAX_FILE_BYTES = 3_000_000

# Don't let one repo with hundreds of transcript files dominate the crawl.
MAX_FILES_PER_REPO = 6

# Paths that mark a transcript as a test fixture / example rather than a real
# coding session — these poison the corpus with "test", "hello", etc.
FIXTURE_PATH = re.compile(
    r"(fixture|sample|example|/test|test[s]?/|mock|demo|/spec|__|dummy|template)",
    re.IGNORECASE,
)

# Politeness / network throttle.
SEARCH_DELAY = 2.0   # GitHub code search is heavily rate-limited (~10/min auth)
FETCH_DELAY = 0.3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url, accept="application/vnd.github+json", timeout=30, raw=False):
    req = urllib.request.Request(url)
    req.add_header("Accept", accept)
    req.add_header("User-Agent", "vibe-tax-prompt-crawler")
    if GITHUB_TOKEN and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data if raw else json.loads(data)


def search_code(query, page, per_page=100):
    q = urllib.parse.quote(query)
    url = (
        f"https://api.github.com/search/code"
        f"?q={q}&per_page={per_page}&page={page}"
    )
    return _get(url)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def raw_url_for(item):
    """Build a raw.githubusercontent.com URL from a search result item."""
    repo = item.get("repository", {})
    full_name = repo.get("full_name") if isinstance(repo, dict) else repo
    path = item.get("path")
    # ref lives in the contents API url: .../contents/<path>?ref=<sha>
    ref = "HEAD"
    api_url = item.get("url", "")
    if "ref=" in api_url:
        ref = api_url.split("ref=")[-1]
    if not full_name or not path:
        return None
    return f"https://raw.githubusercontent.com/{full_name}/{ref}/{urllib.parse.quote(path)}"


def discover(limit, log):
    """Page through searches and collect unique candidate transcript files.

    Caps files-per-repo for diversity and skips obvious fixture paths so the
    crawl budget is spent on real sessions, not test data.
    """
    seen = set()
    per_repo = {}
    candidates = []
    for query in SEARCH_QUERIES:
        if len(candidates) >= limit:
            break
        page = 1
        while len(candidates) < limit:
            log(f"  search '{query}' page {page} ...")
            try:
                res = search_code(query, page)
            except Exception as e:
                log(f"    search error: {e}")
                break
            items = res.get("items", [])
            if not items:
                break
            for it in items:
                sha = it.get("sha")
                key = sha or raw_url_for(it)
                if not key or key in seen:
                    continue
                seen.add(key)
                path = it.get("path") or ""
                if FIXTURE_PATH.search(path):
                    continue
                if SIDECHAIN_FILENAME.search(path):
                    continue
                repo = it.get("repository", {})
                repo_name = repo.get("full_name") if isinstance(repo, dict) else repo
                if per_repo.get(repo_name, 0) >= MAX_FILES_PER_REPO:
                    continue
                url = raw_url_for(it)
                if not url:
                    continue
                per_repo[repo_name] = per_repo.get(repo_name, 0) + 1
                candidates.append({
                    "repo": repo_name,
                    "path": path,
                    "sha": sha,
                    "raw_url": url,
                })
                if len(candidates) >= limit:
                    break
            if len(items) < 100:
                break
            page += 1
            time.sleep(SEARCH_DELAY)
        time.sleep(SEARCH_DELAY)
    return candidates


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# Substrings that mark a "user" turn as machine-injected rather than human-typed.
NOISE_MARKERS = (
    "[Request interrupted",
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<system-reminder>",
    "<bash-",
    "tool_use_id",
    "Caveat: The messages below",
    "This session is being continued",
    "<user-prompt-submit-hook>",
)


def _turn_text(message):
    """Pull plain human text out of a transcript message; '' if none/non-human."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            # tool_result blocks are machine output, not a human prompt
            if p.get("type") == "tool_result":
                return ""
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
        return " ".join(parts).strip()
    return ""


def extract_prompts(jsonl_text):
    """Yield genuine human prompt strings from one transcript file."""
    prompts = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "user":
            continue
        if o.get("isMeta"):
            continue
        # Sub-agent / sidechain turns are not human-typed prompts.
        if o.get("isSidechain"):
            continue
        # When the transcript records a userType, require it to be the human one.
        # (Older transcripts may omit the field — keep those rather than lose data.)
        utype = o.get("userType")
        if utype is not None and utype != "external":
            continue
        msg = o.get("message") or {}
        if msg.get("role") and msg.get("role") != "user":
            continue
        text = _turn_text(msg)
        if not text:
            continue
        if any(m in text for m in NOISE_MARKERS):
            continue
        if text.startswith("<"):           # leftover xml-ish wrappers
            continue
        if text.startswith("/"):           # slash command
            continue
        prompts.append(text)
    return prompts


# ---------------------------------------------------------------------------
# Dedupe + scoring
# ---------------------------------------------------------------------------

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


CODING_HINTS = re.compile(
    r"\b(fix|add|implement|refactor|function|bug|test|error|build|run|class|"
    r"method|api|file|code|script|component|deploy|install|import|update|"
    r"create|write|change|make|remove|debug|commit|feature|endpoint)\b",
    re.IGNORECASE,
)

POLITE = re.compile(r"\b(please|could you|would you|thanks|thank you|can you)\b", re.IGNORECASE)


def informality_score(text):
    """Rough 0.0 (formal spec) .. 1.0 (terse vibe) heuristic."""
    t = text.strip()
    if not t:
        return 0.5
    score = 0.5
    words = t.split()
    n = len(words)
    # short prompts read as vibe
    if n <= 6:
        score += 0.25
    elif n <= 15:
        score += 0.1
    elif n >= 60:
        score -= 0.25
    # no terminal punctuation -> casual
    if t[-1] not in ".!?":
        score += 0.1
    # all lowercase first letter -> casual
    if t[:1].islower():
        score += 0.1
    # politeness markers -> slightly more formal/considered
    if POLITE.search(t):
        score -= 0.05
    # bullet/numbered structure -> formal spec
    if re.search(r"(\n\s*[-*\d]|```)", t):
        score -= 0.2
    return max(0.0, min(1.0, round(score, 3)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


# Trivial greetings / filler that are technically "user" turns but carry no task.
TRIVIAL = re.compile(
    r"^(hi|hey|hello|yo|ohayo|thanks|thank you|ok|okay|yes|no|yep|nope|cool|"
    r"nice|great|test|testing|warmup|hello[, ]+claude|continue|go|done)\b",
    re.IGNORECASE,
)


def run(limit, max_prompt_chars, min_prompt_chars, coding_only):
    log("=" * 60)
    log("VIBE TAX — Real Prompt Crawler")
    log("=" * 60)
    log(f"Token: {'set' if GITHUB_TOKEN else 'NONE (low rate limit)'}")

    log(f"[1/4] Discovering up to {limit} transcript files ...")
    candidates = discover(limit, log)
    log(f"  found {len(candidates)} candidate files")

    log("[2/4] Fetching + extracting ...")
    corpus = []
    seen_norm = set()
    transcripts_used = 0
    transcripts_fetched = 0
    repos = set()

    for i, c in enumerate(candidates):
        try:
            raw = _get(c["raw_url"], raw=True, timeout=30)
        except Exception as e:
            log(f"  [{i+1}/{len(candidates)}] fetch fail {c['repo']}: {e}")
            continue
        transcripts_fetched += 1
        if len(raw) > MAX_FILE_BYTES:
            log(f"  [{i+1}/{len(candidates)}] skip (too big) {c['repo']}")
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        prompts = extract_prompts(text)
        kept_here = 0
        for p in prompts:
            if not (min_prompt_chars <= len(p) <= max_prompt_chars):
                continue
            if TRIVIAL.match(p.strip()):
                continue
            is_coding = bool(CODING_HINTS.search(p))
            if coding_only and not is_coding:
                continue
            norm = normalize(p)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            corpus.append({
                "prompt": p,
                "repo": c["repo"],
                "path": c["path"],
                "n_words": len(p.split()),
                "n_chars": len(p),
                "looks_coding": is_coding,
                "informality": informality_score(p),
            })
            kept_here += 1
        if kept_here:
            transcripts_used += 1
            repos.add(c["repo"])
        log(f"  [{i+1}/{len(candidates)}] {c['repo']}: +{kept_here} prompts")
        time.sleep(FETCH_DELAY)

    log("[3/4] Saving corpus ...")
    corpus.sort(key=lambda x: x["informality"], reverse=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    coding = [c for c in corpus if c["looks_coding"]]
    stats = {
        "crawled_at": datetime.now().isoformat(),
        "candidate_files": len(candidates),
        "transcripts_fetched": transcripts_fetched,
        "transcripts_with_prompts": transcripts_used,
        "hit_rate": f"{transcripts_used / transcripts_fetched * 100:.0f}%" if transcripts_fetched else "N/A",
        "unique_repos": len(repos),
        "total_prompts": len(corpus),
        "coding_prompts": len(coding),
        "avg_words": round(sum(c["n_words"] for c in corpus) / len(corpus), 1) if corpus else 0,
        "informality_buckets": {
            "formal_0.0-0.4": sum(1 for c in corpus if c["informality"] < 0.4),
            "mid_0.4-0.6": sum(1 for c in corpus if 0.4 <= c["informality"] < 0.6),
            "vibe_0.6-1.0": sum(1 for c in corpus if c["informality"] >= 0.6),
        },
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log("[4/4] Writing readable sample ...")
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        f.write("REAL VIBE-CODING PROMPTS — sample (sorted most-vibe first)\n")
        f.write("=" * 60 + "\n\n")
        for c in corpus[:120]:
            f.write(f"[informality {c['informality']:.2f} | {c['n_words']}w | {c['repo']}]\n")
            f.write(c["prompt"] + "\n")
            f.write("-" * 40 + "\n")

    log("=" * 60)
    log("DONE")
    log(f"  candidate files     : {stats['candidate_files']}")
    log(f"  transcripts fetched : {stats['transcripts_fetched']}")
    log(f"  transcripts w/prompt: {stats['transcripts_with_prompts']}")
    log(f"  unique repos        : {stats['unique_repos']}")
    log(f"  total prompts       : {stats['total_prompts']}")
    log(f"  coding prompts      : {stats['coding_prompts']}")
    log(f"  informality buckets : {stats['informality_buckets']}")
    log(f"  corpus -> {CORPUS_FILE}")
    log(f"  sample -> {SAMPLE_FILE}")
    log(f"  stats  -> {STATS_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80, help="max transcript files to crawl")
    ap.add_argument("--max-chars", type=int, default=2000, help="max prompt length to keep")
    ap.add_argument("--min-chars", type=int, default=8, help="min prompt length to keep")
    ap.add_argument("--all-prompts", action="store_true",
                    help="keep non-coding prompts too (default: coding-only)")
    args = ap.parse_args()
    run(args.limit, args.max_chars, args.min_chars, coding_only=not args.all_prompts)
