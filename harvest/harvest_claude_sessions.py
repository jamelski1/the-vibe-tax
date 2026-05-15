"""
Harvest public GitHub artifacts that reference Claude Code session links.

Claude Code appends a footer like

    https://claude.ai/code/session_01P51qUYxv7MMiiHmTBdCLhQ

to the commit messages, PR descriptions, and issue bodies it produces.
The session URL itself is private (it deep-links into the original
user's Claude account and requires auth to view), but the GitHub
artifact that contains it is public. The human-authored text around the
link -- the PR body, issue body, commit message, review thread -- is a
faithful, public record of the natural-language prompts a user gave to
Claude Code.

This script harvests those public artifacts so they can later be mined
for "vibe prompts" with a clean provenance trail.

Strategy
--------
* Use GitHub's search APIs (/search/issues, /search/commits) to find
  every PR / issue / commit whose body contains the phrase
  "claude.ai/code/session".
* GitHub's search API caps each query at 1000 results, so slice by
  date range (default 7-day windows) and recurse if a single window
  trips the cap.
* Write one JSON record per match to a JSONL file. Re-running the
  script reads the existing file and skips duplicates, so it is safe
  to resume after a crash or rate-limit pause.
* Authenticated search is limited to 30 req/min; we sleep 1s between
  paginated calls and let the script back off on 403/429.

Usage
-----
    GITHUB_TOKEN=ghp_xxx python harvest/harvest_claude_sessions.py \
        --since 2024-06-01 --until 2026-05-15 \
        --kinds pr,issue,commit \
        --out harvest/claude_session_refs.jsonl

The token only needs `public_repo` scope (or no scope at all for a
fine-grained PAT with read-only public access).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from typing import Iterator

import requests

API = "https://api.github.com"
SESSION_RE = re.compile(r"https?://claude\.ai/code/session_[A-Za-z0-9_-]+")
SEARCH_PHRASE = '"claude.ai/code/session"'
SEARCH_CAP = 1000  # GitHub-imposed per-query result cap


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def gh_get(path: str, params: dict, token: str) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "vibe-tax-harvester",
    }
    url = f"{API}{path}"
    for attempt in range(6):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429):
            reset = r.headers.get("x-ratelimit-reset")
            if reset and reset.isdigit():
                wait = max(2, int(reset) - int(time.time()) + 2)
            else:
                wait = min(60, 2 ** (attempt + 1))
            print(f"  rate-limited ({r.status_code}); sleeping {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code == 422:
            # malformed query, or window has too many results
            return r
        r.raise_for_status()
    r.raise_for_status()  # final
    return r  # unreachable


# ---------------------------------------------------------------------------
# Date slicing
# ---------------------------------------------------------------------------

def slice_dates(since: date, until: date, step_days: int
                ) -> Iterator[tuple[date, date]]:
    cur = since
    while cur <= until:
        end = min(cur + timedelta(days=step_days - 1), until)
        yield cur, end
        cur = end + timedelta(days=1)


# ---------------------------------------------------------------------------
# Search drivers
# ---------------------------------------------------------------------------

def _iter_search(path: str, base_query: str, date_field: str,
                 lo: date, hi: date, token: str,
                 step_days: int) -> Iterator[dict]:
    """Paginate one date window, recursively halving on 1000-result cap."""
    q = f"{base_query} {date_field}:{lo}..{hi}"
    page = 1
    seen_count = 0
    total = None
    while True:
        r = gh_get(path, {"q": q, "per_page": 100, "page": page}, token)
        if r.status_code != 200:
            if r.status_code == 422 and (hi - lo).days > 0:
                # window too big -- split in half
                mid = lo + (hi - lo) // 2
                yield from _iter_search(path, base_query, date_field,
                                        lo, mid, token, step_days)
                yield from _iter_search(path, base_query, date_field,
                                        mid + timedelta(days=1), hi,
                                        token, step_days)
                return
            print(f"  {r.status_code} on {lo}..{hi}: {r.text[:160]}",
                  file=sys.stderr)
            return

        data = r.json()
        if total is None:
            total = data.get("total_count", 0)
            if total > SEARCH_CAP and (hi - lo).days > 0:
                # window will be truncated -- subdivide
                mid = lo + (hi - lo) // 2
                yield from _iter_search(path, base_query, date_field,
                                        lo, mid, token, step_days)
                yield from _iter_search(path, base_query, date_field,
                                        mid + timedelta(days=1), hi,
                                        token, step_days)
                return

        items = data.get("items", [])
        for item in items:
            yield item
        seen_count += len(items)
        if len(items) < 100 or seen_count >= SEARCH_CAP:
            return
        page += 1
        time.sleep(1.0)


def search_issues_or_prs(kind: str, since: date, until: date,
                         token: str, step_days: int) -> Iterator[dict]:
    assert kind in ("issue", "pr")
    base = f"{SEARCH_PHRASE} in:body type:{kind}"
    for lo, hi in slice_dates(since, until, step_days):
        for item in _iter_search("/search/issues", base, "created",
                                 lo, hi, token, step_days):
            body = item.get("body") or ""
            m = SESSION_RE.search(body)
            if not m:
                continue
            repo_url = item.get("repository_url", "")
            repo = "/".join(repo_url.rsplit("/", 2)[-2:]) if repo_url else None
            yield {
                "kind": kind,
                "repo": repo,
                "number": item.get("number"),
                "title": item.get("title"),
                "body": body,
                "html_url": item.get("html_url"),
                "user": (item.get("user") or {}).get("login"),
                "created_at": item.get("created_at"),
                "session_url": m.group(0),
            }
        time.sleep(1.0)


def search_commits(since: date, until: date, token: str,
                   step_days: int) -> Iterator[dict]:
    base = SEARCH_PHRASE
    for lo, hi in slice_dates(since, until, step_days):
        for item in _iter_search("/search/commits", base, "committer-date",
                                 lo, hi, token, step_days):
            commit = item.get("commit") or {}
            msg = commit.get("message") or ""
            m = SESSION_RE.search(msg)
            if not m:
                continue
            author = item.get("author") or {}
            commit_author = commit.get("author") or {}
            yield {
                "kind": "commit",
                "repo": (item.get("repository") or {}).get("full_name"),
                "sha": item.get("sha"),
                "title": msg.split("\n", 1)[0],
                "body": msg,
                "html_url": item.get("html_url"),
                "user": author.get("login") or commit_author.get("name"),
                "created_at": commit_author.get("date"),
                "session_url": m.group(0),
            }
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Output / dedupe
# ---------------------------------------------------------------------------

def record_key(rec: dict) -> tuple:
    if rec["kind"] == "commit":
        return ("commit", rec.get("repo"), rec.get("sha"))
    return (rec["kind"], rec.get("repo"), rec.get("number"))


def load_seen(path: str) -> set[tuple]:
    seen: set[tuple] = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--until", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--kinds", default="pr,issue,commit",
                    help="comma-separated subset of {pr,issue,commit}")
    ap.add_argument("--out", default="harvest/claude_session_refs.jsonl")
    ap.add_argument("--step-days", type=int, default=7,
                    help="initial date-window size; auto-halves on cap")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN env var required", file=sys.stderr)
        return 1

    since = date.fromisoformat(args.since)
    until = date.fromisoformat(args.until)
    if since > until:
        print("ERROR: --since must be <= --until", file=sys.stderr)
        return 1

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    bad = [k for k in kinds if k not in ("pr", "issue", "commit")]
    if bad:
        print(f"ERROR: unknown kinds {bad}", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    seen = load_seen(args.out)
    print(f"resuming with {len(seen)} existing records in {args.out}")

    with open(args.out, "a", encoding="utf-8") as out:
        for kind in kinds:
            print(f"=== {kind} ===  {since}..{until}")
            if kind == "commit":
                gen = search_commits(since, until, token, args.step_days)
            else:
                gen = search_issues_or_prs(kind, since, until, token,
                                           args.step_days)
            n_new = 0
            for rec in gen:
                k = record_key(rec)
                if k in seen:
                    continue
                seen.add(k)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_new += 1
                if n_new % 50 == 0:
                    print(f"  {kind}: {n_new} new so far")
            print(f"  {kind}: {n_new} new records")

    return 0


if __name__ == "__main__":
    sys.exit(main())
