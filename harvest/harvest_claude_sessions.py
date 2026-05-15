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
    # Probe first: see how big the corpus is before committing
    GITHUB_TOKEN=ghp_xxx python harvest/harvest_claude_sessions.py \\
        --since 2026-05-01 --until 2026-05-15 --probe

    # Real run with safety caps
    GITHUB_TOKEN=ghp_xxx python harvest/harvest_claude_sessions.py \\
        --since 2026-05-01 --until 2026-05-15 \\
        --kinds pr,issue,commit \\
        --max-records 5000 --max-runtime 3600 \\
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import gh_get, require_token, setup_logging  # noqa: E402

import logging  # noqa: E402

log = logging.getLogger("harvest")

SESSION_RE = re.compile(r"https?://claude\.ai/code/session_[A-Za-z0-9_-]+")
SEARCH_PHRASE = '"claude.ai/code/session"'
SEARCH_CAP = 1000  # GitHub-imposed per-query result cap


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
                 lo: date, hi: date, token: str) -> Iterator[dict]:
    """Paginate one date window, recursively halving on 1000-result cap."""
    q = f"{base_query} {date_field}:{lo}..{hi}"
    page = 1
    seen_count = 0
    total = None
    while True:
        r = gh_get(path, {"q": q, "per_page": 100, "page": page}, token)
        if r.status_code != 200:
            if r.status_code == 422 and (hi - lo).days > 0:
                mid = lo + (hi - lo) // 2
                yield from _iter_search(path, base_query, date_field,
                                        lo, mid, token)
                yield from _iter_search(path, base_query, date_field,
                                        mid + timedelta(days=1), hi, token)
                return
            log.warning("%s on %s..%s: %s", r.status_code, lo, hi,
                        r.text[:160])
            return

        data = r.json()
        if total is None:
            total = data.get("total_count", 0)
            log.debug("window %s..%s total_count=%d", lo, hi, total)
            if total > SEARCH_CAP and (hi - lo).days > 0:
                mid = lo + (hi - lo) // 2
                log.info("window %s..%s has %d results > cap; halving",
                         lo, hi, total)
                yield from _iter_search(path, base_query, date_field,
                                        lo, mid, token)
                yield from _iter_search(path, base_query, date_field,
                                        mid + timedelta(days=1), hi, token)
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
                                 lo, hi, token):
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
                                 lo, hi, token):
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
# Probe mode
# ---------------------------------------------------------------------------

def probe(since: date, until: date, kinds: list[str], step_days: int,
          token: str) -> None:
    """Report total_count per (kind, window) without paginating or writing."""
    log.info("PROBE MODE: %s..%s, step=%dd, kinds=%s",
             since, until, step_days, ",".join(kinds))
    grand_total = 0
    for kind in kinds:
        if kind == "commit":
            path, base, df = "/search/commits", SEARCH_PHRASE, "committer-date"
        else:
            path = "/search/issues"
            base = f"{SEARCH_PHRASE} in:body type:{kind}"
            df = "created"
        kind_total = 0
        for lo, hi in slice_dates(since, until, step_days):
            q = f"{base} {df}:{lo}..{hi}"
            r = gh_get(path, {"q": q, "per_page": 1}, token)
            if r.status_code == 200:
                t = r.json().get("total_count", 0)
                kind_total += t
                cap_note = "  *CAPPED*" if t > SEARCH_CAP else ""
                log.info("  %-6s %s..%s  total=%d%s", kind, lo, hi, t,
                         cap_note)
            else:
                log.warning("  %-6s %s..%s  status=%d", kind, lo, hi,
                            r.status_code)
            time.sleep(2.0)  # search is 30 req/min
        log.info("  %s total: %d (sum across windows; will hit cap on any "
                 "window >%d)", kind, kind_total, SEARCH_CAP)
        grand_total += kind_total
    log.info("GRAND TOTAL across kinds (raw, pre-dedupe): %d", grand_total)


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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--since", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--until", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--kinds", default="pr,issue,commit",
                    help="comma-separated subset of {pr,issue,commit}")
    ap.add_argument("--out", default="harvest/claude_session_refs.jsonl")
    ap.add_argument("--step-days", type=int, default=7,
                    help="initial date-window size; auto-halves on cap")
    ap.add_argument("--max-records", type=int, default=0,
                    help="stop after writing N new records (0 = unlimited)")
    ap.add_argument("--max-runtime", type=int, default=0,
                    help="stop after N seconds of wall time (0 = unlimited)")
    ap.add_argument("--probe", action="store_true",
                    help="report total_count per window, write nothing, exit")
    ap.add_argument("--log", default="harvest/harvest_claude_sessions.log",
                    help="log file path (empty string disables)")
    args = ap.parse_args()

    setup_logging(args.log or None)
    token = require_token()

    try:
        since = date.fromisoformat(args.since)
        until = date.fromisoformat(args.until)
    except ValueError as e:
        log.error("bad date: %s", e)
        return 1
    if since > until:
        log.error("--since must be <= --until")
        return 1

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    bad = [k for k in kinds if k not in ("pr", "issue", "commit")]
    if bad:
        log.error("unknown kinds %s", bad)
        return 1

    if args.probe:
        probe(since, until, kinds, args.step_days, token)
        return 0

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    seen = load_seen(args.out)
    log.info("resuming with %d existing records in %s", len(seen), args.out)
    log.info("range %s..%s, kinds=%s, step=%dd, caps: records=%s runtime=%ss",
             since, until, ",".join(kinds), args.step_days,
             args.max_records or "off", args.max_runtime or "off")

    start = time.time()
    counts: dict[str, int] = {k: 0 for k in kinds}
    total_new = 0
    stop_reason = "completed"
    last_window: tuple[str, str] | None = None

    try:
        with open(args.out, "a", encoding="utf-8") as out:
            for kind in kinds:
                log.info("=== %s ===  %s..%s", kind, since, until)
                if kind == "commit":
                    gen = search_commits(since, until, token, args.step_days)
                else:
                    gen = search_issues_or_prs(kind, since, until, token,
                                               args.step_days)
                for rec in gen:
                    k = record_key(rec)
                    if k in seen:
                        continue
                    seen.add(k)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    counts[kind] += 1
                    total_new += 1
                    last_window = (kind, rec.get("created_at") or "")
                    if counts[kind] % 50 == 0:
                        log.info("  %s: %d new so far", kind, counts[kind])
                    if args.max_records and total_new >= args.max_records:
                        stop_reason = f"hit --max-records={args.max_records}"
                        log.warning(stop_reason)
                        raise _Stop()
                    if (args.max_runtime
                            and (time.time() - start) >= args.max_runtime):
                        stop_reason = f"hit --max-runtime={args.max_runtime}s"
                        log.warning(stop_reason)
                        raise _Stop()
                log.info("  %s: %d new records this run", kind, counts[kind])
    except KeyboardInterrupt:
        stop_reason = "interrupted by user (SIGINT)"
        log.warning(stop_reason)
    except _Stop:
        pass
    except Exception as e:
        stop_reason = f"crashed: {type(e).__name__}: {e}"
        log.exception("unhandled exception")
    finally:
        elapsed = time.time() - start
        log.info("---- summary ----")
        log.info("stop reason: %s", stop_reason)
        log.info("elapsed: %.1fs (%.1fm)", elapsed, elapsed / 60)
        log.info("new records by kind: %s", counts)
        log.info("new records total: %d", total_new)
        if last_window:
            log.info("last record: kind=%s created_at=%s", *last_window)
        log.info("output: %s", args.out)
        if args.log:
            log.info("log:    %s", args.log)

    return 0 if stop_reason in ("completed",) or stop_reason.startswith(
        ("hit --", "interrupted")) else 1


class _Stop(Exception):
    """Internal control-flow signal for hitting a configured cap."""


if __name__ == "__main__":
    sys.exit(main())
