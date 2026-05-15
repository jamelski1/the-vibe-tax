"""
Pull comment threads for each PR/issue captured by harvest_claude_sessions.

For each PR or issue in refs.jsonl, fetches:
* parent metadata (state, merged_at, closed_at, commit/comment counts,
  additions/deletions/changed_files for PRs)
* issue comments (PR/issue conversations)
* pull-request review comments (line-level discussions, PRs only)
* pull-request reviews (review summaries, PRs only)

Writes one JSONL line per comment plus one `parent_meta` line per
PR/issue. Re-running skips parents already processed.

Rate-limit budget: ~3-4 REST calls per PR (1 meta + 1-3 comment lists).
Authenticated REST is 5000 req/hr, so ~1200-1500 PRs/hr is the
practical ceiling.

Usage:
    GITHUB_TOKEN=ghp_xxx python harvest/harvest_comments.py \\
        --refs harvest/claude_session_refs.jsonl \\
        --out  harvest/claude_session_comments.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import gh_get, iter_paginated, require_token


def load_seen_parents(path: str) -> set[tuple]:
    """Return set of (kind, repo, number) for parents already processed."""
    seen: set[tuple] = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "parent_meta":
                seen.add((rec["parent_kind"], rec["repo"], rec["number"]))
    return seen


def fetch_pr_meta(repo: str, number: int, token: str) -> dict:
    r = gh_get(f"/repos/{repo}/pulls/{number}", None, token)
    if r.status_code != 200:
        return {"kind": "parent_meta", "parent_kind": "pr", "repo": repo,
                "number": number, "skipped": True, "status": r.status_code}
    d = r.json()
    return {
        "kind": "parent_meta",
        "parent_kind": "pr",
        "repo": repo,
        "number": number,
        "state": d.get("state"),
        "merged_at": d.get("merged_at"),
        "closed_at": d.get("closed_at"),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        "commits": d.get("commits"),
        "additions": d.get("additions"),
        "deletions": d.get("deletions"),
        "changed_files": d.get("changed_files"),
        "comments": d.get("comments"),
        "review_comments": d.get("review_comments"),
        "user": (d.get("user") or {}).get("login"),
        "draft": d.get("draft"),
        "title": d.get("title"),
        "base_repo": ((d.get("base") or {}).get("repo") or {}).get("full_name"),
        "head_repo": ((d.get("head") or {}).get("repo") or {}).get("full_name"),
        "labels": [l.get("name") for l in d.get("labels") or []],
    }


def fetch_issue_meta(repo: str, number: int, token: str) -> dict:
    r = gh_get(f"/repos/{repo}/issues/{number}", None, token)
    if r.status_code != 200:
        return {"kind": "parent_meta", "parent_kind": "issue", "repo": repo,
                "number": number, "skipped": True, "status": r.status_code}
    d = r.json()
    return {
        "kind": "parent_meta",
        "parent_kind": "issue",
        "repo": repo,
        "number": number,
        "state": d.get("state"),
        "closed_at": d.get("closed_at"),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        "comments": d.get("comments"),
        "user": (d.get("user") or {}).get("login"),
        "title": d.get("title"),
        "labels": [l.get("name") for l in d.get("labels") or []],
    }


def iter_issue_comments(repo: str, number: int, token: str):
    for c in iter_paginated(f"/repos/{repo}/issues/{number}/comments",
                            {}, token):
        yield {
            "comment_kind": "issue_comment",
            "id": c.get("id"),
            "user": (c.get("user") or {}).get("login"),
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
            "body": c.get("body"),
        }


def iter_review_comments(repo: str, number: int, token: str):
    for c in iter_paginated(f"/repos/{repo}/pulls/{number}/comments",
                            {}, token):
        yield {
            "comment_kind": "review_comment",
            "id": c.get("id"),
            "user": (c.get("user") or {}).get("login"),
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
            "body": c.get("body"),
            "path": c.get("path"),
            "line": c.get("line"),
            "in_reply_to_id": c.get("in_reply_to_id"),
        }


def iter_reviews(repo: str, number: int, token: str):
    for c in iter_paginated(f"/repos/{repo}/pulls/{number}/reviews",
                            {}, token):
        yield {
            "comment_kind": "review",
            "id": c.get("id"),
            "user": (c.get("user") or {}).get("login"),
            "submitted_at": c.get("submitted_at"),
            "state": c.get("state"),
            "body": c.get("body"),
        }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--refs", default="harvest/claude_session_refs.jsonl")
    ap.add_argument("--out", default="harvest/claude_session_comments.jsonl")
    ap.add_argument("--kinds", default="pr,issue",
                    help="parent kinds to enrich (commit threads not pulled)")
    args = ap.parse_args()

    token = require_token()
    if not os.path.exists(args.refs):
        print(f"ERROR: refs file not found: {args.refs}", file=sys.stderr)
        return 1

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    seen = load_seen_parents(args.out)
    print(f"resuming with {len(seen)} parents already processed")

    todo: list[tuple[str, str, int]] = []
    with open(args.refs, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind")
            if kind not in kinds:
                continue
            repo, number = rec.get("repo"), rec.get("number")
            if not repo or number is None:
                continue
            key = (kind, repo, number)
            if key in seen:
                continue
            todo.append(key)

    todo = list(dict.fromkeys(todo))  # preserve order, drop dupes
    print(f"to process: {len(todo)} parents")

    with open(args.out, "a", encoding="utf-8") as out:
        for i, (kind, repo, number) in enumerate(todo, 1):
            meta = (fetch_pr_meta(repo, number, token) if kind == "pr"
                    else fetch_issue_meta(repo, number, token))
            out.write(json.dumps(meta, ensure_ascii=False) + "\n")

            if not meta.get("skipped"):
                for c in iter_issue_comments(repo, number, token):
                    rec = {"kind": "comment", "parent_kind": kind,
                           "repo": repo, "parent_number": number, **c}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if kind == "pr":
                    for c in iter_review_comments(repo, number, token):
                        rec = {"kind": "comment", "parent_kind": kind,
                               "repo": repo, "parent_number": number, **c}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for c in iter_reviews(repo, number, token):
                        rec = {"kind": "comment", "parent_kind": kind,
                               "repo": repo, "parent_number": number, **c}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            out.flush()
            if i % 25 == 0:
                print(f"  processed {i}/{len(todo)}")
            time.sleep(0.5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
