"""
Enrich the harvest with per-repo and per-user metadata.

Reads refs.jsonl (and optionally comments.jsonl), collects all unique
repo full-names and user logins seen anywhere in the corpus, then hits
`GET /repos/{r}` and `GET /users/{u}` once each and caches the result
to JSONL. Re-running skips entries already cached.

Two output files:
* repos.jsonl  — one JSON record per repo
* users.jsonl  — one JSON record per user / organization

Each pull is ~1 REST call. With the 5000/hr authenticated budget that's
roughly 5000 unique repos+users per hour.

Usage:
    GITHUB_TOKEN=ghp_xxx python harvest/enrich_metadata.py \\
        --refs     harvest/claude_session_refs.jsonl \\
        --comments harvest/claude_session_comments.jsonl \\
        --repos-out harvest/repos.jsonl \\
        --users-out harvest/users.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gh import gh_get, require_token, setup_logging

log = logging.getLogger("harvest")


class _Stop(Exception):
    """Internal control-flow signal for hitting --max-runtime."""


REPO_FIELDS = [
    "full_name", "id", "created_at", "updated_at", "pushed_at",
    "size", "language", "stargazers_count", "watchers_count",
    "forks_count", "open_issues_count", "subscribers_count",
    "topics", "description", "fork", "archived", "disabled",
    "default_branch", "visibility", "has_issues", "has_wiki",
    "homepage",
]
USER_FIELDS = [
    "login", "id", "type", "created_at", "updated_at",
    "public_repos", "public_gists", "followers", "following",
    "bio", "company", "location", "blog", "name",
    "hireable", "twitter_username",
]


def load_seen(path: str, key: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get(key)
            if v:
                seen.add(v)
    return seen


def collect_targets(refs_path: str, comments_path: str | None
                    ) -> tuple[set[str], set[str]]:
    repos: set[str] = set()
    users: set[str] = set()
    paths = [refs_path] + ([comments_path] if comments_path else [])
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("repo"):
                    repos.add(rec["repo"])
                u = rec.get("user")
                if isinstance(u, str) and u:
                    users.add(u)
                if rec.get("kind") == "parent_meta":
                    for k in ("base_repo", "head_repo"):
                        v = rec.get(k)
                        if v:
                            repos.add(v)
    return repos, users


def fetch_repo(full_name: str, token: str) -> dict:
    r = gh_get(f"/repos/{full_name}", None, token)
    if r.status_code != 200:
        return {"full_name": full_name, "skipped": True,
                "status": r.status_code}
    d = r.json()
    out = {k: d.get(k) for k in REPO_FIELDS}
    out["license_spdx_id"] = (d.get("license") or {}).get("spdx_id")
    out["owner_login"] = (d.get("owner") or {}).get("login")
    out["owner_type"] = (d.get("owner") or {}).get("type")
    return out


def fetch_user(login: str, token: str) -> dict:
    r = gh_get(f"/users/{login}", None, token)
    if r.status_code != 200:
        return {"login": login, "skipped": True, "status": r.status_code}
    d = r.json()
    return {k: d.get(k) for k in USER_FIELDS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--refs", default="harvest/claude_session_refs.jsonl")
    ap.add_argument("--comments",
                    default="harvest/claude_session_comments.jsonl")
    ap.add_argument("--repos-out", default="harvest/repos.jsonl")
    ap.add_argument("--users-out", default="harvest/users.jsonl")
    ap.add_argument("--max-runtime", type=int, default=0,
                    help="stop after N seconds of wall time (0 = unlimited)")
    ap.add_argument("--log", default="harvest/enrich_metadata.log",
                    help="log file path (empty string disables)")
    args = ap.parse_args()

    setup_logging(args.log or None)
    token = require_token()
    for path in (args.repos_out, args.users_out):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    repos, users = collect_targets(args.refs, args.comments)
    have_repos = load_seen(args.repos_out, "full_name")
    have_users = load_seen(args.users_out, "login")
    new_repos = sorted(repos - have_repos)
    new_users = sorted(users - have_users)
    log.info("repos: %d unique, %d new", len(repos), len(new_repos))
    log.info("users: %d unique, %d new", len(users), len(new_users))

    start = time.time()
    n_repos = n_users = 0
    n_repos_skipped = n_users_skipped = 0
    stop_reason = "completed"

    try:
        with open(args.repos_out, "a", encoding="utf-8") as out:
            for i, r in enumerate(new_repos, 1):
                rec = fetch_repo(r, token)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if rec.get("skipped"):
                    n_repos_skipped += 1
                n_repos += 1
                if i % 50 == 0:
                    log.info("  repos %d/%d", i, len(new_repos))
                if (args.max_runtime
                        and (time.time() - start) >= args.max_runtime):
                    stop_reason = f"hit --max-runtime={args.max_runtime}s"
                    log.warning(stop_reason)
                    raise _Stop()
                time.sleep(0.25)

        with open(args.users_out, "a", encoding="utf-8") as out:
            for i, u in enumerate(new_users, 1):
                rec = fetch_user(u, token)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if rec.get("skipped"):
                    n_users_skipped += 1
                n_users += 1
                if i % 50 == 0:
                    log.info("  users %d/%d", i, len(new_users))
                if (args.max_runtime
                        and (time.time() - start) >= args.max_runtime):
                    stop_reason = f"hit --max-runtime={args.max_runtime}s"
                    log.warning(stop_reason)
                    raise _Stop()
                time.sleep(0.25)
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
        log.info("repos fetched: %d   skipped: %d", n_repos, n_repos_skipped)
        log.info("users fetched: %d   skipped: %d", n_users, n_users_skipped)
        log.info("outputs: %s , %s", args.repos_out, args.users_out)
        if args.log:
            log.info("log:     %s", args.log)

    return 0 if stop_reason in ("completed",) or stop_reason.startswith(
        ("hit --", "interrupted")) else 1


if __name__ == "__main__":
    sys.exit(main())
