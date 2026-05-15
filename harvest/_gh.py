"""Shared GitHub REST helpers for the harvest scripts."""

from __future__ import annotations

import os
import sys
import time

import requests

API = "https://api.github.com"


def make_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "vibe-tax-harvester",
    }


def gh_get(path: str, params: dict | None, token: str) -> requests.Response:
    """GET with retry on rate-limit + 5xx. 404/410/451/422 are returned."""
    url = path if path.startswith("http") else f"{API}{path}"
    headers = make_headers(token)
    for attempt in range(6):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code in (404, 410, 451, 422):
            return r
        if r.status_code in (403, 429):
            remaining = r.headers.get("x-ratelimit-remaining")
            reset = r.headers.get("x-ratelimit-reset")
            if remaining == "0" and reset and reset.isdigit():
                wait = max(2, int(reset) - int(time.time()) + 2)
            else:
                wait = min(60, 2 ** (attempt + 1))
            print(f"  rate-limited ({r.status_code}); sleeping {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
            continue
        if 500 <= r.status_code < 600:
            time.sleep(min(60, 2 ** attempt))
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r  # unreachable


def iter_paginated(path: str, params: dict | None, token: str):
    """Yield items from a paginated REST endpoint."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    page = 1
    while True:
        params["page"] = page
        r = gh_get(path, params, token)
        if r.status_code != 200:
            return
        items = r.json()
        if not isinstance(items, list):
            return
        for item in items:
            yield item
        if len(items) < params["per_page"]:
            return
        page += 1
        time.sleep(0.25)


def require_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN env var required", file=sys.stderr)
        sys.exit(1)
    return token
