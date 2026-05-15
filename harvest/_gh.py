"""Shared GitHub REST helpers and logging for the harvest scripts."""

from __future__ import annotations

import logging
import os
import sys
import time

import requests

API = "https://api.github.com"

log = logging.getLogger("harvest")


def setup_logging(log_path: str | None) -> logging.Logger:
    """Configure the 'harvest' logger: INFO to console, DEBUG to file."""
    logger = logging.getLogger("harvest")
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_path:
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


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
            log.warning("rate-limited (%s); sleeping %ds", r.status_code, wait)
            time.sleep(wait)
            continue
        if 500 <= r.status_code < 600:
            wait = min(60, 2 ** attempt)
            log.warning("%s on %s; retrying in %ds", r.status_code, url, wait)
            time.sleep(wait)
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
        sys.stderr.write("ERROR: GITHUB_TOKEN env var required\n")
        sys.exit(1)
    return token
