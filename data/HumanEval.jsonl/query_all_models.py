"""
Query ChatGPT, Claude, and Cursor with all 250 zero-shot prompts from vibe_spectrum_data.json.

50 problems × 5 formality levels × 3 models = 750 total completions.

Usage:
    python query_all_models.py

Environment variables required:
    OPENAI_API_KEY     - For ChatGPT (gpt-5.3)
    ANTHROPIC_API_KEY  - For Claude (claude-opus-4-6)
    CURSOR_API_KEY     - For Cursor (OpenAI-compatible endpoint)
    CURSOR_BASE_URL    - Cursor API base URL (default: https://api.cursor.com/v1)

Set CURSOR_MODEL to override the Cursor model name (default: composer-2).
"""

import json
import os
import sys
import time
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.3")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
CURSOR_MODEL = os.getenv("CURSOR_MODEL", "composer-2")
CURSOR_BASE_URL = os.getenv("CURSOR_BASE_URL", "https://api.cursor.com/v1")

# Delay between API calls (seconds) to respect rate limits
API_DELAY = float(os.getenv("API_DELAY", "1.0"))

DATA_FILE = os.path.join(os.path.dirname(__file__), "vibe_spectrum_data.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "all_model_responses.json")
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "query_progress.json")

LEVEL_KEYS = [
    "level_1_formal",
    "level_2",
    "level_3",
    "level_4",
    "level_5_vibe",
]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Given a function signature and docstring, write the complete function implementation. "
    "Return ONLY the Python code for the function body (the implementation after the docstring). "
    "Do not include the function signature, docstring, imports, or any explanation. "
    "Do not wrap the code in markdown code blocks."
)


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def make_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set — ChatGPT queries will be skipped.")
        return None
    return OpenAI(api_key=api_key)


def make_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY not set — Claude queries will be skipped.")
        return None
    return Anthropic(api_key=api_key)


def make_cursor_client():
    api_key = os.getenv("CURSOR_API_KEY")
    if not api_key:
        print("WARNING: CURSOR_API_KEY not set — Cursor queries will be skipped.")
        return None
    return OpenAI(api_key=api_key, base_url=CURSOR_BASE_URL)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def query_openai(client, prompt, model=None):
    """Send a zero-shot prompt to the OpenAI API and return the completion."""
    model = model or OPENAI_MODEL
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def query_anthropic(client, prompt):
    """Send a zero-shot prompt to the Anthropic API and return the completion."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        temperature=0,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.content[0].text


def query_cursor(client, prompt):
    """Send a zero-shot prompt to Cursor's OpenAI-compatible API."""
    return query_openai(client, prompt, model=CURSOR_MODEL)


# ---------------------------------------------------------------------------
# Progress / resume helpers
# ---------------------------------------------------------------------------

def load_progress():
    """Load previously saved progress so we can resume after interruption."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def progress_key(task_id, level, model_name):
    return f"{task_id}|{level}|{model_name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_user_prompt(entry, level_key, entry_point):
    """Construct the user message sent to each model."""
    docstring = entry[level_key]
    return (
        f"Implement the Python function `{entry_point}` described below.\n\n"
        f"{docstring}\n\n"
        f"Write only the function body implementation."
    )


def run():
    # Load data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems from {DATA_FILE}")
    print(f"Levels per problem: {len(LEVEL_KEYS)}")

    # Set up clients
    clients = {}
    openai_client = make_openai_client()
    if openai_client:
        clients["chatgpt"] = ("openai", openai_client)

    anthropic_client = make_anthropic_client()
    if anthropic_client:
        clients["claude"] = ("anthropic", anthropic_client)

    cursor_client = make_cursor_client()
    if cursor_client:
        clients["cursor"] = ("cursor", cursor_client)

    if not clients:
        print("ERROR: No API keys configured. Set at least one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, CURSOR_API_KEY")
        sys.exit(1)

    total_queries = len(problems) * len(LEVEL_KEYS) * len(clients)
    print(f"\nModels configured: {list(clients.keys())}")
    print(f"Total queries to run: {total_queries}")

    # Load existing progress
    progress = load_progress()
    already_done = len(progress)
    if already_done:
        print(f"Resuming — {already_done} completions already done.")

    # Collect results
    results = []
    completed = 0
    errors = 0
    start_time = datetime.now()

    for i, entry in enumerate(problems):
        task_id = entry["task_id"]
        entry_point = entry["entry_point"]

        for level_key in LEVEL_KEYS:
            level_label = level_key  # e.g. "level_1_formal"

            for model_name, (api_type, client) in clients.items():
                pkey = progress_key(task_id, level_label, model_name)

                # Skip if already completed
                if pkey in progress:
                    results.append(progress[pkey])
                    continue

                user_prompt = build_user_prompt(entry, level_key, entry_point)

                try:
                    if api_type == "openai":
                        completion = query_openai(client, user_prompt)
                    elif api_type == "anthropic":
                        completion = query_anthropic(client, user_prompt)
                    elif api_type == "cursor":
                        completion = query_cursor(client, user_prompt)
                    else:
                        completion = f"ERROR: unknown api_type {api_type}"

                    result_entry = {
                        "task_id": task_id,
                        "problem_number": entry["problem_number"],
                        "entry_point": entry_point,
                        "level": level_label,
                        "model": model_name,
                        "prompt_text": entry[level_key],
                        "completion": completion,
                        "error": None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    completed += 1

                except Exception as e:
                    result_entry = {
                        "task_id": task_id,
                        "problem_number": entry["problem_number"],
                        "entry_point": entry_point,
                        "level": level_label,
                        "model": model_name,
                        "prompt_text": entry[level_key],
                        "completion": None,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    errors += 1

                results.append(result_entry)
                progress[pkey] = result_entry
                save_progress(progress)

                total_done = already_done + completed + errors
                print(
                    f"[{total_done}/{total_queries}] "
                    f"{task_id} | {level_label} | {model_name} — "
                    f"{'OK' if result_entry['error'] is None else 'ERROR: ' + result_entry['error']}"
                )

                time.sleep(API_DELAY)

    # Save final output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"Done! {completed} completions, {errors} errors")
    print(f"Elapsed: {elapsed}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Progress saved to: {PROGRESS_FILE}")

    # Clean up progress file on successful full run
    if errors == 0 and total_queries == len(results):
        os.remove(PROGRESS_FILE)
        print("(Progress file removed — full run complete.)")


if __name__ == "__main__":
    run()
