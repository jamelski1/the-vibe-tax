"""
Query ChatGPT, Claude, and Codestral with all 250 zero-shot prompts from vibe_spectrum_data.json.

50 problems x 5 formality levels x 3 models = 750 total completions.

Usage:
    python query_all_models.py

Environment variables required:
    OPENAI_API_KEY     - For ChatGPT (gpt-5.4)
    ANTHROPIC_API_KEY  - For Claude (claude-opus-4-6)
    CODESTRAL_API_KEY  - For Codestral (codestral-latest)

Set CODESTRAL_MODEL to override the model name (default: codestral-latest).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
CODESTRAL_MODEL = os.getenv("CODESTRAL_MODEL", "codestral-latest")
CODESTRAL_BASE_URL = "https://codestral.mistral.ai/v1"

# Delay between API calls (seconds) to respect rate limits
API_DELAY = float(os.getenv("API_DELAY", "1.0"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "vibe_spectrum_data.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "all_model_responses.json")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, "query_progress.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "query_all_models.log")

# Save results to disk after every N completions (first checkpoint at 15
# to verify the first problem's 5 levels x 3 models all saved correctly)
FIRST_CHECKPOINT = 15
CHECKPOINT_INTERVAL = 50

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
# Logging setup — writes to both terminal and log file
# ---------------------------------------------------------------------------

def setup_logging():
    logger = logging.getLogger("vibe_tax")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — DEBUG and above (captures everything)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def make_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.warning("OPENAI_API_KEY not set -- ChatGPT queries will be skipped.")
        return None
    return OpenAI(api_key=api_key)


def make_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set -- Claude queries will be skipped.")
        return None
    return Anthropic(api_key=api_key)


def make_codestral_client():
    api_key = os.getenv("CODESTRAL_API_KEY")
    if not api_key:
        log.warning("CODESTRAL_API_KEY not set -- Codestral queries will be skipped.")
        return None
    return OpenAI(api_key=api_key, base_url=CODESTRAL_BASE_URL)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def query_openai(client, prompt):
    """Send a zero-shot prompt to the OpenAI API and return the completion."""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        max_completion_tokens=2048,
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


def query_codestral(client, prompt):
    """Send a zero-shot prompt to Mistral's Codestral API."""
    response = client.chat.completions.create(
        model=CODESTRAL_MODEL,
        temperature=0,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


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
# Save helpers
# ---------------------------------------------------------------------------

def save_results(results, path):
    """Write the results list to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s (%d entries)", path, len(results))


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
    log.info("=" * 60)
    log.info("VIBE TAX — Model Query Run Started")
    log.info("=" * 60)

    # Load data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        problems = json.load(f)

    log.info("Loaded %d problems from %s", len(problems), DATA_FILE)
    log.info("Levels per problem: %d", len(LEVEL_KEYS))

    # Set up clients
    clients = {}
    openai_client = make_openai_client()
    if openai_client:
        clients["chatgpt"] = ("openai", openai_client)

    anthropic_client = make_anthropic_client()
    if anthropic_client:
        clients["claude"] = ("anthropic", anthropic_client)

    codestral_client = make_codestral_client()
    if codestral_client:
        clients["codestral"] = ("codestral", codestral_client)

    if not clients:
        log.error("No API keys configured. Set at least one of:")
        log.error("  OPENAI_API_KEY, ANTHROPIC_API_KEY, CODESTRAL_API_KEY")
        sys.exit(1)

    total_queries = len(problems) * len(LEVEL_KEYS) * len(clients)
    log.info("Models configured: %s", list(clients.keys()))
    log.info("Total queries to run: %d", total_queries)

    # Load existing progress
    progress = load_progress()
    already_done = len(progress)
    if already_done:
        log.info("Resuming -- %d completions already done.", already_done)

    # Collect results
    results = []
    new_completed = 0
    errors = 0
    start_time = datetime.now()

    for i, entry in enumerate(problems):
        task_id = entry["task_id"]
        entry_point = entry["entry_point"]
        log.info("-" * 40)
        log.info("Problem %d/%d: %s (%s)", i + 1, len(problems), task_id, entry_point)

        for level_key in LEVEL_KEYS:
            level_label = level_key  # e.g. "level_1_formal"

            for model_name, (api_type, client) in clients.items():
                pkey = progress_key(task_id, level_label, model_name)

                # Skip if already completed
                if pkey in progress:
                    results.append(progress[pkey])
                    log.debug("SKIP (cached): %s | %s | %s", task_id, level_label, model_name)
                    continue

                user_prompt = build_user_prompt(entry, level_key, entry_point)
                log.debug("Sending prompt to %s (%d chars)", model_name, len(user_prompt))

                try:
                    query_start = time.time()

                    if api_type == "openai":
                        completion = query_openai(client, user_prompt)
                    elif api_type == "anthropic":
                        completion = query_anthropic(client, user_prompt)
                    elif api_type == "codestral":
                        completion = query_codestral(client, user_prompt)
                    else:
                        completion = f"ERROR: unknown api_type {api_type}"

                    elapsed_secs = time.time() - query_start

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
                    new_completed += 1

                    log.info(
                        "[%d/%d] OK   %s | %s | %s (%.1fs, %d chars returned)",
                        already_done + new_completed + errors,
                        total_queries,
                        task_id,
                        level_label,
                        model_name,
                        elapsed_secs,
                        len(completion),
                    )
                    log.debug("Response preview: %.200s", completion.replace("\n", "\\n"))

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

                    log.error(
                        "[%d/%d] FAIL %s | %s | %s -- %s",
                        already_done + new_completed + errors,
                        total_queries,
                        task_id,
                        level_label,
                        model_name,
                        str(e),
                    )

                results.append(result_entry)
                progress[pkey] = result_entry
                save_progress(progress)

                # Checkpoint: save results to output file at key intervals
                total_new = new_completed + errors
                if total_new == FIRST_CHECKPOINT:
                    log.info(
                        ">>> FIRST CHECKPOINT: %d completions done — saving to verify output...",
                        FIRST_CHECKPOINT,
                    )
                    save_results(results, OUTPUT_FILE)
                elif total_new > FIRST_CHECKPOINT and total_new % CHECKPOINT_INTERVAL == 0:
                    log.info(">>> CHECKPOINT at %d completions — saving...", total_new)
                    save_results(results, OUTPUT_FILE)

                time.sleep(API_DELAY)

    # Final save
    save_results(results, OUTPUT_FILE)

    elapsed = datetime.now() - start_time
    log.info("=" * 60)
    log.info("DONE! %d completions, %d errors", new_completed, errors)
    log.info("Elapsed: %s", elapsed)
    log.info("Results saved to: %s", OUTPUT_FILE)
    log.info("Log saved to: %s", LOG_FILE)

    # Clean up progress file on successful full run
    if errors == 0 and total_queries == len(results):
        os.remove(PROGRESS_FILE)
        log.info("(Progress file removed -- full run complete.)")


if __name__ == "__main__":
    run()
