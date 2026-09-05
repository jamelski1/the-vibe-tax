"""
Run the Vibe Tax v2 experiment: send every prompt in vibe_tax_prompts.json to
ChatGPT, Claude, and Codestral, and save completions for scoring.

274 prompts x 3 models = 822 completions (~50 problems x 6 medium-grounded
conditions; the two paste conditions cover the 37 problems with a viable
injected bug).

The output schema matches data/HumanEval.jsonl/all_model_responses.json with
the experiment condition stored in the "level" field, so the existing scorer
(data/HumanEval.jsonl/run_tests.py) works on it unchanged — just point
RESPONSES_FILE at this run's output (or copy the file over).

Usage (run locally, where your API keys live):
    python run_vibe_tax.py

Environment variables:
    OPENAI_API_KEY     - ChatGPT        (model: OPENAI_MODEL, default gpt-5.4)
    ANTHROPIC_API_KEY  - Claude         (model: ANTHROPIC_MODEL, default claude-opus-4-6)
    CODESTRAL_API_KEY  - Codestral      (model: CODESTRAL_MODEL, default codestral-latest)
    API_DELAY          - seconds between calls (default 1.0)

Models with a missing key are skipped. Progress is checkpointed to
vibe_tax_progress.json, so an interrupted run resumes where it left off.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths are env-overridable so the same runner works on v2 and the v3 calibrated
# prompts without editing code, e.g. (PowerShell):
#   $env:PROMPTS_FILE = "..\vibe_tax_v3\vibe_tax_v3_prompts.json"
#   $env:OUTPUT_FILE  = "..\vibe_tax_v3\vibe_tax_v3_responses.json"
#   $env:PROGRESS_FILE = "..\vibe_tax_v3\vibe_tax_v3_progress.json"
PROMPTS_FILE = os.getenv("PROMPTS_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_prompts.json"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_responses.json"))
PROGRESS_FILE = os.getenv("PROGRESS_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_progress.json"))
STATS_FILE = os.getenv("STATS_FILE", os.path.join(SCRIPT_DIR, "vibe_tax_run_stats.json"))
LOG_FILE = os.getenv("RUN_VIBE_TAX_LOG", os.path.join(SCRIPT_DIR, "run_vibe_tax.log"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
CODESTRAL_MODEL = os.getenv("CODESTRAL_MODEL", "codestral-latest")
CODESTRAL_BASE_URL = "https://api.mistral.ai/v1"
API_DELAY = float(os.getenv("API_DELAY", "1.0"))
CHECKPOINT_INTERVAL = 50

# Deliberately minimal and medium-neutral: real users don't ship an elaborate
# system prompt, and per-condition instructions live in the user prompt itself.
# The scorer's clean_completion() handles full-function or body-only replies.
SYSTEM_PROMPT = (
    "You are a helpful Python programming assistant. "
    "When you provide code, output plain Python without markdown fences."
)


def setup_logging():
    logger = logging.getLogger("vibe_tax_v2")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logging()


# Exact model id per provider — stamped into every record for reproducibility.
MODEL_ID = {"openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL, "codestral": CODESTRAL_MODEL}

# Optional: run only a subset of providers, e.g. ONLY_MODELS=chatgpt for a
# single-model ablation (leave unset to run every provider with a key).
ONLY_MODELS = {m.strip().lower() for m in os.getenv("ONLY_MODELS", "").split(",") if m.strip()}


def make_clients():
    clients = {}
    if os.getenv("OPENAI_API_KEY"):
        clients["chatgpt"] = ("openai", OpenAI())
    else:
        log.warning("OPENAI_API_KEY not set -- ChatGPT skipped.")
    if os.getenv("ANTHROPIC_API_KEY"):
        clients["claude"] = ("anthropic", Anthropic())
    else:
        log.warning("ANTHROPIC_API_KEY not set -- Claude skipped.")
    if os.getenv("CODESTRAL_API_KEY"):
        clients["codestral"] = ("codestral", OpenAI(
            api_key=os.getenv("CODESTRAL_API_KEY"), base_url=CODESTRAL_BASE_URL))
    else:
        log.warning("CODESTRAL_API_KEY not set -- Codestral skipped.")
    if ONLY_MODELS:
        clients = {k: v for k, v in clients.items() if k in ONLY_MODELS}
        log.info("ONLY_MODELS=%s -> running: %s", sorted(ONLY_MODELS), list(clients))
    return clients


def query(api_type, client, prompt):
    if api_type == "anthropic":
        r = client.messages.create(
            model=ANTHROPIC_MODEL, temperature=0, max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    model = OPENAI_MODEL if api_type == "openai" else CODESTRAL_MODEL
    kwargs = dict(model=model, temperature=0,
                  messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}])
    if api_type == "openai":
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 2048
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content


def run():
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        prompts = json.load(f)
    clients = make_clients()
    if not clients:
        log.error("No API keys configured.")
        sys.exit(1)

    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        log.info("Resuming -- %d completions already done.", len(progress))
    already_done = len(progress)   # fixed baseline; progress grows during the run

    total = len(prompts) * len(clients)
    log.info("Models: %s | prompts: %d | total queries: %d",
             list(clients), len(prompts), total)

    results, new, errors = [], 0, 0
    start = datetime.now()
    for entry in prompts:
        for model_name, (api_type, client) in clients.items():
            key = f"{entry['task_id']}|{entry['condition']}|{model_name}"
            if key in progress:
                results.append(progress[key])
                continue
            try:
                t0 = time.time()
                completion = query(api_type, client, entry["prompt"])
                record = {
                    "task_id": entry["task_id"],
                    "problem_number": entry["problem_number"],
                    "entry_point": entry["entry_point"],
                    "level": entry["condition"],       # scorer-compatible field
                    "medium": entry["medium"],
                    "model": model_name,
                    "model_id": MODEL_ID[api_type],    # exact model string, for reproducibility
                    "prompt_text": entry["prompt"],
                    "completion": completion,
                    "error": None,
                    "timestamp": datetime.now().isoformat(),
                }
                new += 1
                log.info("[%d/%d] OK   %s | %-22s | %-9s (%.1fs)",
                         already_done + new + errors, total, entry["task_id"],
                         entry["condition"], model_name, time.time() - t0)
            except Exception as e:
                record = {
                    "task_id": entry["task_id"],
                    "problem_number": entry["problem_number"],
                    "entry_point": entry["entry_point"],
                    "level": entry["condition"],
                    "medium": entry["medium"],
                    "model": model_name,
                    "model_id": MODEL_ID[api_type],
                    "prompt_text": entry["prompt"],
                    "completion": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                errors += 1
                log.error("[%d/%d] FAIL %s | %s | %s -- %s",
                          already_done + new + errors, total, entry["task_id"],
                          entry["condition"], model_name, e)
            results.append(record)
            progress[key] = record
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            if (new + errors) % CHECKPOINT_INTERVAL == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                log.info(">>> checkpoint: %d saved", len(results))
            time.sleep(API_DELAY)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    stats = {
        "run_timestamp": start.isoformat(),
        "elapsed_seconds": (datetime.now() - start).total_seconds(),
        "models": {"chatgpt": OPENAI_MODEL, "claude": ANTHROPIC_MODEL,
                   "codestral": CODESTRAL_MODEL},
        "total": len(results),
        "errors": errors,
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("DONE: %d completions (%d errors) -> %s", len(results), errors, OUTPUT_FILE)
    log.info("Score with: data/HumanEval.jsonl/run_tests.py "
             "(point RESPONSES_FILE at %s)", OUTPUT_FILE)


if __name__ == "__main__":
    run()
