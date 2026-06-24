"""
Pull an UNBIASED web-chat corpus straight from HuggingFace (WildChat-1M or
LMSYS-Chat-1M), for environments that can reach HF and have a token.

This is the clean replacement for pull_webchat_corpus.py, which sourced a
curated GitHub *mirror* of WildChat because HuggingFace was firewalled in the
cloud sandbox. Running this on your own machine removes that selection bias.

Output schema matches real_prompts_corpus.json / webchat_corpus.json, so
classify_prompts.py runs on it unchanged.

Setup (one time):
    pip install "datasets>=2.0" huggingface_hub
    export HF_TOKEN=hf_xxx            # a token that has accepted the dataset terms
    # WildChat-1M and LMSYS-Chat-1M are GATED: visit the dataset page once and
    # click "Agree and access" with the same HF account as your token.

Usage:
    python pull_webchat_hf.py --dataset wildchat --limit 1000
    python pull_webchat_hf.py --dataset lmsys    --limit 1000
"""

import argparse
import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reuse the exact same coding / error / code detectors as the GitHub puller so
# the two web-chat corpora are directly comparable.
try:
    from pull_webchat_corpus import CODING_HINTS, ERROR_PASTE, CODE_PASTE, NON_ASCII, normalize
except Exception:  # pragma: no cover - fallback if run standalone
    CODING_HINTS = re.compile(r"\b(fix|bug|error|function|code|python|javascript|compile|traceback|exception|class|method|api|script|implement|debug|import)\b", re.I)
    ERROR_PASTE = re.compile(r"(Traceback \(most recent call last\)|\b\w*(Error|Exception)\b\s*:|SyntaxError|TypeError|panic:)", re.I | re.M)
    CODE_PASTE = re.compile(r"```|\bdef \w+\s*\(|\bclass \w+|\bfunction \w+\s*\(|=>|import \w+")
    NON_ASCII = re.compile(r"[^\x00-\x7f]")
    def normalize(t):
        return re.sub(r"\s+", " ", t.strip().lower())

DATASETS = {
    "wildchat": {
        "id": "allenai/WildChat-1M",
        "split": "train",
        "conv_field": "conversation",   # list of {role, content, language, ...}
    },
    "lmsys": {
        "id": "lmsys/lmsys-chat-1m",
        "split": "train",
        "conv_field": "conversation",   # list of {role, content}
    },
}


def iter_user_turns(row, conv_field):
    """Yield (text, turn_language_or_None) for each human turn in a row."""
    conv = row.get(conv_field) or []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "user":
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            yield content.strip(), turn.get("language") or row.get("language")


def run(dataset_key, limit, max_chars):
    cfg = DATASETS[dataset_key]
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a token that has accepted the dataset's terms).")

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit('pip install "datasets>=2.0" huggingface_hub')

    out_file = os.path.join(SCRIPT_DIR, f"webchat_{dataset_key}_corpus.json")
    print(f"Streaming {cfg['id']} (split={cfg['split']}) ...")
    ds = load_dataset(cfg["id"], split=cfg["split"], streaming=True, token=token)

    corpus = []
    seen = set()
    scanned = 0
    for row in ds:
        scanned += 1
        for text, lang_meta in iter_user_turns(row, cfg["conv_field"]):
            if not (8 <= len(text) <= max_chars):
                continue
            is_coding = bool(CODING_HINTS.search(text))
            has_error = bool(ERROR_PASTE.search(text))
            has_code = bool(CODE_PASTE.search(text))
            if not (is_coding or has_error or has_code):
                continue
            norm = normalize(text)
            if norm in seen:
                continue
            seen.add(norm)
            corpus.append({
                "prompt": text,
                "repo": cfg["id"],
                "path": f"{dataset_key}:train",
                "source": "webchat",
                "language_meta": lang_meta,     # dataset-provided language (ground truth)
                "n_words": len(text.split()),
                "n_chars": len(text),
                "looks_coding": is_coding,
                "non_english": bool(NON_ASCII.search(text)),
                "has_error_paste": has_error,
                "has_code_paste": has_code,
            })
        if len(corpus) >= limit:
            break
        if scanned % 2000 == 0:
            print(f"  scanned {scanned} convs, kept {len(corpus)} coding prompts")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    err = sum(1 for c in corpus if c["has_error_paste"])
    code = sum(1 for c in corpus if c["has_code_paste"])
    nonen = sum(1 for c in corpus if c["non_english"])
    n = len(corpus) or 1
    print("=" * 60)
    print(f"scanned conversations : {scanned}")
    print(f"coding prompts kept   : {len(corpus)}")
    print(f"  pasted error        : {err} ({err/n*100:.0f}%)")
    print(f"  pasted code         : {code} ({code/n*100:.0f}%)")
    print(f"  non-English         : {nonen} ({nonen/n*100:.0f}%)")
    print(f"corpus -> {out_file}")
    print(f"\nNext: python classify_prompts.py --in {os.path.basename(out_file)} "
          f"--out vibe_spectrum_{dataset_key}_corpus.json "
          f"--stats vibe_spectrum_{dataset_key}_stats.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="wildchat")
    ap.add_argument("--limit", type=int, default=1000, help="coding prompts to collect")
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()
    run(args.dataset, args.limit, args.max_chars)
