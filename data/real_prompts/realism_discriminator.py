"""
Realism discriminator: how distinguishable are our SYNTHETIC prompts from REAL
ones? Trains a classifier to tell "real corpus prompt" from "synthetic prompt".
Its test accuracy is a realism score:

    ~50%  -> synthetic prompts are indistinguishable from real (very realistic)
    ~100% -> trivially separable (unrealistic; a calibrated generator is worth
             building, and the top "tells" below say exactly what to fix)

Pure standard library (no numpy / sklearn / network / signals) so it runs
identically in the cloud sandbox and on your local machine.

Data (auto-discovered — uses whatever is present):
  REAL       any data/real_prompts/*_corpus.json with a "prompt" field
             (real_prompts_corpus.json = agentic; webchat_*_corpus.json = chat)
  SYNTHETIC  data/HumanEval.jsonl/vibe_spectrum_data.json (the 5 hand-written
             formality levels) + data/vibe_tax_v2/vibe_tax_prompts.json
             (the v2 experiment conditions)

Usage:
    python realism_discriminator.py                 # all real corpora vs synthetic
    python realism_discriminator.py --real agentic  # restrict real set
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SYNTH_SPECTRUM = os.path.join(ROOT, "data", "HumanEval.jsonl", "vibe_spectrum_data.json")
SYNTH_V2 = os.path.join(ROOT, "data", "vibe_tax_v2", "vibe_tax_prompts.json")
STATS_OUT = os.path.join(SCRIPT_DIR, "realism_discriminator_stats.json")

LEVEL_KEYS = ["level_1_formal", "level_2", "level_3", "level_4", "level_5_vibe"]


# ---------------------------------------------------------------------------
# Features (interpretable — the weights become the "tells")
# ---------------------------------------------------------------------------

FEATURES = [
    "n_chars", "n_words", "avg_word_len", "frac_upper", "frac_digit",
    "frac_punct", "frac_non_ascii", "starts_lower", "ends_terminal",
    "type_token_ratio", "n_newlines", "has_doctest", "has_triplequote",
    "has_code_fence", "has_signature", "has_traceback", "has_url",
    "polite", "has_question", "first_person",
]

_POLITE = re.compile(r"\b(please|thanks|thank you|could you|would you|can you)\b", re.I)
_FIRSTP = re.compile(r"\b(i|i'm|i've|my|we|let's|i'd)\b", re.I)


def featurize(text):
    t = text or ""
    n = len(t) or 1
    words = t.split()
    nw = len(words) or 1
    letters = [c for c in t if c.isalpha()]
    nl = len(letters) or 1
    f = {
        "n_chars": len(t),
        "n_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / nw,
        "frac_upper": sum(c.isupper() for c in letters) / nl,
        "frac_digit": sum(c.isdigit() for c in t) / n,
        "frac_punct": sum(c in ".,;:!?()[]{}\"'`" for c in t) / n,
        "frac_non_ascii": sum(ord(c) > 127 for c in t) / n,
        "starts_lower": 1.0 if t[:1].islower() else 0.0,
        "ends_terminal": 1.0 if t.rstrip()[-1:] in ".!?" else 0.0,
        "type_token_ratio": len(set(w.lower() for w in words)) / nw,
        "n_newlines": t.count("\n"),
        "has_doctest": 1.0 if ">>>" in t else 0.0,
        "has_triplequote": 1.0 if ('"""' in t or "'''" in t) else 0.0,
        "has_code_fence": 1.0 if "```" in t else 0.0,
        "has_signature": 1.0 if re.search(r"\bdef \w+\s*\(", t) else 0.0,
        "has_traceback": 1.0 if re.search(r"Traceback|Error:|Exception", t) else 0.0,
        "has_url": 1.0 if "http" in t else 0.0,
        "polite": 1.0 if _POLITE.search(t) else 0.0,
        "has_question": 1.0 if "?" in t else 0.0,
        "first_person": 1.0 if _FIRSTP.search(t) else 0.0,
    }
    return [float(f[k]) for k in FEATURES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real(which):
    prompts = []
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "*_corpus.json"))):
        name = os.path.basename(path)
        is_agentic = name == "real_prompts_corpus.json"
        if which == "agentic" and not is_agentic:
            continue
        if which == "webchat" and is_agentic:
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for x in data:
            p = x.get("prompt")
            if p and len(p.strip()) >= 5:
                prompts.append(p)
    return prompts


def load_synthetic(which="all"):
    prompts = []
    if which in ("all", "spectrum") and os.path.exists(SYNTH_SPECTRUM):
        for e in json.load(open(SYNTH_SPECTRUM, encoding="utf-8")):
            for k in LEVEL_KEYS:
                if e.get(k):
                    prompts.append(e[k])
    if which in ("all", "v2") and os.path.exists(SYNTH_V2):
        for x in json.load(open(SYNTH_V2, encoding="utf-8")):
            if x.get("prompt"):
                prompts.append(x["prompt"])
    return prompts


def _h(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Pure-python logistic regression (standardized features, L2, GD)
# ---------------------------------------------------------------------------

def standardize(X, mean, std):
    return [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in X]


def fit(X, y, iters=800, lr=0.2, l2=0.01):
    m, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for i in range(m):
            z = b + sum(w[j] * X[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            err = p - y[i]
            gb += err
            for j in range(d):
                gw[j] += err * X[i][j]
        b -= lr * gb / m
        for j in range(d):
            w[j] -= lr * (gw[j] / m + l2 * w[j])
    return w, b


def predict_prob(w, b, x):
    z = b + sum(w[j] * x[j] for j in range(len(x)))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def auc(pairs):
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(which, synth_which="all"):
    real = load_real(which)
    synth = load_synthetic(synth_which)
    if not real:
        print("No real corpora found in", SCRIPT_DIR)
        print("Generate one first, e.g.:")
        print("  python crawl_real_prompts.py --limit 500 --deep --per-repo 40")
        print("  (and/or run pull_webchat_hf.py locally for the web-chat corpora)")
        return
    if not synth:
        print("No synthetic sources found."); return

    # Balance classes deterministically (hash-sorted, take equal counts).
    real = sorted(set(real), key=_h)
    synth = sorted(set(synth), key=_h)
    n = min(len(real), len(synth))
    real, synth = real[:n], synth[:n]

    rows = [(featurize(p), 1) for p in real] + [(featurize(p), 0) for p in synth]
    texts = real + synth
    # deterministic 75/25 split by text hash
    train, test = [], []
    for (x, y), txt in zip(rows, texts):
        (test if _h(txt) % 4 == 0 else train).append((x, y))

    Xtr = [r[0] for r in train]; ytr = [r[1] for r in train]
    d = len(FEATURES)
    mean = [sum(row[j] for row in Xtr) / len(Xtr) for j in range(d)]
    std = [(sum((row[j] - mean[j]) ** 2 for row in Xtr) / len(Xtr)) ** 0.5 or 1.0 for j in range(d)]
    Xtr_s = standardize(Xtr, mean, std)
    w, b = fit(Xtr_s, ytr)

    Xte = standardize([r[0] for r in test], mean, std)
    yte = [r[1] for r in test]
    probs = [predict_prob(w, b, x) for x in Xte]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(pr == yt for pr, yt in zip(preds, yte)) / len(yte)
    a = auc(list(zip(probs, yte)))

    # tells: standardized-feature weights (positive => pushes toward REAL)
    tells = sorted(zip(FEATURES, w), key=lambda kv: -abs(kv[1]))

    stats = {
        "real_set": which, "n_real": len(real), "n_synthetic": len(synth),
        "n_train": len(train), "n_test": len(test),
        "test_accuracy": round(acc, 3), "test_auc": round(a, 3) if a else None,
        "top_tells": [{"feature": f, "weight": round(wt, 3),
                       "leans": "real" if wt > 0 else "synthetic"} for f, wt in tells[:10]],
    }
    json.dump(stats, open(STATS_OUT, "w", encoding="utf-8"), indent=2)

    print("=" * 60)
    print(f"REAL ({which}): {len(real)}   SYNTHETIC: {len(synth)}   "
          f"(balanced; train {len(train)} / test {len(test)})")
    print(f"\nDiscriminator test accuracy: {acc*100:.1f}%   AUC: {a:.3f}")
    print("  (50% = indistinguishable/realistic; 100% = trivially separable)")
    print("\nTop tells (what gives a prompt away):")
    for f, wt in tells[:10]:
        print(f"  {f:18s} {wt:+.2f}  -> {'REAL' if wt>0 else 'SYNTHETIC'}")
    print(f"\nstats -> {STATS_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", choices=["all", "agentic", "webchat"], default="all")
    ap.add_argument("--synth", choices=["all", "spectrum", "v2"], default="all")
    a = ap.parse_args()
    run(a.real, a.synth)
