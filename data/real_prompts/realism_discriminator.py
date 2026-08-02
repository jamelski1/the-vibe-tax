"""
Realism discriminator: how distinguishable are our SYNTHETIC prompts from REAL
ones? Trains a classifier to tell "real corpus prompt" from "synthetic prompt".
Its test accuracy is a realism score:

    ~50%  -> synthetic prompts are indistinguishable from real (very realistic)
    ~100% -> trivially separable (unrealistic; a calibrated generator is worth
             building, and the top "tells" below say exactly what to fix)

MODEL: feature-based logistic regression (sigmoid + cross-entropy, batch
gradient descent), implemented in pure standard library — no numpy / sklearn /
network / signals — so it runs identically in the cloud sandbox and locally.
Chosen for INTERPRETABILITY: the weights ARE the "tells". 20 hand-crafted
features (lengths, char-class fractions, and binary flags like has_doctest /
has_triplequote / has_signature); no embeddings.

REPRODUCIBLE: no RNG anywhere. Class-balancing and the 75/25 train/test split
are keyed on md5 hashes of the prompt text (not Python's salted hash()); weights
init to zero; hyperparameters are fixed. Same inputs -> identical output on any
machine, every run.

Importable:
    from realism_discriminator import RealismDiscriminator, featurize, FEATURES
    d = RealismDiscriminator().fit(real_texts, synth_texts)
    d.evaluate()            # {'test_accuracy':..., 'test_auc':...}
    d.tells()               # [(feature, weight), ...] most-telling first
    d.predict_proba(text)   # P(real) in [0,1]

Data (auto-discovered — uses whatever is present):
  REAL       any data/real_prompts/*_corpus.json with a "prompt" field
             (real_prompts_corpus.json = agentic; webchat_*_corpus.json = chat)
  SYNTHETIC  data/HumanEval.jsonl/vibe_spectrum_data.json (5 hand-written levels)
             + data/vibe_tax_v2/vibe_tax_prompts.json (v2 experiment conditions)

CLI:
    python realism_discriminator.py                 # all real corpora vs all synthetic
    python realism_discriminator.py --real agentic --synth v2

NOTE: this is a deliberately WEAK discriminator — the features include literal
code-scaffolding flags, so current synthetic prompts separate trivially. It is
right for "is our synthetic obviously fake?" and as a first-pass fitness
function for a calibrated generator. Certifying subtle style realism needs a
stronger discriminator (TF-IDF n-grams or an embedding/LLM judge) — see
RESEARCH_LOG open threads.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re

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


def _h(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _auc(pairs):
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# The discriminator — pure-python logistic regression, deterministic
# ---------------------------------------------------------------------------

class RealismDiscriminator:
    """Reproducible logistic-regression discriminator (real=1 vs synthetic=0).

    fit() balances the classes and makes a hash-keyed 75/25 split internally,
    trains on the train split, and keeps the test split for evaluate(). No RNG.
    """

    def __init__(self, features=FEATURES, iters=800, lr=0.2, l2=0.01, test_mod=4):
        self.features = features
        self.iters, self.lr, self.l2, self.test_mod = iters, lr, l2, test_mod
        self.w = None
        self.b = 0.0
        self.mean = self.std = None
        self._test = []
        self.n_real = self.n_synthetic = self.n_train = self.n_test = 0

    def _standardize(self, X):
        return [[(row[j] - self.mean[j]) / self.std[j] for j in range(len(row))] for row in X]

    def _proba(self, xs):  # xs already standardized
        z = self.b + sum(self.w[j] * xs[j] for j in range(len(xs)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def fit(self, real_texts, synth_texts):
        real = sorted(set(real_texts), key=_h)
        synth = sorted(set(synth_texts), key=_h)
        n = min(len(real), len(synth))
        real, synth = real[:n], synth[:n]
        self.n_real, self.n_synthetic = len(real), len(synth)

        rows = [(featurize(p), 1) for p in real] + [(featurize(p), 0) for p in synth]
        texts = real + synth
        train, self._test = [], []
        for (x, y), t in zip(rows, texts):
            (self._test if _h(t) % self.test_mod == 0 else train).append((x, y))
        self.n_train, self.n_test = len(train), len(self._test)

        Xtr = [r[0] for r in train]
        ytr = [r[1] for r in train]
        d = len(self.features)
        self.mean = [sum(r[j] for r in Xtr) / len(Xtr) for j in range(d)]
        self.std = [(sum((r[j] - self.mean[j]) ** 2 for r in Xtr) / len(Xtr)) ** 0.5 or 1.0
                    for j in range(d)]
        Xs = self._standardize(Xtr)

        w, b = [0.0] * d, 0.0
        m = len(Xs)
        for _ in range(self.iters):
            gw, gb = [0.0] * d, 0.0
            for i in range(m):
                z = b + sum(w[j] * Xs[i][j] for j in range(d))
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                e = p - ytr[i]
                gb += e
                for j in range(d):
                    gw[j] += e * Xs[i][j]
            b -= self.lr * gb / m
            for j in range(d):
                w[j] -= self.lr * (gw[j] / m + self.l2 * w[j])
        self.w, self.b = w, b
        return self

    def predict_proba(self, text):
        """P(real) for an arbitrary prompt string."""
        return self._proba(self._standardize([featurize(text)])[0])

    def evaluate(self):
        Xte = self._standardize([r[0] for r in self._test])
        yte = [r[1] for r in self._test]
        probs = [self._proba(x) for x in Xte]
        acc = sum((p >= 0.5) == bool(y) for p, y in zip(probs, yte)) / len(yte) if yte else None
        a = _auc(list(zip(probs, yte)))
        return {
            "n_real": self.n_real, "n_synthetic": self.n_synthetic,
            "n_train": self.n_train, "n_test": self.n_test,
            "test_accuracy": round(acc, 3) if acc is not None else None,
            "test_auc": round(a, 3) if a is not None else None,
        }

    def tells(self, k=10):
        """(feature, weight) most-telling first; weight>0 leans REAL."""
        return sorted(zip(self.features, self.w), key=lambda kv: -abs(kv[1]))[:k]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real(which="all"):
    prompts = []
    for path in sorted(glob.glob(os.path.join(SCRIPT_DIR, "*_corpus.json"))):
        is_agentic = os.path.basename(path) == "real_prompts_corpus.json"
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


# ---------------------------------------------------------------------------
# CLI
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
        print("No synthetic sources found.")
        return

    d = RealismDiscriminator().fit(real, synth)
    ev = d.evaluate()
    tells = d.tells()

    stats = {
        "real_set": which, "synth_set": synth_which, **ev,
        "top_tells": [{"feature": f, "weight": round(wt, 3),
                       "leans": "real" if wt > 0 else "synthetic"} for f, wt in tells],
    }
    json.dump(stats, open(STATS_OUT, "w", encoding="utf-8"), indent=2)

    print("=" * 60)
    print(f"REAL ({which}): {ev['n_real']}   SYNTHETIC ({synth_which}): {ev['n_synthetic']}   "
          f"(balanced; train {ev['n_train']} / test {ev['n_test']})")
    print(f"\nDiscriminator test accuracy: {ev['test_accuracy']*100:.1f}%   AUC: {ev['test_auc']}")
    print("  (50% = indistinguishable/realistic; 100% = trivially separable)")
    print("\nTop tells (what gives a prompt away):")
    for f, wt in tells:
        print(f"  {f:18s} {wt:+.2f}  -> {'REAL' if wt > 0 else 'SYNTHETIC'}")
    print(f"\nstats -> {STATS_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", choices=["all", "agentic", "webchat"], default="all")
    ap.add_argument("--synth", choices=["all", "spectrum", "v2"], default="all")
    a = ap.parse_args()
    run(a.real, a.synth)
