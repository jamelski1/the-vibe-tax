"""
Resolve the EXACT model versions used in the study, for documentation /
reproducibility. Run LOCALLY with the same API keys the study used.

For each provider it prints:
  - the id you REQUESTED (gpt-5.4 / claude-opus-4-6 / codestral-latest),
  - the CANONICAL id it resolves to (important for '-latest' aliases),
  - the provider's `created` / `created_at` timestamp (a release-ish date),
  - the display name where available.

IMPORTANT: none of these APIs return the TRAINING / KNOWLEDGE CUTOFF — the date
that actually governs benchmark contamination. That number is only in each
provider's model docs (linked in the printout). This script pins *which* model
you called; you still record the cutoff by hand from the docs.

Writes model_provenance.json (commit it alongside the results).

    # same env the study used
    #   OPENAI_MODEL (default gpt-5.4), ANTHROPIC_MODEL (default claude-opus-4-6),
    #   CODESTRAL_MODEL (default codestral-latest)
    python model_provenance.py
"""
import datetime as dt
import json
import os
import sys

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
CODESTRAL_MODEL = os.getenv("CODESTRAL_MODEL", "codestral-latest")


def _iso(ts):
    """Unix seconds -> ISO date, or pass through an existing string."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return dt.datetime.utcfromtimestamp(ts).isoformat() + "Z"
    return str(ts)


def openai_provenance():
    from openai import OpenAI
    c = OpenAI()
    m = c.models.retrieve(OPENAI_MODEL)
    d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
    return {
        "provider": "OpenAI",
        "requested": OPENAI_MODEL,
        "canonical_id": d.get("id"),
        "created": _iso(d.get("created")),
        "owned_by": d.get("owned_by"),
        "cutoff_docs": "https://platform.openai.com/docs/models  (knowledge cutoff per model)",
        "raw": d,
    }


def anthropic_provenance():
    from anthropic import Anthropic
    c = Anthropic()
    m = c.models.retrieve(ANTHROPIC_MODEL)
    d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
    return {
        "provider": "Anthropic",
        "requested": ANTHROPIC_MODEL,
        "canonical_id": d.get("id"),
        "created": _iso(d.get("created_at")),
        "display_name": d.get("display_name"),
        "cutoff_docs": "https://docs.anthropic.com/en/docs/about-claude/models  (training cutoff per model)",
        "raw": d,
    }


def codestral_provenance():
    # Mistral/Codestral: '-latest' is an ALIAS that resolves to a dated version
    # (e.g. codestral-2508 = 2025-08). The version suffix IS the release month.
    from openai import OpenAI  # Mistral is OpenAI-API-compatible
    key = os.getenv("CODESTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
    c = OpenAI(api_key=key, base_url="https://api.mistral.ai/v1")
    canonical = None
    created = None
    try:
        m = c.models.retrieve(CODESTRAL_MODEL)
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        canonical = d.get("id")
        created = _iso(d.get("created"))
    except Exception as e:
        d = {"retrieve_error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {
        "provider": "Mistral (Codestral)",
        "requested": CODESTRAL_MODEL,
        "canonical_id": canonical,   # look for a dated suffix, e.g. codestral-2508
        "created": created,
        "cutoff_docs": "https://docs.mistral.ai/getting-started/models/models_overview/",
        "raw": d,
    }


def main():
    out = []
    for name, fn in (("OpenAI", openai_provenance),
                     ("Anthropic", anthropic_provenance),
                     ("Codestral", codestral_provenance)):
        try:
            out.append(fn())
        except Exception as e:
            out.append({"provider": name, "error": f"{type(e).__name__}: {str(e)[:160]}",
                        "hint": "set the provider's API key in this shell and retry"})

    stamp = dt.datetime.utcnow().isoformat() + "Z"
    json.dump({"resolved_at": stamp, "models": out},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "model_provenance.json"), "w"), indent=2)

    print("=" * 74)
    print(f"MODEL PROVENANCE  (resolved {stamp})")
    print("=" * 74)
    for r in out:
        print(f"\n{r['provider']}:")
        if "error" in r:
            print(f"  ERROR: {r['error']}\n  {r.get('hint','')}")
            continue
        print(f"  requested     : {r.get('requested')}")
        print(f"  canonical id  : {r.get('canonical_id')}")
        print(f"  created       : {r.get('created')}")
        if r.get("display_name"):
            print(f"  display name  : {r['display_name']}")
        print(f"  cutoff (docs) : {r['cutoff_docs']}")
    print("\n-> model_provenance.json")
    print("\nNOTE: 'created' is a release-ish date, NOT the training cutoff. Record the")
    print("training/knowledge cutoff by hand from each 'cutoff (docs)' link above —")
    print("that is the date that governs LiveCodeBench contamination.")


if __name__ == "__main__":
    sys.exit(main())
