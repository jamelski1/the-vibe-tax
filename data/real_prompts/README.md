# Real Vibe-Coding Prompt Corpus

Harvests **authentic** prompts that real developers typed while coding with
Claude Code, by crawling committed session transcripts on public GitHub, then
classifies them along the **Vibe Spectrum** — a multi-axis taxonomy of how
people actually prompt.

This complements the synthetic `vibe_spectrum_data.json` (which derives 5
formality levels from HumanEval docstrings): instead of *inventing* informal
prompts, we collect how people *actually* prompt — to calibrate or sanity-check
the synthetic spectrum.

**→ See [`VIBE_SPECTRUM.md`](VIBE_SPECTRUM.md) for the taxonomy, distributions,
and what they mean for the Vibe Tax study.**

Scripts:
- `crawl_real_prompts.py` — harvest the **agentic-CLI** corpus from committed
  Claude Code transcripts.
- `pull_webchat_corpus.py` — harvest a parallel **web-chat** corpus from real
  WildChat (ChatGPT) conversations mirrored on GitHub (HuggingFace is firewalled
  in the cloud sandbox, so we source the mirror over the allowed GitHub access).
- `pull_webchat_hf.py` — the **unbiased** web-chat pull: streams WildChat-1M /
  LMSYS-Chat-1M directly from HuggingFace (run locally with an HF token).
- `classify_prompts.py` — label any corpus across the five spectrum axes
  (`--in/--out/--stats` to point it at either corpus).

The two media prompt very differently (web-chat is far more paste-heavy and
verbose; agentic CLI is terse) — see `VIBE_SPECTRUM.md` for the comparison.

## Running locally with an HF token (unbiased web-chat pull)

The cloud sandbox can't reach HuggingFace, so `pull_webchat_corpus.py` sourced a
GitHub *mirror* of WildChat (curated subset → selection bias). On your own
machine, pull WildChat-1M / LMSYS-Chat-1M directly for unbiased rates:

```bash
git fetch origin claude/blissful-hopper-dy0rvr
git checkout claude/blissful-hopper-dy0rvr
cd data/real_prompts

pip install "datasets>=2.0" huggingface_hub

# WildChat-1M and LMSYS-Chat-1M are GATED — visit each dataset page on HF once,
# click "Agree and access" with the account that owns your token, then:
export HF_TOKEN=hf_xxx

python pull_webchat_hf.py --dataset wildchat --limit 1000
python classify_prompts.py --in webchat_wildchat_corpus.json \
    --out vibe_spectrum_wildchat_corpus.json --stats vibe_spectrum_wildchat_stats.json
```

`pull_webchat_hf.py` streams the dataset (no full download), keeps coding turns,
flags pasted error/code, and writes the same schema the classifier consumes.
Compare its stats against the agentic `vibe_spectrum_stats.json` for unbiased
cross-medium rates. The dataset also provides a ground-truth `language` per turn
(stored as `language_meta`), better than the classifier's script heuristic.

## How it works

`crawl_real_prompts.py` runs a 6-step pipeline:

1. **Discover** — GitHub code-search API (paginated) for `*.jsonl` files
   carrying Claude Code's distinctive transcript fields. Leads with the
   `"userType":"external"` phrase to target **main sessions** (which hold
   human-typed turns) over sub-agent sidechains. Caps files-per-repo for
   diversity and skips fixture/test/example paths and `agent-*.jsonl`
   sidechain files.
2. **Fetch** — downloads each raw file from `raw.githubusercontent.com`.
3. **Extract** — parses JSONL, keeps genuine human `type:"user"` turns
   (requires `userType == "external"`, rejects `isSidechain` turns), drops
   tool results, `[Request interrupted]`, system-reminders, `<command-*>`
   wrappers, and slash-commands.
4. **Dedupe** — exact + whitespace-normalized, plus a long-prefix collapse
   that removes templated machine-generated prompts (e.g. SWE-bench-style
   harnesses that share a 60+ char prefix and differ only in a trailing id).
5. **Score** — a rough `informality` heuristic, `0.0` (formal spec) →
   `1.0` (terse vibe), based on length, punctuation, casing, politeness, and
   structure (bullets/code blocks read as formal).
6. **Save** — corpus JSON + readable sample + stats.

## Usage

```bash
python crawl_real_prompts.py --limit 80        # pilot (default)
python crawl_real_prompts.py --limit 1000      # larger crawl
python crawl_real_prompts.py --all-prompts     # keep non-coding chatter too
```

`GITHUB_TOKEN` / `GH_TOKEN` (optional) raises the search rate limit.

## Outputs

| File | Contents |
|------|----------|
| `real_prompts_corpus.json` | Each prompt with `repo`, `path`, word/char counts, `looks_coding`, `informality`. |
| `real_prompts_sample.txt`  | Human-readable sample, sorted most-vibe first. |
| `real_prompts_stats.json`  | Aggregate counts + informality buckets. |

## Crawl results (inclusive run)

A 13-query crawl (`--limit 1000`) surfaced **220** candidate transcript files
and yielded **~200** unique human prompts across **~40** repos. Making the keep
gate multilingual- and error/code-paste-aware (rather than English-coding-keyword
only) roughly **doubled** the yield and surfaced patterns the strict gate
silently dropped — ~11% non-English prompts and the pasted-error/spec patterns.
Discovery breadth (number of distinct query angles), not fetch budget, is the
binding constraint on corpus size; add more query variants to `SEARCH_QUERIES`
to grow it further.

By default the crawler is **inclusive** (keeps any substantive turn from a
coding session, in any language); pass `--strict-coding` for the old
English-keyword-only gate.

## Pilot findings

- Committed Claude Code transcripts are plentiful (~10k `.jsonl` files match
  the field signatures), but many are **agent sidechain logs** (`agent-*.jsonl`)
  with no human turns, or **test fixtures** in transcript-tooling repos
  (parsers, editors, exporters) — both filtered out.
- Leading discovery with `"userType":"external"` and dropping sidechains
  roughly **doubled the hit rate** (share of fetched files yielding a real
  prompt) from ~17% to ~32%.
- Real prompts skew terse and informal: `"commit changes"`, `"Fix the
  network"`, `"Run the agent task"`, alongside fuller asks like `"change nvim
  config so that suggestions dont come up automatically but some keypress like
  ctrl + space"`. This supports the vibe-tax premise that real prompts live
  toward the informal end of the spectrum.

## Provenance & ethics

Reads only **public** GitHub data. Every prompt retains its source `repo` and
`path` for auditability. This is for research characterization of prompt
*style*; treat the text as third-party public content.
