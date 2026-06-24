# Real Vibe-Coding Prompt Corpus

Harvests **authentic** prompts that real developers typed while coding with
Claude Code, by crawling committed session transcripts on public GitHub.

This complements the synthetic `vibe_spectrum_data.json` (which derives 5
formality levels from HumanEval docstrings): instead of *inventing* informal
prompts, we collect how people *actually* prompt — to calibrate or sanity-check
the synthetic spectrum.

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

## Crawl results (scaled run)

A 13-query crawl (`--limit 1000`) surfaced **220** candidate transcript files
and yielded **89** unique, de-templated coding prompts across **22** repos.
Informality skew: 11 formal / 11 mid / **67 vibe** — most real prompts land at
the informal end. Discovery breadth (number of distinct query angles), not
fetch budget, is the binding constraint on corpus size; add more query
variants to `SEARCH_QUERIES` to grow it further.

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
