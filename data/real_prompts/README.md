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
   carrying Claude Code's distinctive transcript fields (`parentUuid`,
   `sessionId`, `userType`, `isSidechain`, `cwd`). Caps files-per-repo for
   diversity and skips fixture/test/example paths.
2. **Fetch** — downloads each raw file from `raw.githubusercontent.com`.
3. **Extract** — parses JSONL, keeps genuine human `type:"user"` turns, drops
   tool results, `[Request interrupted]`, system-reminders, `<command-*>`
   wrappers, slash-commands, and sub-agent sidechain noise.
4. **Dedupe** — exact + whitespace-normalized.
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

## Pilot findings

- Committed Claude Code transcripts are plentiful (~10k `.jsonl` files match
  the field signatures), but many are **agent sidechain logs** (`agent-*.jsonl`)
  with no human turns, or **test fixtures** in transcript-tooling repos
  (parsers, editors, exporters) — both filtered out.
- Real prompts skew terse and informal: `"commit changes"`, `"Fix the
  network"`, `"Run the agent task"`, alongside fuller asks like `"change nvim
  config so that suggestions dont come up automatically but some keypress like
  ctrl + space"`. This supports the vibe-tax premise that real prompts live
  toward the informal end of the spectrum.

## Provenance & ethics

Reads only **public** GitHub data. Every prompt retains its source `repo` and
`path` for auditability. This is for research characterization of prompt
*style*; treat the text as third-party public content.
