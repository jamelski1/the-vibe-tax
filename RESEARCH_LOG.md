# The Vibe Tax — research log

A single running record of what we did, what we found, and why we made each
decision. Newest entries at the bottom. Numbers link to committed stats files;
`git log` is the commit-level record.

**Core question:** does prompting an LLM the way real people actually do —
informal, terse, paste-heavy, multilingual — cost code correctness ("the vibe
tax")? And first: what does realistic prompting even look like ("the vibe
spectrum")?

---

## Phase 0 — Synthetic baseline (v1, pre-existing)

- **Design:** 50 HumanEval problems × 5 hand-written docstring formality levels
  (formal → "yo so like…") × 3 models (ChatGPT, Claude, Codestral) = 750
  completions, scored by HumanEval tests.
- **Weakness identified:** the 5 levels were researcher-invented, not grounded
  in how people actually prompt.
- Artifacts: `data/HumanEval.jsonl/` (`vibe_spectrum_data.json`, `query_all_models.py`, `run_tests.py`).

## Phase 1 — Harvest real prompts (agentic)

- Built `crawl_real_prompts.py`: mines committed Claude Code session transcripts
  from public GitHub → genuine human turns.
- Decision: use GitHub (proxy-allowed) since we can't invent realistic prompts.

## Phase 2 — Define the Vibe Spectrum (reframe)

- **Reframe (user):** can't measure the *tax* before defining the *spectrum*
  empirically. Two blind spots caught: multilingual prompts (English-keyword
  filter was dropping them) and copy-paste-error prompts.
- Built `classify_prompts.py` — 5 axes: spec level, intent, context provision,
  language, register.
- Output: `data/real_prompts/VIBE_SPECTRUM.md` + `*_stats.json`.

## Phase 3 — Web-chat arm (WildChat / LMSYS)

- **Key finding origin:** agentic transcripts have ~2% error-paste because the
  *agent* reads errors via tools — error-pasting is a web-chat behavior.
- HuggingFace firewalled in sandbox → first used a GitHub mirror; then user ran
  unbiased HF pulls locally: **WildChat-1M n=5,000**, **LMSYS-Chat-1M n=5,000**
  (`pull_webchat_hf.py`).

## Phase 4 — Scale, rigor, hygiene

- **LMSYS replication check:** split findings into *medium effects* (replicate:
  detail, register, code-paste) vs *population effects* (don't: error-paste
  12%→3%, multilingual 42%→16%).
- **Data hygiene:** GitHub push-protection caught a real OpenAI key a user had
  pasted into a WildChat conversation → now gitignore all raw corpora, commit
  only aggregate stats.
- **Agentic scale-up:** deep repo-tree enumeration + 31 query angles took the
  agentic corpus 198 → 2,356 → **4,675 prompts / 163 repos**; top-10-repo
  concentration 88% → 36%. Profile stayed stable (findings not a power-user
  artifact).

### Spectrum headline (committed in VIBE_SPECTRUM.md)

| Axis | Agentic (n=4,675) | WildChat (n=5,000) | LMSYS (n=5,000) |
|---|--:|--:|--:|
| Any paste | 8% | 54% | 36% |
| References files instead | 21% | ~4% | ~4% |
| Pasted error | 2% | 12% | 3% |
| Casual/vague spec | 87% | 55% | 53% |
| Multilingual | 23% | 42% | 16% |

**Conclusion:** the medium shapes the prompt. Agentic users point at files;
web-chat users paste. Robust across both web-chat datasets; error-paste and
language mix are population-, not medium-, driven.

## Phase 5 — Vibe Tax v2 experiment

- Built `data/vibe_tax_v2/`: 6 conditions per problem, each pegged to a measured
  real prompting mode with its prevalence (agentic terse/casual; web-chat
  detailed / code-paste / **error-paste with real captured tracebacks** /
  multilingual). Bug injection mutates canonical solutions, verified to fail
  the tests. 274 prompts × 3 models = **822 completions** (user ran locally).

### Scoring was the hard part (critical lesson)

Raw score with the v1 tester gave `webchat_error_paste = 0%` for all models — a
**scoring artifact**, not a result (conversational "fix it" replies broke the
body extractor). Fixing it took several rounds:

| Scorer state | overall | note |
|---|--:|---|
| v1 tester | (error_paste 0%) | first-code-line extractor grabs buggy snippet |
| + last-`def` extractor | 78.2% | recovers conversational replies |
| + v1 body normalizer | 90.5% | fixes bare bodies |
| + first-line-indent repair | 95.0% | Claude 84%→98% (indent artifact) |
| + fenced-block isolation | **96.2%** | final; residual 31 fails, 24 genuine |

**Lesson:** for prompt-style studies, code extraction can swing a condition by
90+ points. Any tax number is meaningless without a paste-robust scorer.

### v2 result (base HumanEval, `vibe_tax_scored_stats.json`)

All conditions **93–99%**; agentic 96.3% ≈ web-chat 96.2%. Small tax, only at
the extreme (agentic_terse/L5 vibe = 93.3% vs agentic_casual/L3 = 99.3%). No
medium effect; no paste premium; language tax only for the weak model
(Codestral multilingual 88%). **Dominant caveat: HumanEval is saturated
(ceiling).**

## Phase 6 — Harder benchmark: HumanEval+ re-score (no new API calls)

- `score_vibe_tax_plus.py` re-scored the same 822 completions against HumanEval+
  edge-case tests (approximate harness: output-equivalence, ~60 inputs/problem).
- Overall 96.2% → **88.7%** (−7.5); condition ranking *changed*.
- **error_paste fell from mid-pack to lowest** (95.5 → 83.8, −11.7). Validated
  genuine: all 3 models fail the same Fibonacci-family problems (HE/63,39,55).
  Interpretation: fixing the *shown* bug passes the visible test but is less
  edge-case-robust than write-from-scratch.
- **Conclusion:** condition ranking is test-dependent; the base result was
  partly a weak-test artifact. Full de-saturation likely needs harder
  *problems* (LiveCodeBench), not just harder tests.

## Literature positioning (from web search)

- Closest prior: [WildCode](https://arxiv.org/html/2512.04259) (coding
  conversations from WildChat, but no cross-medium / correctness measurement);
  [When Prompt Under-Specification Improves Code Correctness](https://arxiv.org/html/2604.24712v1)
  (−15% pass@1, sometimes reversed); [ReCode](https://aclanthology.org/2023.acl-long.773/)
  (mechanical docstring perturbations); [Mind Your Tone](https://arxiv.org/abs/2510.04950)
  (rude > polite on accuracy).
- **"vibe tax"/"vibe spectrum" as terms: unclaimed.** Our defensible novelty:
  (1) two-medium contrast under one classifier; (2) conditions pegged to
  measured prevalence; (3) error-paste with real tracebacks; (4) prevalence-
  weighted per-medium tax.

## Open threads / next steps

- [ ] **Discriminator diagnostic** (cheap, no API keys): real-vs-synthetic
      classifier to quantify how unrealistic the current synthetic prompts are →
      decides whether the calibrated generator is worth building.
- [ ] **LiveCodeBench port** for harder *problems* (real de-saturation).
- [ ] **Calibrated generator** (needs API keys): LLM rewriting + classifier
      validation so conditions are statistically indistinguishable from real
      prompts. Realism upgrade, not a measurement upgrade.
- [ ] Re-run W3 with the failing-assert line elided (it currently leaks one test).

## Standing caveats (carry into any writeup)

- Agentic corpus = ~163 committers, not a random developer sample (style skew,
  though rates are stable).
- Web-chat = WildChat/LMSYS, 2023-era (GPT-3.5/4), gated, contain PII/secrets.
- HumanEval+ scorer here is an approximation (exact-equality, sampled inputs) —
  directional, not the official EvalPlus number.
- Everything on base HumanEval sits near the ceiling for frontier models.
