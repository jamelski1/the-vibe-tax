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

## Phase 7 — Realism discriminator (motivates the calibrated generator)

- `data/real_prompts/realism_discriminator.py` — the `RealismDiscriminator`
  class: real corpus prompt vs synthetic prompt. Test accuracy = realism score
  (50% = indistinguishable, 100% = trivially separable).
  - **Model:** feature-based **logistic regression** (sigmoid + cross-entropy,
    batch gradient descent), pure standard library — no numpy/sklearn/network/
    signals — chosen for interpretability (weights = the "tells"). 20 hand-
    crafted features (lengths, char-class fractions, binary flags like
    has_doctest/has_triplequote/has_signature); no embeddings.
  - **Reproducible:** no RNG — class-balancing and the 75/25 split are md5-hash-
    keyed (not Python's salted hash()), weights init to zero, fixed
    hyperparameters. Same inputs → identical output on any machine (verified:
    two runs bit-identical).
  - **Importable:** `from realism_discriminator import RealismDiscriminator`;
    `.fit(real, synth) / .evaluate() / .tells() / .predict_proba(text)`.
- Demo (real WildChat web-chat n=207 vs synthetic, balanced):
  - vs OLD 5-level docstrings: **100% acc, AUC 1.000**
  - vs v2 conditions: **100% acc, AUC 1.000**
- **Top tells (both):** `has_triplequote`, `has_signature`, `has_doctest`,
  `frac_punct` → SYNTHETIC. Our prompts are dressed-up HumanEval stubs; real
  ones are conversational messages.
- **Caveat:** 100% is partly trivial — synthetic prompts literally contain code
  scaffolding real chat prompts lack, so one feature nearly separates them. The
  diagnostic's real value: (1) confirms the current synthetic spectrum is
  unrealistic, (2) becomes the **fitness function** for a calibrated generator
  (generate → discriminate → drive accuracy toward 50%).
- **Decision:** calibrated generator IS worth building; the tells say how (strip
  signature/docstring/doctest scaffolding; phrase tasks as prose messages).

## Phase 8 — Calibrated generator (v3)

- `data/vibe_tax_v3/generate_calibrated_prompts.py` — rewrites each HumanEval
  task into a realistic prompt (preserving exact function name + params so it
  stays gradeable), gated by the `RealismDiscriminator`. Pluggable backend
  (anthropic / openai / **mock** for keyless testing). Output schema = v2's, so
  it feeds `run_vibe_tax.py` unchanged.
- **Mock-backend demo result:** prose conditions read realistic —
  agentic_terse/casual 0.94–0.95, webchat_detailed 0.94, multilingual 0.98
  P(real) — vs the old synthetic **0.00**. Big realism gain, harness validated.
- **Finding — the weak discriminator breaks on paste conditions.** Error/code-
  paste prompts legitimately contain pasted code (`def` line) + traceback, which
  the feature discriminator mis-reads as synthetic (reads_real ≈ 0.01) and a
  blanket scaffolding check would reject. Resolved with **condition-aware
  gating**: paste conditions gated on function-name preservation only, with the
  discriminator score recorded as advisory. This concretely motivates the
  stronger-discriminator thread — you can't certify realistic pasted code until
  the discriminator can tell it from a synthetic stub.
- Needs API keys to produce real (non-mock) rewrites; run locally.

## Phase 9 — v2 vs v3 result (the payoff)

- User generated real v3 prompts (Claude backend) + ran all 3 models (822
  completions). Scored with the robust base scorer AND HumanEval+ (see
  `data/vibe_tax_v3/RESULTS.md`).
- **Realistic prompts get BETTER code than researcher-written ones.** v3 ≥ v2 in
  almost every cell; clearest on HumanEval+: overall **88.7% (v2) → 91.7% (v3)**,
  +4.6 on agentic_terse, +4.5 on error_paste.
- **Interpretation:** the researcher-written v2 conditions **overstated the vibe
  tax.** Real informal phrasing is in-distribution for RLHF'd models; invented
  informality ("yo so like…", one-word "finish this") is weird/out-of-
  distribution and confuses them more. A caution for the perturbation/under-
  specification literature that relies on artificial degradations.
- **Replicates:** paste/fix conditions remain hardest on HumanEval+ (both
  versions); base HumanEval still saturated; no *large* tax on easy problems.
- **Significance (paired McNemar, v2 vs v3 on HumanEval+):** 822 paired items,
  771 agree; of 51 discordant, 38 favor v3 vs 13 favor v2 → **p=0.001**, delta
  **+3.0 ± 1.7 pts**. So the effect is REAL, not noise — but **small and
  ceiling-suppressed**: at ~90% there's no room for it to grow. Significant ≠
  large. Can't size the true tax on a saturated benchmark → need harder problems.

## Phase 10 — LiveCodeBench port (design)

- Chosen next benchmark: **LiveCodeBench** (hard + contamination-free; frontier
  pass ~30–70% = real headroom). `data/vibe_tax_lcb/DESIGN.md` +
  `prepare_lcb.py`.
- Scoping (probed): LCB and BigCodeBench are **HF-firewalled** in the sandbox
  (fetch locally); MBPP+ was reachable (drop-in) but too easy for real headroom.
- Design decisions: reuse LCB's **official loader + evaluator** (don't reimpl the
  finicky test decoding); **functional problems only** first (keeps method-name
  trick + call-based scoring); **contamination date-filter**; paste conditions
  via **correct-then-mutate** (LCB has no canonical solutions). Generator's
  realism machinery (6 conditions, discriminator gate, name preservation) ports
  directly; only task text + name source change.
- Status: design done; blocked on a local LCB fetch (`prepare_lcb.py` →
  `lcb_sample.json`) so the generator/scorer adapters are built against the real
  schema rather than a reconstructed one.

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

- [x] **Discriminator diagnostic** (Phase 7): DONE — current synthetic is 100%
      separable from real; calibrated generator justified.
- [~] **LiveCodeBench port** (Phase 10): design + `prepare_lcb.py` done; blocked
      on a local LCB fetch to finalize the generator/scorer adapters. THE key
      step — the only way to size the tax with real headroom.
- [ ] **Calibrated generator** (needs API keys): LLM rewriting + classifier
      validation so conditions are statistically indistinguishable from real
      prompts. Realism upgrade, not a measurement upgrade. Uses the
      `RealismDiscriminator` as its first-pass fitness function.
- [ ] **Stronger discriminator** (pairs with the generator): the current
      `RealismDiscriminator` (logistic regression on 20 hand features incl.
      literal code-scaffolding flags) is deliberately WEAK — it separates
      current synthetic trivially and can't certify subtle style realism. Once
      the generator strips scaffolding, upgrade to TF-IDF n-grams + logistic
      regression, or an embedding / LLM-judge discriminator, to drive a
      meaningful accuracy-toward-50% signal.
- [ ] Re-run W3 with the failing-assert line elided (it currently leaks one test).

## Standing caveats (carry into any writeup)

- Agentic corpus = ~163 committers, not a random developer sample (style skew,
  though rates are stable).
- Web-chat = WildChat/LMSYS, 2023-era (GPT-3.5/4), gated, contain PII/secrets.
- HumanEval+ scorer here is an approximation (exact-equality, sampled inputs) —
  directional, not the official EvalPlus number.
- Everything on base HumanEval sits near the ceiling for frontier models.
