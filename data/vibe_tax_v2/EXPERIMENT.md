# Vibe Tax v2 — the two-medium experiment

Measures whether *realistic* prompting styles cost correctness — with conditions
grounded in the empirical **Vibe Spectrum** (`data/real_prompts/VIBE_SPECTRUM.md`)
instead of the original synthetic 5-level formality ladder.

The spectrum research established that vibe coding is **not one distribution**:
agentic-CLI users are terse and reference context; web-chat users are verbose
and paste context (code, errors), and are far more multilingual. So v2 runs the
tax measurement **per medium**, with each condition modeled on a documented
real-world prompting mode and pegged to its observed prevalence.

## Hypotheses

- **H1 (spec tax):** pass rate rises with specification level within each
  medium (detailed > casual > terse).
- **H2 (medium interaction):** the tax differs by medium — terse agentic-style
  prompts may lose less than their brevity suggests because the signature
  context does the work; verbose web-chat prompts may not gain proportionally.
- **H3 (paste premium):** error-paste beats code-paste — a real traceback
  localizes the bug, so models should fix it more reliably than from
  "it's not working" alone.
- **H4 (language tax):** Chinese-wrapped (code-switched) prompts score at or
  near English casual prompts — or reveal a measurable multilingual tax.

## Conditions

| # | Condition | Medium | Models this real pattern | Observed prevalence |
|---|-----------|--------|--------------------------|--------------------:|
| A1 | `agentic_terse` | agentic | signature+vibe docstring as "file", ask = "finish this" | agentic casual/vague = 87% |
| A2 | `agentic_casual` | agentic | lowercase casual ask + L3 docstring | modal agentic prompt |
| W1 | `webchat_detailed` | webchat | polite conversational chat + full formal spec | webchat detailed/formal = 45% |
| W2 | `webchat_code_paste` | webchat | "I wrote this, it's not working" + **buggy code** | 28% of WildChat |
| W3 | `webchat_error_paste` | webchat | buggy code + **real captured traceback** | 12% of WildChat |
| W4 | `webchat_multilingual` | webchat | Chinese instructions around English code (code-switching) | 42% of WildChat non-English; Chinese largest |

Prompt *content* reuses the existing per-problem docstring rewrites
(`vibe_spectrum_data.json` L1/L3/L5); the *framing* per condition is patterned
on real prompts from the harvested corpora.

## Materials

- **Problems:** the same 50 HumanEval problems as v1 (comparable results).
- **Bug injection (W2/W3):** the canonical solution is mutated with small
  realistic operators (flipped comparison, `==`→`!=`, off-by-one, `and`/`or`
  swap, index shift). A mutation is kept only if the mutated solution
  **actually fails** the official HumanEval tests; the real error output is
  captured for W3 (paths scrubbed to `solution.py`). Coverage: **37/50**
  problems (74%) have a viable bug; W2/W3 run on those 37.
- **Prompt set:** `vibe_tax_prompts.json` — **274 prompts**
  (50×4 full-coverage conditions + 37×2 paste conditions).
- **Models:** ChatGPT / Claude / Codestral (same trio as v1), temperature 0.
  274 × 3 = **822 completions**.

## Procedure

```bash
# 1. (already done, committed) generate the prompt set
python generate_experiment_prompts.py

# 2. query the models — run locally where your API keys live
python run_vibe_tax.py

# 3. score with the EXISTING v1 scorer (schema-compatible: the condition is
#    stored in the "level" field)
#    point RESPONSES_FILE in data/HumanEval.jsonl/run_tests.py at
#    vibe_tax_responses.json (or copy it over all_model_responses.json)
```

## Analysis plan

1. **Primary:** pass@1 per condition × model; per-medium averages.
2. **Paired contrasts** (same problems, so per-problem paired comparisons —
   report deltas and a sign test):
   - A1 vs A2 vs W1 → spec tax (H1), across-media (H2)
   - W2 vs W3 (37-problem subset) → paste premium (H3)
   - W4 vs A2/W1 → language tax (H4)
3. **Weighted "real-world tax":** weight each condition's pass rate by its
   observed prevalence per medium to estimate the *expected* tax a real
   agentic vs web-chat user pays.
4. Slice by problem category (the scorer already buckets by category).

## Caveats (read before trusting results)

- **W3 leaks one assertion.** The pasted traceback shows the failing test call
  (real users do paste failing test output). All conditions already leak the
  docstring examples, but W3's extra hint means H3 should be phrased as
  "traceback helps" — part of that help is the revealed case. A follow-up can
  re-run W3 with the assert line elided.
- **W2/W3 are fix tasks, not synthesis tasks.** The other four are write-from-
  scratch. Compare W2↔W3 with each other (both fix), and against others only
  with that framing in mind.
- **W4 is wrapper-level code-switching** (Chinese instructions, English spec),
  the dominant observed pattern — not full translation. A translated-docstring
  arm needs an LLM pass and is future work.
- **Agentic conditions simulate file context inline.** Real agentic users rely
  on the agent opening files; here the signature+docstring stand in for that
  context so completions stay scoreable by HumanEval tests.
- Bug mutations skew simple (60% flipped `==`/comparison); real-world bugs are
  messier. Treat W2/W3 as a *lower bound* on fix difficulty.

## Provenance

- No third-party user text appears in any prompt (all content derives from
  HumanEval + our own rewrites) — the prompt set is safe to commit, unlike the
  raw harvested corpora.
- Empirical grounding: `data/real_prompts/VIBE_SPECTRUM.md` (agentic n=4,675 /
  163 repos; WildChat n=5,000; LMSYS n=5,000).
