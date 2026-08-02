# Vibe Tax v3 — calibrated prompt generator

v2's conditions were researcher-written and are **100% distinguishable** from
real prompts (`realism_discriminator`). v3 *rewrites* each HumanEval task into a
prompt styled like real usage and **gates** each rewrite through the
`RealismDiscriminator`, so obviously-synthetic scaffolding is rejected.

The two-layer trick is preserved: the **content** is always a gradeable
HumanEval task (the rewrite keeps the exact function name + parameter order, so
`run_tests.py` / `score_vibe_tax*.py` still work); only the **style** is made
realistic.

## Pipeline

For each problem × condition:
1. Build a rewrite instruction: keep the exact `entry_point(params)`, convey the
   behaviour (docstring prose, doctests stripped), target the condition's real
   style, and **forbid the scaffolding tells** (`def` line, `"""` docstring,
   `>>>` doctests) — phrase it as a chat message.
2. An LLM backend produces the message.
3. **Gate** it, then keep / regenerate.

## Run it

```bash
# no keys — deterministic mock backend, exercises the whole harness
python generate_calibrated_prompts.py --backend mock --limit 5

# realistic rewrites (local, with keys) — auto-picks anthropic/openai
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
python generate_calibrated_prompts.py --retries 2
#   -> vibe_tax_v3_prompts.json  (same schema as v2; feed to run_vibe_tax.py)
```

The discriminator gate needs a real corpus present in `data/real_prompts/`
(`real_prompts_corpus.json` and/or `webchat_*_corpus.json`). Without one the
gate disables (scaffolding + name checks still apply) and warns.

## Condition-aware gating (and the finding it exposed)

| Gate check | prose conditions | paste conditions |
|------------|:----------------:|:----------------:|
| function name preserved (gradeable) | ✅ required | ✅ required |
| scaffold-free (`def`/`"""`/`>>>`) | ✅ required | ⬜ not applied |
| discriminator `P(real) ≥ threshold` | ✅ required | ⬜ advisory only |

**Why paste conditions are gated differently — a real finding.** An
error/code-paste prompt legitimately *contains* code (a `def` line, sometimes a
docstring) and a traceback — that's the whole point. Our current
`RealismDiscriminator` is deliberately weak (feature flags incl. `has_signature`
/ `has_traceback`), so it mis-reads pasted code as synthetic
(`reads_real ≈ 0.01`), and a blanket scaffolding check would reject it. So paste
conditions are gated on **function-name preservation only**, with the
discriminator score recorded as advisory.

Mock-backend demo (per-condition avg `reads_real`, discriminator P(real)):

| Condition | reads_real |
|-----------|-----------:|
| agentic_terse | 0.95 |
| agentic_casual | 0.94 |
| webchat_detailed | 0.94 |
| webchat_multilingual | 0.98 |
| webchat_code_paste | 0.01 *(advisory)* |
| webchat_error_paste | 0.02 *(advisory)* |

Prose conditions land ~0.95 (vs the old synthetic **0.00**) — a real realism
gain. The paste conditions can't be certified until the **stronger
discriminator** (RESEARCH_LOG open thread) exists — one that distinguishes
*realistic* pasted code from a synthetic stub.

## Caveats

- The `mock` backend is a deterministic placeholder to test the harness; real
  realism needs an LLM backend (your keys).
- Passing this weak discriminator means "scaffolding removed," not "certified
  realistic." It is a floor and a first-pass fitness function, not a ceiling.
- `vibe_tax_v3_prompts.json` is gitignored (regenerated per backend).
