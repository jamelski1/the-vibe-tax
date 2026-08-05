# Vibe Tax on LiveCodeBench — port design

**Why:** HumanEval(+) is saturated (~90%), so the measured tax is real but
ceiling-suppressed (v2→v3 was +3.0 pts, McNemar p=0.001 — significant but small).
LiveCodeBench (LCB) is **hard and contamination-free** (problems dated after model
cutoffs), with frontier pass@1 ~30–70%. That headroom is the whole point: it lets
the tax express its true size instead of being crushed against 100%.

**Status:** design only. LCB is HF-gated and firewalled in the cloud sandbox, so
data fetch + runs happen on the user's machine. This doc de-risks the build; the
adapters get written against a real sample (see the fetch steps).

## Guiding principle: reuse LCB's own tooling, don't reimplement

LCB's test cases are version-specific and partly compressed (base64+zlib on the
private tests), and its evaluator handles stdin/stdout vs functional dispatch,
timeouts, and output normalization. Reimplementing that blind is the main risk.
So:

- **Load + score with LCB's official package/harness.** We do NOT rewrite its
  scorer. Our job is only: (a) generate prompts, (b) collect completions, (c)
  hand completions to LCB's evaluator.

## Scoping decisions

1. **Functional problems only (first pass).** LCB has two problem types:
   - *functional* — ships `starter_code` (a `class Solution` with a method); the
     model implements the method. This keeps our "preserve the exact
     function/method name" gradeability trick and a clean call-based scorer.
   - *stdin/stdout* — the whole program reads stdin and prints; no function.
     Different harness, and our function-name trick doesn't apply.
   Restricting to functional problems ports the machinery with least rework.
   stdin/stdout is a documented follow-up, not v1.

2. **Contamination filter.** Keep only problems with `contest_date` after each
   model's training cutoff (per-model if cutoffs differ). This is LCB's core
   value — don't lose it by including pre-cutoff problems.

3. **Difficulty for headroom.** Prefer `medium` + `hard` (that's where pass rates
   fall to 30–60%). Include some `easy` only as a bridge to the HumanEval range.

4. **Paste conditions (W2/W3) need a known-correct solution to mutate — LCB has
   none.** Options:
   - **(a) correct-then-mutate (recommended):** first get a solution that passes
     ALL of a problem's LCB tests (from a strong model), then apply our bug
     operators and verify it now fails → real buggy code + real error to paste.
     Costs one extra generation+eval per paste problem, but yields authentic
     paste conditions consistent with v2/v3.
   - **(b) defer paste conditions:** run only the 4 non-paste conditions on LCB.
     Simpler; loses the (already hardest, most interesting) paste arm.
   Recommend (a); fall back to (b) if the correct-solution yield is low on hard
   problems.

## Component port map

| Component | HumanEval(+) version | LCB change |
|-----------|----------------------|------------|
| Problem source | `vibe_spectrum_data.json` + HE jsonl | LCB loader → normalized `lcb_problems.jsonl` (functional, dated, medium/hard) |
| Task spec for generator | docstring prose | `question_content` (full NL statement) |
| Function-name trick | `entry(params)` from signature | method name + params parsed from `starter_code` (note: it's `Solution.method`) |
| Generator (v2 + v3) | `generate_experiment_prompts.py` / `generate_calibrated_prompts.py` | same conditions/styles; feed `question_content`; preserve method name; discriminator gate unchanged |
| Bug injection (paste) | mutate `canonical_solution` | correct-then-mutate (option a) |
| Runner | `run_vibe_tax.py` (env-overridable) | unchanged — prompts in, completions out |
| Scorer | `score_vibe_tax*.py` | **replace with LCB's official evaluator** (functional dispatch, its test cases) |

The generator's realism machinery (6 conditions, discriminator gate,
name-preservation) ports directly — only the *task text* and the
*name source* change.

## Normalized schema (what the fetch step produces)

`prepare_lcb.py` (run locally) writes `lcb_problems.jsonl`, one object per problem:

```json
{
  "task_id": "lcb/<platform>/<id>",
  "entry_point": "methodName",
  "class_name": "Solution",
  "params": "a, b",
  "starter_code": "class Solution:\n    def methodName(self, a, b):\n",
  "question_content": "<full natural-language problem statement>",
  "difficulty": "medium",
  "contest_date": "2024-11-01",
  "n_tests": 42
}
```
(Test cases stay in LCB's own structures for its evaluator; we only carry
metadata + the statement for prompt generation.)

## Local workflow (user)

1. `pip install livecodebench` (or clone their repo) + accept the HF dataset terms.
2. Run `prepare_lcb.py` → filter (functional + dated + medium/hard) →
   `lcb_problems.jsonl` (+ push a ~10-problem sample so the adapters can be
   built/tested against the real schema).
3. Generate prompts: v2-style and v3-style over `lcb_problems.jsonl`
   (adapters built once the sample confirms the schema).
4. Run models (`run_vibe_tax.py`, env-overridable paths).
5. Score with **LCB's evaluator**; compare v2 vs v3 with the paired McNemar test.

## Risks / open questions (resolve against the real sample)

- Exact field names + version tag of `code_generation_lite` (self-reported by
  `prepare_lcb.py`).
- Private-test decoding — prefer letting LCB's harness handle it rather than us.
- `Solution.method` calling convention in the scorer (instantiate, then call).
- Correct-solution yield on hard problems (affects paste-condition coverage).
- Cost: correct-then-mutate + 6 conditions × 3 models over a few hundred hard
  problems is more API spend than HumanEval — size the subset accordingly.

## Why this is worth the lift

If v3>v2 holds on LCB with a *larger* magnitude → "researcher-written prompts
overstate the tax, and here's its real size." If it stays ~3% with headroom →
"the tax is genuinely small, robustly." Either is a real, contamination-free
result — which HumanEval cannot give.
