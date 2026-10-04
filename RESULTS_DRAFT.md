# Results (draft, Section 4: "The artifact")

Draft prose for the core-result section of framing (A), built around Fig. 1 and the
three-extractor table. Numbers trace to `THREE_EXTRACTOR_RESULTS.md` (pass-rates, discordant
counts) and to a fresh recomputation from `lcb_v3_responses.json` (compile-rates, prose volume,
failure reasons). See **Notes for the author** at the end before using any number.

---

## 4 The register effect is an extraction artifact

We first report what a standard-but-naive extraction pipeline concludes about prompt register
(Section 4.1), then re-score the same stored completions with two independent robust extractors
(Section 4.2), and finally trace the discrepancy to a single mechanism (Section 4.3). All analyses
use the capable-model slice (ChatGPT and Claude; 167 problems x 2 models = 334 paired observations
per register condition, 1,336 completions per extractor). Every comparison re-scores the *same*
completions; no model is re-queried, so differences between extractors reflect scoring alone.

### 4.1 Naive extraction yields a large, highly significant "politeness tax"

Under the naive extractor, terse prompts appear to beat the polite, detailed wrapper by a wide
margin. Terse prompts pass 73.4% of problems; detailed prompts pass 57.5% (Fig. 1b, orange). The
paired difference is **+15.9 points** (95% CI 10.2 to 21.6). Of the 103 discordant pairs, terse
alone passed 78 and detailed alone passed 25 (exact McNemar p = 1.6 x 10^-7; Fig. 1c). Casual
(66.5%) and multilingual (66.2%) prompts fall between the two.

Read in isolation, this result supports a strong and publishable claim: polite, detailed
prompting costs a capable model roughly one problem in six on a de-saturated benchmark. The claim
has the same shape as effects reported in the politeness-prompting literature
[Mind Your Tone; Cai et al.], and nothing in the naive pipeline's output signals a problem.

### 4.2 The effect vanishes under two independent robust extractors

We re-scored the identical completions with our robust extractor (`score_lcb.py`) and with
EvalPlus's `sanitize` algorithm [Liu et al. 2023; evalplus repository], an independent,
widely used implementation built on a different design (tree-sitter parsing, longest valid
substring, reachability from the entry point). Table 3 reports all three.

**Table 3.** Pass-rate (%) by register condition and the paired terse minus detailed difference,
under three extractors applied to the same 1,336 completions. Capable models; full official LCB
test suites.

| Condition | Naive | Ours | EvalPlus `sanitize` |
|---|---:|---:|---:|
| terse | 73.4 | 81.4 | 66.8 |
| casual | 66.5 | 81.4 | 68.0 |
| multilingual | 66.2 | 84.1 | 70.1 |
| detailed (polite) | 57.5 | 81.4 | 69.2 |
| **terse minus detailed (pts)** | **+15.9** | **0.0** | **-2.4** |
| 95% CI (Wald, paired) | [10.2, 21.6] | [-3.0, 3.0] | [-6.1, 1.3] |
| discordant pairs (terse-only / detailed-only) | 78 / 25 | 13 / 13 | 16 / 24 |
| exact McNemar p | 1.6 x 10^-7 | 1.00 | 0.27 |

Both robust extractors erase the effect. Our extractor produces a perfectly balanced set of
discordant pairs (13 vs 13; p = 1.00). `sanitize` produces a small difference in the *opposite*
direction that does not approach significance (-2.4 points; p = 0.27). The two robust confidence
intervals both include zero and exclude the naive point estimate entirely (Fig. 1c).

The agreement matters because our own extractor sits under suspicion in a paper about extractors.
A lenient extractor could, in principle, manufacture a null. `sanitize` rules out that reading: a
tool we did not write, designed for a different benchmark format, reaches the same conclusion.

**Absolute levels differ by design; read the asymmetry.** `sanitize` scores 12 to 15 points below
our extractor in every condition. Its reachability step targets top-level functions, while
LiveCodeBench's functional problems define a method on `class Solution`, so `sanitize` often falls
back to its longest-valid-substring heuristic and drops some correct code. That loss is roughly
uniform across conditions, however. The quantity our thesis concerns, the *condition asymmetry*,
collapses from a 15.9-point spread (naive) to 2.7 points (ours) and 3.3 points (`sanitize`)
across all four conditions.

### 4.3 Mechanism: register drives prose volume, and prose breaks naive extraction

The system prompt asks for plain Python without markdown fences. Models frequently comply with
the "no fences" instruction yet still append an explanation after the code ("This checks every
adjacent pair..."). With no fence to mark where the code ends, the naive extractor passes code
plus trailing English to the interpreter, which raises `SyntaxError` on otherwise correct
solutions.

Register controls how much of that trailing prose appears. Table 4 traces the chain.

**Table 4.** Prose volume and naive-extraction failure by register condition (334 completions per
condition). "Non-code text" counts characters outside the code our extractor recovers.

| Condition | Median reply length (chars) | Replies with >20 chars non-code text | Naive `SyntaxError` failures | Naive compile-rate | Robust compile-rate (ours / `sanitize`) |
|---|---:|---:|---:|---:|---:|
| terse | 1,053 | 29.6% | 40 | 88.0% | 100.0 / 81.4 |
| casual | 1,117 | 37.7% | 63 | 81.1% | 100.0 / 84.1 |
| multilingual | 1,336 | 43.4% | 76 | 77.2% | 100.0 / 83.2 |
| detailed (polite) | 1,424 | 58.1% | 89 | 73.4% | 100.0 / 84.1 |

Three observations establish the mechanism.

1. **The ordering is monotone.** Reply length, the share of replies carrying explanatory prose,
   and the count of naive failures all rise in the same order: terse, casual, multilingual,
   detailed. The polite, detailed wrapper roughly doubles the share of prose-bearing replies
   relative to terse (58.1% vs 29.6%).
2. **Every naive failure is a parse failure.** All 268 naive compile failures across the four
   conditions are `SyntaxError`s; the naive extractor never fails to locate the target class or
   function. The extractor finds the right code and then chokes on the words after it.
3. **The asymmetry exists before a single test runs.** Compile-rate (Fig. 1a) requires only
   `ast.parse`, with zero test execution. The naive compile-rate falls 14.6 points from terse to
   detailed, almost exactly the 15.9-point pass-rate gap. Paired over the same 334 observations,
   naive compiles terse-only in 73 cases and detailed-only in 24 (exact McNemar p = 6.4 x 10^-7).
   Our extractor compiles 100% in every condition; `sanitize` stays within a 2.7-point band with
   no significant terse-detailed asymmetry (12 vs 21; p = 0.16).

An analogy helps here. Picture a grader who stops reading at the first sentence that does not
parse as an answer. A student who writes the answer and stops gets full marks; a student who
writes the same answer and then politely explains it gets zero. The grader has measured
politeness, not correctness. The naive extractor behaves exactly this way, and the register
manipulation changes how often models "explain afterwards."

The robust extractors cannot inflate correctness to produce the null. Each one only removes text;
the extracted code must still pass every official test. A completion that our extractor rescues
was already correct code that the naive pipeline discarded for its trailing prose.

### 4.4 Sanity checks

- **No test sampling.** All 167 problems carry 33 to 44 official tests (median 42), so the
  `MAX_TESTS = 60` setting runs every test for every problem. Re-running with an unbounded cap
  would produce byte-identical results.
- **Contamination cannot create the register effect.** Absolute pass-rates may reflect
  memorization (model training cutoffs postdate the problem window), but each problem appears in
  all four conditions for the same model, so memorization shifts every condition equally. We
  describe the problem set as contamination-*window*, not contamination-free.
- **No re-querying.** All extractor comparisons re-score one fixed set of stored completions,
  which removes sampling noise between conditions from the extractor comparison.

### Figure 1 caption (draft)

> **Figure 1. A significant "politeness tax" that exists only under naive extraction.** Three
> extractors score the same 1,336 completions (ChatGPT and Claude, 167 LiveCodeBench problems,
> four register wrappers around an identical problem statement). **(a)** Share of completions
> whose extracted code parses and defines the target; no tests run. Naive extraction degrades as
> the wrapper grows wordier; both robust extractors stay flat. **(b)** Pass-rate on the full
> official test suites. **(c)** Paired terse minus detailed pass-rate difference with Wald 95% CI
> and exact McNemar p (n = 334 pairs). The +15.9-point effect under naive extraction disappears
> under our extractor (0.0) and under EvalPlus `sanitize` (-2.4).

Files: `figures/fig1.pdf` (vector, for submission), `figures/fig1.svg`, `figures/fig1.png`;
regenerate with `python figures/make_fig1.py`.

---

## Notes for the author (resolve before submission)

1. **Two different "naive" headline numbers exist in the repo. Pick one and use it everywhere.**
   `PAPER_FRAMING.md` (one-sentence contribution, Section 4 outline) and
   `data/vibe_tax_lcb/RESULTS.md` report the *original* pre-fix pipeline: **+9.3 pts, p = 0.007**
   (capable models), with the robust null as **+0.6, p = 0.845**. `THREE_EXTRACTOR_RESULTS.md`
   reports the notebook's re-implemented naive extractor: **+15.9, p = 1.6 x 10^-7**, robust
   **0.0, p = 1.0**. This draft uses the three-extractor numbers because they come from one
   reproducible run with all three extractors on identical inputs. The difference likely
   reflects scorer details in the original pipeline (for example, the per-test timeout added
   with the fix), but I have not confirmed the cause. Options: (a) use +15.9 throughout and
   mention +9.3 only in the introduction's narrative as "what we first observed"; or (b) re-run
   the original pipeline and explain the gap in a footnote. The framing doc and abstract need
   updating either way.
2. **Two runs of the three-extractor pass-rates exist** (they differ by up to 0.3 points per cell,
   attributed to timeout noise). This draft and Fig. 1 use the *second* run, because its
   pass-rates match the McNemar discordant counts exactly (e.g., naive 73.4 - 57.5 = 15.9 =
   (78 - 25)/334). Report this as scoring noise in Methods or pin one run.
3. **Pass-rates are not re-verified here.** Re-scoring needs the HF-gated `lcb_tests.jsonl`. I
   reproduced the compile-rates exactly and recomputed every McNemar p and CI from the discordant
   counts; the pass-rate cells themselves come from `THREE_EXTRACTOR_RESULTS.md`.
4. **Table 4 is new.** The prose-volume and failure-reason columns come from a fresh analysis of
   `lcb_v3_responses.json` (2026-10-04). "Non-code text" is a coarse proxy (reply length minus the
   length of the code our extractor recovers), so it also counts any leading prose and fence
   markers. Reproduce with `python figures/mechanism_table.py` (pulls the extractors verbatim from
   the three-extractor notebook).
5. **"Politeness" vs "detail."** The detailed wrapper is both polite and longer ("A polite, clear
   chat message asking for help"), so it varies two things at once. The Results prose says
   "polite, detailed" deliberately; Limitations should say the design cannot separate politeness
   from verbosity, which is fine for an artifact paper because the mechanism runs through prose
   volume either way.
6. **Citations to add** (verified entries exist in `RELATED_WORK_NOTES.md`): EvalPlus (Liu, Xia,
   Wang, Zhang, NeurIPS 2023) plus a repository citation for `sanitize`, since the paper itself
   does not describe it; LiveCodeBench (Jain et al., ICLR 2025; verify before citing, it is not
   in the notes file); McNemar (1947), *Psychometrika* 12(2): 153-157; Agresti, *Categorical Data
   Analysis*, 3rd ed. (2013) for the paired-proportion Wald interval.
