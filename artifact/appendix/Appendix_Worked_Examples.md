---
title: "Appendix: Worked Examples for the Pitfall Catalog"
lang: en
---

Table II of the paper catalogs fifteen pipeline pitfalls. This appendix gives one
worked example for each, in catalog order. Every example is a record from the
study's stored data. None is constructed. For each one, the relevant extractor
or harness was run on the stored record, and the section reports what it did.

**How to read the sections.** Each section gives the problem and condition, an
excerpt of the stored reply or prompt (`[...]` marks trimmed text), what the
extractors or the harness do with it, and the verdict.

**Where the records come from.**

- *The 1,336.* The paper's slice: 167 LiveCodeBench problems, two models, four
  registers, in `data/lcb_v3_responses_capable.json`. A "stored verdict" is the
  pass or fail recorded for that reply in `data/lcb_scored_sieve.json`, which
  was produced with Sieve and LiveCodeBench's official tests.
- *Official tests.* The full test file is gated and is not on this branch. The
  branch holds the official tests for two problems (lcb/3517 and lcb/3527, 37
  tests each), and sections 6, 10 and 14 run the harness on those. Elsewhere,
  extracted code is run on the examples printed in the stored problem statement
  ("statement examples"), and the stored verdict is quoted for the full suite.
- *HumanEval phase.* Pitfalls 3, 7, 8 and 13 belong to the study's earlier
  HumanEval phase and have no instance among the 1,336. Their examples come from
  that phase's stored replies in the research tree of this branch
  (`data/politeness_tax_v2/`), with HumanEval's own tests. These sections say so.

**Reproducing.** `appendix/verify_examples.py` re-runs every check quoted here,
and `appendix/verify_output.txt` is its captured output. `appendix/README.md`
explains how to rebuild this PDF.

| # | Pitfall | Record used | Source | Status |
|--:|---|---|---|---|
| 1 | Trailing prose after unfenced code | lcb/3747, GPT-5.4, detailed | the 1,336 | Verified |
| 2 | Leading prose before code | lcb/3535, Claude Opus 4.6, casual | the 1,336 | Verified |
| 3 | Buggy snippet quoted first in "fix my error" | HumanEval/52, GPT-5.4, error-paste | HumanEval phase | Verified |
| 4 | Helper defined above Solution dropped | lcb/3680, GPT-5.4, terse | the 1,336 | Verified |
| 5 | Code split across several fenced blocks | lcb/3789, Claude Opus 4.6, terse | the 1,336 | Verified |
| 6 | Shape mismatch: bare, nested or renamed | none mis-scored | n/a | No instance in the stored data |
| 7 | Full def appended under the prompt's def | none with the cataloged failure | n/a | No instance in the stored data |
| 8 | Ragged first-line indentation | HumanEval/35, Claude Opus 4.6, casual | HumanEval phase | Verified |
| 9 | Fence-marker variants, unbalanced backticks | lcb/3785, Claude Opus 4.6, multilingual | the 1,336 | Verified |
| 10 | Over-aggressive extraction returns nothing | lcb/3517, GPT-5.4, terse | the 1,336 | Verified |
| 11 | Whole-suite timeout instead of per test | lcb/3551, GPT-5.4, terse | the 1,336 | Stored runs only; not re-executed |
| 12 | Token cap: empty or truncated mid-code | lcb/3700, Claude Opus 4.6, casual | the 1,336 | Verified (truncated); no empty reply stored |
| 13 | Weak tests pass subtly wrong code | HumanEval/55, GPT-5.4, detailed | HumanEval phase | Verified |
| 14 | Binary pass@1 hides near-misses | lcb/3527, GPT-5.4, detailed | the 1,336 | Verified |
| 15 | Non-ASCII prompt corrupted by code page | lcb/3540, GPT-5.4, multilingual | the 1,336 | Verified in the stored file; not in the prompt as sent |

: The fifteen pitfalls and the record used for each.

# 1. Trailing prose after unfenced code

**Problem and condition.** LiveCodeBench problem 3747 (`maxAdjacentDistance`,
easy), GPT-5.4, detailed register. This is the example the paper's Mechanism
section points to.

**The reply, in full.** Correct code followed by explanatory prose, with no code
fence:

```
class Solution:
    def maxAdjacentDistance(self, nums):
        n = len(nums)
        ans = 0

        for i in range(n):
            ans = max(ans, abs(nums[i] - nums[(i + 1) % n]))

        return ans

This checks every adjacent pair, including the circular pair formed by the last and first elements using modulo.

Time complexity: O(n)
Space complexity: O(1)
```

**What the extractors do.**

| Extractor | What it extracts | Result |
|---|---|---|
| Naive | The code together with the prose that follows it (379 characters) | `SyntaxError` at line 11 of the extracted text. Fail |
| Sieve | The code only (220 characters) | Compiles; returns 3 and 5 on the two statement examples, as the statement says. Pass |
| sanitize (EvalPlus) | The code only | Same as Sieve. Pass |
| LCB-official | Nothing | Fail (no code) |

**Verdict.** The stored verdict is PASS. The naive extractor feeds the prose to
the interpreter as part of the program, so a correct reply is scored as a
failure. The two robust extractors strip the prose and the same reply passes.
The benchmark's own extractor finds no code in the form it expects and scores
the reply as empty.

# 2. Leading prose before code

**Problem and condition.** LiveCodeBench problem 3535 (`countOfPairs`, hard),
Claude Opus 4.6, casual register.

**The reply.** 64 lines and no code fence. Lines 1 to 31 are a derivation in
prose. The code starts on line 32 and runs to the last line, so nothing follows
it:

```
Looking at this problem, I need to count pairs (arr1, arr2) where:
- arr1 is non-decreasing
- arr2 is non-increasing
- arr1[i] + arr2[i] = nums[i]
[...]
The answer is the sum of all dp[n-1][v].

class Solution:
    def countOfPairs(self, nums):
        MOD = 10**9 + 7
        n = len(nums)
        [...]
        return sum(dp) % MOD
```

**What the extractors do.**

| Extractor | What it extracts | Result |
|---|---|---|
| Naive | The whole reply, prose first (2,246 characters) | `SyntaxError` at line 1 of the extracted text. Fail |
| Sieve | The code only (1,020 characters) | Compiles; returns 4 and 126 on the two statement examples, as the statement says. Pass |
| sanitize (EvalPlus) | The code only | Same as Sieve. Pass |
| LCB-official | Nothing | Fail (no code) |

**Verdict.** The stored verdict is PASS. This is the mirror image of pitfall 1:
the prose sits before the code, and the naive extractor fails on the reply's
first line.

# 3. Buggy snippet quoted first in "fix my error"

**Source.** No instance among the 1,336: the LiveCodeBench study has no "fix my
error" condition. The record is from the HumanEval phase, which did
(`data/politeness_tax_v2/politeness_tax_responses.json`).

**Problem and condition.** HumanEval/52 (`below_threshold`), GPT-5.4,
error-paste condition. The prompt pastes a traceback and a function whose
comparison is inverted, and asks how to fix it.

**The reply.** It quotes the buggy function, then gives the fix, then a compact
version. The target function is defined three times, on lines 9, 19 and 37:

```
Your logic is backwards.
[...]
But your code currently says:

def below_threshold(l: list, t: int):
    for e in l:
        if e < t:
            return False
    return True
[...]
Fix it like this:

def below_threshold(l: list, t: int):
    [...]
    for e in l:
        if e >= t:
            return False
    return True
[...]
You could also write it more compactly as:

def below_threshold(l: list, t: int):
    return all(e < t for e in l)
[...]
```

**What the scorers do.**

| What is run against HumanEval's official tests | Result |
|---|---|
| The first `def` in the reply (line 9, the quoted bug) | `AssertionError`. Fail |
| The last `def` in the reply (line 37) | Passes |
| The first scorer (`run_tests.py`). Its cleaning step starts at the first code-like line and keeps the body of the first `def`, the quoted bug, plus the text after it | Strategy 1 (prompt + body): `SyntaxError`. Strategy 2 (standalone): `IndentationError`. Fail |
| The current scorer (`score_politeness_tax.py`), which tries the last `def` first | Pass |

**Verdict.** The stored verdict under the current scorer is PASS. Over all 74
error-paste replies of the two capable models, the first scorer passes none and
the current scorer passes 72. In 3 of the 74, the first definition of the target
fails the tests while the last one passes. The first scorer fails the other 71
as well, including replies whose first definition is correct.

# 4. Helper defined above Solution dropped

**Problem and condition.** LiveCodeBench problem 3680 (`countComponents`, hard),
GPT-5.4, terse register.

**The reply.** Code only, 63 lines. A union-find helper class is defined above
`class Solution`, which uses it:

```
from math import gcd

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n
    [...]

class Solution:
    def countComponents(self, nums, threshold):
        n = len(nums)
        dsu = DSU(n)
        [...]
        return len(roots)
```

**What the extractors do.** The extractor the study used before this fix sliced
the reply from the `class Solution` anchor. It is kept verbatim in
`appendix/historical.py`.

| Extractor | What it extracts | Result on the two statement examples |
|---|---|---|
| Pre-fix extractor (anchor slice) | From `class Solution` on (984 characters). `class DSU` is gone | Compiles, then `NameError: name 'DSU' is not defined` on both |
| Sieve | The whole reply (1,488 characters), helper included | Returns 4 and 2, as the statement says |

**Verdict.** The same stored reply is FAIL in the stored scoring run made with
the pre-fix extractor and PASS in the current Sieve run. The code was correct
both times. In the 1,336, the pre-fix extractor drops a definition placed above
`Solution` in 10 replies.

# 5. Code split across several fenced blocks

**Problem and condition.** LiveCodeBench problem 3789 (`maxSubarrays`, hard),
Claude Opus 4.6, terse register.

**The reply.** 88 lines with two fenced blocks. The first holds the complete
solution (55 lines). The second repeats one line of it:

~~~
Looking at this problem, I need to find which conflicting pair to remove [...]

```python
class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: list[list[int]]) -> int:
        [...]
        max_gain = max(gain)
        return base_count + max_gain
```

Wait, I need to handle the case where multiple pairs share the same pair index
[...] After testing against examples, this logic works. Let me return the solution:

```python
return base_count + max(gain)
```

The solution handles both examples correctly.
~~~

**What the extractors do.**

| Extractor | Block it takes | Result |
|---|---|---|
| Naive | Block 1, the first block that names `class Solution` | Compiles; returns 9 and 12 on the two statement examples, as the statement says |
| Sieve | Block 1 | Same. Pass |
| sanitize (EvalPlus) | Block 1 | Same |
| LCB-official | Block 2, the text between the last two fence markers: `return base_count + max(gain)` | `SyntaxError: 'return' outside function`. Fail |

**Verdict.** The stored verdict is PASS. An extractor that takes a single block
by position scores this passing reply as a failure, because the last block is
not the solution. In the 1,336, 26 replies have two or more fenced blocks, and
in 23 of them Naive and LCB-official extract different text. No reply in the 1,336
puts a helper in one block and `Solution` in another, the form the catalog
names first.

# 6. Shape mismatch: bare, nested or renamed

**No instance in the stored data.** No stored reply was mis-scored because of
its shape.

**What the stored data do contain.** Sieve's output was parsed for all 2,004
stored replies:

| Model | Method on a top-level `class Solution` | Bare top-level function | Renamed or nested class |
|---|--:|--:|--:|
| GPT-5.4 | 668 | 0 | 0 |
| Claude Opus 4.6 | 668 | 0 | 0 |
| Codestral (outside the 1,336) | 596 | 71 | 0 |

One Codestral reply is a refusal with no code. So the bare-function shape occurs
only outside the paper's slice, and the nested and renamed shapes do not occur
at all.

**A bare-function reply, and what the harness does with it.** LiveCodeBench
problem 3517, Codestral, terse register. The prompt asks for
`Solution.shortestDistanceAfterQueries(n, queries)`. The reply defines a
module-level function instead:

~~~
Here's a Python solution to the problem:

```python
from collections import defaultdict, deque

def shortestDistanceAfterQueries(n, queries):
    # Initialize the graph with the initial roads
    graph = defaultdict(list)
    [...]
    return answer
```
[...]
~~~

After the extracted code is executed, no name `Solution` exists. The harness's
resolver binds the bare function, and `score_lcb.grade` passes it on all 37
official tests stored for this problem. The stored verdict is PASS. The
resolver in the study's first LiveCodeBench scorer also fell back to a bare
function, so this shape was never mis-scored.

**Where the pitfall was observed.** Not in a model reply. The study made its
resolver shape-agnostic as hardening, and its test
`data/politeness_tax_lcb/test_class_shape.py` shows that the earlier resolver,
which looked up the name `Solution`, fails on a renamed class and on a nested
class. Those two inputs are synthetic shapes written for the test. The same
script scans the stored replies and finds none that the earlier resolver
misses.

# 7. Full def appended under the prompt's def

**No instance in the stored data** of the failure the catalog describes, a
double signature that breaks the assembled program.

**What was checked.** HumanEval's convention is to append the completion under
the prompt, which ends inside a `def`. Across the two HumanEval-phase reply
files, 586 replies from the two capable models are a clean, complete, correct
function. Appended under the prompt unchanged, all 586 still pass: the second
`def` simply redefines the first.

**A related failure that does occur.** The first scorer does not append the
reply unchanged. It converts a full function to a bare body and appends that.
Two of the 586 fail there. One is HumanEval/71 (`triangle_area`), GPT-5.4,
casual register, in `data/politeness_tax_v2/he4_responses.json`. The reply, in
full:

```
import math

def triangle_area(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a:
        return -1

    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    return round(area, 2)
```

| What is run against HumanEval's official tests | Result |
|---|---|
| The reply appended under the prompt, unchanged | Passes |
| The reply on its own | Passes |
| First scorer, strategy 1: the prompt's `def` plus the body it cut out of the reply. The `import math` line above the reply's `def` is not part of that body | `NameError: name 'math' is not defined`. Fail |
| First scorer, strategy 2: that body on its own | `IndentationError`. Fail |
| Current scorer, which tries the complete function first | Pass |

**Verdict.** The stored verdict under the current scorer is PASS. The ambiguity
between a body and a full function is real in the stored data, but it shows up
as a dropped import, not as a double signature.

**Where the cataloged form was observed.** Not in a stored reply. The catalog
entry describes the body-append path of the first scorer, and the stored
replies do not reproduce the double-signature failure it names.

# 8. Ragged first-line indentation

**Source.** No instance among the 1,336: LiveCodeBench replies are whole
classes, not bare bodies. The record is from the HumanEval phase
(`data/politeness_tax_v2/politeness_tax_responses.json`).

**Problem and condition.** HumanEval/35 (`max_element`), Claude Opus 4.6,
casual register. The prompt shows the signature and docstring and says "just
write the body".

**The reply, in full.** Five lines. The first is indented by three spaces and
the rest by four or more:

```
   m = l[0]
    for e in l:
        if e > m:
            m = e
    return m
```

**What the scorers do.**

| Step | Indentation it produces (spaces per line) | Result on HumanEval's official tests |
|---|---|---|
| First scorer: re-indents every line relative to the first line | 4, 8, 12, 16, 8 | Strategy 1 (prompt + body): `IndentationError`. Strategy 2 (standalone): `IndentationError`. Fail |
| Current scorer: raises the short first line to match the rest, then re-indents | 4, 4, 8, 12, 4 | Passes |

**Verdict.** The stored verdict under the current scorer is PASS. One missing
space on the first line makes the first scorer shift every later line, and a
correct five-line body fails to parse. Across the two HumanEval-phase reply
files, 41 bare-body replies have a first line indented less than every later
line, all from Claude Opus 4.6, and 40 of them fail under the first scorer.

# 9. Fence-marker variants, unbalanced backticks

**Problem and condition.** LiveCodeBench problem 3785 (`countArrays`, medium),
Claude Opus 4.6, multilingual register.

**The reply.** 39 lines. It has three fenced blocks, and none of them is code:
each holds one line of the derivation. The solution follows from line 27,
outside every fence:

~~~
Looking at this problem, I need to find how many valid arrays `copy` exist [...]

```
copy[i] = copy[0] + (original[i] - original[0])
```
[...]
```
bounds[i][0] <= copy[0] + d[i] <= bounds[i][1]
```

Which gives:

```
bounds[i][0] - d[i] <= copy[0] <= bounds[i][1] - d[i]
```
[...]
class Solution:
    def countArrays(self, original: list[int], bounds: list[list[int]]) -> int:
        [...]
        return max(0, hi - lo + 1)
~~~

**What the extractors do.**

| Extractor | What it extracts | Result |
|---|---|---|
| Naive | No fenced block names the class, so it falls back to the whole reply with the fence lines removed (1,386 characters) | `SyntaxError` at line 1 of the extracted text. Fail |
| Sieve | The class (444 characters) | Compiles; returns 2, 4 and 0 on the three statement examples, as the statement says. Pass |
| sanitize (EvalPlus) | The class | Same as Sieve. Pass |
| LCB-official | The text between the last two fence markers: `bounds[i][0] - d[i] <= copy[0] <= bounds[i][1] - d[i]` | Valid Python, but it defines nothing. Fail |

**Verdict.** The stored verdict is PASS. Two of the four extractors treat a
fence as proof of code and fail a correct reply.

**The unbalanced form.** LiveCodeBench problem 3733 (`lenOfVDiagonal`, hard),
Claude Opus 4.6, detailed register, has fence markers on lines 21, 96 and 110:
a closed block, then a second block that is opened and never closed. LCB-official
takes the text between the last two markers, which is the prose between the
blocks, and gets a `SyntaxError`. Four replies in the 1,336 have an odd number
of fence markers. All four were cut off; see pitfall 12.

# 10. Over-aggressive extraction returns nothing

**Problem and condition.** LiveCodeBench problem 3517
(`shortestDistanceAfterQueries`, medium), GPT-5.4, terse register. The official
tests for this problem are on the branch.

**The reply.** 30 lines of code and nothing else. The whole reply compiles as
Python. It has no code fence, as the system prompt asked:

```
from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]
        [...]
        return ans
```

**What the extractors and the harness do.**

| Extractor | What it extracts | `score_lcb.grade` on the 37 official tests |
|---|---|---|
| LCB-official | The empty string. It needs two fence markers and the reply has none | `(False, 'no_code')`. Fail |
| Sieve | The whole reply | `(True, 'pass')`. Pass |

**Verdict.** The stored verdict is PASS. The extractor returns nothing although
the reply is nothing but runnable, correct code. LCB-official returns the empty
string for 1,035 of the 1,336 replies.

# 11. Whole-suite timeout instead of per test

**Status.** Evidence from two stored scoring runs. The timing itself was not
re-executed, because the official tests for this problem are gated.

**Problem and condition.** LiveCodeBench problem 3551 (`maximumSubarrayXor`,
hard), GPT-5.4, terse register.

**The reply.** 29 lines of code. It fills two n-by-n tables, then answers each
query by lookup:

```
class Solution:
    def maximumSubarrayXor(self, nums, queries):
        n = len(nums)

        # score[i][j] = XOR score of subarray nums[i..j]
        score = [[0] * n for _ in range(n)]
        [...]
        return [best[l][r] for l, r in queries]
```

**What the harness did.** The study's earlier harness gave one 8-second budget
to the whole test suite of a problem. The current one gives each test 6 seconds.
The stored verdicts for all eight replies to this problem, in the scoring run
before the change and in the current run:

| Reply | 8 s for the whole suite | 6 s per test | Extracted code identical in both runs |
|---|---|---|---|
| GPT-5.4, terse | FAIL | PASS | yes |
| GPT-5.4, casual | FAIL | PASS | yes |
| GPT-5.4, detailed | FAIL | PASS | yes |
| GPT-5.4, multilingual | FAIL | PASS | yes |
| Claude Opus 4.6, terse | FAIL | PASS | no |
| Claude Opus 4.6, casual | FAIL | PASS | yes |
| Claude Opus 4.6, detailed | FAIL | PASS | yes |
| Claude Opus 4.6, multilingual | FAIL | PASS | yes |

For the GPT-5.4 terse reply, the earlier extractor and Sieve return
byte-identical code, so the change of verdict does not come from extraction. Run
here, that code returns `[12, 60, 60]` and `[7, 14, 11, 14, 5]` on the two
statement examples, as the statement says.

**Verdict.** Stored FAIL under the whole-suite budget, stored PASS under the
per-test timeout, same code. Across the 1,336, 75 replies go from FAIL to PASS
between the two runs and none goes the other way; 46 of the 75 have
byte-identical extracted code in both. In the other 29 the two extractors return
different code, as in pitfall 4.

**What could and could not be re-run.** The earlier whole-suite check is kept
verbatim in `appendix/historical.py`. On the two problems whose official tests
are on the branch, it agrees with the current per-test check on all 16 replies
from the two capable models (6 pass under both, 10 fail under both), so neither
of those problems shows the pitfall.

# 12. Token cap: empty or truncated mid-code

**Problem and condition.** LiveCodeBench problem 3700
(`subsequencesWithMiddleMode`, hard), Claude Opus 4.6, casual register. The
study collected replies with an output cap of 2,048 tokens.

**The reply.** 112 lines, 6,496 characters. Prose, then a code fence that
opens on line 25 and never closes. The reply stops partway through a comment:

~~~
[...]
```python
class Solution:
    def subsequencesWithMiddleMode(self, nums):
        MOD = 10**9 + 7
        [...]
                    # Actually let's just do inclusion properly:
                    # #(v>=2) = #(pick 3 nonm with v>=2)
                    # = total with v in all 3: C(lv,1)*C
~~~

**What the extractor and the harness do.** Sieve extracts all 87 lines from
`class Solution` to the cut. They compile, because the cut falls inside a
comment and everything before it is syntactically complete. The method contains
no `return` statement of its own. On the three statement examples it
returns `None` where the statement says 6, 4 and 0.

**Verdict.** The stored verdict is FAIL, and the harness records it as an
ordinary wrong answer. Nothing in the verdict says the reply was cut off. The
stop reason was not stored, so the cut is inferred from the open fence and the
unfinished line. Seven replies in the 1,336 end this way, with an open code
fence or a last line that is a comment inside a block. All seven are Claude
Opus 4.6 on hard problems, and all seven are stored as FAIL:

| Problem | Register | Length | Last characters of the reply |
|---|---|--:|---|
| lcb/3700 | terse | 6,692 | `= always (since we pick 4` |
| lcb/3700 | casual | 6,496 | `# = total with v in all 3: C(lv,1)*C` |
| lcb/3700 | detailed | 6,704 | `(1 m left, 1 non-m left, 2 non` |
| lcb/3696 | terse | 5,815 | `# For each i_mod = i%period` |
| lcb/3733 | detailed | 6,555 | `# alt20[d][r][c] = max run starting` |
| lcb/3701 | multilingual | 6,442 | `+ (1 if chars[i] <= v else 0` |
| lcb/3784 | terse | 6,902 | `#` |

**The empty form.** No instance in the stored data: none of the 1,336 replies
is empty. Empty replies were observed in a separate GPT-5.6 ablation run at the
same cap, described in `data/politeness_tax_lcb/ABLATION_gpt56.md`. Only that
run's scored output is stored, not its replies, so no empty reply can be shown.

# 13. Weak tests pass subtly wrong code

**Source.** The record is from the HumanEval phase
(`data/politeness_tax_v2/politeness_tax_responses.json`), where the study met
this pitfall by re-scoring the same replies on HumanEval+ inputs.

**Problem and condition.** HumanEval/55 (`fib`), GPT-5.4, detailed register.

**The reply.** The complete function. It rejects `n = 0`:

```
def fib(n: int):
    """Return n-th Fibonacci number.
    [...]
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

**What the tests do.** HumanEval's base test for this problem has five
assertions, for n = 10, 1, 8, 11 and 12. None calls `fib(0)`.

| Check | Result |
|---|---|
| The reply against the base test | Passes |
| `fib(0)`: HumanEval's reference solution | Returns 0 |
| `fib(0)`: the reply | Raises `ValueError: n must be a positive integer` |
| `fib(1)`, `fib(2)`, `fib(10)`: reference and reply | Agree (1, 1, 55) |

**Verdict.** The stored verdict is PASS on the base tests and FAIL on the
HumanEval+ inputs, which include 0. Of the 822 stored cells of this phase, 65
pass the base tests and fail the HumanEval+ inputs, 41 of them from the two
capable models. The HumanEval+ file is a public EvalPlus release and is not
redistributed on this branch, so the script re-runs the base test and the single
input 0, and quotes the stored HumanEval+ verdict.

# 14. Binary pass@1 hides near-misses

**Problem and condition.** LiveCodeBench problem 3527
(`numberOfAlternatingGroups`, hard), GPT-5.4, detailed register. The official
tests for this problem are on the branch: 2 public and 35 private.

**The reply.** 180 lines of code: a Fenwick-tree helper class, then `Solution`:

```
class Fenwick:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 2)
    [...]

class Solution:
    def numberOfAlternatingGroups(self, colors, queries):
        [...]
```

**What the harness does.** Run on every test, without stopping at the first
failure:

| Tests | Passed |
|---|--:|
| Public | 2 of 2 |
| Private | 30 of 35 |
| All | 32 of 37 (86.5%) |

`score_lcb.grade` on all 37 tests returns `(False, 'wrong_answer')`. On the two
public tests alone it returns `(True, 'pass')`.

**Verdict.** The stored verdict is FAIL. Pass@1 gives this reply the same score
as the same model's replies to the same problem in the other three registers,
which pass 2, 3 and 3 of the 37 tests. The study's stored partial-credit run
(`data/politeness_tax_lcb/full_partial_credit.json`) lists this problem as never
solved in 12 attempts, with a best attempt of 86.5%, the same figure. The reply
is also a second example of pitfall 13: it passes both public tests and fails
five private ones.

# 15. Non-ASCII prompt corrupted by code page

**Problem and condition.** LiveCodeBench problem 3540 (`stringHash`, medium),
GPT-5.4, multilingual register. The multilingual template wraps the English
problem statement in a Chinese sentence. This statement also contains the
arrow `→`.

**The stored prompt.** In the stored record, both are garbled in the way UTF-8
bytes look when read as code page 437. Encoding the stored text as CP437 and
decoding it as UTF-8 restores it:

```
stored:     σ╕«µêæΦºúσå│Φ┐ÖΣ╕¬Θù«Θóÿ∩╝îΦ»╖σ«₧τÄ░ `Solution` τ▒╗τÜä `stringHash(s, k)` µû╣µ│ò∩╝Ü
recovered:  帮我解决这个问题，请实现 `Solution` 类的 `stringHash(s, k)` 方法：

stored:     (e.g., 'a' ΓåÆ 0, 'b' ΓåÆ 1, ..., 'z' ΓåÆ 25)
recovered:  (e.g., 'a' → 0, 'b' → 1, ..., 'z' → 25)
```

**The reply.** An 11-line class, with no prose. Sieve extracts it, and it
returns `'bf'` and `'i'` on the two statement examples, as the statement says.
The stored verdict is PASS.

**Where the corruption sits.** Two checks on the stored data place it in the
stored file, after the replies were collected, and not in the prompt as sent.

| Check | Result |
|---|---|
| The checkpoint written while replies were collected (`data/politeness_tax_lcb/lcb_v3_progress.json`) holds this record with the same timestamp | Its prompt is clean Chinese, and equals the recovered text exactly |
| All 1,488 records present in both files (the 124 medium and hard problems, three models) | Same timestamps. 828 fields differ: 453 prompts and 375 replies. All 828 become identical after the CP437-to-UTF-8 recovery. No field differs in any other way. The checkpoint has no garbled field |
| Stored replies in the 1,336 that are garbled although their prompt is pure ASCII | 237. Here the garbled characters are ones the model itself wrote, such as arrows and inequality signs in its explanation |

**Verdict.** A genuine instance of code-page corruption of non-ASCII text, in
302 stored prompts and 371 stored replies of the 1,336. It did not change a
verdict here, and all 1,336 replies still compile under Sieve. The stored data
do not show a model receiving a garbled prompt: prompts and replies are garbled
together in the stored file, and the checkpoint written at collection time
holds the clean text of both. `ENCODING_DEFECT.md` describes
the defect as present in the prompt the model received and predates this check.
