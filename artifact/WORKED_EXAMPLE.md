# Worked example: one reply, four scorings

This is the worked example the paper's Mechanism section points to. It is one of
the 1,336 stored replies in `data/lcb_v3_responses_capable.json`: GPT-5.4,
detailed register, LiveCodeBench problem 3747.

The reply is correct code followed by explanatory prose:

```python
class Solution:
    def maxAdjacentDistance(self, nums):
        n = len(nums)
        ans = 0
        for i in range(n):
            ans = max(ans, abs(nums[i] - nums[(i + 1) % n]))
        return ans

This checks every adjacent pair, including the circular pair formed by the
last and first elements using modulo.

Time complexity: O(n)
Space complexity: O(1)
```

Scored four ways (see `extractors.py` and `score_all_extractors.py`):

| Extractor | What it extracts | Verdict |
|---|---|---|
| Naive | The code together with the prose that follows it | Fail (SyntaxError) |
| Sieve | The code only | Pass |
| sanitize (EvalPlus) | The code only | Pass |
| LCB-official | Nothing | Fail (no code) |

The naive extractor feeds the prose to the interpreter as part of the program,
so a correct reply is scored as a failure. The two robust extractors strip the
prose and the same reply passes. The benchmark's own extractor finds no code
in the form it expects and scores the reply as empty.
