# Paper prose — input-side encoding artifact + the true-error pivot (paste-ready)

Draft prose for three connected pieces of the paper. Figure refs: `\ref{fig:mechanism}` =
the output-side extraction artifact (`figures/fig_mechanism.pdf`), `\ref{fig:encoding}` =
the input-side encoding artifact (`figures/fig_encoding.pdf`). Numbers trace to
`encoding_robustness.py` (run it to regenerate). Semantic-split numbers are provisional
(all three models) until the capable-models run (`score_lcb.py --models chatgpt claude`)
is dropped in — placeholders marked ⟨CAP⟩.

---

## Piece 1 — Results: an input-side artifact that does not bite

While auditing the corpus we found a second, independent measurement artifact — this one on
the *input* side — and we report it because it makes the opposite point to the extraction
result, and because it is exactly the kind of confound that could have silently produced our
numbers. A prompt-construction step corrupted non-ASCII characters: UTF-8 bytes were decoded
through the CP437 code page, so the arrow `→` (bytes `E2 86 92`) reached the model as the
mojibake `ΓåÆ`. The corruption is losslessly recoverable
(`"ΓåÆ".encode("cp437").decode("utf-8") == "→"`), confirming it changed the *encoding* of the
text, not its content. It affected two places: the Chinese framing of the multilingual
condition (124 of 167 problems, since Chinese is entirely non-ASCII), and the problem
*statement* of 9 problems that contained a non-ASCII symbol (arrows, `×`, a curly apostrophe,
or a zero-width space), which — because the statement is shared — were corrupted identically
across all four register conditions (Fig.~\ref{fig:encoding}).

This corruption did not bite. Three checks establish it:

1. **No refusals or safety behavior.** Across all 372 corrupted multilingual prompts, 100%
   produced code and none produced a refusal, safety response, or "this input is unreadable"
   message. The models treated the garbled framing as noise and solved from the intact English
   problem body — the "ignore it as garbage" behavior, not the guardrail behavior that
   impolite or adversarial input can trigger [MYT].
2. **No pass-rate penalty.** Within each (problem × model), the multilingual pass rate equals
   that problem's mean under the English registers (mean difference ≈ 0), so the corruption
   carried no cost once problem difficulty is held fixed.
3. **No effect on the headline.** Excluding the 9 corrupted-statement problems leaves the
   result unchanged: the naive-extraction register gap holds ($p = 6.4\times10^{-7}$ on all
   167 problems vs.\ $p = 6.7\times10^{-8}$ on the clean 158), and the robust extractors remain
   null in both. The encoding bug can neither create nor hide the register effect.

The reason it does not bite is the same property that makes the register effect an artifact
rather than a behavior: the models are *consistent*. They separate noisy or stylistic framing
from the task and succeed or fail on the basis of the problem itself, not its surface form.
This is why we keep the multilingual condition in the design but do not read it as a
multilingual *capability* measurement: it is a single, hand-written, encoding-corrupted Chinese
wrapper that conflates language with register, and its only clean message is a robustness one —
even heavy input corruption does not move the result.

## Piece 2 — Pivot: what the models actually got wrong

With both artifacts controlled — extraction on the output side, encoding on the input side —
the remaining failures are genuine. Of the capable models' failed completions, ⟨CAP: XX⟩%
are *semantic* (the extracted code compiled, ran, and returned a wrong or too-slow answer) and
only ⟨CAP: X⟩% are extraction/structure failures. We decompose the semantic failures, which
prior analyses collapsed into a single "wrong answer" bucket, into three kinds:
**wrong-answer** (incorrect algorithm), **runtime-error** (an exception during execution), and
**timeout** (a correct-looking solution that exceeds the per-test time limit — "right idea,
too slow"). The split is ⟨CAP: wrong-answer / runtime-error / timeout⟩. Failure tracks
difficulty, not framing: ⟨CAP: easy XX% / medium XX% / hard XX%⟩ pass-rate, flat across the
four register conditions. The models fail where the *problem* is hard, and the hardest category
is dynamic programming / combinatorial counting — not where the *prompt* is polite, terse, or
garbled.

## Piece 3 — Discussion: report your extractor, and validate your encoding

Our study surfaces a measurement confound on each side of the model. On the **output** side,
code extraction can fabricate a large, significant effect that is purely an artifact of how
text is cut out of the reply (Fig.~\ref{fig:mechanism}); the remedy is to report the extractor,
measure its by-condition compile rate, and confirm results under an independent extractor. On
the **input** side, constructing prompts that contain non-ASCII text — a different human
language, mathematical symbols, or typographic punctuation — risks silent encoding corruption
that a reader never sees in a pass-rate table. Non-English prompting is especially exposed:
because scripts like Chinese are entirely non-ASCII, a single encoding mistake corrupts the
*whole* prompt, where an English prompt is corrupted only where a stray symbol appears (here,
74% of multilingual framing vs.\ 9 English-statement problems). Our finding that capable models
shrug off this corruption is reassuring, but it is model- and task-dependent — it held because
the task (an English-stated coding problem) survived intact. The general lesson for
prompt-engineering studies is symmetric to the extraction one: **report and validate the exact
bytes the model received**, and treat any non-ASCII input as a place to check the encoding
before attributing a result to the prompt.
