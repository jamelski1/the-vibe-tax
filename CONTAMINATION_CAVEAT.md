# Contamination caveat (paste-ready for Limitations / Threats to Validity)

**Decision:** We do **not** re-run on a contamination-free problem slice. The central
result — the register comparison — is *contamination-immune by construction* (paired
within problem × model), so a re-pull would change nothing about the finding. We add
the short paragraph below so a reviewer cannot read the absolute pass-rates as a
contamination-free capability claim, and because the shrinking post-cutoff window is
itself part of the closing-window argument.

Drop this verbatim into Threats to Validity (trim the bracketed cite tags to match the
`.bib` keys).

---

## Contamination.

LiveCodeBench time-stamps every problem and is designed for contamination-free
evaluation by *time-segmentation*: one evaluates a model only on problems released
after its training cutoff [jain2025livecodebench]. Our models were queried in
August 2026, and their cutoffs postdate the problem window we drew from
(Claude-Opus-4-6, cutoff May 2025; Codestral, July 2025; GPT-5.4, a 2026-03 snapshot),
so our problems are **not** strictly post-cutoff for every model. We therefore describe
the set as *contamination-window*, not contamination-free, and we do **not** claim the
absolute pass-rates are free of memorization.

This does **not** threaten our central result. The register comparison is paired within
each problem and model: the *same* problem is presented to the *same* model under all
four register wrappers, and we score the four stored completions with one fixed
extractor. Any memorization of a problem's solution shifts all four of its conditions
by the same amount and cancels in the within-problem difference. Contamination is thus a
threat to the *descriptive capability numbers* (the absolute pass-rates), not to the
*extraction-artifact finding* (the terse−detailed asymmetry and its collapse under robust
extraction). The compile-rate analysis (Fig. 1a) makes this especially clear: it runs no
tests at all, so it cannot be inflated by a memorized test outcome, yet it reproduces the
same terse−detailed asymmetry under naive extraction and the same flat null under robust
extraction.

Finally, the direction of this limitation reinforces, rather than weakens, the paper's
thesis. As models are trained on ever-more-recent data, the window of genuinely
post-cutoff problems on any fixed benchmark shrinks, so the slice on which absolute
numbers are trustworthy narrows over time — exactly the closing-window dynamic that makes
scoring artifacts harder to detect precisely when benchmarks saturate.
