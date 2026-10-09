# Data notes — known issues and the robustness checks that bound them

Audit-trail for data-quality issues found during analysis, each with the check that shows it
does or does not affect the headline. Reproduce everything with
`python data/vibe_tax_lcb/encoding_robustness.py` (reads only committed files; no model calls).

## 1. A prompt-encoding bug (UTF-8 bytes decoded as CP437)

At prompt-generation time, non-ASCII characters were corrupted by a UTF-8 → CP437 mishandling:
the UTF-8 bytes of a character were written/read through the wrong code page, so e.g. the arrow
`→` (U+2192, bytes `E2 86 92`) became the mojibake `ΓåÆ` (CP437 of those three bytes). It is
losslessly recoverable: `s.encode("cp437").decode("utf-8")` restores the original text.

This affected **two** places:

| where | extent |
|---|---|
| **multilingual framing** (Chinese is all non-ASCII) | 372 / 501 prompts mojibake; only 129 clean Chinese |
| **English-condition problem statements** with a non-ASCII symbol (→, ≤, ×, …) | **27 / 501 per condition** = **9 distinct problems** (0 easy, 4 medium, 5 hard), corrupted in *every* condition because the body is shared |

So it is a **dataset-wide** encoding bug, not a multilingual-only one. Root cause to fix before
any re-run: force `encoding="utf-8"` end-to-end in prompt generation/IO.

## 2. It does NOT bias the register / extraction headline

The terse-vs-detailed comparison is unchanged when the 9 corrupted-spec problems are excluded:

| problem set | NAIVE compile-tax (terse−det) | ROBUST pass (terse−det) |
|---|---|---|
| **all 167** | b=73, c=24, Δ=+49, p = 6.4×10⁻⁷ | b=14, c=12, Δ=+2, p = 0.85 (null) |
| **clean 158** (excl. corrupted) | b=73, c=21, Δ=+52, p = 6.7×10⁻⁸ | b=12, c=11, Δ=+1, p = 1.00 (null) |

The naive artifact is unchanged (slightly stronger) and the robust null stays null. The encoding
bug does not create or hide the register effect.

## 3. The corruption did not trigger refusals or safety behavior

Checked against the Mind-Your-Tone concern (garbled/odd input → guardrail/refusal). Across all 372
mojibake multilingual prompts: **100% produced code, 0 genuine refusals / safety responses**. The
models treated the garbled framing as ignorable noise and solved from the intact **English**
problem body. Within-problem, multilingual pass ≈ the same problem's English-register mean
(mean Δ ≈ +0.03, i.e. neutral), so the corruption carried no pass-rate penalty once difficulty is
controlled.

## 4. The multilingual condition is not a valid language manipulation

Independent of the encoding bug, the multilingual condition is: (a) a *single* hand-written Chinese
template (`帮我解决这个问题，请实现 \`cls\` 类的 \`entry\` 方法：`), not native-speaker-validated;
(b) a fixed register (brief + mildly polite), so it **conflates language with register** — it is
not "terse/casual/polite Chinese"; and (c) 74% encoding-corrupted. The clean/corrupt split is also
perfectly confounded with difficulty (clean = all 43 easy; corrupt = all 124 medium/hard).
**Decision:** the register axis is carried by the three clean English conditions
(terse / casual / detailed); multilingual is reported only as a disclosed robustness aside, not as
a multilingual/Chinese claim. A legitimate multilingual result would need a fixed-encoding,
native-validated, register-matched redesign and a re-query (separate follow-up).

## 5. Capability numbers (minor caveat)

The 9 corrupted-spec problems (4 medium, 5 hard) had mangled symbols in their statements. On hard
problems they pass 46.7% vs 61.2% for clean-spec hard problems, but n = 5 problems, so whether the
garble hurt is inconclusive. Absolute capability numbers should be reported on the clean 158 (or
flag the 9); the register/extraction headline is unaffected (§2).
