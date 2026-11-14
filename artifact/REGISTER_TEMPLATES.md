# The four register templates (terse, casual, detailed, multilingual)

The study wraps each LiveCodeBench problem statement in one of four register templates, varying
only the surrounding words — the problem statement itself is inserted **verbatim and in English**
in all four. Source: `generate_lcb_prompts.py`.

## Two modes (and which the study used)

`generate_lcb_prompts.py` has two paths:

1. **LLM-rewrite mode** — the `CONDITIONS` dict (lines ~40–49) gives a *style description* that an
   LLM uses to rewrite the problem into that register.
2. **Deterministic mock mode** — `_mock()` (lines ~85–97) emits a *fixed template*.

**The shipped run used the deterministic `_mock` templates.** This was verified against the stored
`prompt_text` in `data/lcb_v3_responses.json`, which matches the mock strings exactly. So the
register manipulation is four fixed, hand-written wrappers, not naturally varied phrasing — a
limitation stated in the paper.

## The four templates (verbatim from `_mock`)

With `{cls}` = `Solution`, `{entry}` = the method name, `{params}` = its parameters, and
`{problem}` = the full English problem statement:

| condition key | register | template |
|---|---|---|
| `agentic_terse` | terse | `need `​`{cls}.{entry}({params})`​` for this:\n\n{problem}` |
| `agentic_casual` | casual | `hey can you solve this? need a `​`{entry}`​` method on a `​`{cls}`​` class\n\n{problem}` |
| `webchat_detailed` | detailed / polite | `Hi! Could you help me solve this problem? I need a method `​`{entry}({params})`​` on a class `​`{cls}`​`.\n\n{problem}\n\nThank you!` |
| `webchat_multilingual` | Chinese framing | `帮我解决这个问题，请实现 `​`{cls}`​` 类的 `​`{entry}({params})`​` 方法：\n\n{problem}` |

The multilingual framing translates to: *"Help me solve this problem, please implement the
`{entry}` method of the `{cls}` class:"* — a single fixed Chinese wrapper (not native-speaker
validated), followed by the English problem statement. It conflates *language* with *register*
(it is one phrasing, not terse/casual/polite Chinese).

## The `CONDITIONS` style descriptions (LLM-rewrite mode, lines ~40–49)

- `agentic_terse` → *"Extremely terse and lowercase, like a quick note-to-self — minimal framing,
  just the problem and what you need."*
- `agentic_casual` → *"Casual and lowercase, like messaging a teammate for help."*
- `webchat_detailed` → *"A polite, clear chat message asking for help."*
- `webchat_multilingual` → *"Write ALL of your own framing text in Chinese (keep the problem
  statement and the method name in English)."*

## Important: encoding defect in the shipped prompts

The multilingual framing and 9 problem statements were corrupted by a UTF-8 → CP437 encoding
defect at prompt-generation time (`→` became `ΓåÆ`, Chinese became mojibake). This is **disclosed
in the paper** and documented in `ENCODING_DEFECT.md`, with the audit script
`encoding_robustness.py` and the robustness check that the study's headline is unchanged.
