# The Vibe Spectrum

An empirically-grounded taxonomy of how people *actually* prompt while coding
with an AI agent — built from real prompts harvested off public GitHub Claude
Code transcripts. This is the foundational research that feeds the **Vibe Tax**
study (which measures whether informal prompting produces worse code).

The original `vibe_spectrum_data.json` modeled "vibe" as a single 1-D axis: 5
formality levels synthesized from HumanEval docstrings. Looking at real prompts
shows that **formality is only one of several axes**, and that the synthetic
levels don't cover some common real patterns at all (multilingual prompts,
pasted errors, conversational follow-ups). This document defines the richer
spectrum so the study can sample from it representatively.

## Data

Two parallel corpora, labeled by the same classifier (see the medium comparison
below):

- **Agentic CLI** — **4,675** human-typed prompts from **163** repos, drawn from
  committed Claude Code session transcripts (`crawl_real_prompts.py --deep`,
  ~31 query angles + full per-repo session-tree enumeration).
- **Web-chat** — **5,000** coding prompts from real **WildChat-1M** (ChatGPT)
  conversations, streamed directly from HuggingFace (`pull_webchat_hf.py`).
  *(An earlier 174-prompt run used a GitHub mirror of WildChat,
  `pull_webchat_corpus.py`, for the HF-firewalled cloud sandbox; the unbiased
  HF pull confirmed and strengthened every cross-medium gap below.)*

For the agentic corpus:
- Only genuine human turns (`userType:"external"`, non-sidechain); tool output,
  system injections, slash-commands, failed-skill artifacts, scheduled-task
  messages, and benchmark-harness templates are filtered out.
- Labeled by `classify_prompts.py` → `vibe_spectrum_corpus.json` +
  `vibe_spectrum_stats.json`.

> ⚠️ **Classification is heuristic** (regex / keyword / Unicode-script). It is
> meant to define the axes and their rough shape, not to be a gold-standard
> labeling. LLM-assisted relabeling is the obvious next refinement.

## The five axes

A prompt is a point in a 5-dimensional space, not a single formality number.

> The per-axis count tables below are from the original **n=198** agentic sample
> (kept for their hand-checked examples). The headline rates are confirmed on the
> full **n=4675** corpus (163 repos) — see the cross-medium table — and stayed
> stable as the crawl scaled 24×.

### Axis 1 — Specification level (the original "formality" axis)

How much the prompt pins down vs. leaves to the model.

| Level | Definition | Count | Example |
|-------|------------|------:|---------|
| L1 formal-spec | Structured requirements / constraints / signature | 7 | *"write a little text explaining that the only way to disable is via flag because: 1. …"* |
| L2 detailed | Clear task with specifics, multi-sentence | 12 | *"working on day 6 part 2. This is going to need a very different parser. each column…"* |
| L3 casual-task | Clear intent, terse, one or two sentences | 87 | *"make a very small update to a comment in the CLI - im just trying to test something"* |
| L4 vague | Underspecified one-liner | 84 | *"commit the staged changes"*, *"Fix the network"* |
| L5 pure-vibe | Minimal / referential | 8 | *"commit changes"*, *"run /init"* |

**Finding:** real prompts pile up at **L3–L4 (86%)**. People rarely write
formal specs *or* pure one-word vibes — the bulk is "casual but intelligible."
The synthetic spectrum's L1/L5 endpoints are real but rare in the wild.

### Axis 2 — Intent

What the user wants done.

| Intent | Count | Example |
|--------|------:|---------|
| explain_understand | 47 | *"explain slices to me"*, *"whats our git status"* |
| modify_run | 41 | *"list all skills we have"*, *"just remove the prepend pilot_"* |
| feature_build | 23 | *"add a Pause/Resume feature"* |
| config_devops | 16 | *"claude config set -g autoUpdates true"* |
| bugfix | 16 | *"part 2 is the issue, part 1 worked fine"* |
| probe_meta | 13 | *"respond \"test ok\""* — sanity-checking the tool |
| git_meta | 11 | *"revert last commit"* |
| refactor | 6 | *"remove the extra one we added"* |
| debug_error_paste | 3 | *(see Axis 3)* |
| other | 21 | *"react vite from day 1"* |

**Finding:** the single biggest intent is **understanding/Q&A**, not building.
A large share of "coding" prompts are conversational — asking what something
does, why a result is wrong, what git status is. The study's prompts (which are
all "implement this function") capture only the `feature_build` slice.

### Axis 3 — Context provision

How the user supplies context — this is where the **"copy-paste"** patterns live.

| Mode | Count | Example |
|------|------:|---------|
| pure_nl | 152 | natural language only |
| references_files | 37 | *"Read the claude file in that dir"* |
| pastes_long_context | 5 | a pasted problem statement / spec (e.g. an Advent-of-Code description) |
| pastes_error | 3 | *"thread 5921302 panic: integer overflow …day01.zig:56:26"* |
| pastes_code | 1 | *"Implement the following plan: # Plan: …"* |

**Key methodological finding — error-paste is rare in agentic transcripts.**
You expected lots of "here's my error, fix it" prompts. They barely appear
(**2%**). The reason is structural: in an *agentic* tool the model runs the code
and reads the error itself via tool execution, so the human says *"fix the
failing test"* and never pastes the stack trace. Manual error-pasting is
predominantly a **web-chat (ChatGPT) behavior**. There *is* a "copy-paste
context" pattern here, but it skews toward pasted **problem descriptions**, not
errors. **Implication:** to study error-paste prompts at volume, mine web-chat
corpora (WildChat, LMSYS-Chat-1M, ShareGPT), not agentic-CLI transcripts.

### Axis 4 — Language

| Language | Count |
|----------|------:|
| English | 176 |
| Chinese | 13 |
| Japanese | 5 |
| Korean | 1 |
| Cyrillic / other | 3 |

**Finding:** **~11% of prompts are non-English** — and the originally-synthetic
spectrum is 100% English. Multilingual vibe coding is a real, visible
phenomenon worth treating as a first-class axis. Notable: non-English prompts
span the same intents (a Japanese *"明日やることをtasksに入れてほしい"* = "add tomorrow's
to-dos to tasks"; a Korean bug report about an API not connecting; Chinese
*"读取 /etc/hostname 文件内容"* = "read the contents of /etc/hostname"). Code-switching
also occurs (English commands embedded in Japanese sentences).

### Axis 5 — Register / tone

| Register | Count |
|----------|------:|
| terse_imperative | 108 |
| conversational | 76 |
| polite | 14 |

**Finding:** **38% are conversational** — multi-sentence, first-person,
thinking-out-loud ("*yeah i don't actually care about the continuation per se --
i'm more imagining that…*"). These are mid-session turns in an ongoing dialogue,
a register the one-shot synthetic spectrum has no analogue for. Explicit
politeness is rare (7%).

## Two media, two distributions — the medium shapes the prompt

The agentic-CLI corpus (Claude Code) is only half the picture. To cover the
patterns it structurally lacks — chiefly manual error/code pasting — we labeled a
parallel **web-chat** corpus of **5,000** coding prompts from real **WildChat-1M**
(ChatGPT) conversations, streamed from HuggingFace (`pull_webchat_hf.py`). Both
corpora were labeled by the **same** classifier, so the axes are directly
comparable.

| Axis | Agentic CLI (n=4675) | Web-chat (n=5000) |
|------|--------------------:|------------------:|
| **Any paste** (error+code+long-context) | **8%** | **54%** |
| └ pasted error | 2% | 12% |
| └ pasted code | 2% | 28% |
| **References files** (agent opens them) | **21%** | ~4% |
| Spec level: casual/vague (L3–L5) | **87%** | 55% |
| Spec level: detailed/formal (L1–L2) | 13% | **45%** |
| Register: terse-imperative | **57%** | 38% |
| Register: conversational | 38% | **58%** |
| Intent: debug / error-paste | 2% | **12%** |
| Intent: feature-build (whole thing) | 13% | **22%** |
| Multilingual | 23% | **42%** |

**The medium shapes the prompt.** In an *agentic CLI*, the model fetches its own
context (reads files, runs code, sees errors), so users are terse, vague, and
rarely paste anything — *"fix the failing test"*. In a *web chatbox*, the model
can't reach the user's machine, so **over half** of prompts paste something,
users write longer and more detailed specs, converse more, and supply the error
themselves:

> *"Error ImportError: Failed to import test module: tests.test_solution
> Traceback (most recent call last): File \"/usr/local/lib/python…\""* — WildChat

The cleanest single signal is **how context is supplied**: agentic users
*reference files* (21% — *"read the config in that dir"*) because the agent can
open them; web-chat users *paste* (over half of prompts) because the model
can't. Same need, opposite mechanism — a direct consequence of the medium.

The intent mix also flips: web-chat's top *codeable* intent is *"build me this
whole thing"* (feature-build), while agentic users mostly *explain* and *modify*
an existing codebase the agent can already see. **Both corpora are now at scale**
(agentic n=4675 from 163 repos, web-chat n=5000) and the gaps are stable:
scaling the agentic crawl from 198 → 2356 → 4675 prompts barely moved its
profile — casual/vague held at ~87–90%, pasting stayed negligible (~6–8%).

This is the single most important finding for the study: **"vibe coding" is not
one distribution.** Where the prompt is typed changes its formality, its length,
its language mix, and whether it carries pasted context. Any vibe-tax measurement
must state which medium it is modeling.

## Replication across web-chat sources — what's a *medium* effect vs a *population* effect

To check whether the web-chat numbers are a real medium effect or an artifact of
one dataset, we ran the same pipeline on a second web-chat source,
**LMSYS-Chat-1M** (Chatbot Arena), at the same n=5000.

| Axis | Agentic (n=4675) | WildChat (n=5000) | LMSYS (n=5000) | replicates? |
|------|-----------------:|------------------:|---------------:|:-----------:|
| Detailed/formal spec (L1–L2) | 13% | 45% | 47% | ✅ |
| Conversational register | 38% | 58% | 54% | ✅ |
| Feature-build intent | 13% | 22% | 28% | ✅ |
| Any paste | 8% | 54% | 36% | ✅ |
| Pasted code | 2% | 28% | 20% | ✅ |
| **Pasted error** | 2% | 12% | **3%** | ❌ |
| **Multilingual** | 23% | 42% | **16%** | ❌ |

**Robust (true medium effects).** On *both* web-chat datasets, users write more
detailed specs, converse more, ask for whole features, and paste code far more
than agentic-CLI users. The structural "medium shapes the prompt" claim holds.

**Not robust (population effects).** Two gaps fail to replicate: error-paste
(WildChat 12% vs LMSYS 3% ≈ agentic) and multilingual (42% vs 16%). These track
*who uses the tool*, not the medium. WildChat is a free public ChatGPT proxy —
broad, international, real users debugging real code (they paste stack traces, in
many languages). LMSYS is Chatbot Arena — English-heavy enthusiasts comparing
models, who paste *code* to test but rarely paste a raw error.

**Takeaway:** "web-chat" is not one population. Treat the formality / register /
code-paste gaps as medium effects, but treat **error-paste prevalence and
language mix as population-dependent** — the error-paste arm of the study must
name its population, and **WildChat (not LMSYS) is the source for it**.

> **This design is now implemented** — see `data/vibe_tax_v2/EXPERIMENT.md` for
> the two-medium experiment built on these distributions.

## What this means for the Vibe Tax study

1. **Sample across all five axes, not just formality.** A faithful "vibe"
   prompt set should mix intents, registers, languages, and context modes — not
   just dial docstring formality up and down.
2. **Center the distribution on L3–L4 casual.** That's where real prompting
   actually lives; L1 formal and L5 one-word are tail cases.
3. **Add a multilingual arm.** ~1 in 9 real prompts is non-English; the tax may
   differ by language.
4. **Add a conversational/follow-up arm.** Much real coding is dialogue, not
   one-shot.
5. **Model the two media separately.** Run the tax once per medium (agentic vs
   web-chat) rather than blending them — they have different formality, paste,
   and language distributions. The web-chat corpus is the right source for the
   error-paste / code-paste arm; the agentic corpus for terse one-liners.

## Limitations

- **Heuristic labels** — expect some misclassification; treat counts as
  directional, not exact. On the web-chat corpus the `other` intent is ~22%
  (the keyword rules fray on web-chat's topic diversity), so the intent axis
  there is the least reliable — LLM-assisted relabeling is the next refinement.
- **Agentic sampling bias** — repos that *commit* their transcripts skew toward
  tutorials, Advent-of-Code, demos, dotfiles, and tooling experiments. Expanding
  to ~31 query angles lifted the corpus to **4,675 prompts from 163 repos**, and
  flattened concentration sharply: the top-10 repos are now **36%** of prompts
  (down from 88% at 60 repos), the largest single repo 5.5%. Still ~163 committers
  rather than a random developer sample, but far more representative than before.
- **n** — both corpora are now comparably at scale (agentic 4,675, web-chat
  5,000), with sample-stable rates. Committed Claude Code transcripts remain
  scarcer than web-chat logs; more query angles could push the agentic n higher
  but with diminishing returns.
- **Web-chat populations differ** — checked against a second source,
  LMSYS-Chat-1M (see the replication section): the formality/register/code-paste
  gaps replicate, but error-paste and multilingual rates are population-specific.
  Don't generalize those two from a single dataset.

## Reproduce

```bash
# Agentic-CLI corpus (Claude Code transcripts) — --deep fully enumerates each
# repo's session tree, bypassing code search's 1000-result cap
python crawl_real_prompts.py --limit 6000 --deep --per-repo 40
python classify_prompts.py
#   -> vibe_spectrum_corpus.json + vibe_spectrum_stats.json

# Web-chat corpus — unbiased, direct from HuggingFace (needs HF_TOKEN + gated
# dataset access; run locally, HF is firewalled in the cloud sandbox)
python pull_webchat_hf.py --dataset wildchat --limit 1000
python classify_prompts.py --in webchat_wildchat_corpus.json \
    --out vibe_spectrum_wildchat_corpus.json --stats vibe_spectrum_wildchat_stats.json

# (fallback used in the firewalled sandbox: GitHub mirror of WildChat)
# python pull_webchat_corpus.py --limit 300
```
