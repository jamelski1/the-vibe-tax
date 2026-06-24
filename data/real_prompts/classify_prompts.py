"""
Classify the harvested real-prompt corpus along the Vibe Spectrum axes, to turn
a flat list of prompts into an empirically-grounded taxonomy.

Input : real_prompts_corpus.json   (from crawl_real_prompts.py)
Output: vibe_spectrum_corpus.json  (each prompt + axis labels)
        vibe_spectrum_stats.json   (distributions across every axis)

Classification is HEURISTIC (regex / keyword / Unicode-script based). It is a
first-pass characterization meant to define the spectrum's dimensions and their
rough distribution — not a gold-standard labeling. LLM-assisted relabeling is a
natural future refinement once API keys are available.

Axes
----
1. spec_level         1 formal-spec .. 5 pure-vibe   (how much is specified)
2. intent             what the user wants done
3. context_provision  how the user supplies context (NL / files / code / error)
4. language           English or detected non-English language
5. register           terse_imperative / polite / conversational

Usage:
    python classify_prompts.py
"""

import json
import os
import re
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(SCRIPT_DIR, "real_prompts_corpus.json")
OUT_FILE = os.path.join(SCRIPT_DIR, "vibe_spectrum_corpus.json")
STATS_FILE = os.path.join(SCRIPT_DIR, "vibe_spectrum_stats.json")


# ---------------------------------------------------------------------------
# Axis 1 — specification level (formal spec .. pure vibe)
# ---------------------------------------------------------------------------

SPEC_LABELS = {
    1: "L1 formal-spec",
    2: "L2 detailed",
    3: "L3 casual-task",
    4: "L4 vague",
    5: "L5 pure-vibe",
}


def spec_level(p):
    """Map a prompt to a 1 (formal) .. 5 (pure vibe) specification level."""
    t = p["prompt"].strip()
    words = p["n_words"]
    structured = bool(re.search(r"(\n\s*[-*\d]\.|\n\s*[-*]\s|```)", t))
    # Structured / long / spec-like prompts are formal.
    if structured and words >= 25:
        return 1
    if structured or words >= 60:
        return 2
    if words >= 18:
        return 3
    if words >= 7:
        return 3 if t[-1:] in ".!?" else 4
    if words >= 3:
        return 4
    return 5  # <=2 words: "commit changes", "fix the network"


# ---------------------------------------------------------------------------
# Axis 2 — intent (single best label, evaluated in priority order)
# ---------------------------------------------------------------------------

# Each intent is matched by an English regex and an optional CJK-keyword regex
# (so the 20+ Chinese prompts don't all collapse into "other"). Order matters:
# the first rule that fires wins.
INTENT_RULES = [
    ("debug_error_paste",
     r"^$",                       # primarily driven by the has_error_paste flag
     None),
    ("explain_understand",
     r"^(what|why|how|explain|tell me|summari[sz]e|describe|walk me|"
     r"can you (explain|tell|read|restate|summari)|so\b.*\?)|"
     r"\b(explain|understand|what does .* do|restate)\b",
     r"(解释|说明|什么意思|是什么|为什么|怎么|如何)"),
    ("git_meta",
     r"^(commit|push|merge|rebase|stage|amend|revert|revent|run /|/)\b|"
     r"\b(commit|git ?ignore|pull request|\.envrc|versioned|staged changes|save this conversation)\b",
     r"(提交|推送|合并|版本)"),
    ("test",
     r"\b(unit test|e2e|test case|add (some )?tests|coverage|pytest|jest|"
     r"failing test|write tests?)\b",
     r"(测试|单元测试)"),
    ("config_devops",
     r"\b(config|\.env|docker|kubernetes|k8s|ci/?cd|deploy|install|setup|set up|"
     r"direnv|devenv|nvim|vim|dotfile|environment|emulator|aspire|proxy|ssl|certificate)\b",
     r"(配置|安装|部署|环境)"),
    ("bugfix",
     r"\b(fix|bug|broken|does ?n'?t work|not working|isn'?t working|issue|wrong|"
     r"fails?|failing|error|too low|too high|incorrect)\b",
     r"(修复|修正|报错|错误|不工作|问题|解决)"),
    ("refactor",
     r"\b(refactor|clean ?up|simplif|rename|extract|reorganiz|tidy|dedupe|"
     r"de-?duplicate|remove|delete|get rid of)\b",
     r"(重构|清理|删除|去掉|简化)"),
    ("feature_build",
     r"\b(add|implement|create|build|make|write|support|new|generate|hook up|"
     r"set up|scaffold)\b",
     r"(添加|增加|实现|创建|生成|写一?个|新建|支持)"),
    ("modify_run",
     r"\b(update|change|replace|convert|edit|adjust|tweak|modify|use|run|list|"
     r"show|read|search|solve|apply|move|copy|check|look at|enable|disable|set)\b",
     r"(修改|改代码|改\b|更新|更改|替换|运行|执行|使用|检查|查看|移动|复制|列出|读取|入れて|してほしい|やること|作成|削除|確認)"),
]
_INTENT_COMPILED = [
    (label, re.compile(en, re.IGNORECASE), re.compile(cjk) if cjk else None)
    for label, en, cjk in INTENT_RULES
]


# Probe / sanity-check turns: people testing that the tool/skill responds.
PROBE = re.compile(
    r"^(just )?(output|respond|reply|say|answer|print)\b|只输出|只回答|"
    r"^just say hello|test-?ok|^ping\b",
    re.IGNORECASE)


def intent(p):
    t = p["prompt"].strip()
    if p["has_error_paste"]:
        return "debug_error_paste"
    if PROBE.search(t):
        return "probe_meta"
    # Plain questions without an action verb read as explain/understand.
    if t.endswith("?") and not re.search(r"\b(add|fix|implement|create|make|change|write|update|remove)\b", t, re.IGNORECASE):
        return "explain_understand"
    for label, en_rx, cjk_rx in _INTENT_COMPILED:
        if label == "debug_error_paste":
            continue
        if en_rx.search(t) or (cjk_rx and cjk_rx.search(t)):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Axis 3 — context provision
# ---------------------------------------------------------------------------

FILE_REF = re.compile(
    r"(\b[\w./-]+\.(py|js|ts|tsx|jsx|go|rs|java|rb|php|c|cpp|h|hpp|sh|md|ya?ml|"
    r"json|toml|txt|ex|exs|vue|css|html|sql|cfg|ini|lock)\b|/\w[\w/-]+)",
    re.IGNORECASE,
)


def context_provision(p):
    if p["has_error_paste"]:
        return "pastes_error"
    if p["has_code_paste"]:
        return "pastes_code"
    # Large NL blobs are usually a pasted spec / problem description (e.g. an
    # Advent-of-Code statement) — the "copy-paste context" pattern, spec flavor.
    if p["n_words"] >= 120:
        return "pastes_long_context"
    if FILE_REF.search(p["prompt"]):
        return "references_files"
    return "pure_nl"


# ---------------------------------------------------------------------------
# Axis 4 — language (Unicode script + small stopword heuristic)
# ---------------------------------------------------------------------------

SCRIPT_RANGES = [
    # Japanese (kana) and Korean (hangul) MUST precede Chinese: Japanese text
    # contains kanji that live in the CJK Han range, so a Chinese-first check
    # would mislabel it. Kana/hangul are unambiguous, Han-only implies Chinese.
    ("Japanese", re.compile(r"[぀-ヿ]")),
    ("Korean", re.compile(r"[가-힯]")),
    ("Chinese", re.compile(r"[一-鿿]")),
    ("Cyrillic", re.compile(r"[Ѐ-ӿ]")),
    ("Arabic", re.compile(r"[؀-ۿ]")),
    ("Devanagari", re.compile(r"[ऀ-ॿ]")),
    ("Thai", re.compile(r"[฀-๿]")),
    ("Hebrew", re.compile(r"[֐-׿]")),
]

# Latin-script non-English: detect by common function words / diacritic words.
LATIN_LANG = [
    ("Vietnamese", re.compile(r"\b(này|của|là|được|giải thích|hãy|không|với|cho|tôi|file này)\b", re.IGNORECASE)),
    ("Spanish", re.compile(r"\b(el|la|los|las|que|por favor|con|una|para|cómo|qué|añade|archivo)\b", re.IGNORECASE)),
    ("Portuguese", re.compile(r"\b(você|não|por favor|arquivo|função|adicione|faça|está)\b", re.IGNORECASE)),
    ("French", re.compile(r"\b(le|la|les|avec|dans|s'il vous plaît|fichier|ajoute|fais|peux-tu)\b", re.IGNORECASE)),
    ("German", re.compile(r"\b(der|die|das|und|nicht|bitte|datei|füge|mach|kannst du)\b", re.IGNORECASE)),
]


def language(p):
    t = p["prompt"]
    for name, rx in SCRIPT_RANGES:
        if rx.search(t):
            return name
    if p["non_english"]:
        for name, rx in LATIN_LANG:
            if rx.search(t):
                return name
        return "Other-non-English"
    return "English"


# ---------------------------------------------------------------------------
# Axis 5 — register
# ---------------------------------------------------------------------------

POLITE = re.compile(r"\b(please|could you|would you|thanks|thank you|can you|if you (could|don't mind))\b", re.IGNORECASE)
CONVERSATIONAL = re.compile(
    r"\b(i think|i'?m|i'?d|i want|i don'?t|yeah|actually|so\b|maybe|let'?s|"
    r"we (probably |actually )?(want|should)|hmm|i guess|imagining)\b", re.IGNORECASE)


def register(p):
    t = p["prompt"].strip()
    n_sentences = len(re.findall(r"[.!?]+", t))
    if CONVERSATIONAL.search(t) or n_sentences >= 3:
        return "conversational"
    if POLITE.search(t):
        return "polite"
    return "terse_imperative"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def classify(p):
    lvl = spec_level(p)
    return {
        **p,
        "spec_level": lvl,
        "spec_label": SPEC_LABELS[lvl],
        "intent": intent(p),
        "context_provision": context_provision(p),
        "language": language(p),
        "register": register(p),
    }


def dist(items, key):
    return dict(Counter(x[key] for x in items).most_common())


# Turns that are CLI/tool/automation artifacts logged as user messages rather
# than human-typed prompts: failed skills, scheduled-task injections (the
# Chinese phrase means "auto-sent by the system, not from the user"), and the
# SWE-bench-style harness template.
NOISE = re.compile(
    r"^(Unknown skill:|API Error:|\[Request|\[定时任务|"
    r"Fix this bug to solve the issue based on manual\.yaml)",
    re.IGNORECASE)


def is_noise(text):
    t = text.strip()
    return bool(NOISE.match(t)) or "并非来自用户" in t


def run(in_file=IN_FILE, out_file=OUT_FILE, stats_file=STATS_FILE):
    with open(in_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus = [p for p in corpus if not is_noise(p["prompt"])]
    labeled = [classify(p) for p in corpus]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)

    stats = {
        "total_prompts": len(labeled),
        "by_spec_level": {SPEC_LABELS[k]: sum(1 for x in labeled if x["spec_level"] == k)
                          for k in sorted(SPEC_LABELS)},
        "by_intent": dist(labeled, "intent"),
        "by_context_provision": dist(labeled, "context_provision"),
        "by_language": dist(labeled, "language"),
        "by_register": dist(labeled, "register"),
        "multilingual_share": f"{sum(1 for x in labeled if x['language'] != 'English') / len(labeled) * 100:.0f}%" if labeled else "N/A",
        "error_paste_share": f"{sum(1 for x in labeled if x['has_error_paste']) / len(labeled) * 100:.0f}%" if labeled else "N/A",
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Classified {len(labeled)} prompts -> {out_file}")
    for axis in ["by_spec_level", "by_intent", "by_context_provision", "by_language", "by_register"]:
        print(f"\n{axis}:")
        for k, v in stats[axis].items():
            print(f"  {v:3d}  {k}")
    print(f"\nmultilingual: {stats['multilingual_share']} | error-paste: {stats['error_paste_share']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", default=IN_FILE)
    ap.add_argument("--out", dest="out_file", default=OUT_FILE)
    ap.add_argument("--stats", dest="stats_file", default=STATS_FILE)
    a = ap.parse_args()
    run(a.in_file, a.out_file, a.stats_file)
