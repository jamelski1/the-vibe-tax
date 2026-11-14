"""Fig. 1 — the extraction artifact in one picture.

(a) Compile-rate by register condition under three extractors (zero tests run).
(b) Pass-rate by condition under the same three extractors (full LCB test suites).
(c) Paired terse - detailed difference with 95% CI and exact McNemar p.

Data provenance (capable models = ChatGPT + Claude; 167 problems x 2 models = 334 paired
observations per condition; 1,336 records per extractor):
  * Compile-rates: recomputed from data/vibe_tax_lcb/lcb_v3_responses.json with the extractors
    in The_Vibe_Tax_Three_Extractors.ipynb (reproduced exactly on 2026-10-04).
  * Pass-rates + discordant counts: THREE_EXTRACTOR_RESULTS.md, "second run, with McNemar".
    Re-scoring requires the HF-gated lcb_tests.jsonl (not in repo).

Run:  python figures/make_fig1.py   ->  figures/fig1.{pdf,svg,png}
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binomtest

OUT = Path(__file__).resolve().parent
N_PAIRS = 334

CONDS = ["terse", "casual", "multi-\nlingual", "detailed"]  # paper names for the 4 wrappers
# keys in the data: agentic_terse, agentic_casual, webchat_multilingual, webchat_detailed

COMPILE = {  # % of completions whose extracted code parses AND defines the target
    "naive":    [88.0, 81.1, 77.2, 73.4],
    "ours":     [100.0, 100.0, 100.0, 100.0],
    "sanitize": [81.4, 84.1, 83.2, 84.1],
}
PASS = {  # % passing every official LCB test (MAX_TESTS=60 runs all; max 44 tests/problem)
    "naive":    [73.4, 66.5, 66.2, 57.5],
    "ours":     [81.4, 81.4, 84.1, 81.4],
    "sanitize": [66.8, 68.0, 70.1, 69.2],
}
DISCORDANT = {  # (terse-only pass, detailed-only pass) over the 334 paired observations
    "naive": (78, 25),
    "ours": (13, 13),
    "sanitize": (16, 24),
}

LABEL = {"naive": "Naive", "ours": "Ours (robust)", "sanitize": "EvalPlus sanitize"}
SHORT = {"naive": "Naive", "ours": "Ours", "sanitize": "sanitize"}
COLOR = {"ours": "#2a78d6", "naive": "#eb6834", "sanitize": "#1baf7a"}  # validated, CVD-safe
MARKER = {"naive": "o", "ours": "s", "sanitize": "D"}                    # secondary encoding
ORDER = ["naive", "ours", "sanitize"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def paired_diff_ci(b, c, n, z=1.96):
    """Difference in paired proportions (b-c)/n with Wald 95% CI (Agresti 2013, sec. 10.1)."""
    d = (b - c) / n
    se = (((b + c) - (b - c) ** 2 / n) ** 0.5) / n
    return 100 * d, 100 * (d - z * se), 100 * (d + z * se)


def fmt_p(p):
    if p >= 0.01:
        return f"p = {p:.2f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"p = {mant}$\times$10$^{{{int(exp)}}}$"


def dot_panel(ax, data, title, ylim, yticks):
    offs = {"naive": -0.18, "ours": 0.0, "sanitize": 0.18}
    for k in ORDER:
        xs = [i + offs[k] for i in range(len(CONDS))]
        ax.plot(xs, data[k], ls="none", marker=MARKER[k], ms=5.2, mfc=COLOR[k],
                mec="white", mew=0.8, color=COLOR[k], label=LABEL[k], zorder=3)
    # direct-label the naive extremes (the asymmetry) only
    nv = data["naive"]
    for i in (0, 3):
        ax.annotate(f"{nv[i]:.1f}", (i + offs["naive"], nv[i]), xytext=(0, -6.5),
                    textcoords="offset points", ha="center", va="top", fontsize=6.8, color=INK)
    ax.set_xticks(range(len(CONDS)), CONDS)
    ax.set_xlim(-0.55, len(CONDS) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, axis="x", pad=4)
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_ylabel("% of completions")


fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7),
                         gridspec_kw={"width_ratios": [1.25, 1.25, 0.9], "wspace": 0.45})

dot_panel(axes[0], COMPILE, "(a) Compile-rate (no tests)", (60, 102), range(60, 101, 10))
dot_panel(axes[1], PASS, "(b) Pass-rate (full test suites)", (50, 90), range(50, 91, 10))

# (c) paired difference panel
ax = axes[2]
for j, k in enumerate(ORDER):
    b, c = DISCORDANT[k]
    d, lo, hi = paired_diff_ci(b, c, N_PAIRS)
    p = binomtest(b, b + c, 0.5).pvalue
    y = len(ORDER) - 1 - j
    ax.plot([lo, hi], [y, y], color=COLOR[k], lw=2, solid_capstyle="round", zorder=2)
    ax.plot(d, y, marker=MARKER[k], ms=6, mfc=COLOR[k], mec="white", mew=0.8, zorder=3)
    ax.annotate((f"{d:+.1f} pts" if abs(d) >= 0.05 else "0.0 pts") + f"\n{fmt_p(p)}", (max(hi, 2), y), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=6.6,
                color=INK, linespacing=1.15)
ax.axvline(0, color=INK2, lw=0.8, zorder=1)
ax.set_yticks(range(len(ORDER)), [SHORT[k] for k in reversed(ORDER)])
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.set_xlim(-10, 38)
ax.set_xticks([-10, 0, 10, 20])
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(length=0, axis="y")
ax.set_xlabel("terse − detailed, pass-rate (pts)")
ax.set_title("(c) Paired \"politeness tax\"", loc="left", color=INK, fontweight="bold")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=7.5, handletextpad=0.3, columnspacing=1.6)

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig1.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
print("wrote", *(OUT / f"fig1.{e}" for e in ("pdf", "svg", "png")))
for k in ORDER:
    b, c = DISCORDANT[k]
    print(k, [round(x, 1) for x in paired_diff_ci(b, c, N_PAIRS)], binomtest(b, b + c).pvalue)
