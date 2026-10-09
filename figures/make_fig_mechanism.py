"""Mechanism schematic — "how prompt register becomes a false capability signal".

A citable, 2D vector diagram (the paper's core idea in one picture, in the spirit of a
clean architecture figure): one model reply, two extractors, divergent verdict. Register
modulates how much trailing prose a reply carries; only the naive extractor is sensitive to
it, so it manufactures a register gap that both robust extractors erase.

Vector output (editable text, infinitely scalable) — NOT a raster AI image.
Run:  python figures/make_fig_mechanism.py  ->  figures/fig_mechanism.{pdf,svg,png}
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

# palette shared with fig1 (CVD-safe)
NAIVE, OURS, CODE = "#eb6834", "#2a78d6", "#1baf7a"
PROSE = "#c9c7c1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANEL = "#f6f5f3"
BAD, GOOD = "#d1462f", "#1baf7a"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def box(ax, x, y, w, h, text, fc="white", ec=INK2, tc=INK, fs=8.2, weight="normal",
        lw=1.0, rounding=0.025, ha="center", style="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"{style},pad=0,rounding_size={rounding*100}",
                       mutation_aspect=1, fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.4
        ax.text(tx, y + h / 2, text, ha=ha, va="center", color=tc, fontsize=fs,
                weight=weight, zorder=4, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                 color=color, lw=lw, ls=ls, zorder=2,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.set_xlim(0, 170); ax.set_ylim(-2, 84); ax.axis("off")

def T(*a, **k):
    k.setdefault("clip_on", False)
    return ax.text(*a, **k)

# ---------- A. one problem, four register wrappers ----------
T(15, 80, "One problem,\nfour register wrappers", ha="center", va="top",
  fontsize=8.4, weight="bold", color=INK, linespacing=1.2)
WRAP = [("terse", 59), ("casual", 50), ("multilingual", 41), ("detailed (polite)", 32)]
for name, y in WRAP:
    box(ax, 4, y, 23, 6.4, name, fc="white", ec=INK2, fs=7.6)
ax.annotate("", xy=(1.6, 31), xytext=(1.6, 65.4),
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
T(0.2, 48, "politeness / length", rotation=90, ha="center", va="center", fontsize=6.6, color=INK2)

# ---------- B. model ----------
box(ax, 31, 44.5, 14, 10, "LLM", fc=PANEL, ec=INK2, fs=9, weight="bold")
for _, y in WRAP:
    arrow(ax, 27, y + 3.2, 31, 49.5, color=GRID, lw=0.9, rad=0.05)

# ---------- C. model reply (code + trailing prose) ----------
rx, ry, rw = 51, 39, 25
box(ax, rx, ry, rw, 21, "", fc=PANEL, ec=INK2)
T(rx + rw / 2, ry + 23.3, "model reply", ha="center", fontsize=7.8, weight="bold", color=INK)
box(ax, rx + 2.5, ry + 11.5, rw - 5, 7.5, "solution code", fc=CODE, ec="white", tc="white", fs=7.6, weight="bold")
box(ax, rx + 2.5, ry + 2.5, rw - 5, 7.5, "trailing prose", fc=PROSE, ec="white", tc=INK, fs=7.6)
arrow(ax, 45, 49.5, 51, 49.5)
T(rx + rw / 2, ry - 3.4, "prose volume grows with politeness", ha="center",
  fontsize=6.7, style="italic", color=INK2)

# fork point
fx = rx + rw
arrow(ax, fx, 54, fx + 6, 65, color=NAIVE, lw=1.5, rad=-0.15)
arrow(ax, fx, 45, fx + 6, 27, color=OURS, lw=1.5, rad=0.15)

# ---------- D. NAIVE lane (top) ----------
ny = 61
box(ax, 83, ny, 23, 10, "Naive extractor\nkeeps the whole reply", fc="white", ec=NAIVE, tc=INK, fs=7.3, lw=1.4)
arrow(ax, 106, ny + 5, 110, ny + 5, color=NAIVE, lw=1.4)
box(ax, 110, ny, 20, 10, "✗ SyntaxError\non trailing prose", fc="#fdeee8", ec=NAIVE, tc=BAD, fs=7.1, lw=1.2)
arrow(ax, 130, ny + 5, 135, ny + 5, color=NAIVE, lw=1.4)
box(ax, 135, ny, 19, 10, "scored\nFAIL", fc=NAIVE, ec="white", tc="white", fs=7.8, weight="bold")

# ---------- E. ROBUST lane (bottom) ----------
oy = 17
box(ax, 83, oy, 23, 10, "Robust extractor\ntrims to compilable code", fc="white", ec=OURS, tc=INK, fs=7.3, lw=1.4)
arrow(ax, 106, oy + 5, 110, oy + 5, color=OURS, lw=1.4)
box(ax, 110, oy, 20, 10, "✓ compiles,\nrun official tests", fc="#e9f1fb", ec=OURS, tc="#1a5fb0", fs=7.1, lw=1.2)
arrow(ax, 130, oy + 5, 135, oy + 5, color=OURS, lw=1.4)
box(ax, 135, oy, 19, 10, "true\nPASS / FAIL", fc=OURS, ec="white", tc="white", fs=7.8, weight="bold")

# "same completions, only the extractor differs" between the lanes
T(95, 44.5, "same completions —\nonly the extractor differs", ha="center", va="center",
  fontsize=6.8, style="italic", color=INK2, linespacing=1.2)

# ---------- F. outcomes ----------
T(96, 78.5, "terse appears to beat polite:  +15.9 pts,  $p=1.6\\times10^{-7}$   (artifact)",
  ha="center", fontsize=7.4, color=NAIVE, weight="bold")
T(96, 9.5, "no register effect:  0.0 pts,  $p=1.0$ (ours)   /   $-2.4$ pts,  $p=0.27$ (EvalPlus sanitize)",
  ha="center", fontsize=7.4, color=OURS, weight="bold")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_mechanism.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    print("wrote", OUT / f"fig_mechanism.{ext}")
