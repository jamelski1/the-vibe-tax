"""Encoding-corruption schematic — the INPUT-side artifact that does NOT bite.

Companion to fig_mechanism (the output-side extraction artifact that does). A UTF-8->CP437
mishandling garbled non-ASCII input (the Chinese framing, and symbols like -> in some English
problem statements). The model ignores the noise and solves from the intact English body, so the
headline is unchanged -- turning a data bug into evidence that the finding is robust and that the
models separate noisy framing from the task.

Chinese glyphs render via WenQuanYi Zen Hei; mojibake + arrows via DejaVu Sans. Vector output.
Run:  python figures/make_fig_encoding.py  ->  figures/fig_encoding.{pdf,svg,png}
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
NAIVE, OURS, CODE = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, GRID, PANEL = "#0b0b0b", "#52514e", "#e4e3df", "#f6f5f3"
BAD, GOOD = "#d1462f", "#178a5f"
CJK = "WenQuanYi Zen Hei" if any("WenQuanYi Zen Hei" == f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def box(ax, x, y, w, h, fc="white", ec=INK2, lw=1.0, r=2.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=2))


def T(ax, *a, **k):
    k.setdefault("clip_on", False)
    return ax.text(*a, **k)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                 color=color, lw=lw, zorder=3))


fig, ax = plt.subplots(figsize=(7.4, 3.1))
ax.set_xlim(0, 164); ax.set_ylim(0, 72); ax.axis("off")

# ---------- intended prompt (clean) ----------
lx, lw_, ty, th = 3, 57, 42, 26
box(ax, lx, ty, lw_, th, fc=PANEL, ec=INK2)
T(ax, lx + lw_ / 2, ty + th + 2.2, "intended prompt", ha="center", fontsize=8, weight="bold", color=INK)
T(ax, lx + 3, ty + th - 6, "framing (Chinese):", fontsize=6.6, color=INK2)
T(ax, lx + 4, ty + th - 13, "帮我解决这个问题，请实现 Solution 类的方法：",
  fontsize=8.0, color=INK, fontfamily=CJK)
T(ax, lx + 3, ty + 9, "problem body (English):", fontsize=6.6, color=INK2)
T(ax, lx + 4, ty + 3.5, "roads go  i → i+1 → i+2 …", fontsize=8, color=INK, fontfamily="DejaVu Sans Mono")

# ---------- encoding bug arrow (gap = 22) ----------
gx = lx + lw_                 # 60
arrow(ax, gx + 2, ty + th / 2, gx + 20, ty + th / 2, color=BAD, lw=1.8)
T(ax, gx + 11, ty + th / 2 + 7.5, "encoding bug", ha="center", fontsize=6.9, weight="bold", color=BAD)
T(ax, gx + 11, ty + th / 2 - 5.5, "UTF-8 bytes\nread as CP437", ha="center", fontsize=6.2,
  color=BAD, linespacing=1.1)

# ---------- prompt as sent (corrupted) ----------
rx = gx + 22                  # 82
box(ax, rx, ty, lw_, th, fc="#fdeee8", ec=NAIVE, lw=1.3)
T(ax, rx + lw_ / 2, ty + th + 2.2, "prompt as sent to the model", ha="center", fontsize=8, weight="bold", color=NAIVE)
T(ax, rx + 3, ty + th - 6, "framing (garbled):", fontsize=6.6, color=INK2)
T(ax, rx + 4, ty + th - 13, "σ╕«µêæΦºúσå│Φ┐ÖΣ╕¬Θù«Θóÿ∩╝î… Solution …",
  fontsize=7.4, color=BAD, fontfamily="DejaVu Sans Mono")
T(ax, rx + 3, ty + 9, "problem body (symbols garbled):", fontsize=6.6, color=INK2)
T(ax, rx + 4, ty + 3.5, "roads go  i ΓåÆ i+1 ΓåÆ i+2 …", fontsize=8, color=BAD, fontfamily="DejaVu Sans Mono")

# extent + recovery note
T(ax, rx + lw_ / 2, ty - 3.4,
  "74% of multilingual framing  ·  9 problems' symbols (→, ≤) in every condition",
  ha="center", fontsize=6.7, color=INK2)
T(ax, rx + lw_ / 2, ty - 7.6,
  "losslessly recoverable:  'ΓåÆ'.encode('cp437').decode('utf-8') = '→'   (same text, wrong code page)",
  ha="center", fontsize=6.4, style="italic", color=INK2)

# ---------- verdict banner ----------
by, bh = 6, 15
box(ax, 3, by, 158, bh, fc="#eaf6f0", ec=GOOD, lw=1.2)
T(ax, 7, by + bh - 4,
  "The model ignores the noise and solves from the intact English body "
  "— an input-side artifact that does NOT bite.",
  fontsize=7.6, weight="bold", color=GOOD)
T(ax, 7, by + 4,
  "100% still produce code  ·  0 refusals / safety responses  ·  headline unchanged: "
  "naive artifact $p=6.4{\\times}10^{-7}$ (all 167) → $6.7{\\times}10^{-8}$ (excl. 9 corrupted),  robust null both",
  fontsize=7.0, color=INK)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_encoding.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    print("wrote", OUT / f"fig_encoding.{ext}")
