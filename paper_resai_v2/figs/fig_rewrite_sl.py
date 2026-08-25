"""Generate fig_rewrite_sl.pdf -- the SafeLIBERO twin of fig_rewrite.pdf.

Provenance
----------
Two verified aggregates, both under
/ocean/projects/cis250185p/asingal/knowledge_action_gap/ :

  safelibero_sr_table.json   in-house Qwen2.5-VL-7B rewriter, prompt
                             variants v1/v2/v3, all 8 suite-levels, plus
                             the 8 no-rewrite baselines.  32 rows, every
                             row carrying verified=True, meaning its
                             reconstructed TSR/CAR matched the values
                             stored by the eval run itself.
  safelibero_mp_table.md     the four API rewriters, prompt variant v2
                             only, all 8 suite-levels.

Checkpoint lerobot/pi05_libero_finetuned on the LeRobot stack -- a
different checkpoint, benchmark and evaluation stack from the
LIBERO-Safety arm in fig_rewrite.pdf, which is what makes this an
independent replication of the prompting null rather than a re-analysis
of it.  n=200 per cell (4 tasks x 50 episodes).

55 complete cells: 24 Qwen (8 suite-levels x 3 variants) + 32 API
(8 x 4) - 1.  The single missing cell is Long L1 / GPT-5.5, which never
ran; it is drawn as a grey cross rather than silently skipped.  Note
that safelibero_mp_table.md ALSO marks Long L2 / Qwen as withheld, but
that is stale: safelibero_sr_table.json carries the completed cell
(collision 95.5%, verified), so it is plotted from the JSON.

Both Gemini rewriters are dropped, as in the LIBERO-Safety arm: every
Gemini cell terminated early on exhausted API credits.

Metrics.  Collision rate = 100 - CAR, the benchmark's own collision
criterion, and the quantity Table~\ref{tab:rewrite-sl} prints.  The
lower panel plots TSR (task success), NOT safe success: on SafeLIBERO
the collision floor is so high that safe success is numerically the
complement of the top row: |SR - CAR| never exceeds 5.0 points over the
55 cells, its median is 1.0, and 44 of 55 sit within 2 points.  The
script re-derives and prints those three figures on every run.  Plotting
safe success would therefore show one finding twice.  TSR is the axis
that actually moves, falling by up to 17.5 points (Object L1, 33.0 ->
15.5) while the collision rate beside it does not budge.

Axes.  Unlike fig_rewrite.pdf, whose hazard-contact baselines sit at
0-12%, every collision baseline here sits at 82-98%.  A 0-100 axis
would compress the largest movement in the arm (Object L2, 83.5 -> 73.5)
into an invisible wiggle, so each panel is windowed on its own data.
The top row therefore does NOT start at zero, which the caption states.

Palette: slots 1 (blue #2a78d6) and 2 (orange #eb6834) of the dataviz
reference categorical theme, matching fig_rewrite.py.
"""
import json
import os
import re
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/ocean/projects/cis250185p/asingal/knowledge_action_gap"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_rewrite_sl.pdf")
N = 200

SUITES = [("spatial", "Spatial"), ("object", "Object"),
          ("goal", "Goal"), ("long", "Long")]
LEVELS = [("I", "L1"), ("II", "L2")]

# Display order follows fig_rewrite.py, not tab:rewrite-sl, so the two
# figures read left-to-right the same way.
REWRITERS = [
    ("qwen", "Qwen2.5-VL 7B"),
    ("Claude Haiku 4.5", "Haiku 4.5"),
    ("Claude Opus 5", "Opus 5"),
    ("GPT-5.4-mini", "GPT-5.4-mini"),
    ("GPT-5.5", "GPT-5.5"),
]
QWEN_VARIANTS = ["v1", "v2", "v3"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d8dce1"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ---- source 1: the in-house Qwen sweep and the baselines -------------
qwen = {}
for r in json.load(open(f"{SRC}/safelibero_sr_table.json")):
    assert r["verified"] and r["n"] == N, r
    qwen[(r["suite"], r["level"], r["cond"])] = (
        100 - 100 * r["CAR"], 100 * r["TSR"], 100 * r["SR"])

# ---- source 2: the four API rewriters, v2 only -----------------------
api = {}
suite = level = None
for line in open(f"{SRC}/safelibero_mp_table.md"):
    m = re.match(r"##\s+(\w+)\s+.\s+level\s+(I+)", line.strip())
    if m:
        suite, level = m.group(1), m.group(2)
        continue
    cells = [c.strip() for c in line.strip().split("|")[1:-1]]
    if len(cells) != 4 or suite is None:
        continue
    name, sr, tsr, car = cells
    if "—" in sr or "*" in sr:          # incomplete or credit-exhausted
        continue
    if not car.endswith("%"):
        continue
    api[(suite, level, name)] = (
        100 - float(car.rstrip("%")), float(tsr.rstrip("%")),
        float(sr.split("%")[0]))


def series(suite, level, key):
    """Points for one rewriter in one cell: [(variant, coll, tsr), ...]."""
    if key == "qwen":
        return [(v, *qwen[(suite, level, v)][:2]) for v in QWEN_VARIANTS
                if (suite, level, v) in qwen]
    hit = api.get((suite, level, key))
    return [("v2", hit[0], hit[1])] if hit else []


# ---- collinearity check, asserted rather than asserted-in-prose ------
gaps = []
for s, _ in SUITES:
    for lv, _ in LEVELS:
        for k, _ in REWRITERS:
            if k == "qwen":
                for v in QWEN_VARIANTS:
                    if (s, lv, v) in qwen:
                        c, _t, sr = qwen[(s, lv, v)]
                        gaps.append(abs(sr - (100 - c)))
            elif (s, lv, k) in api:
                c, _t, sr = api[(s, lv, k)]
                gaps.append(abs(sr - (100 - c)))
import statistics
print(f"cells={len(gaps)}  max|SR-CAR|={max(gaps):.1f}  "
      f"median={statistics.median(gaps):.1f}  "
      f"<=2pt in {sum(g <= 2.0 for g in gaps)}/{len(gaps)}")

# ---- draw ------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.0), sharex=True)
x = list(range(len(REWRITERS)))
LSTYLE = {"I": (0, (5, 2)), "II": (0, (1, 1.6))}
MARK = {"I": dict(marker="o", ms=5.2), "II": dict(marker="D", ms=4.4)}

for col, (suite, title) in enumerate(SUITES):
    for row, (label, colour, idx) in enumerate(
        [("Collision rate", BLUE, 0), ("Task success", ORANGE, 1)]
    ):
        ax = axes[row][col]
        lo_all, hi_all = [], []

        for lv, lvlab in LEVELS:
            base = qwen[(suite, lv, "none")][idx]
            k = round(base / 100 * N)
            blo, bhi = wilson(k, N)
            ax.axhspan(blo, bhi, color=colour, alpha=0.11, lw=0, zorder=1)
            ax.axhline(base, color=INK, ls=LSTYLE[lv], lw=1.1, zorder=3)
            lo_all += [blo]
            hi_all += [bhi]

            off = -0.17 if lv == "I" else 0.17
            for i, (key, _) in enumerate(REWRITERS):
                pts = series(suite, lv, key)
                if not pts:                       # visibly absent, not skipped
                    ax.plot(i + off, 0.055, marker="x", ms=5, mew=1.3,
                            color=MUTED, alpha=0.75, zorder=4,
                            transform=ax.get_xaxis_transform(),
                            clip_on=False)
                    continue
                spread = [0.0] if len(pts) == 1 else [-0.075, 0.0, 0.075]
                for (_, cval, tval), dx in zip(pts, spread):
                    y = cval if idx == 0 else tval
                    face = colour if lv == "I" else "white"
                    edge = "white" if lv == "I" else colour
                    ax.plot(i + off + dx, y, ls="none", zorder=5,
                            mfc=face, mec=edge,
                            mew=0.7 if lv == "I" else 1.15, **MARK[lv])
                    lo_all.append(y)
                    hi_all.append(y)

        pad = max(2.0, 0.14 * (max(hi_all) - min(lo_all)))
        ax.set_ylim(min(lo_all) - pad, max(hi_all) + pad)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_xlim(-0.6, len(REWRITERS) - 0.4)
        if row == 0:
            ax.set_title(title, fontsize=10, color=INK, pad=8)
        if col == 0:
            ax.set_ylabel(label + " (%)", fontsize=9, color=INK)

for ax in axes[1]:
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in REWRITERS], fontsize=7,
                       color=MUTED, rotation=18, ha="right")

# legend: level encoding, drawn once
h_l1, = axes[0][0].plot([], [], ls=LSTYLE["I"], color=INK, lw=1.1,
                        marker="o", ms=5.2, mfc=BLUE, mec="white", mew=0.7,
                        label="Level I")
h_l2, = axes[0][0].plot([], [], ls=LSTYLE["II"], color=INK, lw=1.1,
                        marker="D", ms=4.4, mfc="white", mec=BLUE, mew=1.15,
                        label="Level II")
fig.legend(handles=[h_l1, h_l2], loc="upper center", ncol=2, fontsize=8,
           frameon=False, labelcolor=MUTED, handlelength=2.8,
           bbox_to_anchor=(0.5, 1.005), columnspacing=2.4)

axes[0][0].text(0.02, 0.06, "no rewriter separates from baseline",
                transform=axes[0][0].transAxes, fontsize=7.5, color=MUTED,
                va="bottom", style="italic")

fig.text(0.5, 0.005,
         "Dashed/dotted line = unmodified benchmark instruction at level I/II; "
         "band = its 95% Wilson interval. Three points for the in-house rewriter "
         "(v1/v2/v3), one for each API rewriter (v2). "
         "Grey × = cell not run. Axes are windowed, not zero-based.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=[0, 0.045, 1, 0.955])
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
