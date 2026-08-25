"""VLM hazard-identification accuracy on the 19-scene SafeLIBERO benchmark,
by how much authority the VLM is given.

Ledger sources (results_tables/fgd_tier_gt_results.md):
- whole-scene WITH images: gpt-5.5 84.2%, sonnet 57.9%, gemini 47.4% (~l.743)
- rule composition critics: haiku 13/16, gemini 11/13 (top-1/top-2 of 19, l.1146)
- prior-file bake-off: haiku-anchored+floor argmax 16/19, top-2 19/19 (l.829)
- hand table 19/19 (l.828)
"""
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LT, ORANGE, GRAY = "#2a78d6", "#9ec5f4", "#eb6834", "#898781"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})

# rows bottom-to-top: (label, top1 %, top2 % or None, color, group)
ROWS = [
    ("Gemini-2.5-Flash",            47.4, None,  GRAY,  "A"),
    ("Claude Sonnet",               57.9, None,  GRAY,  "A"),
    ("GPT-5.5",                     84.2, None,  GRAY,  "A"),
    ("Gemini-2.5-Flash critic",     57.9, 68.4,  ORANGE, "B"),
    ("Claude Haiku critic",         68.4, 84.2,  ORANGE, "B"),
    ("Hand-coded predicate table",  100.0, None, "#104281", "C"),
    ("VLM-grounded (Haiku priors)", 84.2, 100.0, BLUE,  "C"),
]
GROUPS = {
    "A": "VLM decides the scene\n(whole-scene identification, with images)",
    "B": "VLM composes the rule\n(runtime FOL composition)",
    "C": "VLM grounds predicates,\nFOL rule decides  (ours)",
}

fig, ax = plt.subplots(figsize=(6.6, 2.9))
ys, ylabels = [], []
y = 0.0
prev_group = None
group_span = {}
for label, t1, t2, color, grp in ROWS:
    if prev_group is not None and grp != prev_group:
        y += 0.65  # gap between groups
    if t2 is not None:  # top-2 guard set as a lighter extension
        ax.barh(y, t2, height=0.62, color=color, alpha=0.32, zorder=2)
        ax.annotate(f"{t2:.0f}", (t2 + 1.2, y), va="center", fontsize=6.5,
                    color=INK2)
    ax.barh(y, t1, height=0.62, color=color, zorder=3)
    ax.annotate(f"{t1:.0f}", (t1 - 1.5, y), va="center", ha="right",
                fontsize=6.5, color="white",
                fontweight="bold")
    ys.append(y); ylabels.append(label)
    group_span.setdefault(grp, [y, y])[1] = y
    prev_group = grp
    y += 1.0

ax.set_yticks(ys)
ax.set_yticklabels(ylabels, fontsize=7.5)
ax.set_xlim(0, 108)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Hazard identification accuracy (%), 19 SafeLIBERO scenes",
              fontsize=8)
ax.set_axisbelow(True)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)

# group brackets on the right margin
for grp, (y0, y1) in group_span.items():
    ax.annotate(GROUPS[grp], xy=(103.5, (y0 + y1) / 2), fontsize=6.6,
                color=INK2, va="center", ha="left", annotation_clip=False)

# legend for solid vs light
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=INK2, label="top-1 (argmax)"),
    Patch(color=INK2, alpha=0.32, label="top-2 guard set (deployed)"),
], loc="lower right", frameon=False, fontsize=6.6, handlelength=1.1,
   bbox_to_anchor=(0.995, 0.02))

fig.subplots_adjust(right=0.72)
fig.savefig(__file__.replace(".py", ".png"), dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight",
            facecolor="white")
print("saved fig_vlm_id_accuracy.png/.pdf")
