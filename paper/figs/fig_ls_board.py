"""LIBERO-Safety five-arm board figure (workshop paper).

TSR per suite (top) + violation rate (bottom), five arms.
Numbers: frozen FINAL BOARD in results_tables/fgd_tier_gt_results.md with the
2026-08-09 batch-aware corrections (oa base/GT n=450; hs base/noGT n=450;
corrected macros). Regenerate after the uniform-300 rerun lands.
"""
import matplotlib.pyplot as plt
import numpy as np

SUITES = ["Obstacle\navoid (oa)", "Hands\n(hs)", "Obst.+hands\n(oah)", "Affordance\n(aff)", "Macro"]

ARMS = [
    # name, TSR per suite [oa, hs, oah, aff, macro], viol per suite, color, hatch
    ("$\\pi_{0.5}$ base",      [73.6, 85.8, 69.3, 56.7, 71.4], [4.7, 0.0, 5.3, 0.0, 2.5], "#898781", None),
    ("Post-hoc shield",        [75.3, 37.3, 46.0, 51.3, 52.5], [2.7, 0.0, 2.0, 0.0, 1.2], "#eb6834", None),
    ("In-denoise reimpl.",     [70.7, 32.7, 48.0, 48.7, 50.0], [0.7, 0.0, 1.3, 0.0, 0.5], "#eda100", "///"),
    ("Ours (GT)",              [77.8, 78.7, 63.2, 58.7, 69.6], [4.2, 0.0, 1.9, 0.0, 1.5], "#86b6ef", None),
    ("Ours (no-GT)",           [68.7, 86.9, 58.9, 56.0, 67.6], [0.7, 0.0, 1.7, 0.0, 0.6], "#2a78d6", None),
]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

fig, (ax_tsr, ax_v) = plt.subplots(
    2, 1, figsize=(7.0, 3.4), sharex=True,
    gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12},
)

n_arms = len(ARMS)
n_suites = len(SUITES)
group_w = 0.82
bar_w = group_w / n_arms
x = np.arange(n_suites)

for ax, key in ((ax_tsr, 1), (ax_v, 2)):
    for i, arm in enumerate(ARMS):
        name, color, hatch = arm[0], arm[3], arm[4]
        vals = arm[key]
        pos = x - group_w / 2 + (i + 0.5) * bar_w
        bars = ax.bar(
            pos, vals, width=bar_w * 0.9, color=color, hatch=hatch,
            edgecolor="white" if hatch else "none", linewidth=0.4,
            label=name if key == 1 else None, zorder=3,
        )
        if key == 1:  # direct value labels on TSR bars
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v + 1.2),
                            ha="center", va="bottom", fontsize=6.2, color=INK2)
        else:  # label only nonzero violation bars
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v + 0.25),
                                ha="center", va="bottom", fontsize=6.0, color=INK2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)

ax_tsr.set_ylim(0, 100)
ax_tsr.set_yticks([0, 25, 50, 75, 100])
ax_tsr.set_ylabel("Task success rate (%)", fontsize=8)
ax_v.set_ylim(0, 6.9)
ax_v.set_yticks([0, 3, 6])
ax_v.set_ylabel("Violation\nrate (%)", fontsize=8)
ax_v.set_xticks(x)
ax_v.set_xticklabels(SUITES, fontsize=7.5)

ax_tsr.legend(
    ncol=5, loc="lower left", bbox_to_anchor=(-0.01, 1.01), frameon=False,
    fontsize=7, handlelength=1.1, handletextpad=0.5, columnspacing=1.0,
)

fig.align_ylabels([ax_tsr, ax_v])
fig.savefig(__file__.replace("fig_ls_board.py", "fig_ls_board.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(__file__.replace("fig_ls_board.py", "fig_ls_board.pdf"),
            bbox_inches="tight", facecolor="white")
print("saved fig_ls_board.png/.pdf")
