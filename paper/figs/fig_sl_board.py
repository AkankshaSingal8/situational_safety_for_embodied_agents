"""SafeLIBERO five-arm board figure — sibling of fig_ls_board.py.

TSR per cell (top) + collision avoidance rate (bottom), five arms.
Numbers: Table tab:board in main.tex (frozen board, n=200/cell, >1mm
GT-hazard displacement criterion). Long L2 no-GT carries the
benchmark-definition-divergence dagger (table-prior arm 44.0/79.5).
"""
import matplotlib.pyplot as plt
import numpy as np

CELLS = ["Spatial\nL1", "Spatial\nL2", "Goal\nL1", "Goal\nL2",
         "Object\nL1", "Object\nL2", "Long\nL1", "Long\nL2"]

ARMS = [
    # name, TSR per cell, CAR per cell, color, hatch
    ("$\\pi_{0.5}$ base",  [67.0, 55.5, 51.0, 66.5, 40.5, 74.0, 58.0, 51.0],
                           [14.0, 12.0, 23.0, 35.0, 14.0, 25.0, 15.0, 16.5], "#898781", None),
    ("Post-hoc shield",    [71.5, 66.5, 72.0, 69.5, 63.5, 79.0, 27.5, 46.5],
                           [28.5, 21.0, 19.0, 48.5, 16.0, 34.5, 32.5, 52.0], "#eb6834", None),
    ("In-denoise reimpl.", [62.0, 71.5, 71.0, 75.0, 53.0, 76.0, 30.5, 43.0],
                           [28.5, 25.5, 27.5, 49.5, 15.5, 33.5, 35.5, 59.5], "#eda100", "///"),
    ("Ours (GT)",          [71.0, 85.0, 77.5, 76.0, 53.5, 83.0, 42.5, 47.0],
                           [23.5, 76.5, 51.0, 60.0, 28.0, 74.5, 71.0, 77.0], "#86b6ef", None),
    ("Ours (no-GT)",       [70.0, 80.5, 73.5, 77.5, 41.5, 75.5, 47.5, 29.0],
                           [46.5, 57.5, 60.5, 52.0, 54.0, 74.5, 56.5, 18.5], "#2a78d6", None),
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

fig, (ax_tsr, ax_car) = plt.subplots(
    2, 1, figsize=(7.0, 3.8), sharex=True,
    gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.14},
)

n_arms = len(ARMS)
n_cells = len(CELLS)
group_w = 0.84
bar_w = group_w / n_arms
x = np.arange(n_cells)

for ax, key in ((ax_tsr, 1), (ax_car, 2)):
    for i, arm in enumerate(ARMS):
        name, color, hatch = arm[0], arm[3], arm[4]
        vals = arm[key]
        pos = x - group_w / 2 + (i + 0.5) * bar_w
        bars = ax.bar(
            pos, vals, width=bar_w * 0.9, color=color, hatch=hatch,
            edgecolor="white" if hatch else "none", linewidth=0.4,
            label=name if key == 1 else None, zorder=3,
        )
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v + 1.5),
                        ha="center", va="bottom", fontsize=5.4, color=INK2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])

ax_tsr.set_ylabel("Task success (%)", fontsize=8)
ax_car.set_ylabel("Collision\navoidance (%)", fontsize=8)
ax_car.set_xticks(x)
ax_car.set_xticklabels(CELLS, fontsize=7.5)

# Long L2 no-GT benchmark-definition-divergence dagger
i_nogt = 4
pos_ll2 = 7 - group_w / 2 + (i_nogt + 0.5) * bar_w
ax_tsr.annotate("$\\ddagger$", (pos_ll2 + 0.12, 30.2), ha="left", fontsize=7,
                color=INK2)
ax_car.annotate("$\\ddagger$ table-prior arm: 44.0 / 79.5", (0.995, 0.90),
                xycoords="axes fraction", ha="right", fontsize=6.0,
                color=MUTED)

ax_tsr.legend(
    ncol=5, loc="lower left", bbox_to_anchor=(-0.01, 1.02), frameon=False,
    fontsize=7, handlelength=1.1, handletextpad=0.5, columnspacing=1.0,
)

fig.align_ylabels([ax_tsr, ax_car])
fig.savefig(__file__.replace("fig_sl_board.py", "fig_sl_board.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(__file__.replace("fig_sl_board.py", "fig_sl_board.pdf"),
            bbox_inches="tight", facecolor="white")
print("saved fig_sl_board.png/.pdf")
