"""Figures for report.tex. Numbers are copied from RESULTS.md / results_*.json.
    python report/make_report_figs.py   (run from experiments/residual_flow_poc)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# reference palette, light mode, first three categorical slots (validated all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# Fig 1: Exp 1, grouped horizontal bars, A-ok&success vs B-ok&success per condition
conds = ["Base (no correction)", "Model, empty text", "Model, A text", "Model, B text",
         "No-text model", "Shuffled-text model, A text", "Shuffled-text model, B text",
         "Mean residual, A", "Mean residual, B"]
a_ok = [0.00, 0.00, 0.96, 0.00, 0.37, 0.37, 0.41, 0.00, 0.00]
b_ok = [0.00, 0.00, 0.00, 1.00, 0.40, 0.41, 0.35, 0.00, 0.00]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
y = range(len(conds))
h = 0.36
for i, (a, b) in enumerate(zip(a_ok, b_ok)):
    ax.barh(i - h / 2 - 0.01, a, h, color=BLUE, edgecolor=SURF, linewidth=1,
            label="Laptop avoided (A) and task done" if i == 0 else None)
    ax.barh(i + h / 2 + 0.01, b, h, color=ORANGE, edgecolor=SURF, linewidth=1,
            label="Mug kept upright (B) and task done" if i == 0 else None)
    for off, v in ((-h / 2, a), (h / 2, b)):
        ax.text(v + 0.015, i + off, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels(conds); ax.invert_yaxis()
ax.set_xlim(0, 1.12); ax.set_xlabel("Fraction of 100 test scenes")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=8)
save(fig, "exp1_swap")

# Fig 2: Exp 2, per-phrasing dot plot, sorted, with training-phrasing reference
A = [("Make sure the cup never travels over the laptop.", 0.98),
     ("Go around the laptop.", 0.90),
     ("Don't pass above the electronics. (PDF)", 0.88),
     ("The beverage must not be carried across the top of the computer.", 0.88),
     ("Stay clear of the space above the notebook computer.", 0.85),
     ("Route around the keyboard rather than over it.", 0.24),
     ("Keep the drink away from the airspace over the PC.", 0.16),
     ("Liquids should never be moved over electronic devices.", 0.13)]
B = [("Maintain the cup's vertical orientation. (PDF)", 1.00),
     ("Don't let the drink slosh out.", 1.00), ("Keep the coffee from spilling.", 1.00),
     ("The mug should stay level the whole time.", 1.00),
     ("No tipping or tilting of the cup during transport.", 1.00),
     ("Carry it as if it were full to the brim.", 1.00),
     ("Ensure the beverage container remains upright.", 1.00),
     ("Avoid rotating the mug sideways.", 1.00)]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
rows = [(t, v, BLUE, "o") for t, v in A] + [(t, v, ORANGE, "s") for t, v in B]
for i, (t, v, c, m) in enumerate(rows):
    ax.plot([0, v], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(v, i, s=42, color=c, marker=m, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.text(v + 0.025, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
ax.axvline(0.972, color=BLUE, lw=1, ls=(0, (3, 2)))
ax.text(0.955, 6, "training phrasings, A: 0.97", ha="right", va="center", fontsize=7.5, color=INK2)
ax.axhline(7.5, color=INK2, lw=0.6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.invert_yaxis(); ax.set_xlim(0, 1.12); ax.grid(axis="y", visible=False)
ax.set_xlabel("Target constraint satisfied and task done (100 scenes each)")
ax.scatter([], [], color=BLUE, marker="o", label="Laptop (A) paraphrases")
ax.scatter([], [], color=ORANGE, marker="s", label="Upright (B) paraphrases")
ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.02), ncol=2, fontsize=8)
save(fig, "exp2_paraphrases")

# Fig 3: Exp 3 arms
arms = ["Combined sentence, fed directly", "Summing the two flows (velocity composition)",
        "Clauses applied in turn, A then B", "Clauses applied in turn, B then A",
        "Two corrections added together", "Reference correction (oracle A+B)"]
vals = [0.00, 0.03, 0.965, 0.978, 0.969, 1.00]
cols = [ORANGE, ORANGE, BLUE, BLUE, BLUE, INK2]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for i, (v, c) in enumerate(zip(vals, cols)):
    ax.barh(i, max(v, 0.004), 0.6, color=c, edgecolor=SURF)
    ax.text(v + 0.015, i, f"{v:.3f}".rstrip("0").rstrip(".") if v not in (0, 1) else f"{v:.2f}",
            va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms); ax.invert_yaxis()
ax.axhline(1.5, color=INK2, lw=0.6)
ax.set_xlim(0, 1.12); ax.grid(axis="y", visible=False)
ax.set_xlabel("Both constraints satisfied and task done (1,200 runs per row; reference: 100)")
save(fig, "exp3_compose")

# Fig 4: augmentation, dumbbell per fresh held-out A phrasing
H2 = [("Do not let the mug hover over the laptop at any point.", 0.96, 0.98),
      ("Keep anything you carry off the space above the computer.", 0.95, 0.98),
      ("Detour so the cup is never above the notebook.", 0.08, 0.60),
      ("Avoid moving the drink over the screen and keyboard.", 0.08, 0.89),
      ("Spills on the laptop would ruin it, so never pass over it.", 0.96, 0.98),
      ("Take a path that doesn't cross over the laptop.", 0.97, 0.98),
      ("Steer clear of the computer when moving the cup.", 0.92, 0.98),
      ("Never transport the beverage above the device.", 0.00, 0.00)]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for i, (t, m, a) in enumerate(H2):
    ax.plot([m, a], [i, i], color=GRID, lw=2.5, zorder=1)
    ax.scatter(m, i, s=40, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3,
               label="Original model" if i == 0 else None)
    ax.scatter(a, i, s=40, color=AQUA, marker="D", edgecolor=SURF, linewidth=1.5, zorder=3,
               label="With embedding mixing" if i == 0 else None)
    if a - m > 0.2:
        ax.text(a + 0.025, i, f"{m:.2f} to {a:.2f}", va="center", fontsize=7.5, color=INK2)
ax.text(0.03, 7, "0.00 for both", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(H2))); ax.set_yticklabels([r[0] for r in H2], fontsize=7.5)
ax.invert_yaxis(); ax.set_xlim(-0.02, 1.12); ax.grid(axis="y", visible=False)
ax.set_xlabel("Laptop avoided and task done (100 scenes each)")
ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, fontsize=8)
save(fig, "aug_heldout2")
print("wrote", sorted(os.listdir(OUT)))
