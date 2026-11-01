"""Figures for report_v2.tex, read from data/, data_v2/ and runs_v2/results_v2.json.
    python report/make_report_figs_v2.py [--runs runs_v2]   (run from experiments/residual_flow_poc)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.texts import HELDOUT_A, HELDOUT2_A, COMBINED, COMBINED_FAR  # noqa: E402
from src.traj import CARRY  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs_v2")
ap.add_argument("--out", default=os.path.join(HERE, "figs_v2"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# reference palette, light mode, first three categorical slots (validated all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
TRAIN_C, TEST_C = BLUE, ORANGE
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(a.out, name + ".pdf"))
    plt.close(fig)


# ---------------- Fig 1: v1 vs v2 scene and base-trajectory variation ----------------
def laptop_poly(sc):
    lw, ld = sc["laptop_size"]; c, s = np.cos(sc["laptop_yaw"]), np.sin(sc["laptop_yaw"])
    pts = np.array([[-lw / 2, -ld / 2], [lw / 2, -ld / 2], [lw / 2, ld / 2], [-lw / 2, ld / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + sc["laptop_center"]


fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
tilts = {}
for ax, (name, d) in zip(axs[:2], (("v1", "data"), ("v2", "data_v2"))):
    D = np.load(os.path.join(d, "train.npz"), allow_pickle=True)
    for i in range(30):
        sc = D["scenes"][i]
        ax.add_patch(Polygon(laptop_poly(sc), closed=True, fill=False, edgecolor=INK2, lw=0.5, alpha=0.5))
        b = D["base"][i]
        ax.plot(b[CARRY[0]:CARRY[1] + 1, 0], b[CARRY[0]:CARRY[1] + 1, 1], color=BLUE, lw=0.8, alpha=0.7)
        ax.plot(*sc["mug_start"], "o", ms=3, color=ORANGE)
        ax.plot(*sc["target"], "s", ms=3, color=AQUA)
    ax.set_xlim(-0.33, 0.33); ax.set_ylim(-0.3, 0.3); ax.set_aspect("equal")
    ax.set_title(f"{name}: 30 training scenes, top view", fontsize=9, color=INK)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    tilts[name] = np.degrees(np.linalg.norm(D["base"][:, :, 3:], axis=-1).max(1))
axs[0].plot([], [], color=BLUE, label="base carry path"); axs[0].plot([], [], "o", color=ORANGE, ms=3, label="mug start")
axs[0].plot([], [], "s", color=AQUA, ms=3, label="target"); axs[0].plot([], [], color=INK2, lw=0.6, label="laptop")
axs[0].legend(loc="lower center", bbox_to_anchor=(1.1, -0.42), ncol=4, fontsize=8)
bins = np.arange(20, 75, 2.5)
axs[2].hist(tilts["v2"], bins=bins, color=ORANGE, alpha=0.75, label="v2", edgecolor=SURF)
axs[2].hist(tilts["v1"], bins=bins, color=BLUE, histtype="step", linewidth=2, label="v1")
axs[2].axvline(15, color=INK2, lw=1, ls=(0, (3, 2))); axs[2].text(16, axs[2].get_ylim()[1] * 0.9, "spill limit", fontsize=7.5, color=INK2)
axs[2].set_xlim(10, 75)
axs[2].set_xlabel("peak base tilt (deg)"); axs[2].set_ylabel("training scenes"); axs[2].set_title("base tilt", fontsize=9, color=INK)
axs[2].legend(fontsize=8)
save(fig, "variation")

R = json.load(open(os.path.join(a.runs, "results_v2.json")))
SPL = [s for s in ("train", "test") if s in R]
COL = {"train": TRAIN_C, "test": TEST_C}
LAB = {"train": "Train scenes (seen in training)", "test": "Test scenes (unseen)"}


def paired_bars(rows, xlabel, name, height=3.6, note_n=None):
    """rows: list of (label, {split: value})"""
    fig, ax = plt.subplots(figsize=(6.6, height))
    h = 0.38
    for i, (lab, vals) in enumerate(rows):
        for j, s in enumerate(SPL):
            v = vals.get(s)
            if v is None:
                continue
            y = i + (j - 0.5) * (h + 0.02)
            ax.barh(y, max(v, 0.004), h, color=COL[s], edgecolor=SURF, linewidth=1, label=LAB[s] if i == 0 else None)
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=7.2, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 1.13); ax.grid(axis="y", visible=False); ax.set_xlabel(xlabel)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, fontsize=8)
    save(fig, name)


S = lambda s, c, m: R[s]["summary"][c][m]["p"] if c in R[s]["summary"] else None  # noqa: E731

# ---------------- Fig 2: Exp 1 ----------------
rows = [("Base: A fixed (laptop avoided)", {s: S(s, "base", "A_ok&succ") for s in SPL}),
        ("A text: A fixed", {s: S(s, "main|A", "A_ok&succ") for s in SPL}),
        ("A text: B also fixed (should be 0)", {s: S(s, "main|A", "B_ok&succ") for s in SPL}),
        ("B text: B fixed (mug upright)", {s: S(s, "main|B", "B_ok&succ") for s in SPL}),
        ("B text: A also fixed (should be 0)", {s: S(s, "main|B", "A_ok&succ") for s in SPL}),
        ("No-text model: A fixed", {s: S(s, "notext", "A_ok&succ") for s in SPL}),
        ("No-text model: B fixed", {s: S(s, "notext", "B_ok&succ") for s in SPL}),
        ("Shuffled-text model, A text: A fixed", {s: S(s, "shuffled|A", "A_ok&succ") for s in SPL}),
        ("Shuffled-text model, B text: B fixed", {s: S(s, "shuffled|B", "B_ok&succ") for s in SPL}),
        ("Mean residual A: A fixed", {s: S(s, "mean_A", "A_ok&succ") for s in SPL}),
        ("Mean residual B: B fixed", {s: S(s, "mean_B", "B_ok&succ") for s in SPL})]
paired_bars(rows, "Fraction of runs (\"fixed\" = constraint met and mug placed)", "exp1", height=4.2)

# ---------------- Fig 3: Exp 2, laptop paraphrases per phrasing ----------------
fig, ax = plt.subplots(figsize=(6.6, 5.2))
items = [("heldout", j, t) for j, t in enumerate(HELDOUT_A)] + [("heldout2", j, t) for j, t in enumerate(HELDOUT2_A)]
for i, (tag, j, t) in enumerate(items):
    vals = [R[s]["exp2"].get(f"main|{tag}_A{j}", {}).get("target_ok_succ", {}).get("p") for s in SPL]
    vv = [v for v in vals if v is not None]
    if vv:
        ax.plot([min(vv), max(vv)], [i, i], color=GRID, lw=2.5, zorder=1)
    for s, v, mk in zip(SPL, vals, ("o", "D")):
        if v is not None:
            ax.scatter(v, i, s=34, color=COL[s], marker=mk, edgecolor=SURF, linewidth=1.2, zorder=3,
                       label=LAB[s] if i == 0 else None)
for s, ls in zip(SPL, ((0, (3, 2)), (0, (1, 2)))):
    ax.axvline(S(s, "main|A", "A_ok&succ"), color=COL[s], lw=1, ls=ls)
ax.axhline(len(HELDOUT_A) - 0.5, color=INK2, lw=0.6)
ax.text(0.01, len(HELDOUT_A) - 0.7, "set 1", fontsize=7.5, color=INK2, va="bottom")
ax.text(0.01, len(HELDOUT_A) - 0.3, "set 2", fontsize=7.5, color=INK2, va="top")
ax.set_yticks(range(len(items))); ax.set_yticklabels([t for _, _, t in items], fontsize=7.3)
ax.invert_yaxis(); ax.set_xlim(-0.02, 1.08); ax.grid(axis="y", visible=False)
ax.set_xlabel("Laptop avoided and mug placed (100 scenes each; dashed lines: training phrasings)")
ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, fontsize=8)
save(fig, "exp2_A")

rows = []
for tag, lab in (("heldout", "set 1"), ("heldout2", "set 2")):
    for cls, cl in (("A", "laptop"), ("B", "upright")):
        rows.append((f"Unseen {cl} paraphrases, {lab} (8)", {s: R[s][f"exp2_{tag}_{cls}"]["p"] for s in SPL}))
rows = [("Training laptop phrasings (5)", {s: S(s, "main|A", "A_ok&succ") for s in SPL}),
        ("Training upright phrasings (5)", {s: S(s, "main|B", "B_ok&succ") for s in SPL})] + rows
paired_bars(rows, "Target constraint met and mug placed (pooled)", "exp2_pooled", height=3.0)

# ---------------- Fig 4: Exp 3 ----------------
E3 = [("main|combined", "No training: sentence fed directly"),
      ("compose|combined", "No training: velocity composition"),
      ("stack_AB|combined", "No training: clauses in turn, A then B"),
      ("stack_BA|combined", "No training: clauses in turn, B then A"),
      ("sum|combined", "No training: two corrections added"),
      ("shuffled|combined", "Shuffled-text model"),
      ("main_ab|combined", "With training: close-wording set"),
      ("main_ab|trainAB", "With training: its own training phrasings"),
      ("main|far", "No training: far-wording set, sentence fed directly"),
      ("stack_AB|far", "No training: far-wording set, clauses in turn A then B"),
      ("main_ab|far", "With training: far-wording set")]
rows = [(lab, {s: R[s]["exp3"][pre]["AB_ok&succ"]["p"] for s in SPL if pre in R[s]["exp3"]}) for pre, lab in E3]
rows.append(("Reference correction (oracle A+B)", {s: S(s, "oracle_AB", "AB_ok&succ") for s in SPL}))
paired_bars(rows, "Both constraints met and mug placed", "exp3", height=4.8)

# ---------------- Fig 5: Exp 3 per sentence (test scenes), both unseen sets ----------------
if "test" in R:
    T = R["test"]
    arms = [("main|", "sentence fed directly (no training)", ORANGE, "o"),
            ("stack_AB|", "clauses in turn A→B (no training)", BLUE, "s"),
            ("main_ab|", "trained on combined phrasings", AQUA, "D")]
    sets = [("close-wording set", COMBINED, T["exp3_per_sentence"], "combined"),
            ("far-wording set", COMBINED_FAR, T["exp3_far_per_sentence"], "far")]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    for ax, (title, L, ps, tag) in zip(axs, sets):
        for i, (t, _) in enumerate(L):
            for k, (pre, lab, c, mk) in enumerate(arms):
                v = ps[t].get(f"{pre}{tag}")
                if v is None:
                    continue
                ax.scatter(v, i + (k - 1) * 0.2, s=30, color=c, marker=mk, edgecolor=SURF, linewidth=1.1, zorder=3,
                           label=lab if (i == 0 and ax is axs[0]) else None)
        ax.set_yticks(range(len(L)))
        ax.set_yticklabels([f"{t}  ({ps[t]['cos_trainAB']:.2f})" if "cos_trainAB" in ps[t] else
                            f"{t}  ({R['test']['exp3_cos_trainAB'][t]:.2f})" for t, _ in L], fontsize=7)
        ax.invert_yaxis(); ax.set_xlim(-0.03, 1.05); ax.grid(axis="y", visible=False)
        ax.set_title(title + " (max cosine to the combined training phrasings in brackets)", fontsize=8, color=INK, loc="left")
    axs[1].set_xlabel("Both constraints met and mug placed (test scenes)")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.35, 1.12), ncol=3, fontsize=7.5)
    save(fig, "exp3_sentences")
print("wrote", sorted(os.listdir(a.out)))
