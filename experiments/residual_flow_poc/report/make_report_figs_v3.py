"""Figures for the October 26 update section of report_v2.tex: seed spread, wording vs a
nearest-wording lookup, and how close A corrections come to the laptop.
    python report/make_report_figs_v3.py   (run from experiments/residual_flow_poc)
Palette checked with the dataviz validator (blue/orange, light mode: all checks pass).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs_v3")
os.makedirs(OUT, exist_ok=True)
RUNS = {"0": "runs_v2", "7": "runs_v2_s7", "42": "runs_v2_s42"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.bbox": "tight",
    "legend.frameon": False,
})
R = {s: json.load(open(os.path.join(d, "results_v2.json"))) for s, d in RUNS.items()}
p = lambda x: x["k"] / x["n"]  # noqa: E731


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# ---------------- Fig A: three seeds, test scenes ----------------
ROWS = [
    ("Exp 1, A text: laptop avoided", lambda T: p(T["summary"]["main|A"]["A_ok&succ"])),
    ("Exp 1, B text: mug upright", lambda T: p(T["summary"]["main|B"]["B_ok&succ"])),
    ("Exp 1, model trained without text: laptop avoided", lambda T: p(T["summary"]["notext"]["A_ok&succ"])),
    ("Exp 2, unseen laptop wordings, set 1", lambda T: p(T["exp2_heldout_A"])),
    ("Exp 2, unseen laptop wordings, set 2", lambda T: p(T["exp2_heldout2_A"])),
    ("Exp 3, sentence given directly", lambda T: p(T["exp3"]["main|combined"]["AB_ok&succ"])),
    ("Exp 3, close wording, clauses in turn", lambda T: p(T["exp3"]["stack_AB|combined"]["AB_ok&succ"])),
    ("Exp 3, close wording, trained on combined", lambda T: p(T["exp3"]["main_ab|combined"]["AB_ok&succ"])),
    ("Exp 3, far wording, clauses in turn", lambda T: p(T["exp3"]["stack_AB|far"]["AB_ok&succ"])),
    ("Exp 3, far wording, trained on combined", lambda T: p(T["exp3"]["main_ab|far"]["AB_ok&succ"])),
]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
for i, (lab, f) in enumerate(ROWS):
    vals = [f(R[s]["test"]) for s in RUNS]
    m = float(np.mean(vals))
    ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(vals, [i] * 3, s=22, facecolor=SURF, edgecolor=INK2, linewidth=1.1, zorder=3,
               label="one training seed" if i == 0 else None)
    ax.plot([m, m], [i - 0.3, i + 0.3], color=BLUE, lw=2.2, zorder=4, label="mean of three seeds" if i == 0 else None)
    ax.text(1.04, i, f"{m:.2f}", va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS]); ax.invert_yaxis()
ax.set_xlim(-0.02, 1.02); ax.grid(axis="y", visible=False)
ax.set_xlabel("Fraction of runs meeting the instruction and placing the mug (test scenes)")
ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, fontsize=8)
save(fig, "seeds")

# ---------------- Fig B: wording vs a nearest-wording lookup ----------------
fig, ax = plt.subplots(figsize=(6.6, 3.4))
pts = {"A": [], "B": []}
for cond, e in R["0"]["test"]["exp2"].items():
    margin = (e["cos_A"] - e["cos_B"]) if e["cls"] == "A" else (e["cos_B"] - e["cos_A"])
    succ = float(np.mean([p(R[s]["test"]["exp2"][cond]["target_ok_succ"]) for s in RUNS]))
    pts[e["cls"]].append((margin, succ, e["text"]))
for cls, col, mk, lab in (("A", BLUE, "o", "laptop wording (16)"), ("B", ORANGE, "s", "upright wording (16)")):
    xs, ys = [q[0] for q in pts[cls]], [q[1] for q in pts[cls]]
    ax.scatter(xs, ys, s=34, color=col, marker=mk, edgecolor=SURF, linewidth=1.2, zorder=3, label=lab)
ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 2)))
ax.text(-0.004, 0.35, "a lookup would pick the\nwrong constraint left of here", ha="right", va="center", fontsize=7.5, color=INK2)
for m_, s_, t in pts["A"]:
    if s_ < 0.05:
        ax.annotate(f"“{t}”\n(read as the upright instruction)", (m_, s_), xytext=(0.15, 0.22), fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.6))
ax.set_xlim(-0.12, 0.32); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("How much closer the wording sits to its own constraint's training wordings\n(max cosine to own class minus max cosine to the other class)")
ax.set_ylabel("Constraint met, mug placed\n(test scenes, mean of 3 seeds)")
ax.legend(loc="upper left", fontsize=8)
save(fig, "lookup")

# ---------------- Fig C: how close A corrections come to the laptop ----------------
cache = os.path.join(OUT, "clearance.json")
if not os.path.exists(cache):
    C = {"main|A": [], "oracle_A": []}
    for s, d in RUNS.items():
        for line in open(os.path.join(d, "eval_test", "rollouts.jsonl")):
            if '"cond": "main|A"' in line or ('"cond": "oracle_A"' in line and s == "0"):
                r = json.loads(line)
                C[r["cond"]].append(r["min_laptop_clearance"])
    json.dump(C, open(cache, "w"))
C = json.load(open(cache))
model = np.array(C["main|A"]) * 100
ref = np.array(C["oracle_A"]) * 100
bins = np.arange(-3, 6.01, 0.25)
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.hist(np.clip(model, -3, 6), bins=bins, color=BLUE, edgecolor=SURF, linewidth=1, label=f"model, A text ({len(model)} runs, 3 seeds)")
ax.hist(np.clip(ref, -3, 6), bins=bins, histtype="step", color=INK, linewidth=1.6,
        weights=np.full(len(ref), len(model) / len(ref)), label=f"reference correction (scaled to the same total)")
ax.axvline(0, color=ORANGE, lw=1.6)
ax.text(-0.1, ax.get_ylim()[1] * 0.92, "left of this line the\nmug passes over the laptop", ha="right", va="top", fontsize=7.5, color=INK2)
ax.axvline(3, color=INK2, lw=1, ls=(0, (3, 2)))
ax.text(3.1, ax.get_ylim()[1] * 0.92, "3 cm: what the reference\ncorrection aims for", ha="left", va="top", fontsize=7.5, color=INK2)
ax.set_xlabel("Closest the carried mug came to the laptop (cm, test scenes)")
ax.set_ylabel("runs"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=2, fontsize=7.5)
save(fig, "clearance")
print("model clearance cm: median %.2f p5 %.2f p1 %.2f below0 %.3f | ref median %.2f"
      % (np.median(model), np.percentile(model, 5), np.percentile(model, 1), np.mean(model < 0), np.median(ref)))
print("A margins min %.3f, B margins min %.3f" % (min(q[0] for q in pts["A"]), min(q[0] for q in pts["B"])))
print("wrote", sorted(os.listdir(OUT)))
