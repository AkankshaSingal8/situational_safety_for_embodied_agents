"""Figure: one test scene, executed grip paths (top-down) and mug tilt per waypoint
for base / A text / B text / A->B stacking.   python make_figure.py --scene 3"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ap = argparse.ArgumentParser(); ap.add_argument("--scene", type=int, default=3); ap.add_argument("--out", default="figure_scene.png"); a = ap.parse_args()
TE = np.load("data/test.npz", allow_pickle=True); sc = TE["scenes"][a.scene]
recs = {}
for path in ("runs/eval/rollouts.jsonl", "runs/eval_compose/rollouts.jsonl"):
    for l in open(path):
        r = json.loads(l)
        if r["scene"] == a.scene and r["sample"] == 0 and r["cond"] in ("base", "main|A", "main|B", "stack_AB|combined0"):
            recs.setdefault(r["cond"], r)
lab = {"base": "base (no instruction)", "main|A": "\"Avoid carrying the mug over the laptop.\"",
       "main|B": "\"Keep the mug upright to prevent spilling.\"", "stack_AB|combined0": "A+B (clause stacking)"}
col = {"base": "#888888", "main|A": "#1f77b4", "main|B": "#d62728", "stack_AB|combined0": "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
lx, ly = sc["laptop_center"]; lw, ld = sc["laptop_size"]; c, s = np.cos(sc["laptop_yaw"]), np.sin(sc["laptop_yaw"])
corners = np.array([[u, v] for u, v in ((-lw/2, -ld/2), (lw/2, -ld/2), (lw/2, ld/2), (-lw/2, ld/2))])
ax[0].add_patch(Polygon(corners @ np.array([[c, s], [-s, c]]) + [lx, ly], color="k", alpha=0.25, label="laptop"))
ax[0].plot(*sc["mug_start"], "o", color="brown", ms=9, label="mug start"); ax[0].plot(*sc["target"], "*", color="green", ms=14, label="target")
for k, r in recs.items():
    g = np.array(r["exec_grip"]); ax[0].plot(g[:, 0], g[:, 1], "-", color=col[k], lw=2, label=lab[k])
    ax[1].plot(r["exec_tilt"], "-", color=col[k], lw=2, label=lab[k])
ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].set_title(f"executed path, test scene {a.scene}")
fig.legend(*ax[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=4, frameon=False)
ax[1].axhline(15, color="k", ls="--", lw=1); ax[1].text(1, 16, "spill limit 15°", fontsize=8)
ax[1].set_xlabel("waypoint"); ax[1].set_ylabel("mug tilt (deg)"); ax[1].set_title("mug tilt (base hidden under A: identical)")
plt.tight_layout(rect=(0, 0.12, 1, 1)); plt.savefig(a.out, dpi=130); print("saved", a.out, sorted(recs))
