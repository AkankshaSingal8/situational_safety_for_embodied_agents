"""fig_probe.pdf -- layerwise linear-probe AUC for pi0.5 on LIBERO-Safety.

Provenance: knowledge_action_gap/e2_probes_pi05_<suite>_results.json, five suites.
Each file carries model="pi0.5-LIBERO-Safety", n_examples=150, depth=18, and
    layerwise_auc                {visual, instruction, pre_action}  [18]
    layerwise_auc_shuffled_labels{visual, instruction, pre_action}  [18]
The peak of layerwise_auc['pre_action'] reproduces the "Pre-action AUC" column of
Table II exactly for all five suites; the mean of the shuffled series reproduces
the "Shuffled" column.  Nothing here is re-derived from the paper text.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = "/ocean/projects/cis250185p/asingal/knowledge_action_gap"
SUITES = [("obstacle_avoidance",       "Obstacle avoid."),
          ("obstacle_avoidance_human", "Obstacle+hand"),
          ("human_safety",             "Human safety"),
          ("affordance",               "Affordance"),
          ("reasoning_safety",         "Reasoning safety")]
COL = ["#3b6ea5", "#c1442e", "#4c8c56", "#8a6bab", "#a37a2c"]

data = {}
for key, _ in SUITES:
    with open(os.path.join(SRC, f"e2_probes_pi05_{key}_results.json")) as fh:
        data[key] = json.load(fh)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.6))

# (a) pre-action AUC across depth, all suites, with the shuffled-label envelope
shuf = []
for i, (key, label) in enumerate(SUITES):
    d = data[key]
    pa = np.asarray(d["layerwise_auc"]["pre_action"])
    x = np.arange(1, len(pa) + 1) / len(pa) * 100
    axA.plot(x, pa, color=COL[i], lw=1.4, label=label)
    j = int(np.argmax(pa))
    axA.plot(x[j], pa[j], "o", color=COL[i], ms=3.5)
    shuf.append(np.asarray(d["layerwise_auc_shuffled_labels"]["pre_action"]))
shuf = np.vstack(shuf)
xs = np.arange(1, shuf.shape[1] + 1) / shuf.shape[1] * 100
axA.fill_between(xs, shuf.min(0), shuf.max(0), color="0.55", alpha=0.30, lw=0,
                 label="shuffled labels")
axA.axhline(0.5, color="0.35", lw=0.8, ls=":")
axA.set_xlabel("relative depth (%)"); axA.set_ylabel("probe AUC")
axA.set_ylim(0.40, 1.03); axA.set_title("(a) pre-action position", fontsize=9)
axA.legend(fontsize=5.6, loc="lower right", frameon=False, ncol=2)

# (b) the three token positions on the byte-identical suite
d = data["obstacle_avoidance_human"]
for pos, lab, c, ls in [("visual", "visual", "#3b6ea5", "-"),
                        ("instruction", "instruction", "#a37a2c", "--"),
                        ("pre_action", "pre-action", "#c1442e", "-")]:
    y = np.asarray(d["layerwise_auc"][pos])
    x = np.arange(1, len(y) + 1) / len(y) * 100
    axB.plot(x, y, color=c, ls=ls, lw=1.6, label=lab)
sb = np.asarray(d["layerwise_auc_shuffled_labels"]["pre_action"])
axB.plot(np.arange(1, len(sb) + 1) / len(sb) * 100, sb, color="0.55", lw=1.0,
         label="shuffled")
axB.axhline(0.5, color="0.35", lw=0.8, ls=":")
axB.set_xlabel("relative depth (%)")
axB.set_ylim(0.40, 1.03); axB.set_title("(b) obstacle+hand, by position", fontsize=9)
axB.legend(fontsize=6.2, loc="lower right", frameon=False)

for ax in (axA, axB):
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig("fig_probe.pdf", bbox_inches="tight")
print("wrote fig_probe.pdf")
