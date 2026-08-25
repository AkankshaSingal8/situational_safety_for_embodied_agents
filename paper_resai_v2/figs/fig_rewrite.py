"""Generate fig_rewrite.pdf -- the E3 instruction-rewriting invariance figure.

Provenance
----------
Source: /ocean/projects/cis250185p/asingal/knowledge_action_gap/
        e3_todo3_<suite>_row0_results.json          (no-rewrite baseline)
        e3_todo3_<suite>_row2_v{1,2,3}_results.json (Qwen2.5-VL-7B, in-house)
        e3_api150_<suite>_<model>_v{1,2,3}_results.json (API rewriters)

Checkpoint: pi0.5 fine-tuned on LIBERO-Safety. n=150 rollouts/cell
(15 twins x 10 pruned initial states), except affordance where the three
"fork" twins are excluded (n=120) because they resolve against safe_bddl
rather than source_bddl and bank successes the other cells cannot score.
That exclusion matches vlm_pipeline/build_api150_sr_table.py::EXCLUDE_TWINS.

Both Gemini rewriters are DROPPED: their obstacle_avoidance and
obstacle_avoidance_human cells terminated at n=35-40 of 150 ("API credits
exhausted"), and the surviving subset is biased (SR 65-86% against 20-30%
for every complete run). Dropping the model rather than the cell keeps the
design uniform at 5 rewriters x 3 variants x 4 suites = 60 complete cells.

Metrics follow the same builder: hazard contact = any hazard contact during
the episode; safe success = task_success AND NOT hazard_contact.

Palette: slots 1 (blue #2a78d6) and 2 (orange #eb6834) of the dataviz
reference categorical theme. That two-colour subset lies inside the
documented first-three-slot set which validates under --pairs all in both
modes (CVD dE 9.2, normal-vision dE 24.0); the validator itself could not be
re-run here because node is unavailable on this host.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt

SRC = "/ocean/projects/cis250185p/asingal/knowledge_action_gap"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_rewrite.pdf")

SUITES = [
    ("obstacle_avoidance", "Obstacle avoidance"),
    ("obstacle_avoidance_human", "Obstacle + hand"),
    ("human_safety", "Human safety"),
    ("affordance", "Affordance"),
]
# (file token, display label).  None token = the in-house row2 run.
REWRITERS = [
    (None, "Qwen2.5-VL 7B"),
    ("claude_cheap", "Haiku 4.5"),
    ("claude_flagship", "Opus 5"),
    ("gpt_cheap", "GPT-5.4-mini"),
    ("gpt_flagship", "GPT-5.5"),
]
VARIANTS = ["v1", "v2", "v3"]
EXCLUDE = {"affordance": ("pick_up_the_fork_and_place_it_on_the_right_of_the_plate",)}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d8dce1"


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fig_rewrite_cache.json")
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def counts(path, suite):
    """Return (n, hazard-contact count, safe-success count) for one cell."""
    if path in _cache:
        return tuple(_cache[path])
    with open(path) as fh:
        recs = json.load(fh)["per_scenario"]
    excl = EXCLUDE.get(suite, ())
    kept = [r for r in recs if not any(e in r["twin_id"] for e in excl)]
    hz = sum(1 for r in kept if r["hazard_contact"])
    sr = sum(1 for r in kept if r["task_success"] and not r["hazard_contact"])
    _cache[path] = [len(kept), hz, sr]
    return len(kept), hz, sr


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def cell_path(suite, token, variant):
    if token is None:
        return f"{SRC}/e3_todo3_{suite}_row2_{variant}_results.json"
    return f"{SRC}/e3_api150_{suite}_{token}_{variant}_results.json"


fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.0), sharex=True)
x = list(range(len(REWRITERS)))

for col, (suite, title) in enumerate(SUITES):
    bn, bhz, bsr = counts(f"{SRC}/e3_todo3_{suite}_row0_results.json", suite)
    for row, (metric, colour, base_k) in enumerate(
        [("Hazard contact", BLUE, bhz), ("Safe success", ORANGE, bsr)]
    ):
        ax = axes[row][col]
        lo, hi = wilson(base_k, bn)
        ax.axhspan(lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.axhline(100 * base_k / bn, color=INK, ls="--", lw=1.2, zorder=3)

        for i, (token, _) in enumerate(REWRITERS):
            for j, variant in enumerate(VARIANTS):
                p = cell_path(suite, token, variant)
                if not os.path.exists(p):
                    continue
                n, hz, sr = counts(p, suite)
                k = hz if row == 0 else sr
                ax.plot(i + (j - 1) * 0.17, 100 * k / n, "o", ms=5.5,
                        color=colour, mec="white", mew=0.8, zorder=4)

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
            ax.set_ylabel(metric + " (%)", fontsize=9, color=INK)

    # shared y-range per column so the two rows are not falsely comparable
    axes[0][col].set_ylim(-3, max(12, 1.45 * 100 * bhz / bn))
    axes[1][col].set_ylim(-3, max(12, 1.45 * 100 * bsr / bn))

for ax in axes[1]:
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in REWRITERS], fontsize=7, color=MUTED, rotation=18, ha="right")

axes[0][0].text(0.02, 0.93, "no rewriter separates from baseline",
                transform=axes[0][0].transAxes, fontsize=7.5, color=MUTED,
                va="top", style="italic")

fig.text(0.5, 0.005,
         "Dashed line = unmodified benchmark instruction; band = its 95% Wilson interval. "
         "Three points per rewriter, one per prompt variant.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig(OUT, bbox_inches="tight")
json.dump(_cache, open(CACHE, "w"))
print("wrote", OUT)
