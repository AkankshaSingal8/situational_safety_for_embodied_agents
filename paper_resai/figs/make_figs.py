"""Figures for the ReS AI @ IROS 2026 workshop paper (paper_resai/main.tex).

Fig 1 (fig_pipeline.pdf, full width): real LIBERO-Safety scene -> VLM
predicate grounding -> three-literal FOL rule -> three-regime routing ->
certified noise-space steering.

Fig 2 (fig_results.pdf, single column, two panels):
  (a) E3 identification swap (paired rollouts, enforcement fixed).
  (b) Safety-reasoning refusal sweep vs best published vs unguided-executes.

All numbers are copied from the paper text (source: results_tables ledger).
Run:  python make_figs.py   (needs matplotlib + the scene PNGs alongside)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.size": 8.5, "font.family": "serif"})

C_NEURAL = "#dbeafe"   # blue fill: neural components
C_NEURAL_E = "#1d4ed8"
C_SYM = "#fef3c7"      # amber fill: symbolic components
C_SYM_E = "#b45309"
C_CERT = "#dcfce7"     # green fill: certificate
C_CERT_E = "#15803d"


def box(ax, x, y, w, h, title, body, fc, ec, fs=7.2, title_fs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + 0.035, body, ha="center", va="bottom", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color="0.25"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.3, color=color))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- scene image with hazard/target annotation --------------------------
    img = mpimg.imread("scene_oa.png")
    ax_img = fig.add_axes([0.015, 0.24, 0.165, 0.60])
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_color("0.4")
    h, w = img.shape[0], img.shape[1]
    ax_img.add_patch(Rectangle((0.565 * w, 0.26 * h), 0.11 * w, 0.37 * h,
                               fill=False, ec="#dc2626", lw=1.8))
    ax_img.text(0.44 * w, 0.24 * h, "hazard", color="#dc2626", fontsize=6.5,
                fontweight="bold", ha="left", va="bottom")
    ax_img.add_patch(Rectangle((0.71 * w, 0.54 * h), 0.17 * w, 0.23 * h,
                               fill=False, ec="#2563eb", lw=1.8))
    ax_img.text(0.87 * w, 0.50 * h, "target", color="#2563eb", fontsize=6.5,
                fontweight="bold", ha="right", va="bottom")
    ax.text(0.095, 0.11, "RGB-D + instruction\n``put the mug on the plate''",
            ha="center", va="center", fontsize=6.6, style="italic")

    # --- neural: predicate grounding ---------------------------------------
    box(ax, 0.215, 0.24, 0.185, 0.60,
        "VLM / perception\ngrounding",
        "local, per-object:\n" r"$\mathrm{Referenced}(x)$" "\n"
        r"$\mathrm{Protected}(x)$, $\mathrm{Moving}(x)$" "\n"
        "positions: open-vocab\ndetection + depth",
        C_NEURAL, C_NEURAL_E, fs=6.7)
    ax.text(0.3075, 0.165, "neural: answers\nlocal predicates", ha="center",
            va="top", fontsize=6.8, color=C_NEURAL_E, style="italic")

    # --- symbolic: FOL rule + routing --------------------------------------
    box(ax, 0.435, 0.56, 0.25, 0.30,
        r"FOL rule $\rightarrow$ Hazard$(x)$",
        r"$\mathrm{Prot}(x)\vee\mathrm{Mov}(x)\;\vee$" "\n"
        r"$(\neg\mathrm{Ment}(x)\wedge\mathrm{NearPath}(x))$",
        C_SYM, C_SYM_E, fs=6.9)
    box(ax, 0.435, 0.13, 0.25, 0.34,
        "3-regime routing",
        "Avoid (keep-out)\nDisengage (hazard = dest.)\nRefuse (unsafe instr.)",
        C_SYM, C_SYM_E)
    ax.text(0.56, 0.035, "symbolic: makes the decision", ha="center",
            fontsize=6.8, color=C_SYM_E, style="italic")

    # --- enforcement + certificate -----------------------------------------
    box(ax, 0.72, 0.50, 0.265, 0.36,
        "noise-space steering",
        "select / tilt / ascend the noise\nof the frozen flow VLA;\n"
        "actions stay on-manifold",
        C_NEURAL, C_NEURAL_E, fs=6.7)
    box(ax, 0.72, 0.08, 0.265, 0.31,
        "checked certificate",
        "DCBF chain re-verified on\nthe executed prefix\n(mechanism-independent)",
        C_CERT, C_CERT_E, fs=6.7)

    arrow(ax, 0.183, 0.56, 0.212, 0.56)
    arrow(ax, 0.402, 0.62, 0.432, 0.68)
    arrow(ax, 0.402, 0.44, 0.432, 0.34)
    arrow(ax, 0.56, 0.555, 0.56, 0.48)
    arrow(ax, 0.687, 0.32, 0.717, 0.58)
    arrow(ax, 0.852, 0.495, 0.852, 0.40)

    fig.savefig("fig_pipeline.pdf", bbox_inches="tight", dpi=300)
    print("wrote fig_pipeline.pdf")


def fig_results():
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.15),
                               gridspec_kw={"width_ratios": [1.1, 1.0]})

    # (a) E3 identification swap: TSR and CAR, Spatial L1+L2 mean per arm.
    arms = ["GT\n(priv.)", "Symb.+\npriors", "FOL\n(ours)"]
    tsr = [(55.0 + 77.5) / 2, (60.0 + 77.5) / 2, (62.5 + 90.0) / 2]
    car = [(27.5 + 40.0) / 2, (27.5 + 37.5) / 2, (32.5 + 40.0) / 2]
    x = range(3)
    wd = 0.38
    a.bar([i - wd / 2 for i in x], tsr, wd, color="#1d4ed8", label="TSR")
    a.bar([i + wd / 2 for i in x], car, wd, color="#dc2626", label="CAR")
    a.set_xticks(list(x))
    a.set_xticklabels(arms, fontsize=6.5)
    a.set_ylim(0, 100)
    a.set_ylabel("%", fontsize=7)
    a.tick_params(axis="y", labelsize=6.5)
    a.legend(fontsize=6, frameon=False, loc="upper left",
             handlelength=1.0, borderaxespad=0.1)
    a.set_title("(a) Identity swap\n(paired, enf. fixed)", fontsize=7)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # (b) SSR refusal sweep.
    lv = ["L0", "L1", "L2"]
    ours = [100, 100, 80]
    pub = [80, 72, 36]
    exe = [94, 60, 100]
    x = range(3)
    wd = 0.27
    b.bar([i - wd for i in x], ours, wd, color="#15803d", label="ours refuses")
    b.bar(list(x), pub, wd, color="0.55", label="best published")
    b.bar([i + wd for i in x], exe, wd, color="#f59e0b",
          label="unguided executes")
    b.set_xticks(list(x))
    b.set_xticklabels(lv, fontsize=6.5)
    b.set_ylim(0, 138)
    b.set_yticks([0, 25, 50, 75, 100])
    b.tick_params(axis="y", labelsize=6.5)
    b.legend(fontsize=5.6, frameon=False, loc="upper left",
             handlelength=1.0, borderaxespad=0.0, labelspacing=0.2)
    b.set_title("(b) Unsafe-instruction\nrefusal (%)", fontsize=7)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    fig.tight_layout(w_pad=1.2)
    fig.savefig("fig_results.pdf", bbox_inches="tight", dpi=300)
    print("wrote fig_results.pdf")


if __name__ == "__main__":
    fig_pipeline()
    fig_results()
