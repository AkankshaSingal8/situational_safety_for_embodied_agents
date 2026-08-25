"""System-overview block diagram in the style of Brunke et al. (RA-L 2025)
Fig. 2: module boxes, labelled signal arrows carrying the variables, real
thumbnails next to the perception / robot modules, the core module
highlighted, two tiers (per-episode synthesis above, 20 Hz control loop
below) with a feedback arrow.

Every module and signal here is the deployed board configuration
(Tables III/IV): open-vocabulary detection + depth, VLM predicate grounding,
FOL election + routing, barrier synthesis, frozen pi0.5 flow map with the
in-denoising projection sweep (ramp w_k), checked prefix certificate +
selection, robot at 20 Hz executing the 5-step prefix.

Thumbnails: paper/figs/sysfig_assets/ (obs.jpg, guard.jpg, exec.jpg) --
real LIBERO-Safety frames (oah L2 t14).
Output: paper/figs/fig_system_block.png/.pdf
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIGDIR = pathlib.Path(__file__).resolve().parent
ASSETS = FIGDIR / "sysfig_assets"

W, H = 13.6, 5.7
FS = 7.2
INK, MUTED = "#1c232e", "#5a6472"
BOX_FC, BOX_EC = "#ffffff", "#3a4350"
CORE_FC, CORE_EC = "#fff1f0", "crimson"
BAND_TOP, BAND_BOT = "#f4f1ea", "#eef3f8"


def box(ax, x, y, w, h, title, body=None, core=False, z=3):
    fc, ec, lw = (CORE_FC, CORE_EC, 1.4) if core else (BOX_FC, BOX_EC, 0.9)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=z))
    if body:
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top",
                fontsize=FS + 0.8, fontweight="bold",
                color=CORE_EC if core else INK, zorder=z + 1)
        ax.text(x + w / 2, y + (h - 0.3) / 2, body, ha="center", va="center",
                fontsize=FS - 0.6, color=INK, zorder=z + 1, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=FS + 0.8, fontweight="bold",
                color=CORE_EC if core else INK, zorder=z + 1)


def arrow(ax, p, q, label=None, lab_dx=0.0, lab_dy=0.08, ha="center",
          va="bottom", style="-|>", ls="-", color="0.25", z=4, via=None):
    pts = [p] + (via or []) + [q]
    for a, b in zip(pts[:-1], pts[1:]):
        last = b is q
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style if last else "-",
                                     mutation_scale=9, lw=0.9, color=color,
                                     ls=ls, zorder=z, shrinkA=0, shrinkB=0))
    if label:
        m = np.mean(pts, axis=0) if len(pts) == 2 else np.asarray(pts[len(pts) // 2])
        ax.text(m[0] + lab_dx, m[1] + lab_dy, label, fontsize=FS - 1,
                color=MUTED, ha=ha, va=va, zorder=z + 1, linespacing=1.15,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                          alpha=0.85))


def thumb(ax, x, y, s, name, caption):
    img = plt.imread(ASSETS / name)
    ax.imshow(img, extent=[x, x + s, y, y + s], zorder=3, aspect="auto")
    ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor="0.5",
                           lw=0.6, zorder=4))
    ax.text(x + s / 2, y - 0.06, caption, ha="center", va="top",
            fontsize=FS - 1, color=MUTED, zorder=5)


def main():
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal")
    ax.axis("off")

    # tier bands
    ax.add_patch(Rectangle((0.05, 3.05), W - 0.1, 2.6, facecolor=BAND_TOP,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.05, 0.05), W - 0.1, 2.85, facecolor=BAND_BOT,
                           edgecolor="none", zorder=0))
    ax.text(0.15, 5.55, "Identification  ·  once per episode, re-read at every replan",
            fontsize=FS + 0.5, color=MUTED, va="top", fontweight="bold")
    ax.text(0.15, 2.82, "Enforcement  ·  every action chunk (20 Hz, 5-step executed prefix)",
            fontsize=FS + 0.5, color=MUTED, va="top", fontweight="bold")

    # ---------------- top tier ----------------
    thumb(ax, 0.2, 3.3, 1.75, "obs.jpg", "RGB-D, proprioception, instruction ℓ")
    box(ax, 2.55, 4.3, 2.35, 0.8, "Open-vocabulary detection",
        "GroundingDINO + depth back-projection\npercentile-box extents")
    box(ax, 2.55, 3.3, 2.35, 0.8, "Predicate grounding",
        "small VLM, 3 votes, cached per task\n(or hand heuristics + priors, floor 0.5)")
    arrow(ax, (1.95, 4.5), (2.55, 4.7))
    arrow(ax, (1.95, 3.9), (2.55, 3.7))

    box(ax, 5.6, 3.5, 2.55, 1.4, "FOL hazard election + routing",
        "Hazard(x) := Prot ∨ Mov ∨ (Near ∧ ¬Ment ∧ ¬Tgt)\n"
        "ranked protected-first, top-2 guard set\n"
        "avoid  |  duality disengage  |  abstain")
    arrow(ax, (4.9, 4.7), (5.6, 4.55), "positions $o_m$, extents", lab_dy=0.1)
    arrow(ax, (4.9, 3.7), (5.6, 3.85),
          "Protected, Referenced,\nHot / Sharp, Moving (>1 cm)", lab_dy=-0.12,
          va="top")

    box(ax, 8.85, 3.5, 2.5, 1.4, "Barrier synthesis",
        "$B_j(p)=\\min_{m,q}\\|p+q-o_m\\|-(r_{\\mathrm{eff},m}+\\lambda j)$\n"
        "$r_{\\mathrm{eff}}$ = class envelope + eef 0.09 + 0.01\n"
        "corridor ×0.4 along eef→target/dest; companions ±")
    arrow(ax, (8.15, 4.2), (8.85, 4.2), "guard set $G$,\ncorridor anchors",
          lab_dy=0.1)
    thumb(ax, 11.65, 3.3, 1.75, "guard.jpg", "elected guard (real frame)")
    arrow(ax, (11.35, 4.2), (11.65, 4.2))

    # ---------------- bottom tier ----------------
    box(ax, 0.2, 0.7, 3.9, 1.5, "Frozen π$_{0.5}$ flow map",
        "10 Euler steps, $K{=}8$ noise seeds $z \\sim N(0,I)$,\n"
        "one shared vision-language KV cache\n\n\n")
    # inner module: projection sweep
    ax.add_patch(FancyBboxPatch((0.4, 0.85), 3.5, 0.5,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="#fdf3e7", edgecolor="#b45309",
                                lw=0.9, zorder=5))
    ax.text(2.15, 1.10, "in-denoising projection sweep at step $k$ · brake-then-push · weight $w_k$ (ramp: 0, 0, 0.2→1)",
            ha="center", va="center", fontsize=FS - 1, color="#7a3e05", zorder=6)
    ax.text(0.1, 1.45, "$z$", fontsize=FS, ha="right", va="center", color=MUTED)

    box(ax, 4.85, 0.7, 3.3, 1.5, "Checked prefix certificate + selection",
        "re-roll executed prefix, chain $B_j \\geq (1-\\gamma) B_{j-1}$, γ=0.9\n"
        "score: feasible ≻ min margin ≻ progress\n"
        "all fail → argmax-margin, margin reported", core=True)
    arrow(ax, (4.1, 1.45), (4.85, 1.45), "$K$ candidate\nchunks $a^{(1..K)}$", lab_dy=0.12)

    box(ax, 8.9, 0.7, 1.75, 1.5, "Robot",
        "executes 5-step prefix,\nreplans")
    arrow(ax, (8.15, 1.45), (8.9, 1.45), "certified prefix\n$a_{1:5}$", lab_dy=0.12)
    thumb(ax, 11.65, 0.5, 1.75, "exec.jpg", "execution (real keyframe)")
    arrow(ax, (10.65, 1.45), (11.65, 1.45))

    # barrier -> enforcement (both the sweep and the certificate)
    arrow(ax, (10.1, 3.5), (6.5, 2.3), via=[(10.1, 2.95), (6.5, 2.95)],
          label="$B_j$, corridor, $G$", lab_dx=0.0, lab_dy=0.07)
    arrow(ax, (6.5, 2.95), (2.15, 2.3), via=[(2.15, 2.95)], style="-|>")

    # feedback: robot -> observations (dashed)
    arrow(ax, (9.8, 0.7), (1.07, 3.3),
          via=[(9.8, 0.25), (0.12, 0.25), (0.12, 2.6), (1.07, 2.6)],
          ls=(0, (4, 2.5)), color="0.4")
    ax.text(5.0, 0.27, "observations · eef $p$ · object positions (20 Hz)",
            fontsize=FS - 1, color=MUTED, ha="center", va="bottom", zorder=5,
            bbox=dict(facecolor=BAND_BOT, edgecolor="none", pad=0.6))
    arrow(ax, (0.12, 1.95), (0.2, 1.95), ls=(0, (4, 2.5)), color="0.4")

    for e in ("png", "pdf"):
        p = FIGDIR / f"fig_system_block.{e}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("wrote", p)


if __name__ == "__main__":
    main()
