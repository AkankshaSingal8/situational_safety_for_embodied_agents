"""Paper filmstrip: OURS keyframes over an aligned clearance-vs-time plot.

Same episode pair as fig_teaser.py (ls_teaser_render/, oah L2 task 14 ep 1,
identical init state). Top: 4 keyframes of OURS (t = 0, 129, 320, 509).
Bottom, aligned time axis: sphere clearance
    h(t) = ||eef(t) - bottle_pos(t)|| - 0.09   (deployed eef radius)
for OURS (solid blue) and BASE (gray ghost). h=0 line in red, h<0 shaded.
Base's minimum (t=573, h=-0.022 m -- a real sub-zero crossing, 215 steps
below 0) is marked "base violates"; ours dips to +0.021 m at t=129 and stays
positive. Video frame index = t // 5 (one frame per 5 env steps, verified
against transition counts). Vertical dotted guides connect each keyframe to
its t on the curve.

Style matches the other paper figures (FS=8, dpi=300, PNG+PDF, tight).
Needs imageio (run inside the openvla_libero_merged conda env).
Run: python paper/figs/fig_filmstrip.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(FIGDIR))
from fig_teaser import ARMS, FRAME_STRIDE, clearance, load_transitions  # noqa: E402

FS = 8
KEYFRAMES = [0, 129, 320, 509]  # ours: start, avoidance, mug placed, success
BLUE, GRAY = "#2a78d6", "#898781"


def main():
    import imageio.v2 as imageio

    h = {name: clearance(load_transitions(arm["d"]))
         for name, arm in ARMS.items()}
    arm = ARMS["ours"]
    rd = imageio.get_reader(
        arm["d"] / "obstacle_avoidance_human/L2/videos" / arm["video"])
    n = rd.count_frames()
    imgs = [rd.get_data(min(t // FRAME_STRIDE, n - 1)) for t in KEYFRAMES]

    tb_min = int(np.argmin(h["base"]))
    print(f"base h_min={h['base'].min():+.4f} at t={tb_min}; "
          f"ours h_min={h['ours'].min():+.4f} at t={int(np.argmin(h['ours']))}")

    T = len(h["base"])  # x-axis spans the longer (base) episode
    fig = plt.figure(figsize=(6.9, 4.1), dpi=300)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0],
                          left=0.085, right=0.985, top=0.965, bottom=0.115,
                          wspace=0.05, hspace=0.16)
    frame_axes = []
    for j, (t, img) in enumerate(zip(KEYFRAMES, imgs)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.6")
        ax.set_title(f"$t={t}$", fontsize=FS - 0.5, pad=2.5)
        frame_axes.append(ax)

    axc = fig.add_subplot(gs[1, :])
    axc.plot(np.arange(len(h["base"])), h["base"], color=GRAY, lw=1.3,
             label="$\\pi_{0.5}$ base (frozen)", zorder=3)
    axc.plot(np.arange(len(h["ours"])), h["ours"], color=BLUE, lw=1.6,
             label="+ ours (training-free, no-GT)", zorder=4)
    axc.axhline(0.0, color="crimson", lw=0.9, zorder=2)
    ymin = min(h["base"].min(), h["ours"].min()) - 0.03
    ymax = max(h["base"].max(), h["ours"].max()) + 0.02
    axc.axhspan(ymin, 0.0, facecolor="crimson", alpha=0.08, zorder=1)
    axc.set_xlim(0, T - 1)
    axc.set_ylim(ymin, ymax)
    axc.set_xlabel("env step $t$", fontsize=FS)
    axc.set_ylabel("clearance $h(t)$ [m]", fontsize=FS)
    axc.tick_params(labelsize=FS - 1)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    axc.plot(tb_min, h["base"][tb_min], marker="o", ms=4.5, mfc="crimson",
             mec="crimson", zorder=5)
    axc.annotate(f"base violates\n($h={h['base'][tb_min]:+.3f}$ m)",
                 (tb_min, h["base"][tb_min]), xytext=(-8, 26),
                 textcoords="offset points", fontsize=FS - 1.5, ha="right",
                 color="crimson", linespacing=1.15,
                 arrowprops=dict(arrowstyle="-", color="crimson", lw=0.7))
    axc.legend(loc="upper left", fontsize=FS - 1, frameon=False,
               borderaxespad=0.2, handlelength=1.6)

    # dotted guides: keyframe bottom-center -> its t on the ours curve
    for ax, t in zip(frame_axes, KEYFRAMES):
        fig.canvas.draw()  # positions are final (fixed gridspec, no tight_layout)
        top = axc.transData.transform((t, h["ours"][min(t, len(h["ours"]) - 1)]))
        bot_ax = ax.transAxes.transform((0.5, 0.0))
        inv = fig.transFigure.inverted()
        (x0, y0), (x1, y1) = inv.transform([bot_ax, top])
        fig.add_artist(matplotlib.lines.Line2D([x0, x1], [y0, y1],
                                               transform=fig.transFigure,
                                               color="0.55", lw=0.7,
                                               linestyle=(0, (1.5, 2.2)),
                                               zorder=0))
    for ext in ("png", "pdf"):
        p = FIGDIR / f"fig_filmstrip.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("wrote", p)


if __name__ == "__main__":
    main()
