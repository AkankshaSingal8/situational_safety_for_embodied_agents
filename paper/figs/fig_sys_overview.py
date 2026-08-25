"""Fig. 2b -- system overview on the teaser template.

Two rows (frozen pi0.5 base / + ours), four REAL video keyframes each with the
REAL executed end-effector path projected into the camera and overlaid (trail
up to the frame's timestep), and a right column with the top-down executed
paths and the real clearance h(t). Column 1 carries the FOL election overlay
(elected guard in crimson, target in green) evaluated by the deployed code.

Data: ls_teaser_render oah L2 task 14 ep 1 (paired init state), identical to
fig_teaser.py (see its docstring for the video/transition alignment: one
frame per 5 env steps). Camera intrinsics/extrinsics from the ep7 scene cache
(same task, fixed agentview camera; 512 px, video is 256 px -> scale 0.5).
Projection check: the current-eef marker lands on the gripper tip at every
keyframe on both arms.

Run inside an env with imageio (openvla_libero_merged):
  python paper/figs/fig_sys_overview.py
Outputs paper/figs/fig_sys_overview.{png,pdf}
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

FIGDIR = pathlib.Path(__file__).resolve().parent
ROOT = FIGDIR.parents[1]
sys.path.insert(0, str(FIGDIR))
sys.path.insert(0, str(ROOT / "vlm_pipeline"))
import fig_mechanism as fm                                   # noqa: E402
import visual_conditioning as vc                             # noqa: E402
from fig_teaser import ARMS, FRAME_STRIDE, clearance, load_transitions  # noqa: E402

FS = 8
CRIMSON, GREEN = "crimson", "#1faa4b"
BLUE, GRAY = "#2a78d6", "#5c5c5c"
HAZ, TGT = "bottle_of_alcohol_1_with_hand_1", "white_yellow_mug_1"
GUARD_R = 0.20   # drawn envelope for the elected guard (covers the hand+bottle)
TGT_R = 0.075    # mug body incl. handle
CLEAR_R = 0.09   # eef sphere radius: violation <=> ||eef - bottle|| < 0.09
COLORS = {"base": GRAY, "ours": BLUE}


def project(K, ext, pts, sc):
    rc, z = vc._project(K, ext, np.asarray(pts, float).reshape(-1, 3))
    return rc[:, 1] * sc, rc[:, 0] * sc, z


def main():
    import imageio.v2 as iio
    scene, K, ext, simg = fm.load_scene()
    cands = {k: np.asarray(v, float) for k, v in scene["obj_pos"].items()}
    _, guard, tgt, _ = fm.election_truth_table(
        scene["language"], cands, np.asarray(scene["eef_pos"], float))

    data = {}
    for name, arm in ARMS.items():
        rows = load_transitions(arm["d"])
        rd = iio.get_reader(arm["d"] / "obstacle_avoidance_human/L2/videos" / arm["video"])
        n = rd.count_frames()
        frames = [rd.get_data(min(t // FRAME_STRIDE, n - 1)) for t, _, _ in arm["keyframes"]]
        data[name] = (rows, clearance(rows), frames)
    sc = data["base"][2][0].shape[0] / simg.shape[0]

    fig = plt.figure(figsize=(13.4, 5.5), dpi=300)
    gs = GridSpec(2, 4, figure=fig, left=0.035, right=0.745, top=0.93,
                  bottom=0.05, wspace=0.05, hspace=0.16)
    gr = GridSpec(2, 1, figure=fig, left=0.815, right=0.995, top=0.93,
                  bottom=0.05, hspace=0.32)

    for i, (name, arm) in enumerate(ARMS.items()):
        rows, h, frames = data[name]
        P = np.array([r["eef"] for r in rows])
        col = COLORS[name]
        for j, ((t, tag, tcol), img) in enumerate(zip(arm["keyframes"], frames)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img)
            ax.set_xlim(0, img.shape[1]), ax.set_ylim(img.shape[0], 0)
            ax.set_xticks([]), ax.set_yticks([])
            # executed path up to t, faded tail -> solid head
            x, y, _ = project(K, ext, P[:t + 1], sc)
            if t > 0:
                segs = 6
                idx = np.linspace(0, t, segs + 1).astype(int)
                for s in range(segs):
                    a, b = idx[s], idx[s + 1] + 1
                    ax.plot(x[a:b], y[a:b], color=col, lw=1.8,
                            alpha=0.4 + 0.6 * (s + 1) / segs, zorder=5,
                            solid_capstyle="round")
            ax.plot(x[-1], y[-1], "o", ms=4.5, mfc=col, mec="white", mew=0.8, zorder=6)
            if j == 0:
                # FOL election overlay (deployed code): guard crimson, target green
                for k in guard:
                    gx, gy, z = project(K, ext, cands[k], sc)
                    r = vc._pixel_radius(K, ext, cands[k], GUARD_R) * sc
                    for f, a in ((1.0, 0.22), (0.7, 0.18), (0.4, 0.18)):
                        ax.add_patch(Circle((gx[0], gy[0]), r * f, facecolor=CRIMSON,
                                            alpha=a, edgecolor="none", zorder=3))
                    ax.add_patch(Circle((gx[0], gy[0]), r, facecolor="none",
                                        edgecolor=CRIMSON, lw=1.4, zorder=3))
                    ax.text(gx[0], gy[0] - r - 3, "hazard", fontsize=FS - 1.5, color="white",
                            ha="center", va="bottom", fontweight="bold", zorder=7,
                            bbox=dict(facecolor=CRIMSON, edgecolor="none", pad=1.6))
                tx, ty, _ = project(K, ext, cands[tgt], sc)
                rt = vc._pixel_radius(K, ext, cands[tgt], TGT_R) * sc
                ax.add_patch(Circle((tx[0], ty[0]), rt, facecolor=GREEN, alpha=0.22,
                                    edgecolor=GREEN, lw=1.4, zorder=3))
                ax.text(tx[0], ty[0] + rt + 2, "target", fontsize=FS - 1.5, color="white",
                        ha="center", va="top", fontweight="bold", zorder=7,
                        bbox=dict(facecolor=GREEN, edgecolor="none", pad=1.6))
            if tag == "violation":
                # clearance sphere around the hand-held bottle at this step
                b = np.asarray(rows[t]["entities"][HAZ])
                bx, by, _ = project(K, ext, b, sc)
                rb = vc._pixel_radius(K, ext, b, CLEAR_R) * sc
                ax.add_patch(Circle((bx[0], by[0]), rb, facecolor=CRIMSON, alpha=0.25,
                                    edgecolor=CRIMSON, lw=1.2, zorder=4))
            for s in ax.spines.values():
                s.set_color(tcol if tcol else "0.6")
                s.set_linewidth(2.4 if tcol else 0.8)
            if tag:
                ax.text(0.04, 0.955, tag, transform=ax.transAxes, fontsize=FS - 0.5,
                        color="white", va="top", fontweight="bold",
                        bbox=dict(facecolor=tcol, edgecolor="none", pad=2.2))
            ax.text(0.5, -0.035, f"$t={t}$", transform=ax.transAxes, ha="center",
                    va="top", fontsize=FS - 1)
            if j == 0:
                ax.text(-0.07, 0.5, arm["label"], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=FS + 1)

    # right column, top: top-down executed paths (both arms)
    ax = fig.add_subplot(gr[0, 0])
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_facecolor("#f3ede2")
    for s in ax.spines.values():
        s.set_color("0.6")
    rb, hb, _ = data["base"]
    Pb = np.array([r["eef"] for r in rb]); Po = np.array([r["eef"] for r in data["ours"][0]])
    iv = int(np.argmin(hb))
    gc = np.asarray(rb[iv]["entities"][HAZ])[:2]
    mug0 = np.asarray(data["ours"][0][0]["entities"][TGT])[:2]
    for f, a in ((1.0, 0.22), (0.7, 0.18), (0.4, 0.18)):
        ax.add_patch(Circle(gc, CLEAR_R * f, facecolor=CRIMSON, alpha=a, edgecolor="none", zorder=3))
    ax.add_patch(Circle(gc, CLEAR_R, facecolor="none", edgecolor=CRIMSON, lw=1.3, zorder=3))
    ax.add_patch(Circle(mug0, TGT_R, facecolor=GREEN, alpha=0.22, edgecolor=GREEN, lw=1.3, zorder=3))
    ax.plot(Pb[:, 0], Pb[:, 1], color=GRAY, lw=1.1, zorder=4, label="base")
    ax.plot(Po[:, 0], Po[:, 1], color=BLUE, lw=1.7, zorder=5, label="ours")
    ax.plot(Pb[iv, 0], Pb[iv, 1], "x", ms=7, mew=2, color=CRIMSON, zorder=6)
    ax.plot(Po[0, 0], Po[0, 1], "^", ms=6, mfc="k", mec="k", zorder=6)
    keyp = np.vstack([gc, mug0, Po[0, :2]])
    cx, cy = keyp.mean(axis=0)
    half = 0.5 * max(np.ptp(keyp[:, 0]), np.ptp(keyp[:, 1])) + 0.13
    ax.set_xlim(cx - half, cx + half), ax.set_ylim(cy - half, cy + half)
    ax.legend(fontsize=FS - 1.5, frameon=False, loc="upper right", handlelength=1.4)
    ax.set_title("executed end-effector paths (top-down)", fontsize=FS - 0.5, pad=3)

    # right column, bottom: real h(t)
    ax = fig.add_subplot(gr[1, 0])
    ho = data["ours"][1]
    ax.plot(hb, color=GRAY, lw=1.0, label="base")
    ax.plot(ho, color=BLUE, lw=1.5, label="ours")
    ax.axhline(0, color=CRIMSON, lw=0.9)
    ax.fill_between(range(len(hb)), np.minimum(hb, 0), 0, color=CRIMSON, alpha=0.3)
    ax.plot(iv, hb[iv], "x", ms=7, mew=2, color=CRIMSON)
    ax.set_xlabel("timestep $t$", fontsize=FS - 1, labelpad=1)
    ax.set_ylabel("clearance $h(t)$ [m]", fontsize=FS - 1, labelpad=2)
    ax.tick_params(labelsize=FS - 2, length=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("clearance to the hazard", fontsize=FS - 0.5, pad=3)

    fig.suptitle(f'“{scene["language"]}”', fontsize=FS + 1.5, style="italic", y=0.985)
    for e in ("png", "pdf"):
        p = FIGDIR / f"fig_sys_overview.{e}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("wrote", p)
    print(f"base min h {hb.min():+.3f} at t={iv}; ours min h {ho.min():+.3f}")


if __name__ == "__main__":
    main()
