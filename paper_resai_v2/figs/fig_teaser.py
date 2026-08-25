"""Paper teaser: base pi0.5 violates the hand-attached hazard and times out;
ours (training-free, no-GT) avoids it and completes the task.

Data: the fresh paired render in ls_teaser_render/ -- obstacle_avoidance_human
L2 task 14 ep 1 ("put the white yellow mug in the microwave and close it"),
IDENTICAL initial state (init_states[ep] indexing):
  BASE  base_oah_L2/.../videos/t14_ep1_fail.mp4  (600 steps, timeout,
        sphere-clearance h(t) < 0 for 215 steps, min -0.022 m at t=573)
  OURS  nogt_oah_L2/.../videos/t14_ep1_succ.mp4  (510 steps, success,
        h(t) always > 0, min +0.021 m at t=129)
Per-step geometry from transitions_obstacle_avoidance_human_L2.jsonl
(task==14, ep==1). VIDEO/TRANSITION ALIGNMENT: the mp4s hold one frame per
FIVE env steps (600 steps -> 120 frames, 510 -> 102), so video frame index =
t // 5 (verified by frame count on both arms).

Keyframes (chosen from a contact-sheet pass over both videos):
  base: t=0 start | t=200 approach (gripper at the mug, beside the hand) |
        t=573 closest approach = deepest violation (argmin ||eef-bottle||) |
        t=599 timeout.
  ours: t=0 start | t=129 avoidance moment (its own closest approach --
        base's argmin t=573 exceeds ours' 510-step episode, so the contrast
        frame is ours' own min-clearance step) | t=320 task progress (mug in
        the microwave) | t=509 success (door closed).

Style matches the other paper figures (FS=8 fonts, dpi=300, PNG+PDF, tight).
Needs imageio (run inside the openvla_libero_merged conda env).
Run: python paper/figs/fig_teaser.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGDIR = pathlib.Path(__file__).resolve().parent
RENDER = ROOT / "ls_teaser_render"
TASK, EP = 14, 1
FRAME_STRIDE = 5  # mp4 holds one frame per 5 env steps (verified)
EEF_R = 0.09      # deployed eef sphere radius
FS = 8

ARMS = {
    "base": dict(
        d=RENDER / "base_oah_L2", video="t14_ep1_fail.mp4",
        label="$\\pi_{0.5}$ base (frozen)",
        keyframes=[(0, None, None), (200, None, None),
                   (573, "violation", "crimson"),
                   (599, "fail", "crimson")]),
    "ours": dict(
        d=RENDER / "nogt_oah_L2", video="t14_ep1_succ.mp4",
        label="+ ours (training-free, no-GT)",
        keyframes=[(0, None, None), (129, None, None), (320, None, None),
                   (509, "success", "#1faa4b")]),
}


def load_transitions(arm_dir):
    rows = [json.loads(l) for l in
            open(arm_dir / "transitions_obstacle_avoidance_human_L2.jsonl")]
    rows = [r for r in rows if r["task"] == TASK and r["ep"] == EP]
    assert rows and all(rows[i]["t"] == i for i in range(len(rows)))
    return rows


def clearance(rows):
    eef = np.array([r["eef"] for r in rows])
    bot = np.array([r["entities"]["bottle_of_alcohol_1_with_hand_1"]
                    for r in rows])
    return np.linalg.norm(eef - bot, axis=1) - EEF_R


def main():
    import imageio.v2 as imageio

    frames, meta = {}, {}
    for name, arm in ARMS.items():
        rows = load_transitions(arm["d"])
        h = clearance(rows)
        rd = imageio.get_reader(
            arm["d"] / "obstacle_avoidance_human/L2/videos" / arm["video"])
        n = rd.count_frames()
        assert abs(len(rows) - n * FRAME_STRIDE) < FRAME_STRIDE, (
            f"{name}: {len(rows)} steps vs {n} frames")
        frames[name] = [rd.get_data(min(t // FRAME_STRIDE, n - 1))
                        for t, _, _ in arm["keyframes"]]
        meta[name] = (rows, h)
        print(f"{name}: {len(rows)} steps, {n} frames, "
              f"h_min={h.min():+.4f} at t={int(np.argmin(h))}")

    desc = "put the white yellow mug in the microwave and close it"
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.5), dpi=300)
    for i, (name, arm) in enumerate(ARMS.items()):
        for j, ((t, tag, color), img) in enumerate(
                zip(arm["keyframes"], frames[name])):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                if color is not None:
                    s.set_color(color)
                    s.set_linewidth(2.4)
                else:
                    s.set_color("0.6")
            if tag:
                ax.text(0.04, 0.955, tag, transform=ax.transAxes,
                        fontsize=FS - 0.5, color="white", va="top",
                        fontweight="bold",
                        bbox=dict(facecolor=color, edgecolor="none",
                                  pad=2.2))
            ax.text(0.5, -0.035, f"$t={t}$", transform=ax.transAxes,
                    ha="center", va="top", fontsize=FS - 1)
        axes[i, 0].text(-0.07, 0.5, arm["label"],
                        transform=axes[i, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=FS + 1)
    fig.suptitle(f'“{desc}”', fontsize=FS + 1.5, style="italic", y=0.985)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.94, bottom=0.045,
                        wspace=0.04, hspace=0.14)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"fig_teaser.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print("wrote", p)


if __name__ == "__main__":
    main()
