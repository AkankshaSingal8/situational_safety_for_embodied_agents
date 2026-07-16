"""
THROWAWAY follow-up calibration/gating script (not a deliverable). Verifies
the fix applied to scenario_4a.py after job 42144436's first full-pipeline
attempt showed 8/10 episodes with the mug spontaneously toppling during
stack construction: mug z-offsets are now the literal known-good XML site
values (bottom_site=-0.06, top_site=+0.04, matching scenario_1b.py's
convention) instead of an empirically-settled AABB read, and all idle-settle
windows around the mug were shortened from 60-100 steps to 15 steps.

This script directly imports scenario_4a's real build_scene/calibrate_offsets
and replicates its actual stack-construction sequence (yellow_book ->
black_book -> mug, each placed with the SAME offsets/clearances/settle-step
counts scenario_4a.py's run_episode() uses), then checks the mug's tip angle
across 10 repeated stack-construction attempts (fresh reset each time) to
confirm the fix actually holds, before re-submitting the expensive full
10-episode robot-motion pipeline job again.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_4a import (
    build_scene, calibrate_offsets, FLAT_QUAT, IDENTITY_QUAT, STACK_XY_RANGE,
)
from metrics_4a import tip_angle_deg


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    scene, franka, yellow_book, black_book, mug, tray, cam, wrist_cam = build_scene(show_viewer=False)
    offsets = calibrate_offsets(scene, yellow_book, black_book, mug, tray)
    print(f"offsets={offsets}")

    n_trials = 10
    tip_results = []
    for trial in range(n_trials):
        xy = np.array(STACK_XY_RANGE[0]) + trial * 0.001  # tiny xy jitter per trial, avoids identical resets
        yellow_book.set_pos(np.array([xy[0], xy[1], 0.75 + 0.02]))
        yellow_book.set_quat(FLAT_QUAT)
        yellow_book.set_dofs_velocity(np.zeros(6))
        for _ in range(20):
            scene.step()
        yb_pos = _np(yellow_book.get_pos())
        yb_top_z = yb_pos[2] + offsets["yellow_book_top"]

        black_book.set_pos(np.array([yb_pos[0], yb_pos[1], yb_top_z + offsets["black_book_bottom"] + 0.006]))
        black_book.set_quat(FLAT_QUAT)
        black_book.set_dofs_velocity(np.zeros(6))
        for _ in range(20):
            scene.step()
        bb_pos = _np(black_book.get_pos())
        bb_top_z = bb_pos[2] + offsets["black_book_top"]

        mug.set_pos(np.array([bb_pos[0], bb_pos[1], bb_top_z + offsets["mug_bottom"] + 0.006]))
        mug.set_quat(IDENTITY_QUAT)
        mug.set_dofs_velocity(np.zeros(6))
        for _ in range(15):
            scene.step()

        mug_tip = tip_angle_deg(tuple(float(v) for v in _np(mug.get_quat())))
        mug_pos = _np(mug.get_pos())
        tip_results.append(mug_tip)
        print(f"trial {trial}: mug_pos={mug_pos} mug_tip_deg={mug_tip:.2f}")

    tip_arr = np.array(tip_results)
    n_bad = int(np.sum(tip_arr > 30.0))
    print(f"\nSummary over {n_trials} trials: mean_tip={tip_arr.mean():.2f} max_tip={tip_arr.max():.2f} "
          f"n_bad(>30deg)={n_bad}/{n_trials}")
    if n_bad == 0:
        print("MUG_FIX_CHECK: PASS")
    else:
        print("MUG_FIX_CHECK: FAIL")


if __name__ == "__main__":
    main()
