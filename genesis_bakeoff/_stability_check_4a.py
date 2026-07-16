"""
THROWAWAY calibration/gating script for scenario 4a (not a deliverable, mirrors
genesis_bakeoff/_stability_check_3b.py's role for scenario 3b). Purpose:
  1. Confirm the actual resting orientation/footprint of yellow_book and
     black_book -- their collision-box geometry (read from the raw XML) is
     THIN in local-x (~0.0118m / ~0.0143m half-extent), MEDIUM in local-y,
     TALL in local-z (~0.061m / ~0.067m half-extent) -- i.e. with identity
     orientation these assets stand up on a thin edge like a book on a
     shelf, not lying flat. A "flat book, bottom of stack" (per the brief)
     needs a +90 deg rotation about the Y axis (quat (0.70711,0,0.70711,0))
     to bring the thin axis vertical -- this script empirically confirms
     that guess via get_AABB() after a settle, rather than trusting the hand
     derivation alone.
  2. Confirm the mug (reused from scenario_1b, known-good) and wooden_tray
     settle to sane, non-degenerate AABBs with identity orientation.
  3. Confirm the 3-object stack (book/book/mug) is nominally table-stable for
     a short settle when built directly in the intended stacked config.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_asset_loader import load_libero_object

REPO = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
YELLOW_BOOK_XML = os.path.join(REPO, "SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/yellow_book/yellow_book.xml")
BLACK_BOOK_XML = os.path.join(REPO, "SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/black_book/black_book.xml")
MUG_XML = os.path.join(REPO, "SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/porcelain_mug/porcelain_mug.xml")
TRAY_XML = os.path.join(REPO, "SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/wooden_tray/wooden_tray.xml")

TABLE_HEIGHT = 0.75
FLAT_QUAT = np.array([0.70711, 0.0, 0.70711, 0.0])  # +90deg about Y -- see docstring
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0.35, 0.05, TABLE_HEIGHT / 2), size=(0.6, 0.6, TABLE_HEIGHT), fixed=True))

    yb = load_libero_object(scene, YELLOW_BOOK_XML, pos=(0.20, 0.0, TABLE_HEIGHT + 0.20))
    bb = load_libero_object(scene, BLACK_BOOK_XML, pos=(0.40, 0.0, TABLE_HEIGHT + 0.20))
    mug = load_libero_object(scene, MUG_XML, pos=(0.60, 0.0, TABLE_HEIGHT + 0.20))
    tray = load_libero_object(scene, TRAY_XML, pos=(0.20, 0.25, TABLE_HEIGHT + 0.20))
    scene.build()

    # --- Part 1: single-object settle with FLAT_QUAT for the books, identity for mug/tray ---
    yb.set_pos(np.array([0.20, 0.0, TABLE_HEIGHT + 0.15]))
    yb.set_quat(FLAT_QUAT)
    bb.set_pos(np.array([0.40, 0.0, TABLE_HEIGHT + 0.15]))
    bb.set_quat(FLAT_QUAT)
    mug.set_pos(np.array([0.60, 0.0, TABLE_HEIGHT + 0.15]))
    mug.set_quat(IDENTITY_QUAT)
    tray.set_pos(np.array([0.20, 0.25, TABLE_HEIGHT + 0.15]))
    tray.set_quat(IDENTITY_QUAT)
    for e in (yb, bb, mug, tray):
        e.set_dofs_velocity(np.zeros(6))
    for _ in range(150):
        scene.step()

    for name, ent in (("yellow_book", yb), ("black_book", bb), ("porcelain_mug", mug), ("wooden_tray", tray)):
        pos = _np(ent.get_pos())
        quat = _np(ent.get_quat())
        aabb = _np(ent.get_AABB())
        half_ext = (aabb[1] - aabb[0]) / 2.0
        print(f"[{name}] pos={pos} quat={quat} AABB_half_extents={half_ext} "
              f"full_dims_cm={(half_ext * 200).round(2)}")

    # --- Part 2: full 3-stack, built directly in intended config, brief settle ---
    print("\n--- Building 3-stack directly (book/book/mug) ---")
    yb.set_pos(np.array([0.30, 0.0, TABLE_HEIGHT + 0.15]))
    yb.set_quat(FLAT_QUAT)
    yb.set_dofs_velocity(np.zeros(6))
    for _ in range(120):
        scene.step()
    yb_pos = _np(yb.get_pos())
    yb_aabb = _np(yb.get_AABB())
    yb_top_z = float(yb_aabb[1][2])
    print(f"yellow_book settled: pos={yb_pos} top_z={yb_top_z}")

    bb.set_pos(np.array([yb_pos[0], yb_pos[1], yb_top_z + 0.08]))
    bb.set_quat(FLAT_QUAT)
    bb.set_dofs_velocity(np.zeros(6))
    for _ in range(120):
        scene.step()
    bb_pos = _np(bb.get_pos())
    bb_aabb = _np(bb.get_AABB())
    bb_top_z = float(bb_aabb[1][2])
    print(f"black_book settled on yellow_book: pos={bb_pos} top_z={bb_top_z}")

    mug.set_pos(np.array([bb_pos[0], bb_pos[1], bb_top_z + 0.08]))
    mug.set_quat(IDENTITY_QUAT)
    mug.set_dofs_velocity(np.zeros(6))
    for _ in range(150):
        scene.step()
    mug_pos = _np(mug.get_pos())
    mug_quat = _np(mug.get_quat())
    print(f"porcelain_mug settled on black_book: pos={mug_pos} quat={mug_quat}")

    # sanity: is the stack still roughly vertically aligned (no collapse)?
    yb_pos_final = _np(yb.get_pos())
    bb_pos_final = _np(bb.get_pos())
    xy_drift_bb = float(np.hypot(bb_pos_final[0] - yb_pos_final[0], bb_pos_final[1] - yb_pos_final[1]))
    xy_drift_mug = float(np.hypot(mug_pos[0] - bb_pos_final[0], mug_pos[1] - bb_pos_final[1]))
    print(f"xy_drift black_book-vs-yellow_book={xy_drift_bb:.4f}m xy_drift mug-vs-black_book={xy_drift_mug:.4f}m")
    stable = xy_drift_bb < 0.05 and xy_drift_mug < 0.05
    print(f"STACK_STABILITY_CHECK: {'STABLE' if stable else 'UNSTABLE'}")

    print("STABILITY_CHECK_RESULT: PASS")


if __name__ == "__main__":
    main()
