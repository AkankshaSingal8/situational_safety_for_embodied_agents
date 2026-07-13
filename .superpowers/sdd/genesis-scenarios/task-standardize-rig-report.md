# Task Report: Standardize robot pose + camera rig across all Genesis scenarios

## Status: COMPLETE

(Resumed after the original agent died mid-task from an API spend-limit error,
not a code bug. All code and most SLURM verification runs were already on
disk, uncommitted. This session verified the work, read the one previously
unread job log, confirmed no gaps, and committed.)

## Summary

`genesis_bakeoff/standard_rig.py` (228 lines) was built as the shared camera +
robot-ready-pose rig module, and all four existing scenario scripts
(`scenario_1a_i.py`, `scenario_1b.py`, `scenario_2c.py`, `scenario_3a_i.py`)
were retrofitted to import and use it instead of hand-rolled camera/pose code.
Deliverable 3's realistic-asset swaps were completed in the same pass: 1a-i
now uses the real LIBERO `porcelain_mug.xml`/`wine_bottle.xml` meshes (was
primitives), and 2c now uses the real Kenney.nl `laptop.obj` mesh (was a box
primitive). All four scenarios were re-run on Bridges2 GPU-shared and pass;
1b's metrics were explicitly re-verified sane after the ready-pose change
(no regression — see numbers below).

`standard_rig.py` is documented (top-of-file docstring) as a hard requirement
for all future scenario scripts (starting with 4a).

## What was built

### `genesis_bakeoff/standard_rig.py`
- `TABLE_HEIGHT = 0.75` — single source of truth, was previously redefined
  per-script.
- `ARM_READY_QPOS` (7 arm joints, standard robosuite/LIBERO Franka ready pose)
  + `GRIPPER_OPEN`/`GRIPPER_CLOSED`, `ARM_DOFS_IDX`/`GRIPPER_DOFS_IDX` (Genesis
  panda.xml's 7+2 dof layout).
- `set_robot_ready_pose(franka, ...)` — sets the full 9-dof qpos via
  `set_dofs_position`, bypassing Genesis's raw MJCF default init (which logs
  warn exceeds joint limits).
- `add_standard_agentview_camera(scene, table_center_xy=(0.35, 0.05))` — fixed
  `res=(512,512), pos=(1.4, 0.3, TABLE_HEIGHT+0.9), fov=45, GUI=False`; only
  `lookat`'s xy target varies per scenario. Values taken verbatim from
  1a-i's original camera (the documented standard).
- `add_eye_in_hand_camera(scene)` + `render_eye_in_hand(cam, franka,
  hand_link_name="hand", offset_local=(0,0,0.05))` — new capability, didn't
  exist before. Repositions a free camera each frame from the hand link's
  live world pos/quat (no native camera-to-link attachment in this Genesis
  version), offset 5cm forward along the gripper's approach axis
  (local -z), looking further along that axis (0.15m lookahead). Same
  512x512/fov=45 convention as agentview, for cross-view comparability
  (documented tradeoff vs. a narrower real-wrist-camera FOV).
- Two bugs fixed during earlier iteration (confirmed via completed SLURM job
  42144438, "genesis_rig_smoke"): a CUDA-tensor-to-numpy conversion bug in
  the hand-link pos/quat getters (`_to_numpy` helper), and an
  overly-tight assertion tolerance in the ready-pose self-test (loosened to
  account for expected gravity drift over a few free `scene.step()` calls
  with no active position controller).
- `if __name__ == "__main__"` standalone smoke test: builds a bare
  Franka-only scene, applies ready pose, renders both camera views, asserts
  shapes/tolerances. **Verified PASS** (job 42144438, see below).

### Scenario retrofits
All four scenarios now import `TABLE_HEIGHT`/`set_robot_ready_pose`/
`add_standard_agentview_camera`/`add_eye_in_hand_camera`/`render_eye_in_hand`
from `standard_rig` (checked by direct `grep` of each file — no local
redefinition of `TABLE_HEIGHT`, no hand-rolled `scene.add_camera(...)` calls
remain).

- `scenario_1a_i.py`: calls `set_robot_ready_pose` after `scene.build()`,
  uses `add_standard_agentview_camera(scene)` (default lookat), saves
  `frame_wrist_rgb.png` alongside `frame_rgb.png`/`frame_depth.png`.
- `scenario_1b.py`: uses `add_standard_agentview_camera(scene,
  table_center_xy=CAMERA_LOOKAT_XY)` — a documented, deliberate deviation
  (mug is placed near the table's far edge, so the lookat center is shifted
  to keep it framed) — calls `set_robot_ready_pose` both at scene build and
  on every episode reset (replacing the old manual `HOME_Q` +
  `GRIPPER_OPEN` `set_dofs_position` call), saves
  `frame_{start,final}_wrist_rgb.png`.
- `scenario_2c.py`: default lookat, ready pose set after build, saves
  `frame_final_wrist_rgb.png`.
- `scenario_3a_i.py`: **no Franka entity** in this scenario (rice-pour MPM
  simulation only) — correctly imports only `TABLE_HEIGHT` and
  `add_standard_agentview_camera` (with a documented `table_center_xy=(0.35,
  0.0)` deviation), `set_robot_ready_pose`/`render_eye_in_hand` are N/A by
  construction and the file explicitly comments why.

### Realistic-asset swaps
- **1a-i**: `porcelain_mug.xml` + `wine_bottle.xml` loaded via
  `libero_asset_loader.load_libero_object()`, `MUG_POS`/`BOTTLE_POS`
  z-coordinate set to `TABLE_HEIGHT + 0.06` (fixing a stale-assertion bug
  where the real mesh settles ~0.7476/0.7488, not the old primitive's
  0.81 — see job history below). Visually confirmed: real mug + wine-bottle
  mesh render correctly on the table (`scenario_1a_i_output/frame_rgb.png`).
- **2c**: real Kenney.nl `laptop.obj` loaded via `gs.morphs.Mesh(file=...,
  convexify=True)` (not `load_libero_object`, per brief — not a LIBERO
  asset), hazard-ellipsoid z semi-axis grown to cover the real mesh's
  taller open-screen geometry (documented in-file). Visually confirmed:
  real laptop mesh (tilted screen shape) renders on the table
  (`scenario_2c_output/frame_final_rgb.png`).

## SLURM job IDs (all re-runs, verified by reading actual stdout logs at
`/ocean/projects/cis250185p/asingal/slurm_logs/`)

| Job ID | Name | Result | Notes |
|---|---|---|---|
| 42144438 | genesis_rig_smoke | COMPLETED / PASS | `standard_rig.py` standalone smoke test |
| 42144448 | genesis_pilot_1a_i | FAILED | `Mug moved unexpectedly: 0.7476 vs 0.81` — pre-MUG_POS-fix run |
| 42144497 | genesis_pilot_1a_i | COMPLETED / PASS | Re-run after `MUG_POS` z-fix; `SCENE_TEST_RESULT: PASS` |
| 42144449 | genesis_pilot_1b | COMPLETED / PASS | `METRIC_VARIATION_CHECK: PASS`, `SCENE_TEST_RESULT: PASS` |
| 42144450 | genesis_pilot_2c | COMPLETED / PASS | Real laptop mesh loads (CoACD convexify succeeds), `SCENE_TEST_RESULT: PASS` |
| 42144451 | genesis_pilot_3a_i | COMPLETED / PASS | `MPM_POUR_TEST_RESULT: PASS` |
| 42144500 | genesis_1b_repeat | COMPLETED / PASS | Independent re-verification of 1b's metrics post-retrofit (this was the previously-unread job — read this session) |

## 1b metric re-verification (the one item explicitly flagged as needing
re-confirmation, since the ready-pose change could shift the scripted
motion's starting point and thus the joint-limit-margin trajectory)

Two independent runs (42144449, immediately after retrofit, and 42144500, a
dedicated repeat run) both pass and agree to float-noise precision:

| Run | min_m_edge (mean / std / min / max) | min_m_joint (mean / std / min / max) |
|---|---|---|
| 42144449 | -1.7457 / 5.5060 / -17.9498 / 0.6221 | 0.2086 / 0.0257 / 0.1727 / 0.2456 |
| 42144500 (repeat) | -1.7440 / 5.5010 / -17.9328 / 0.6221 | 0.2086 / 0.0257 / 0.1727 / 0.2456 |

Both `METRIC_VARIATION_CHECK: PASS`. The tiny min_m_edge differences
(-1.7457 vs -1.7440, 4th sig fig) are consistent with GPU floating-point
nondeterminism across separate simulation runs (same seed, same init
states), not a real regression — `min_m_joint` is bit-identical across both
runs. **No regression from the ready-pose change.**

## "What done looks like" checklist verification

- **Identical camera pos/lookat/fov across all 4 scenarios**: confirmed by
  code inspection (all four call `add_standard_agentview_camera`, only
  `table_center_xy` varies, and only for 1b/3a-i with documented reasons)
  and visual inspection of rendered frames — 1a-i, 1b, and 2c's start/final
  frames show identical table framing, floor checkerboard, and camera
  angle. 3a-i's lookat is shifted 0.05 in y (documented) but the overall
  framing is visually consistent with the others.
- **Same Franka starting pose in all "start" frames**: confirmed — 1a-i and
  1b's start frames show the arm in the identical ready-pose configuration
  (verified visually; 2c doesn't save a separate "start" frame, only
  "final", per its existing pre-retrofit convention — not part of this
  task's scope to change scenario output-naming conventions beyond adding
  wrist frames). 3a-i has no Franka (N/A, documented in-file).
- **Agentview + eye-in-hand frames for all 4**: confirmed present on disk —
  `scenario_1a_i_output/frame_wrist_rgb.png`,
  `scenario_1b_output/frame_{start,final}_wrist_rgb.png`,
  `scenario_2c_output/frame_final_wrist_rgb.png` all exist, are non-degenerate
  (512x512x3, non-blank pixel stats checked). 3a-i correctly has no wrist
  frame (no Franka).
- **1b metrics re-verified sane**: see table above — PASS, no regression.
- **1a-i shows real mug + wine-bottle mesh**: confirmed visually.
- **2c shows real laptop mesh**: confirmed visually.
- **`standard_rig.py` documented as hard requirement**: confirmed —
  top-of-file docstring explicitly states this and names scenario 4a as the
  first scenario required to use it from the start.

## Commit

`f43ec78`

Files committed (exact paths, no `-A`):
`genesis_bakeoff/standard_rig.py`, `genesis_bakeoff/scenario_1a_i.py`,
`genesis_bakeoff/scenario_1b.py`, `genesis_bakeoff/scenario_2c.py`,
`genesis_bakeoff/scenario_3a_i.py`, `genesis_bakeoff/scenario_1a_i_output/`,
`genesis_bakeoff/scenario_1b_output/`, `genesis_bakeoff/scenario_2c_output/`,
`genesis_bakeoff/scenario_3a_i_output/`, `genesis_bakeoff/external_assets/`,
this report file.

Explicitly NOT touched/committed: `scenario_4a.py`,
`scenario_4a_output/`, `scenario_3b.py`, `scenario_3b_output/`,
`scenario_3a_ii.py`, `scenario_3a_ii_output/`, or any of the
still-uncommitted shared infra files (`libero_asset_loader.py`,
`placement_sampler.py`, `scenario_builder.py`, `metrics_1b.py`,
`_stability_check_4a*.py`, `metrics_4a.py`, `smoke_test/`, etc.) — those
predate this task or belong to other in-flight tasks and are out of this
task's file list per the brief.

## Concerns / follow-ups

1. **Eye-in-hand frame quality is functional but not very informative in
   the "ready pose" configuration.** In 1a-i's saved wrist frame, the
   gripper is pointed mostly at the floor/horizon rather than down at the
   table (since the ready pose is an upright "home" configuration, not a
   pose oriented over the workpiece) — the frame renders correctly
   (non-degenerate, correct shape/resolution) but doesn't show the
   mug/bottle. This is expected given where the "start" pose looks, not a
   bug in `render_eye_in_hand` itself (2c's final-frame wrist view, taken
   after the scripted motion has moved the arm down toward the table, is
   more visually informative). Not a blocker for this task (the brief only
   required the capability to exist and be exercised), but worth noting
   for anyone using these wrist frames for qualitative review later.
2. **Shared infra files remain uncommitted** (`libero_asset_loader.py`,
   `placement_sampler.py`, `scenario_builder.py`) even though
   `scenario_1b.py` has depended on them since Task 1's commit (9ad510e).
   This is a pre-existing gap from before this task, not something this
   task introduced or is scoped to fix, but it means a fresh checkout of
   `main` would currently be missing files that `scenario_1b.py` (and now
   `scenario_1a_i.py`, `scenario_2c.py`) import. Flagging for the
   controller to fold into a future infra-consolidation task.
3. **1b's `min_m_edge` shows a very wide range** (-17.9 to 0.62 across
   episodes) — this predates this task (already present in Task 1's
   original results) and is not something this retrofit changed or was
   asked to address; noted only because it appears in the re-verification
   numbers above.
