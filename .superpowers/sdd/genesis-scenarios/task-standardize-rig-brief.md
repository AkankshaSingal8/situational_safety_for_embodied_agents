# Task: Standardize robot pose + camera rig across all Genesis scenarios

## Why
Every scenario script built so far (`scenario_1a_i.py`, `scenario_1b.py`, `scenario_2c.py`, `scenario_3a_i.py`) picked its own agentview camera position/lookat ad hoc — confirmed by direct inspection, all four differ:
- 1a-i: `pos=(1.4, 0.3, TABLE_HEIGHT+0.9), lookat=(0.35, 0.05, TABLE_HEIGHT+0.05)`
- 1b: `pos=(1.5, 0.5, TABLE_HEIGHT+1.0), lookat=(0.35, 0.1, TABLE_HEIGHT+0.05)`
- 2c: same as 1a-i
- 3a-i: `pos=(0.9, -0.6, TABLE_HEIGHT+0.7), lookat=(0.35, 0.0, TABLE_HEIGHT+0.05)`

And none set an explicit robot "ready" starting pose — every scenario relies on Genesis's raw MJCF default init, which the scene-build logs themselves warn is "Neutral robot position (qpos0) exceeds joint limits" — not something to standardize on as-is.

The project's goal is a standardized benchmark (explicit user requirement, similar to LIBERO's convention): every scenario should start the robot from the same pose, and render agentview + eye-in-hand from the same camera convention, so scenarios are visually and quantitatively comparable to each other.

## Deliverable 1: `genesis_bakeoff/standard_rig.py`

A new shared module, following this project's established pattern (see `genesis_bakeoff/placement_sampler.py` / `libero_asset_loader.py` for style — pure, well-documented, reusable functions, not scenario-specific logic):

1. `TABLE_HEIGHT = 0.75` (constant — already the de facto convention across all existing scripts, just needs to live in one place instead of being redefined per-file).

2. `ARM_READY_QPOS` — the standard Franka "ready" pose used throughout the robosuite/LIBERO ecosystem: 7 arm joints `[0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398]` plus gripper open (`[0.04, 0.04]` finger joints, or whatever Genesis's finger-joint convention is — check `franka.get_dofs_limit()`/joint naming from Task 1's `scenario_1b.py` for the exact finger-joint handling already worked out there). Provide `set_robot_ready_pose(franka)` that sets this via `franka.set_qpos(...)` (or the correct Genesis API for directly setting a full-body qpos — check what's actually available; `set_dofs_position` or similar, following patterns already used elsewhere in `genesis_bakeoff/`).

3. `add_standard_agentview_camera(scene, table_center_xy=(0.35, 0.05))` — returns a camera built with FIXED, standardized parameters: `res=(512, 512), pos=(1.4, 0.3, TABLE_HEIGHT + 0.9), lookat=(table_center_xy[0], table_center_xy[1], TABLE_HEIGHT + 0.05), fov=45, GUI=False`. Use 1a-i's original values as the standard (it was the first scenario built and is closest to this project's documented workspace-bounds convention). The `table_center_xy` parameter allows different scenarios to have their "center of action" at slightly different table coordinates while keeping everything else about the camera identical — do not let scenarios override `pos`/`fov`/`res`, only `lookat`'s target point, and even that should default to a single shared standard center unless a scenario has a good reason to differ (document any deviation explicitly in that scenario's own code comments).

4. **Eye-in-hand camera — new capability, doesn't exist anywhere yet.** Genesis has no native "attach camera to a moving link" API (confirmed during the earlier OpenVLA-Genesis spike — `genesis_bakeoff/spike_openvla_e2e.py` worked around this by manually recomputing camera pose from the hand link's position/quaternion each frame). Build `render_eye_in_hand(cam, franka, hand_link_name="hand", offset_local=(0.0, 0.0, 0.05))`: reads `franka.get_link(hand_link_name).get_pos()` / `.get_quat()`, computes a world-frame camera pose offset slightly forward of the gripper along its approach axis (looking in the direction the gripper points, small standoff distance — this mimics a real wrist-mounted camera, not a camera sitting exactly at the gripper's origin), calls `cam.set_pose(pos=..., lookat=...)` (or whatever the actual Genesis camera repositioning API is — find it; if `set_pose` doesn't exist, check the camera object's available methods) each frame before rendering, and returns the rendered RGB frame. Use the same resolution/fov convention as the agentview camera for consistency (`res=(512, 512), fov=45` unless a real wrist camera's typically-narrower FOV is a better match — your call, document the choice).

## Deliverable 2: retrofit existing scenarios to use the standard rig

Update `scenario_1a_i.py`, `scenario_1b.py`, `scenario_2c.py`, `scenario_3a_i.py` to:
- Import and call `set_robot_ready_pose(franka)` right after `scene.build()`.
- Import and call `add_standard_agentview_camera(scene)` instead of each script's own ad hoc `scene.add_camera(...)` call.
- Add a second `render_eye_in_hand(...)` call alongside every existing `cam.render()` call, saving the wrist-view frame alongside the existing agentview frame (same naming convention, e.g. `frame_start_rgb.png` → also save `frame_start_wrist_rgb.png`).
- Re-run each scenario on the cluster (4 SLURM jobs — these are all lightweight/fast scene-render checks, not full metric-generation runs, except re-confirm 1b's metrics didn't change from the ready-pose change, since 1b's joint-limit-margin metric depends on the robot's pose trajectory — if the ready pose changes the scripted motion's starting point, 1b's numeric results may shift; note this explicitly and re-verify 1b's `METRIC_VARIATION_CHECK` still passes with sane values after the retrofit, don't just assume it's unaffected).
- Regenerate output images for all 4 scenarios (both agentview and eye-in-hand) into their existing `*_output/` directories.

## Deliverable 3: realistic-asset swaps (user-requested, do in the same pass)

**1a-i (mug behind tall bottle)** currently uses primitive shapes, not real meshes. Swap in real LIBERO MJCF assets (already on disk, no external sourcing needed, confirmed present):
- Mug: `SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/porcelain_mug/porcelain_mug.xml` (already used in scenario 1b — reuse the same loading pattern) or `red_coffee_mug`/`white_yellow_mug` if a different look fits the scene better, your call.
- Tall bottle: `SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/wine_bottle/wine_bottle.xml` (a real wine-bottle mesh, not a cylinder primitive).
- Load both via `genesis_bakeoff/libero_asset_loader.py`'s `load_libero_object()` (same as every other scenario) — do not hand-roll new loading logic.
- Adjust `REGION_SPECS`/placement as needed for the real meshes' actual footprints (they will differ from the old primitives' dimensions).

**2c (glass tilt over laptop)** currently uses a box primitive standing in for a laptop. Swap in the real laptop mesh now staged at `genesis_bakeoff/external_assets/laptop/laptop.obj` (CC0-licensed, from Kenney.nl's Furniture Kit — license file at `genesis_bakeoff/external_assets/laptop/License.txt`). Load via `gs.morphs.Mesh(file="genesis_bakeoff/external_assets/laptop/laptop.obj", convexify=True)` (not `load_libero_object`, this isn't a LIBERO asset) + `entity.set_mass(...)` if a reasonable laptop mass matters for the CBF hazard-zone logic already in place (check `scenario_2c.py` for whether mass is used in the constraint, or purely geometric — if purely geometric, mass doesn't need special handling). The mesh's scale/orientation as shipped may not match the scene's existing hazard-zone geometry — visually verify the laptop sits flat on the table at a sensible size (not doll-sized or room-sized) and adjust the hazard-zone bounds in `scenario_2c.py` if the real mesh's footprint differs meaningfully from the old box's.

Note: scenario 3b is a separate, already-completed task (commit `d4f6ef681f11173cff3a3467acaf94aaf318e95b`) — do not touch `scenario_3b.py`, it's out of scope here and not part of this retrofit's file list.

## What "done" looks like
- All 4 existing scenarios render from an IDENTICAL camera position/lookat/fov (visually verify: the table and background framing should look the same across scenario screenshots, only the objects on the table differ).
- All 4 show the Franka arm starting from the SAME visible pose in their "start" frames.
- Every scenario now has both an agentview AND an eye-in-hand frame saved.
- 1b's metrics re-verified sane after the ready-pose change (report the new numbers, confirm no regression).
- 1a-i shows a real mug + real wine-bottle mesh (not primitives); 2c shows the real laptop mesh (not a box primitive).
- This module (`standard_rig.py`) becomes a hard requirement going forward — document this in a comment at the top of the file and in your report, so future scenario tasks (4a next, and everything after) are told to use it from the start rather than retrofitted later. Scenario 3b is already done and out of scope for this retrofit.

## Report file
Write your full report to `/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.superpowers/sdd/genesis-scenarios/task-standardize-rig-report.md`. Final message to controller: status, one-line summary, SLURM job IDs (all 4 re-runs), one-line verification summary, commit hash (STATE IT), concerns.

## Committing
Commit only the files you touched (new `standard_rig.py`, `genesis_bakeoff/external_assets/laptop/` if not already tracked, the 4 retrofitted scenario scripts + regenerated output images) by exact path, not `git add -A`. Clear commit message noting this is an infra/standardization change (rig + realistic-asset swap), not a new scenario.

## Process note
Do not idle-poll `squeue` in a tight loop while jobs are pending — submit and step back, the controller is independently monitoring and will resume you with results if needed, or just use a longer poll interval (e.g. sleep 60+ between checks) if you check yourself.
