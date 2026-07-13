# Task 5 Report: Fix asset-realism issues in 1a-i and 2c

## Status: COMPLETE

All three issues from the brief were diagnosed and fixed. Both scenarios
were re-run on Bridges2 GPU-shared after the fixes; both pass their existing
scripted checks with no regressions, and the rendered frames were visually
(and, for issue 1, pixel-measurement) verified to address the repo owner's
feedback.

## Issue 1: 1a-i bottle should look bigger/taller than the mug

**Diagnosis: NOT a mesh-scale bug. NOT a tip-over/instability bug. It was a
genuine camera-framing/relative-scale perception effect**, root-caused with
Genesis's own ground-truth data rather than assumption:

- A diagnostic SLURM run (job 42162110) settled both objects and read back
  `entity.get_quat()` / a manual tip-angle computation: both mug and bottle
  settle at ~0.00deg tip angle (quat ~= identity) -- neither tips over.
- A second diagnostic run (job 42162147) read `entity.get_AABB()` directly
  from Genesis post-settle (ground truth of what's actually rendered, not a
  re-derivation from the raw MJCF): **mug height = 0.8561 - 0.7500 =
  0.1061m; bottle height = 0.9069 - 0.7499 = 0.1570m** -- the bottle IS
  ~48% taller than the mug, matching the brief's XML-derived expectation.
  I independently cross-checked this against `mujoco.MjModel.from_xml_string`
  + `mj.mesh_vert` (i.e. mujoco's own compiler output, which is what
  Genesis's MJCF loader reads mesh vertices from) and confirmed the `<mesh
  scale="0.5 0.5 0.5">` attribute IS correctly baked into the compiled mesh
  vertices before Genesis ever sees them -- so there is no scale-loss bug in
  `genesis_bakeoff/libero_asset_loader.py` or Genesis's MJCF import path.
- So the geometry loaded into the scene was always correct. The *rendered
  frame* still didn't read that way to a human because: (a) the mug's wide,
  handle-inclusive footprint (world AABB up to ~0.13m across) visually
  dominates the frame against the bottle's narrow ~0.04m-diameter
  silhouette at the standard rig's fixed, fairly distant, elevated-angle
  camera, and (b) with the old placement (`BOTTLE_POS` y=0.14, only 0.15m
  from `MUG_POS`), the mug's cast shadow extended toward the bottle,
  reducing its visual distinctness. I confirmed this directly by
  pixel-measuring the pre-fix render: the bottle's on-screen height was
  visibly *shorter* than the mug's despite being physically taller.

**Fix** (`genesis_bakeoff/scenario_1a_i.py`): moved `BOTTLE_POS`'s y from
0.14 to 0.20. This (a) widens mug/bottle separation from 0.15m to 0.21m so
the mug's shadow no longer crosses the bottle, and (b) moves the bottle a
few cm closer to the camera's y=0.3 while the mug stays equally far, so
real perspective now reinforces the bottle's already-correct height
advantage rather than working against it. No mesh, scale, or collision
geometry was touched -- this is purely a placement change. Per-brief
requirement, the standard camera rig (`standard_rig.py`) was NOT modified.

**Verification**: re-ran the real scenario (not a diagnostic) on SLURM (job
42162159, `SCENE_TEST_RESULT: PASS`, mug/bottle settle at the same
z=0.7476/0.7488 as before -- the existing height-plausibility assertions are
unaffected by the y-only placement change). Pixel-measured the new render
(`genesis_bakeoff/scenario_1a_i_output/frame_rgb.png`): bottle spans ~140px
top-to-bottom vs. the mug's ~65px -- the bottle now unambiguously reads as
taller.

## Issue 2: 2c laptop orientation

**Diagnosis: a genuine axis-mapping bug**, not a placement/camera issue.
Scanning every `v` line in `genesis_bakeoff/external_assets/laptop/laptop.obj`
by hand showed: the base/keyboard-deck vertices have local `y` in
`{0, 0.0132}` (a thin ~1.3cm slab) while spanning local `z` from 0 to
~0.17; the screen vertices rise in local `y` up to 0.1618094 as `z`
increases further to 0.2400652 (the screen tilts backward in depth as it
goes up). **The mesh's native "up" axis is local Y, not Z as the
pre-existing code assumed** -- local Z is actually front-to-back depth. The
original code loaded the mesh with the identity quat, which put the depth
axis into world-vertical -- exactly the "floating diagonal panel" artifact
in the pre-fix render.

**Fix** (`genesis_bakeoff/scenario_2c.py`): load the entity with
`euler=(90.0, 0.0, 0.0)` (scipy extrinsic x-y-z convention; verified via
`scipy.spatial.transform.Rotation.from_euler('xyz',[90,0,0])` that this maps
local +Y -> world +Z and local +Z -> world -Y). Recomputed
`LAPTOP_MESH_POS`'s corner->center offset, `LAPTOP_ELLIPSOID_CENTER`/
`LAPTOP_ELLIPSOID_SEMI_AXES`, and `PLACE_TARGET` against the corrected axis
mapping (true height 0.1618094m, not the pre-fix code's mislabeled 0.24m;
true depth 0.2400652m, width 0.264m unchanged since X is rotation-invariant
here).

**Verification**: re-ran on SLURM (job 42162160, `SCENE_TEST_RESULT: PASS`,
`CBF_PORTABILITY_RESULT: NEVER_ACTIVATED` -- expected, since the corrected,
now-larger hazard ellipsoid was already never triggered by this scripted
straight-line trajectory in the pre-fix run either; the CBF math itself was
untouched). Rendered frame (`scenario_2c_output/frame_final_rgb.png`) now
shows the laptop sitting flat on the table with its screen naturally tilted
back, base flush -- not floating, sideways, or upside-down.

## Issue 3: 2c glass should look like glass, filled with water

**Sourcing decision: used the brief's documented fallback option (b)
(convincing primitive) rather than sourcing a new mesh.** Re-confirmed
LIBERO's asset library has no glass/cup/tumbler mesh (`stable_scanned_objects/`,
`turbosquid_objects/`). Checked Kenney.nl's "Furniture Kit" page (the source
of the laptop mesh) via web search/fetch -- it does not clearly list a
glass/cup asset, and downloading+staging+license-checking a brand-new
external mesh carries real integration risk (as issue 2 itself just
demonstrated: even a "vetted" CC0 mesh had a non-obvious native-axis gotcha
that took direct vertex inspection to catch). Given the scenario's own
docstring already frames this as a CBF-portability test rather than an
asset-realism showcase, I judged the primitive-improvement path the better
risk/time tradeoff for this pass.

**Fix** (`genesis_bakeoff/scenario_2c.py`): replaced the single bare opaque
cylinder with two nested `fixed=True` primitive cylinders:
- Outer "shell": radius 0.035m, height 0.10m, pale cyan
  (`color=(0.80,0.92,0.97)`, `opacity=0.55`).
- Inner "water fill": radius 0.027m (smaller, reads as glass wall
  thickness), height 0.065m (shorter, water level sits below the rim),
  saturated blue (`color=(0.05,0.35,0.80)`, `opacity=0.95`).

Both are `fixed=True` (not left to free-fall like the old single cylinder)
because they geometrically overlap by design (the water sits inside the
glass) -- two independent free rigid bodies with overlapping collision
geometry would fight Genesis's contact solver every step. This is a purely
visual change; the CBF hazard-zone math only ever referenced the laptop
ellipsoid, never the glass, so no hazard-geometry adjustment was needed or
made.

**Iteration during verification**: the first re-render (job 42162160)
revealed two follow-on problems only visible once issue 2 was actually
fixed: (a) the glass at its original position (`GLASS_POS_XY=(0.35,-0.10)`)
was significantly occluded by the laptop's now-correctly-oriented,
screen-tilted-back footprint (the *broken* pre-fix laptop orientation never
extended toward the glass, so this occlusion was invisible until the
orientation bug was fixed), and (b) the rasterizer backend used by
`cam.render()` doesn't render the `opacity` field as strong true alpha
transparency, so the shell read as a flat pale-gray blob rather than
glassy. Fixed both: moved `GLASS_POS_XY` to `(0.30, -0.20)` (clear of the
laptop's real world footprint, x in [0.218,0.482]/y in [-0.020,0.220]), and
pushed the shell/water colors to higher contrast/saturation so the
distinction reads clearly even without correct alpha blending. Re-ran again
(job 42162187, `SCENE_TEST_RESULT: PASS`) and visually confirmed via a
zoomed crop: the glass now shows a clear pale outer rim/wall with a
shorter, distinctly blue inset -- reads unambiguously as a filled glass, not
an empty cup, and is no longer occluded.

## What "done" looks like -- checklist verification

- **1a-i bottle now clearly reads as taller/narrower than the mug**:
  confirmed via pixel measurement of `scenario_1a_i_output/frame_rgb.png`
  (bottle ~140px vs. mug ~65px in a matched crop).
- **2c laptop sits flat and correctly oriented**: confirmed visually in
  `scenario_2c_output/frame_final_rgb.png` (job 42162187's render) --
  base flat on the table, screen tilted back naturally, hinge-side down.
- **2c glass looks like a glass and appears filled with water**: confirmed
  visually via a zoomed crop of the same frame -- pale outer shell + inset
  blue water fill, clearly distinguishable from a flat-color cylinder.
- **No regression to either scenario's existing pass/fail checks**: both
  re-verified on SLURM after all fixes (1a-i job 42162159, 2c job
  42162187 is the final/current one) -- both `SCENE_TEST_RESULT: PASS`.
  1a-i's mug/bottle settle heights (0.7476/0.7488) are unchanged from
  before this task (only the y-placement changed, not the z-settling
  physics), so the existing height-plausibility assertions are unaffected.
  2c's CBF hazard-zone/tilt-angle logic (`certify_action_simple`/
  `evaluate_cbf`) was not touched at all -- only the hazard ellipsoid's
  *size* was recomputed to match the corrected mesh orientation, same as
  was already done in the prior standardize-rig task when the laptop mesh
  was first swapped in.

## SLURM job IDs

| Job ID | Name | Purpose | Result |
|---|---|---|---|
| 42162110 | diag_1a_i | Diagnostic: settle + read quat/tip-angle for mug/bottle | COMPLETED (not a scenario test, no pass/fail marker -- diagnostic-only, deleted from repo after use) |
| 42162147 | diag_1a_i2 | Diagnostic: settle + read ground-truth `get_AABB()` for mug/bottle + close-up render | COMPLETED (diagnostic-only, deleted from repo after use) |
| 42162159 | genesis_pilot_1a_i | Real scenario re-run with the BOTTLE_POS fix | COMPLETED / PASS |
| 42162160 | genesis_pilot_2c | Real scenario re-run with the laptop-orientation + first-pass glass fix | COMPLETED / PASS (this render exposed the glass-occlusion follow-on issue, superseded by 42162187) |
| 42162187 | genesis_pilot_2c | Real scenario re-run with the corrected glass placement/coloring | COMPLETED / PASS (final, current `scenario_2c_output/` on disk) |

All logs read directly from `/ocean/projects/cis250185p/asingal/slurm_logs/`.

## Files changed

- `genesis_bakeoff/scenario_1a_i.py` -- `BOTTLE_POS` y 0.14 -> 0.20, plus an
  extensive in-file comment documenting the Task-5 diagnosis (ground-truth
  AABB numbers, ruled-out hypotheses, and the reasoning for the placement
  fix).
- `genesis_bakeoff/scenario_2c.py` -- laptop loaded with `euler=(90,0,0)`
  and all position/hazard-ellipsoid constants recomputed against the
  corrected axis mapping; glass rebuilt as two nested `fixed=True`
  cylinders (shell + water fill) with a repositioned, higher-contrast
  final placement; module docstring extended to document both fixes.
- `genesis_bakeoff/scenario_1a_i_output/` -- regenerated
  `frame_rgb.png`/`frame_depth.png`/`frame_wrist_rgb.png` (job 42162159).
- `genesis_bakeoff/scenario_2c_output/` -- regenerated
  `frame_final_rgb.png`/`frame_final_wrist_rgb.png`/`h_log.json` (job
  42162187, the final/corrected glass-placement run).

No new mesh assets were sourced (see issue 3's sourcing decision above), so
`genesis_bakeoff/external_assets/glass/` does not exist and there is nothing
to commit there. No new `slurm/*.slurm` files were needed for the real
scenario re-runs -- both `slurm/pilot_1a_i_genesis.slurm` and
`slurm/pilot_2c_genesis.slurm` already existed (pre-dating this task, per
the earlier standardize-rig task) and were reused unmodified. Two throwaway
diagnostic scripts (`genesis_bakeoff/_diag_1a_i.py`,
`genesis_bakeoff/_diag_1a_i2.py`) and their SLURM wrappers were written to
answer issue 1's root-cause question and deleted before committing --
not part of the permanent scenario suite.

## Concerns / follow-ups

1. **`slurm/pilot_1a_i_genesis.slurm` and `slurm/pilot_2c_genesis.slurm`
   remain uncommitted** (untracked in git, per `git status`), even though
   this task (and the prior standardize-rig task before it) both depend on
   them to reproduce these results. This is the same pre-existing gap
   flagged in `task-standardize-rig-report.md`'s concerns section (shared
   infra files not yet committed) -- not introduced or expanded by this
   task, since I reused these files unmodified. Flagging again for the
   controller to fold into a future infra-consolidation task.
2. **The `opacity` surface field does not render as true alpha
   transparency in this rasterizer backend** (confirmed empirically during
   this task's glass iteration -- see issue 3 above). The current glass
   fix compensates with color contrast rather than genuine transparency,
   which is visually convincing but worth knowing about for any future
   scenario that specifically needs see-through materials (e.g. Genesis's
   raytrace renderer, `genesis.vis.raytracer`, may be required for real
   glass transmission -- not investigated further here, out of scope for a
   purely visual primitive fix).
3. **Eye-in-hand wrist frames remain uninformative** for both scenarios in
   their "final" pose (both show mostly featureless dark/checkerboard
   circles) -- this is the same pre-existing, already-documented limitation
   from `task-standardize-rig-report.md` (the ready/scripted poses don't
   point the wrist camera at the tabletop objects), not something this
   task's scope covered or regressed.
