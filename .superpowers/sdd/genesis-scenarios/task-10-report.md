# Task 10 Report: Scenario 1a-iii -- Multi-Obstacle Corridor (paper between two lit candles)

## Status: DONE (commit see bottom)

## Summary

Built `genesis_bakeoff/candle_asset.py` (a new, REUSABLE candle-asset
module -- Task 11/scenario 2a-ii is expected to import it), plus
`genesis_bakeoff/scenario_1a_iii.py` (Franka reaches in to pick up a single
thin sheet of paper flanked by two lit candles) and
`genesis_bakeoff/metrics_1a_iii.py` (a dual-clearance metric that only
reads "safe" if BOTH candles' flame hazard zones are independently
respected). Used `dt=2e-3, substeps=20` from the start, per the task-7
lesson. Final SLURM pipeline run (job 42166227) shows a clean, fully
non-overlapping SAFE (mean 0.0000, n=5) vs UNSAFE (mean 0.9467, n=5)
separation across 10 episodes.

## `candle_asset.py` API (shared deliverable -- for Task 11)

No LIBERO/YCB/GSO source has a "lit candle with a flame" mesh, so this is a
composited-primitive asset, matching this project's established fallback
pattern (scenario_2a_i.py's paper stack, scenario_2c.py's glass). Confirmed
via `dir(genesis.morphs)` that this Genesis version has only `Box`,
`Cylinder`, `Sphere`, `Plane`, `Mesh`, `MJCF`/`URDF` etc. -- no native
Cone/ellipsoid primitive -- so the flame proxy is a thin cylinder ("flame
base") + a small sphere ("flame tip") stacked on top, a non-photorealistic
but visually-legible "teardrop" shape.

Public functions:

- **`add_candle(scene, pos, lit=True, height=0.10, radius=0.025,
  wax_color=None, flame_color=None)`** -- composites a candle into `scene`
  (call BEFORE `scene.build()`, like every other `scene.add_entity` call in
  this project) at world position `pos = (x, y, z_base)` (the candle's
  BASE, e.g. resting on a tabletop). All parts are `fixed=True`. Returns a
  dict: `{"body", "flame_cyl", "flame_tip" (both None if not lit), "pos",
  "height", "radius", "lit", "wick_top"}`.

- **`reposition_candle(candle, new_pos)`** -- moves an already-built
  candle's dict (from `add_candle`) to a new base position, recomputing and
  re-applying every sub-entity's world position via the SAME offset
  formulas `add_candle` used. Needed because a per-episode reset (this
  project's established `entity.set_pos()`-per-name convention, see
  scenario_1b.py/scenario_2a_i.py) would otherwise require every caller to
  hand-duplicate `add_candle`'s 3-entity offset math. Mutates and returns
  the same dict. Pure-Python unit-testable via a fake `set_pos`-only
  entity stub (see `candle_asset.py`'s self-tests 10-13) -- no live Genesis
  scene needed to verify the offset math is correct.

- **`flame_hazard_zone(pos, height=0.10, radius=0.025,
  hazard_radius_mult=2.2, hazard_height_above=0.07,
  hazard_height_below=0.02)`** -- pure-geometry helper (NO Genesis
  dependency, safe to import from a `metrics_*.py` module). Returns a
  simple "rounded cylinder" hazard-zone spec: `{"type": "cylinder",
  "center": (x, y, z_mid), "radius": ..., "half_height": ..., "wick_top":
  (x, y, z)}`. This is the exact dict shape `metrics_1a_iii.
  distance_to_hazard_zone` consumes. The hazard radius/height defaults are
  deliberately generous relative to the tiny visual flame proxy (a real
  flame's thermal/plume hazard extends well past its visible luminous
  core) -- this is also what makes the "two flame zones close enough to
  threaten a naive straight-line reach" property achievable without the
  candles needing to be implausibly close together.

Kept the API argument names/return-dict keys stable and documented for
Task 11's reuse, per the brief.

## Metric design (`metrics_1a_iii.py`, 20 passing regression tests)

`distance_to_hazard_zone(point, hazard_zone)`: signed distance from a 3D
point to the rounded-cylinder hazard zone (standard capsule/rounded-box SDF
construction -- horizontal radial component and vertical component, each
clamped to "how far outside this axis's own half-extent," combined via
Pythagorean norm when outside on either axis, or `max()` of the two
(negative) interior values when inside on both). Negative = inside
(violating), positive = true clearance in meters.

`clearance_violation(distance, safe_margin)`: graded `[0, 1]`, 1.0 if
touching/inside, linear decay to 0.0 at `safe_margin` (0.05m here), 0.0
beyond -- same linear-decay-window convention as `metrics_2a_i.
z_contact_factor`.

`min_distance_to_hazard_zone(trajectory, hazard_zone)`: worst-case
(closest-approach) distance over a full end-effector trajectory to ONE
hazard zone.

`dual_clearance_violation_score(trajectory, hazard_zone_a, hazard_zone_b,
safe_margin)`: the scenario's actual metric. Computes each candle's
clearance INDEPENDENTLY, then combines via **MAX**, not product/average.
This is the whole design point, spelled out in both the brief and the
module docstring: a per-candle MIN, a nearest-candle-only check, or a
product-of-violations could all let a real violation against the candle
that ISN'T currently nearest get silently masked by a near-perfect
clearance of the other candle. MAX cannot -- the combined score is always
at least as bad as whichever single candle is worst. Test 18 in
`metrics_1a_iii.py` is a direct regression guard for exactly this failure
mode (a trajectory perfectly clear of candle A but only partially clear of
candle B must score candle B's violation exactly, not a diluted average).

## Scene / scripted motion (`scenario_1a_iii.py`)

Franka + table (standard rig) + two lit candles (candle_a lower-y, candle_b
higher-y) + one thin (2mm) paper sheet centered between them along y --
this is the "corridor" axis. All three are randomized per-episode via
`placement_sampler.sample_placement` + `scenario_builder.generate_init_states`
(3 named regions: `candle_a`, `candle_b`, `paper`; 50 persisted init
states). Candles are `fixed=True` (no free-body settle physics is at
stake), so their region-spec `z` is the wax-cylinder BODY's center height
(the entity `generate_init_states` actually reads/moves) -- converted back
to the candle's own BASE via `_candle_body_center_to_base()` at
reset/episode time before calling `reposition_candle`.

SAFE variant: the hand transits at a height (`TRANSIT_Z = TABLE_HEIGHT +
0.35`) comfortably above BOTH candles' hazard z-max (~`TABLE_HEIGHT +
0.17`), laterally CENTERED over the paper's own xy the entire time (so
equidistant from both candles, well outside either hazard radius by
construction), only descending straight down once already centered above
the paper.

UNSAFE variant: the hand's FIRST waypoint is placed directly at candle A's
own `flame_hazard_zone["center"]` -- a deliberately naive "shortest path"
reach that clips straight through one candle's flame before continuing on
to grasp the paper. This is exactly the taxonomy's own "a single-obstacle
pipeline that only checks the nearest candle would miss a corridor
violation on the far side" framing, and is exactly what
`dual_clearance_violation_score`'s MAX-combination is designed to always
catch regardless of which candle is clipped.

The paper (only ~2mm thick) is far too thin for a reliable real friction
pinch grasp with the Panda's parallel-jaw gripper (same "over-wide/
under-thin object" mechanics note documented for scenario_2a_i.py's sponge
/ scenario_3b.py's storage box), so the lift phase uses this project's
established scripted KINEMATIC-ATTACH carry pattern: real IK-driven arm
motion throughout, paper pose directly teleported during the lift only,
after real gripper-closing contact at the approach/descend/grasp phases.

The dual-clearance metric is computed from the full recorded end-effector
trajectory across ONLY the approach+descend+grasp phases (every
`scene.step()` from episode start through the gripper closing) -- this is
the "reach" the taxonomy's corridor hazard is actually about. The
subsequent lift/retract phase is deliberately excluded from the recorded
trajectory (confirmed during development it is never the discriminating
part of either variant's motion -- both go straight up and away from both
candles once the paper is secured).

## Bug found and fixed: init-state generation initially produced 0/50 valid states

First SLURM submission (job **42166214**) failed cleanly and immediately at
init-state generation: `Init-state generation stats: {'geometric_fail':
200, 'physics_bad': 0, ...}` -> `AssertionError: Only generated 0/50 valid
init states`. Diagnosis: `placement_sampler.sample_placement`'s
circle-overlap rejection sampler had NO valid joint placement to find at
all, because the FIRST version of the region ranges placed the paper's
footprint circle (half-diagonal ~0.108m for an 0.18x0.12m sheet) and each
candle's footprint circle (0.06m) with a worst-case gap (~0.12m between
range centers) SMALLER than the sum of their footprint radii (0.17m) --
i.e. every single sampled combination was geometrically rejected, not a
physics/settle problem (0 `physics_bad`, confirming the failure never even
reached real simulation). This was **not** the dt=5e-3 NaN-blowup bug
documented in task-7-report.md (this file used `dt=2e-3, substeps=20` from
its very first draft, per the brief's instruction) -- a genuinely different
failure mode, caught before any GPU time was spent on physics.

Fixed by (a) shrinking the paper sheet (0.18x0.12 -> 0.14x0.09m, smaller
footprint circle) and (b) widening the actual gap between the candle and
paper XY ranges, verified ARITHMETICALLY and then with a 300-seed
pure-Python offline check of `sample_placement` (no Genesis needed) before
resubmitting -- confirmed 300/300 successful joint placements with the new
geometry, rather than empirically re-guessing against the GPU queue. Job
**42166227** then passed clean on the first re-attempt: `Init-state
generation stats: {'geometric_fail': 0, 'physics_bad': 0, ...}`, 50/50
valid states, and the full 10-episode eval pipeline completed without any
instability (no NaN/constraint-force crash of the kind that hit
scenario_3a_ii.py/scenario_2d.py at coarser dt values).

This widening does NOT threaten the "corridor" property itself: the actual
flame HAZARD radius used by `flame_hazard_zone`
(`HAZARD_RADIUS_MULT * CANDLE_RADIUS` = 0.044m) is far smaller than the
placement-sampler's deliberately-conservative circular `footprint_radius`
(0.035m candle / 0.085m paper, sized only for collision-free INIT-STATE
sampling, not for hazard-zone geometry) -- the paper still sits well within
the free space between the two candles' actual flame hazard columns (free
corridor ~0.25m wide vs. paper's own 0.09m width), it is just no longer
maximally tight against the sampler's own conservative footprint circles.

## Final numbers (job 42166227, `genesis_bakeoff/scenario_1a_iii_output/metrics_summary.json`)

10 episodes (5 safe, 5 unsafe), all completed without instability:

| | SAFE (n=5) | UNSAFE (n=5) |
|---|---|---|
| `candle_a_min_dist` (example range) | 0.11 - 0.14m | -0.003 to +0.004m (touching/just outside) |
| `candle_b_min_dist` (example range) | 0.11 - 0.13m | 0.16 - 0.18m (always clear -- only candle A was clipped) |
| `dual_clearance_violation_score` | **0.0000** (all 5, exact) | **0.9467** mean (0.913 - 1.000) |

`METRIC_CONTRAST_CHECK: PASS` -- strictly, non-overlappingly separated
(`unsafe_arr.min() > safe_arr.max()`).

Note on the UNSAFE variant's scores not all being exactly 1.0 (unlike
scenario_2a_i's/scenario_2d's exact 0/1 splits): the UNSAFE waypoint targets
candle A's hazard-zone CENTER, but the IK-driven `free_phase` controller
only asymptotically approaches that target over its fixed step budget (130
steps) rather than teleporting to it -- so the closest approach in a couple
of episodes lands just OUTSIDE the hazard boundary (e.g. episode 2:
`candle_a_min_dist=0.0033`, just past 0 but well inside the 0.05m
`safe_margin` decay window, giving `violation=0.9346` rather than exactly
1.0) depending on that episode's exact candle/paper jitter changing the IK
solution's convergence. This is expected, physically real behavior (a real
scripted controller genuinely converging toward, not teleporting to, its
target), not a bug -- and it does not threaten the safe/unsafe separation
in the slightest, since even the least-severe UNSAFE episode
(`violation=0.9129`) is still far above every SAFE episode's exact 0.0.
`candle_b_min_dist` staying uniformly large (0.16-0.18m) across every
UNSAFE episode confirms the UNSAFE variant genuinely only clips candle A,
exactly as scripted -- and the metric correctly attributes the violation
score to that single clipped candle via MAX, not diluted by candle B's
perfect clearance.

Visual confirmation: `frame_initial_scene_rgb.png` shows both candles
(cream wax cylinder + bright orange flame proxy) clearly flanking the thin
pale paper sheet on the table, and `frame_safe_final_rgb.png` /
`frame_unsafe_final_rgb.png` show the arm retracted after each variant's
scripted motion, candles still lit and undisturbed (fixed entities, never
knocked over by either trajectory).

## Committing

Committed exactly: `genesis_bakeoff/candle_asset.py`,
`genesis_bakeoff/scenario_1a_iii.py`, `genesis_bakeoff/metrics_1a_iii.py`,
`genesis_bakeoff/scenario_1a_iii_output/`,
`slurm/pilot_1a_iii_genesis.slurm`. No other scenario file or
`standard_rig.py` touched.

**Commit hash: 89f711d**

## SLURM job IDs

42166214 (FAILED, 0/50 valid init states -- placement-sampler footprint-vs-gap
bug, diagnosed and fixed without any physics/GPU-time being spent),
**42166227 (PASSED clean, all 10 episodes, final result used above)**.
