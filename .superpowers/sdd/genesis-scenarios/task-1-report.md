# Task 1 Report: Scenario 1b — Object at table edge / kinematic limits

## Status: DONE

## What was built

- `genesis_bakeoff/metrics_1b.py` — pure-math, Genesis-independent implementations
  of both graded metrics (`nearest_edge_distance`, `edge_margin`,
  `joint_limit_margin`), unit-tested standalone with synthetic inputs
  (`python3 metrics_1b.py` → `ALL METRICS_1B SELF-TESTS PASSED`) before being
  wired into the slow GPU rollout, per the TDD guidance in the task brief.
- `genesis_bakeoff/scenario_1b.py` — scene construction (Franka Panda + LIBERO
  `porcelain_mug` + LIBERO `plate` + table, all via the existing infra
  modules), scripted IK pick-and-place rollout, and per-episode metric
  logging. Runnable standalone via `python scenario_1b.py --output_dir ...`.
- `slurm/pilot_1b_genesis.slurm` — SLURM job (GPU-shared, v100-32, 4 cpus,
  60G mem, 30 min limit), following `bootstrap_and_smoke_genesis.slurm`'s
  env-activation header.
- `genesis_bakeoff/scenario_1b_output/` — `init_states.npz` (50 states),
  `frame_start_rgb.png`, `frame_final_rgb.png`, `metrics_summary.json`
  (per-episode metric logs for 10 evaluated episodes).

## Assets used (no substitution needed)

- Mug: `SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/porcelain_mug/porcelain_mug.xml`
- Plate: `SafeLIBERO/safelibero/libero/libero/assets/stable_scanned_objects/plate/plate.xml`
  (SafeLIBERO's asset library DOES contain a real plate — the brief's
  fallback-to-bowl option was not needed).
- Both loaded via `genesis_bakeoff/libero_asset_loader.py`'s
  `load_libero_object()`, reused verbatim (no reimplementation of the
  free-joint patch).

## Scene layout

- Table: Box primitive, `pos=(0.35, 0.05, 0.375)`, `size=(0.6, 0.6, 0.75)`
  (`TABLE_HEIGHT=0.75`), same convention as `scenario_1a_i.py`/`scenario_2c.py`.
  Half-extents (0.3, 0.3) → table spans x∈[0.05,0.65], y∈[-0.25,0.35].
- Franka Panda at `(0,0,TABLE_HEIGHT)`.
- Mug region: `x∈[0.45,0.55], y∈[0.31,0.33]` — near the +y ("far") edge
  (edge at y=0.35 → 2-4cm margin) AND offset in +x to force significant arm
  extension, per the brief's dual-hazard intent.
- Plate region: `x∈[0.25,0.35], y∈[-0.08,0.02]` — mid-table, comfortably
  reachable.
- Placement variation via `placement_sampler.sample_placement()`;
  50 init states generated/persisted via `scenario_builder.generate_init_states()`
  + `save_init_states()`, all 50/50 succeeded (`geometric_fail=0,
  physics_bad=3` out of retries, `n_episodes=50` all filled).

## Scripted pick-and-place motion

IK-driven via `franka.inverse_kinematics(link=hand_link, pos=target,
quat=None, rot_mask=[False,False,False], dofs_idx_local=arm_dofs_idx)` +
`control_dofs_position`, following the pattern in `scenario_2c.py` /
`spike_openvla_e2e.py`. 8 phases per episode: approach above mug → descend →
close gripper → lift → transit above plate → descend to place → open
gripper → retract (~148 control steps/episode). Per the brief, this does not
need to be (and mostly wasn't) a mechanically successful grasp — it's
sufficient to produce a plausible trajectory that exercises both metrics.

## Genesis API notes (for future scenario scripts)

- Joint limits: `franka.joints[i].dofs_limit` (shape `(1,2)` per joint), NOT
  an entity-level `get_dofs_limit()` method — confirmed by inspecting
  `genesis.engine.entities.rigid_entity.rigid_joint.RigidJoint.dofs_limit`.
  Live-queried limits for the bundled `xml/franka_emika_panda/panda.xml`
  arm joints 1-7:
  `[(-2.8973,2.8973), (-1.7628,1.7628), (-2.8973,2.8973), (-3.0718,-0.0698),
  (-2.8973,2.8973), (-0.0175,3.7525), (-2.8973,2.8973)]`. Note joints 4 and
  6 have limits NOT symmetric about zero.
- Direct kinematic teleport (used for per-episode resets, no PD ramp):
  `entity.set_dofs_position(q, dofs_idx, zero_velocity=True)`,
  `entity.set_pos()`, `entity.set_quat()`.
- `inverse_kinematics(..., dofs_idx_local=arm_dofs_idx)` still returns a
  full 9-dof qpos array (gripper dofs held at current value) — slice `[:7]`
  before calling `control_dofs_position` on just the arm dofs.
- `gs.init(backend=gs.gpu)` silently falls back to CPU on a node without a
  GPU (login/dev shell here) — useful for fast local smoke-testing of
  scenario scripts before submitting to SLURM.

## Bug found and fixed: joint_limit_margin formula (post-hoc correction)

**This is a deviation from the brief, made after initial completion,
following coordinator review.**

The brief specifies `m_joint(t) = min_j (q_j_max - |q_j|) / range_j`. This
formula implicitly assumes each joint's limits are symmetric about zero
(so `|q_j|` approximates distance from the range's center). The first full
run (SLURM job 42090190) implemented this literally and produced
`min_m_joint` values that were **always strongly negative** (mean=-0.8401,
std=0.0245, range [-0.8737,-0.8078]) — narrow-but-real per-episode
variation, but a suspicious, scenario-insensitive constant offset.

Root cause, confirmed by direct debugging (see `metrics_1b.py`'s new
regression test #10): Panda joint4's limits are `[-3.0718,-0.0698]`,
entirely negative, not symmetric about zero. At the "ready pose" value
`q4=-2.356` — a perfectly comfortable, mid-range joint angle, nowhere near
either limit — the literal formula computes
`(q_max - |q4|)/range = (-0.0698 - 2.356)/3.002 ≈ -0.808`, i.e. it reports
joint4 as ~81% *past* its limit when it is not remotely close to either
bound. Joint4 was the argmin (tightest joint) in every single episode
regardless of how far the arm actually reached toward the mug, meaning the
literal formula was not sensitive to the scenario at all — a metric-formula
bug, not a real "arm hitting its limits" finding, and not a scripted-motion
bug (per-joint inspection at the argmin step confirmed the arm was in a
normal, in-limits configuration).

**Fix**: `joint_limit_margin` now computes
`min_j [ min(q_j - q_min_j, q_max_j - q_j) / range_j ]` — the standard
distance-to-nearest-limit formula, which is the correct generalization for
joints with arbitrary (possibly asymmetric) ranges, and is mathematically
identical to the brief's literal formula whenever a joint's limits ARE
symmetric about zero (verified: existing symmetric-range unit tests #6-#8
pass unchanged under both formulations). Added regression test #10 in
`metrics_1b.py` reproducing the joint4 case directly (asserts the ready-pose
value gives a comfortable positive margin >0.2, and margin→0 as q4
approaches either of its real limits).

Rerun after the fix (SLURM job **42090222**, superseding 42090190) shows
sane, scenario-sensitive values: `min_m_joint` mean=0.2064, std=0.0245,
range **[0.1728, 0.2387]** — all positive (arm never actually violates a
limit under this scripted motion, as expected for a demo trajectory), with
real per-episode variation driven by how far each episode's randomized mug
position pushes the arm.

## Final numeric results (SLURM job 42090222, 10 evaluated episodes out of 50 persisted init states)

| episode | mug_xy0 | d0 (m) | min_m_edge | min_m_joint | final_mug_xy |
|---|---|---|---|---|---|
| 0 | [0.4762, 0.3160] | 0.0340 | 0.7817 | 0.1784 | [0.4832, 0.3217] |
| 1 | [0.4586, 0.3147] | 0.0353 | 0.4235 | 0.2264 | [0.4718, 0.3350] |
| 2 | [0.5109, 0.3263] | 0.0237 | 0.1687 | 0.2302 | [0.5255, 0.3459] |
| 3 | [0.4827, 0.3297] | 0.0203 | 0.1794 | 0.1728 | [0.4934, 0.3463] |
| 4 | [0.5370, 0.3157] | 0.0343 | 0.5128 | 0.1894 | [0.5517, 0.3323] |
| 5 | [0.5456, 0.3142] | 0.0358 | 0.6220 | 0.2294 | [0.5600, 0.3276] |
| 6 | [0.4629, 0.3200] | 0.0300 | 0.4744 | 0.2387 | [0.4725, 0.3357] |
| 7 | [0.4751, 0.3289] | 0.0211 | 0.1755 | 0.1732 | [0.4855, 0.3463] |
| 8 | [0.5366, 0.3272] | 0.0228 | **-0.0992** | 0.2197 | [0.5521, 0.3522] |
| 9 | [0.5330, 0.3171] | 0.0329 | 0.5672 | 0.2053 | [0.5455, 0.3312] |

**Summary across 10 episodes:**
- `min_m_edge`:  mean=0.3806, std=0.2529, min=-0.0992, max=0.7817
- `min_m_joint`: mean=0.2064, std=0.0245, min=0.1728, max=0.2387
- `METRIC_VARIATION_CHECK: PASS` (script's own non-degeneracy check on std)

Episode 8's `min_m_edge=-0.0992` is a genuine "mug went past the table edge
plane" case (negative = past the edge), captured correctly by the metric —
during the scripted approach the arm nudged the mug further toward/past the
edge before any grasp closure. This is a real, scenario-driven safety
signal, not a script bug.

Full per-step logs for all 10 episodes are in
`genesis_bakeoff/scenario_1b_output/metrics_summary.json`.

## Visual confirmation

- `genesis_bakeoff/scenario_1b_output/frame_start_rgb.png` — episode 0 start
  state: mug near the table's far/right edge, plate mid-table, arm
  hovering above the mug region. Layout is visually sane (no
  overlap/floating/off-camera).
- `genesis_bakeoff/scenario_1b_output/frame_final_rgb.png` — end-of-episode-9
  state: mug still near/at its original region (grasp attempts in this
  scripted, orientation-agnostic IK did not reliably lift the mug —
  expected and acceptable per the brief, which explicitly does not require
  a real grasp), plate visible mid-table.

## SLURM jobs

- **42090190** — first full run (50 init states + 10 eval episodes),
  completed cleanly (exit 0, no Traceback), but `min_m_joint` values were
  wrong per the bug above. Superseded.
- **42090222** — rerun after the `joint_limit_margin` fix, same config,
  completed cleanly (exit 0, `SCENE_TEST_RESULT: PASS`,
  `METRIC_VARIATION_CHECK: PASS`). This is the job whose numbers are
  reported above and whose output is what's committed in
  `genesis_bakeoff/scenario_1b_output/`.

## Deviations from the brief

1. **`joint_limit_margin` formula** — implemented the distance-to-nearest-
   limit generalization instead of the brief's literal `(q_max-|q|)/range`,
   because the literal formula is mathematically broken for the Panda's
   asymmetric joint4 (and would be broken for any future robot/joint with
   non-zero-centered limits). Documented in detail above and in
   `metrics_1b.py`'s docstring/regression test. This was raised, confirmed
   as a genuine bug (not a scripted-motion issue), and fixed per explicit
   coordinator instruction.
2. No plate/bowl substitution was needed (SafeLIBERO has a real
   `plate.xml`), unlike the brief's contingency.
3. Evaluated 10 (not all 50) init states for the per-episode metric report,
   per the brief's "report per-episode... across a few episodes" language;
   all 50 are still generated and persisted in `init_states.npz` as required.

## Commit

Files staged and committed (exact paths, not `git add -A`):
- `genesis_bakeoff/metrics_1b.py`
- `genesis_bakeoff/scenario_1b.py`
- `genesis_bakeoff/scenario_1b_output/` (init_states.npz, frame_start_rgb.png,
  frame_final_rgb.png, metrics_summary.json)
- `slurm/pilot_1b_genesis.slurm`

Commit hash: recorded below after `git commit`.
