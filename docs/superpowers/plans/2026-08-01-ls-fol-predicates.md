# Plan: Groundable FOL safety predicates for the LIBERO-Safety guided client

Goal: raise SR on the LIBERO-Safety table (finetuned π0.5 base) by adding three
default-off predicate flags to `vlm_pipeline/run_guided_libero_safety_eval.py`
(this worktree). Each predicate is grounded a different way (proprioceptive
velocity, gripper state, dynamic-hazard extrapolation) and composes with the
existing `--engage_radius` / `--duality_disengage` / `--guard_topk` /
`--corridor` / `--exec_parity` flags.

## Global Constraints

- ALL new flags default off. With none of the new flags passed, the client's
  behavior and the payload sent to the server must be byte-identical to
  current HEAD (existing tests `test_ls_corridor_guard.py`,
  `test_exec_parity.py` must keep passing; add explicit byte-identity tests
  for the new helpers with flags off).
- New logic lives in PURE helper functions (no I/O, no env access inside the
  helper) following the existing `_gate_guard` pattern
  (`run_guided_libero_safety_eval.py:183-204`); wiring reads obs and calls
  the helper at the same chunk-boundary point where `_gate_guard` is called.
- Never use the `_obstacle_` name substring in any no-GT scoring/decision
  logic (house rule). Classifying an entity as a MOVER must use the
  episode's movers metadata (already available to the client), not name
  patterns; "hand" hazards may be identified by the entity names the
  benchmark itself uses (e.g. `left_hand_1`).
- Safety asymmetry: predicates may only RELAX guidance for static obstacles;
  guards on movers/hands must never be suppressed by `--holding_disengage`.
  When grounding signals are missing (first chunk, no previous position),
  fall back to ENGAGED (conservative).
- Record each new flag's value in the results JSON metadata dict alongside
  the existing flags (same place `engage_radius`/`exec_parity` are recorded).
- Tests are CPU-only, live in `vlm_pipeline/tests/`, run with the repo's
  existing pytest invocation; the full client test suite must pass.
- Commit per task on branch `new-steering-method` in worktree
  `.worktrees/flow-guidance-tier-gt` (a parallel session shares this branch —
  do not rebase/reset; plain commits only).

## Task 1: `--engage_cone` — APPROACHING(eef, hazard) velocity gating

Flag: `--engage_cone DEG` (float, default None = off).

Semantics (chunk-boundary gating, composing AFTER `_gate_guard` radius
gating if both are active): a guard entry is kept iff
  (a) dist(eef, guard_pos) <= HARD_R (constant `ENGAGE_CONE_HARD_R = 0.14`;
      always engaged inside this radius), OR
  (b) the eef is closing on it: the angle between the eef velocity vector
      and (guard_pos - eef) is <= DEG degrees AND ||v|| > 1e-4.
Velocity grounding: finite difference of eef position between consecutive
chunk-boundary queries, tracked by the caller and passed in (prev_eef may be
None on the first query → treat every entry as ENGAGED). Runner-up promotion
and empty-guard → no-guidance payload behave exactly as `_gate_guard` does
(reuse/extend that code path rather than duplicating it). Corridor anchors
are never gated. Guards on movers/hands ARE subject to cone gating (the
hard radius protects the close range).

Implement pure helper `_cone_gate(guard, pos_of, eef, prev_eef, cone_deg,
hard_radius)` + wiring + argparse + results-JSON metadata.

Tests (new file `vlm_pipeline/tests/test_fol_predicates.py`): first-query
fallback (prev None → unchanged), closing vs receding vs tangent geometry
cases, hard-radius override, composition with `_gate_guard`, defaults-off
byte-identity of the payload path.

## Task 2: `--holding_disengage` — HOLDING(·) gripper-state suppression

Flag: `--holding_disengage` (store_true, default off).

Semantics: when the robot is HOLDING (gripper closed on something), suppress
guard entries for STATIC obstacles only; mover/hand entries always stay.
HOLDING grounding: `obs["robot0_gripper_qpos"]` (2-vector of finger joint
positions; Panda gripper fully open width ≈ 0.08 m, fully closed ≈ 0).
Holding iff finger-opening width < `HOLDING_WIDTH_MAX = 0.045` AND width >
`HOLDING_WIDTH_MIN = 0.002` (near-zero width = closed on nothing) AND the
gripper has been in that band for >= 2 consecutive chunk-boundary queries
(caller tracks the streak and passes an int). Static vs mover: entry name in
the episode movers set (metadata the client already reads) → mover; else
static. If movers metadata is unavailable, treat ALL entries as movers
(i.e. suppress nothing — conservative).

Pure helper `_holding_filter(guard, movers, is_holding)` + a pure
`_is_holding(width, streak)`-style predicate + wiring + argparse + metadata.

Tests: width thresholds incl. both edges, streak requirement, static
suppressed while mover kept, missing movers metadata → no suppression,
defaults-off byte-identity.

## Task 3: `--mover_lead` — MOVER_LEAD(hazard) predictive barrier anchor

Flag: `--mover_lead STEPS` (float, default None = off).

Semantics: for guard entries classified as movers (movers metadata, as in
Task 2), replace the position sent to the server with the constant-velocity
extrapolation `pos + v * STEPS`, where v is the per-entity per-step velocity
from finite differences between consecutive chunk-boundary queries (caller
tracks previous positions per entity name). Cap the extrapolation
displacement norm at `MOVER_LEAD_CAP = 0.10` m. First query (no previous
position) → send the observed position unchanged. Static entries unchanged.
Extrapolation applies to whatever position source is active (gt or percep) —
the helper takes a name→pos mapping and returns an adjusted mapping used
only for the guard payload (logging/violation accounting still uses observed
positions).

Pure helper `_lead_positions(pos_of, prev_pos_of, movers, lead_steps, cap)`
+ wiring + argparse + metadata.

Tests: extrapolation math, cap saturation, first-query passthrough, static
untouched, missing movers metadata → all untouched, defaults-off
byte-identity.
