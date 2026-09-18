# FOL Safety Filter — V1 (Reference Implementation)

## What It Does
Geometric Control Barrier Function (CBF) wrapper for OpenVLA-OFT robot policy.
Monitors the end-effector (EEF) and arm body checkpoints against the SafeLIBERO
obstacle at each timestep and cancels approach components.

## Where the rules live

The safety behaviour is **data**, in `safety_memory/`. Each record is one
activation condition plus one effect, loaded and validated at episode setup
and lifted over the scene's objects. See `safety_memory/README.md` for the
schema and `docs/superpowers/specs/2026-09-18-reusable-fol-rule-memory-design.md`
for why.

| record | effect | loaded |
|---|---|---|
| `eef_obstacle_avoid` | `avoid_sphere` | default |
| `arm_link_obstacle_avoid` | `avoid_sphere` | default |
| `fragile_speed_limit` | `limit_speed` | opt-in |
| `spillable_rotation_lock` | `limit_angular_rate` | opt-in |
| `overhead_exclusion` | `avoid_region_above` | opt-in |

`_apply_cbf` holds no geometric constant. Deleting a record removes exactly one
behaviour, and editing its radii changes what the robot does — both asserted
in `tests/test_rule_falsifiability.py`, in both directions.

**The results below were produced before that refactor, by the geometry that
was then hardcoded in `_apply_cbf`.** They remain valid because
`tools/fol_parity_trace.py` holds the two default records to `max|delta| == 0`
against that implementation over 2592 rows and 12 env-var combinations. Run
`python tools/fol_parity_trace.py --check fol_safety_filter/tests/data/parity_golden.npz`
before trusting any of them after a change.

Parity's scope is narrower than it sounds: the harness builds `MappedCBFSet`
itself, so it pins the projection **given a constraint set** and does not pin
which rules fire. That is why only the two spatial records load by default.

## Core Rule
Binary 100% approach cancellation + proportional outward push inside the hard radius:

| Point | warning_r | hard_r | push_hard |
|-------|-----------|--------|-----------|
| EEF (robot0_eef_pos) | 0.20 m | 0.10 m | 0.08 |
| Arm link4 (elbow) | 0.15 m | 0.08 m | 0.06 |
| Arm link6 (wrist) | 0.15 m | 0.08 m | 0.06 |

When the EEF or an arm checkpoint enters `warning_r`, 100% of the velocity
component pointing toward the obstacle is removed. If inside `hard_r`, an
additional outward push is applied proportional to penetration depth.

## Results (n=50, FOL v1 clean)

| Suite   | Level | FOL TSR | OVL TSR | ΔTSR    | FOL CAR | OVL CAR | ΔCAR    |
|---------|-------|---------|---------|---------|---------|---------|---------|
| Spatial | I     | 48.0%   | 42.0%   | +6.0pp  | 59.5%   | 12.0%   | +47.5pp |
| Spatial | II    | 68.0%   | 35.0%   | +33.0pp | 56.0%   | 11.5%   | +44.5pp |
| Object  | I     | 19.5%   | 0.0%    | +19.5pp | 7.0%    | 1.5%    | +5.5pp  |
| Goal    | I     | 13.0%   | 18.0%   | **-5pp**| 59.5%   | 15.5%   | +44.0pp |
| Long    | I     | 1.5%    | 3.0%    | -1.5pp  | 72.0%   | 15.0%   | +57.0pp |
| Goal    | II    | TBD     | 21.0%   | —       | TBD     | 6.0%    | —       |
| Object  | II    | TBD     | 42.5%   | —       | TBD     | 26.0%   | —       |
| Long    | II    | TBD     | 16.0%   | —       | TBD     | 8.0%    | —       |

## Known Issue
Goal L1 TSR regresses 18% → 13% because the obstacle is placed near the goal
location (stove, drawer). 100% cancellation prevents the robot from completing
the final ~0.20 m of approach. **Fixed in v2** (see `../v2/fol_safety_filter/README.md`).

## Known defects, pinned rather than fixed

`tests/test_composer_channel_status_quo.py` pins two defects in the composer
path. Neither is repaired here: repairing either changes behaviour, and the
table above was produced with both present.

1. A rotation rule authored as `{"angular_limit": 0.10}` runs at 0.25 rad/s,
   because `cbf_mapper.py:184` reads `omega_max`. The authored value is
   unreachable and nothing reports it.
2. For `moka_pot_obstacle` and `wine_bottle_obstacle` the composer authors a
   rotation lock whose own `IS_SPILLABLE` precondition is permanently false:
   `_props_from_name` keys on `pot`/`bottle` while `_heuristic_ground` sets
   `is_open` only for bowl/cup/glass/mug. The channel is inert on both
   reported obstacles.

Under the record schema the first defect is a load error rather than a shrug,
since each effect has a closed, typed parameter set.

## Files
- `filter.py` — `FOLSafetyFilter` class; `_apply_cbf`, `_apply_point_cbf`
- `safety_memory/` — the rule records (the safety behaviour, as data)
- `rule_memory/` — schema, loader, binding compiler, effect handlers
- `primitives.py` — `RobotState`, `ObjectState`, predicate registry
- `kb.py` — `FOLKnowledgeBase`, `FOLRule`
- `cbf_mapper.py` — `CBFMapper`, `MappedCBFSet`
- `__init__.py` — package exports
- `test_fol_unit.py` — unit tests

## SLURM Scripts (v1)
`../slurm/eval_fol_<suite>_<level>_n50.slurm` — one per suite/level combination.
PYTHONPATH: `$WT:$REPO/vlm_pipeline:...` (resolves `fol_safety_filter` to this directory).
