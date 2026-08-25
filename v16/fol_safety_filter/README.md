# FOL Safety Filter — V1 (Reference Implementation)

## What It Does
Geometric Control Barrier Function (CBF) wrapper for OpenVLA-OFT robot policy.
Monitors the end-effector (EEF) and arm body checkpoints against the SafeLIBERO
obstacle at each timestep and cancels approach components.

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

## Files
- `filter.py` — `FOLSafetyFilter` class; `_apply_cbf`, `_apply_point_cbf`
- `primitives.py` — `RobotState`, `ObjectState`, predicate registry
- `kb.py` — `FOLKnowledgeBase`, `FOLRule`
- `cbf_mapper.py` — `CBFMapper`, `MappedCBFSet`
- `__init__.py` — package exports
- `test_fol_unit.py` — unit tests

## SLURM Scripts (v1)
`../slurm/eval_fol_<suite>_<level>_n50.slurm` — one per suite/level combination.
PYTHONPATH: `$WT:$REPO/vlm_pipeline:...` (resolves `fol_safety_filter` to this directory).
