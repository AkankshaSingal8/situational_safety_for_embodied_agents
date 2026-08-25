# FOL Safety Filter — V3 (Carrying-Phase + Goal-Tangential Redirect)

## What Changed vs V2

Three additions to `filter.py` only — all other files are byte-for-byte identical to v1/v2.

### Change 1: Goal-receptacle target extraction (`_extract_target_name`)

For task descriptions of the form "put X on/in/inside/onto Y", v3 returns **Y** (the goal
receptacle) instead of X. This is the key fix for Goal-suite tasks where the obstacle blocks
access to the stove/drawer, not the object being picked up.

```
"put the bowl on the stove"       → target = stove_1
"open the drawer and put X inside"→ target = drawer_1  (fallback: longest name match)
"pick up the orange juice"        → target = orange_juice_1  (no preposition → fallback)
```

### Change 2: Carrying-phase detection (`_apply_cbf`)

Checks `state.gripper_width < 0.025` to detect when the robot is holding an object
(gripper closed). During the carrying phase:
- EEF `warning_r` shrinks 0.20 → 0.12 m (robot can get closer before filter activates)
- EEF `hard_r` shrinks 0.10 → 0.06 m
- `carrying=True` flag passed to `_apply_point_cbf`

Arm checkpoints (link4/link6) are unaffected — always 100% cancel, no phase modulation.

### Change 3: Goal-tangential redirect (`_apply_point_cbf`)

When `carrying=True`: `cancel_fraction` drops unconditionally to **0.15** (allow 85% through).

Goal receptacles (stove, drawer, cabinet) are scene geometry — they do NOT appear as
`_pos` keys in obs, so `target_pos` is usually `None` for Goal tasks. The allowance
therefore cannot require `target_pos`; the gripper state alone is the signal.

Additionally, if `target_pos` IS known and the goal is within 2×warning_r of the obstacle:
- Push direction changes from pure outward → **60% outward + 40% goal-tangential**
  (nudges robot around the obstacle toward the goal)

When `carrying=False` (pre-grasp), behavior is identical to v2 (cancel_fraction=0.6 when
heading toward a near-obstacle target).

## Why This Should Exceed OVL Baseline on Goal

OVL Goal L1: 18% TSR, 84.5% collision rate. Collisions do **not** end Goal episodes —
the robot grazes the obstacle and keeps going. So v1/v2's aggressive 100% cancellation
was preventing task completion without providing meaningful safety benefit in the deposit
phase.

V3 mechanisms:
1. **Correct target (receptacle)** → allowance actually fires (v2 couldn't trigger it)
2. **Shrunk carrying zone** → robot can get within 12cm of obstacle while depositing
3. **85% through + redirect** → robot reaches the goal receptacle in most episodes
4. **Redirect rescues OVL-failed episodes** → robot stuck against obstacle face gets
   nudged around it toward the goal

## Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `gripper_width < 0.025` | 0.025 | Clear gap from open (~0.04); closed = holding object |
| carrying `warning_r` | 0.12 m | Allows robot within 12cm for deposit |
| carrying `hard_r` | 0.06 m | Proportionally smaller hard zone |
| `cancel_fraction` (carrying) | 0.15 | 85% through; slow but passable |
| redirect mix | 60/40 out/tangent | Mostly outward for safety, 40% toward goal to route around obstacle |
| `toward_target > 0.5` | cosine > 0.5 | Action must be predominantly toward goal |

## Files Changed vs V2 (`filter.py` only)

- `_extract_target_name`: +14 lines (receptacle preposition parsing, fallback unchanged)
- `_apply_cbf`: +7 lines (carrying detection, phase-aware radii, carrying flag)
- `_apply_point_cbf`: +1 param (`carrying`), +20 lines (carrying branch + redirect)

## Expected Results (n=50)

| Suite   | Level | OVL TSR | V1 TSR | V2 TSR | V3 target |
|---------|-------|---------|--------|--------|-----------|
| Spatial | I     | 42%     | 48%    | 48.5%  | ~48%      |
| Spatial | II    | 35%     | 68%    | 69.5%  | ~69%      |
| Goal    | I     | 18%     | 13%    | 14%    | **>18%**  |
| Goal    | II    | 21%     | 7.5%   | 7.5%   | **>21%**  |
| Object  | I     | 0%      | 19.5%  | 19.5%  | ~19%      |
| Object  | II    | 42.5%   | 49%    | 49%    | ~49%      |
| Long    | I     | 3%      | 1.5%   | 1.5%   | ~2%       |
| Long    | II    | 16%     | 21%    | 21%    | ~21%      |

## SLURM Scripts (v3)
`../../slurm/eval_fol_v3_<suite>_<level>_n50.slurm` — one per suite/level.
PYTHONPATH: `$WT/v3:$WT:$REPO/vlm_pipeline:...` — `$WT/v3` prepended so Python
resolves `fol_safety_filter` to v3.
