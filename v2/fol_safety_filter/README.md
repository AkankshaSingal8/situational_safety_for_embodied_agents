# FOL Safety Filter — V2 (Target-Aware Last-Mile Fix)

## What Changed vs V1

Single logical addition, touching only `filter.py`:

**Last-mile allowance** in `_apply_point_cbf` (EEF path only):

When the EEF is inside `warning_r` AND both conditions hold:
1. The identified target object is within `2 × warning_r` (0.40 m) of the obstacle
2. The current action is >50% directed toward the target (`toward_target > 0.5`)

→ `cancel_fraction` drops from **1.0 → 0.6** (allow 40% of approach through).

Arm body checkpoints (link4/link6) are **unchanged** — always 100% cancel.

## Why

Goal L1 TSR regressed 18% → 13% in v1 because task_2 ("put the bowl on the stove")
and task_3 ("open the top drawer and put the bowl inside") have the goal location
within 0.20 m of the obstacle. The filter blocked the final approach entirely.

Target-aware allowance lets the robot approach slowly through the warning zone
when moving toward a goal that happens to be near an obstacle — without lowering
protection when moving in any other direction.

## Diff vs V1 (`filter.py` only)

```diff
 # --- EEF CBF ---
+# Target-aware: allow partial approach when heading toward a near-obstacle goal.
+target_pos: Optional[np.ndarray] = None
+if self._target_name and self._target_name in state.objects:
+    target_pos = state.objects[self._target_name].pos
+
 self._apply_point_cbf(u, ee, obs_pos,
                        warning_r=0.20, hard_r=0.10,
                        push_hard=0.08,
-                       label="EEF")
+                       label="EEF",
+                       target_pos=target_pos)

 # --- Arm body checkpoints (no target awareness — always maximally protected) ---
```

```diff
 def _apply_point_cbf(self, u, point, obs_pos, warning_r, hard_r,
-                     push_hard, label=""):
+                     push_hard, label="", target_pos=None):

-    # Cancel 100% of the approach component
     approach_comp = float(np.dot(u[:3], -out_dir))
     if approach_comp > 0:
-        u[:3] += approach_comp * out_dir
-        logger.debug(f"[FOL-CBF] {label} cancel 1.0: dist={dist:.3f}")
+        cancel_fraction = 1.0
+        if target_pos is not None:
+            obs_to_target = float(np.linalg.norm(target_pos - obs_pos))
+            if obs_to_target < warning_r * 2.0:
+                target_diff = target_pos - point
+                target_dist = float(np.linalg.norm(target_diff))
+                if target_dist > 1e-6:
+                    target_dir = target_diff / target_dist
+                    speed = float(np.linalg.norm(u[:3]))
+                    toward_target = float(np.dot(u[:3] / (speed + 1e-9), target_dir))
+                    if toward_target > 0.5:
+                        cancel_fraction = 0.6
+        u[:3] += approach_comp * cancel_fraction * out_dir
```

## Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `obs_to_target < 2 × warning_r` | < 0.40 m | Goal location is near obstacle. Spatial suite targets are typically >0.40 m from obstacle — allowance stays off. |
| `toward_target > 0.5` | cosine > 0.5 | Requires action to be predominantly toward goal (within 60°), not tangential. |
| `cancel_fraction = 0.6` | 60% cancel | Retains majority protection. Robot approaches at 40% of original speed through the zone. |

## Files Changed vs V1

Only `filter.py` (all other files are byte-for-byte identical to v1):
- `_apply_cbf`: +5 lines (compute `target_pos`, pass to EEF call)
- `_apply_point_cbf`: +1 parameter, +13 lines (allowance logic)

## Expected Results (n=50)

| Suite   | Level | V1 TSR | V2 target TSR | V1 CAR | V2 target CAR |
|---------|-------|--------|---------------|--------|---------------|
| Spatial | I     | 48%    | ~48%          | 59.5%  | ~59%          |
| Spatial | II    | 68%    | ~68%          | 56%    | ~56%          |
| Goal    | I     | 13%    | **≥18%**      | 59.5%  | ~55%          |
| Goal    | II    | TBD    | ≥21%          | TBD    | ≥20%          |
| Object  | I     | 19.5%  | ~19.5%        | 7%     | ~7%           |
| Object  | II    | TBD    | ≥42.5%        | TBD    | ≥26%          |
| Long    | I     | 1.5%   | ≥3%           | 72%    | ~72%          |
| Long    | II    | TBD    | ≥16%          | TBD    | ≥20%          |

## SLURM Scripts (v2)
`../../slurm/eval_fol_v2_<suite>_<level>_n50.slurm` — one per suite/level.
PYTHONPATH: `$WT/v2:$WT:$REPO/vlm_pipeline:...` — `$WT/v2` prepended so Python
resolves `fol_safety_filter` to this directory instead of v1.
