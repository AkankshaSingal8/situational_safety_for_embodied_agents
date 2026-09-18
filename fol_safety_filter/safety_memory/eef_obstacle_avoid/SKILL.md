---
schema_version: 1
id: eef_obstacle_avoid
revision: 1
description: Keep the end-effector out of each grounded obstacle's exclusion zone.
roles:
  subject: {binding: eef}
  target: {binding: obstacles}
applies_when: "NEAR(SUBJECT, TARGET, 0.60)"
requires:
  predicate: avoid_sphere
  warning_r: 0.20
  hard_r: 0.10
  push_hard: 0.08
  shape: ellipsoid_if_available
  cancel_scale_on_target_corridor: 0.6
violation_action: avoid
unknown_action: stop_and_report
---

# End-effector obstacle avoidance

The rule that does most of the work. Inside `warning_r` of a grounded
obstacle, the entire component of the commanded motion pointing at that
obstacle is removed; inside `hard_r`, an outward push proportional to
penetration depth is added on top.

## Where these numbers come from

They are not new. They are the constants that were hardcoded in
`filter.py:_apply_cbf` and that produced every result in
`fol_safety_filter/README.md`:

| parameter | value | was |
|---|---|---|
| `warning_r` | 0.20 | `max(0.20, self._obstacle_radii.get(obs_name, 0.20))` |
| `hard_r` | 0.10 | literal `hard_r=0.10` |
| `push_hard` | 0.08 | literal `push_hard=0.08` |
| `cancel_scale_on_target_corridor` | 0.6 | literal `cancel_scale = 0.6` |

`warning_r` is a **floor**, matching the `max(...)` it replaces. A grounded
per-obstacle radius from the perception stack may expand the zone and may
never shrink it. A record edit that lowered this below what perception asked
for would be ignored; that asymmetry is intentional and is why the field is
documented as a floor rather than a radius.

`shape: ellipsoid_if_available` selects the rotated-ellipsoid zones when the
grounder has produced an oriented extent for this obstacle, and the isotropic
sphere otherwise. `FOL_ELLIPSOID=0` forces the sphere regardless.

`cancel_scale_on_target_corridor` relaxes cancellation from 100% to 60% when
the commanded motion aims at a vision-detected target within 0.30 m. Without
it, an obstacle sphere offset by a few centimetres swallows the grasp corridor
of a target sitting next to that obstacle, and the arm can never complete the
final approach. It applies only to vision-grounded obstacles; oracle-state
obstacles are exactly centred and do not need it.

## Why `applies_when` is wider than `warning_r`

`NEAR(SUBJECT, TARGET, 0.60)` is a **pre-filter, not the constraint**. It
selects which (subject, target) pairs are handed to the exact geometric test;
the zone geometry in `requires` is what actually decides.

The gate has to be wider than `warning_r` because the geometric test engages
when *either* the current position or the position after the commanded step is
inside the zone, while `NEAR` only sees the current position. A gate at 0.20
would drop exactly the steps that are about to enter the zone — the ones that
matter most.

0.60 m is the schema's maximum `warning_r`, so the gate cannot exclude a pair
the exact test would have constrained, **provided a single commanded step is
shorter than `0.60 - warning_r`**. For this policy the per-step position delta
is on the order of a centimetre, so the margin is three orders of magnitude
larger than needed. A policy emitting 40 cm single-step deltas would violate
the assumption, and the gate would then need widening or replacing with a
look-ahead predicate.

## Limitations

This is a proximity constraint on a point. It is not a collision check: the
end-effector is treated as a point, the obstacle as a sphere or ellipsoid, and
the gripper's own finger volume is not represented. The arm links are a
separate rule (`arm_link_obstacle_avoid`) because they are separate monitored
points, not because the arm is a separate hazard.

Nothing here establishes that the obstacle is undamaged — only that the
monitored point stayed out of a declared region. Contact through an
unmonitored body is invisible to this rule.
