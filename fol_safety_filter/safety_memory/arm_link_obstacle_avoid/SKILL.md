---
schema_version: 1
id: arm_link_obstacle_avoid
revision: 1
description: Keep each monitored arm link out of every grounded obstacle's exclusion zone.
roles:
  subject: {binding: arm_checkpoints}
  target: {binding: obstacles}
applies_when: "NEAR(SUBJECT, TARGET, 0.60)"
requires:
  predicate: avoid_sphere
  warning_r: 0.15
  hard_r: 0.08
  push_hard: 0.06
  shape: sphere
violation_action: avoid
unknown_action: stop_and_report
---

# Arm-link obstacle avoidance

The elbow and the wrist sweep obstacles that the end-effector has already
cleared. This rule monitors them with the same projection as
`eef_obstacle_avoid`, at tighter radii.

## This record is the generalization claim

It shares `eef_obstacle_avoid`'s effect predicate exactly. There is no second
avoidance implementation, no arm-specific geometry code, and no new schema
field. The only differences are `roles.subject` and three numbers:

| | `eef_obstacle_avoid` | this record |
|---|---|---|
| subject binding | `eef` | `arm_checkpoints` |
| `warning_r` | 0.20 | 0.15 |
| `hard_r` | 0.10 | 0.08 |
| `push_hard` | 0.08 | 0.06 |

That is what "a rule generalizes" means concretely here: one authored
constraint, rebound to a different set of monitored points, with the
parameters as data. If avoidance needed re-implementing per subject, the
decomposition would be cosmetic.

The radii match the per-body values that `_get_arm_checkpoints` returned for
`robot0_link4` and `robot0_link6`. They now live here, and
`_get_arm_checkpoints` supplies positions. The one exception is the opt-in
`FOL_HAND_CHECKPOINT` bodies (`robot0_link7`, `robot0_right_hand`), which keep
their tighter 0.10 / 0.06 / 0.06 override because they monitor a different
volume; that override is an env-var-gated experiment, not the default path.

## Binding requires registering the checkpoints as pseudo-objects

`primitives.NEAR` resolves the literal symbol `eef` from `state.ee_pos` and
every other symbol from `state.objects` (`primitives.py:96`). An arm
checkpoint is not in `state.objects` by default, so the lookup raises
`KeyError`, `kb.py:57` catches it, and the rule evaluates **False** — silently.

Measured: a checkpoint 1 cm from an obstacle does not fire unless registered.
The filter therefore registers each checkpoint into `state.objects` as a
zero-extent pseudo-object before evaluating the knowledge base. Without that
step this record loads, validates, reports as enabled, and does nothing.
`test_rule_falsifiability.py` exists to keep that from regressing.

`shape: sphere` rather than `ellipsoid_if_available`: the projection for arm
checkpoints was always isotropic, and the oriented extents the grounder
produces describe the obstacle as seen for a gripper approach. Using them for
the elbow would be a change, not a port.

## Limitations

Two points on a seven-link arm. The links between them are unmonitored, so a
mid-forearm contact is invisible to this rule. Radii are not derived from link
geometry — they are tuned values that happened to trade collision avoidance
against task success acceptably on SafeLIBERO, and they carry no guarantee on
another robot or another workspace.
