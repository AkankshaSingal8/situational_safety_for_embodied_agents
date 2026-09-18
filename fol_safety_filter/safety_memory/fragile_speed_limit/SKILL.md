---
schema_version: 1
id: fragile_speed_limit
revision: 1
description: Slow the end-effector while it is close to a fragile object.
roles:
  subject: {binding: eef}
  target: {binding: "objects_with_fact:IS_FRAGILE"}
applies_when: "IS_FRAGILE(TARGET) AND NEAR(SUBJECT, TARGET, 0.15)"
requires:
  predicate: limit_speed
  v_max: 0.10
violation_action: slow
unknown_action: stop_and_report
---

# Speed limit near fragile objects

Within 0.15 m of an object labelled fragile, the commanded translational step
is clamped to 0.10 m/s equivalent. The approach direction is untouched — this
rule does not steer, it only slows.

## Why slowing rather than a wider exclusion zone

A wider avoidance radius was the obvious first answer and it was wrong: it
over-constrains, because a fragile object is frequently the task target or
sits next to it, and pushing the gripper away from a fragile cup means never
picking up the cup. Separating "do not approach" from "approach gently" is the
reason these are two records with two effect predicates rather than one rule
with a mode flag.

`v_max: 0.10` and the 0.15 m radius are ported from the `IS_FRAGILE` branch of
`rule_composer.py`, which capped its radius at 0.15 for the same reason.

## Opt-in, and why

`default_loaded: false`. This is a faithful port, but "faithful" is not
"tested": `_props_from_name` returns `IS_FRAGILE` only for names containing
egg, vase, crystal or porcelain, and none of the reported SafeLIBERO obstacles
match. So the channel has never actually run on the reported configuration,
and enabling it by default would be shipping an untested path, not preserving
a measured one.

The golden parity trace cannot settle this either: it supplies the velocity
channel directly and so does not exercise rule selection at all. Enable this
record with a measured comparison, not with an argument.

## Limitations

A speed cap is not a force or impact bound. It limits the commanded step, not
the realised contact force, and says nothing about the momentum the arm
already carries. `IS_FRAGILE` is a semantic label supplied by an oracle or a
VLM; this rule neither verifies it nor degrades gracefully if it is wrong in
the permissive direction.

Under three-valued facts, an object whose `IS_FRAGILE` is *unknown* is not
selected as a target. That is not a judgement that it is sturdy — it is a
deferral, and `unknown_action: stop_and_report` is what the caller should
honour.
