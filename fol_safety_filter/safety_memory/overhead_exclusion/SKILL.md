---
schema_version: 1
id: overhead_exclusion
revision: 1
description: Do not carry a payload through the column above a fragile object.
roles:
  subject: {binding: currently_grasped}
  target: {binding: "objects_with_fact:IS_FRAGILE"}
applies_when: "IS_FRAGILE(TARGET) AND ABOVE(SUBJECT, TARGET, 0.02)"
requires:
  predicate: avoid_region_above
  clearance_m: 0.02
  z_max: 1.40
  cancel_scale: 1.0
violation_action: avoid
unknown_action: stop_and_report
---

# Overhead exclusion during transport

A carried payload must not pass through the vertical column standing above a
fragile object. If the grasp slips, whatever is directly underneath takes the
impact — so the hazard is a region, not a surface, and avoiding it means
routing around rather than lifting higher.

## Not ordinary collision avoidance

`eef_obstacle_avoid` keeps a point out of a ball centred on an object. This
rule keeps the payload out of the object's horizontal footprint extruded
upward to `z_max`. A payload 40 cm directly above a fragile bowl is far
outside every avoidance sphere in this memory and squarely inside this
region. The two rules are not redundant and neither subsumes the other.

`clearance_m: 0.02` expands the footprint laterally. `z_max: 1.40` matches the
workspace ceiling in `WORKSPACE_BOUNDS`, so the column is capped at the
reachable volume rather than being unbounded.

## The new rule, and the one the demo toggles

This is the only record here that is not a port of existing behaviour — no
prior code applied a projected-region constraint. It is `default_loaded:
false` and its `metadata.json` records `evidence: none`, because it has no
measured episode results at all. Its purpose on this branch is to demonstrate
that the memory is genuinely load-bearing: a record can be added, validated,
and switched on through the manifest without touching filter code, and the
activation demo shows it firing on exactly the scenes where a payload is above
a fragile object and staying silent elsewhere.

Treat that as a mechanism demonstration. It is not evidence that the rule
improves safety, which would take the paired episode comparison this branch
deliberately does not run.

## Known conservatism

`ABOVE` compares object *centres* (`primitives.py:101`), so activation is
centre-over-centre rather than volume-over-footprint. A wide payload whose
edge overhangs a fragile object while its centre does not will not activate
the rule, and a narrow payload centred over a wide object activates it as soon
as the centres line up. The honest version checks the payload's swept occupied
volume against the extruded footprint; that needs extents this pipeline does
not reliably have, since `ObjectState.bbox_half` is a hardcoded
`[0.04, 0.04, 0.04]` for every oracle-state object (`filter.py:249`), and `[0.05, 0.05, 0.05]` on the vision path (`filter.py:601,633`).

So: a point-vs-column approximation with a documented failure direction, not a
swept-volume check. The 0.02 m clearance does not compensate for it.

## Limitations

The region is axis-aligned and static per timestep; a moving target is
re-evaluated but not predicted. Under three-valued facts an object whose
`IS_FRAGILE` is unknown is not selected, so an unknown object gets no
protection — a deferral, not a clearance, and `unknown_action` is what should
decide it. Nothing here establishes that a dropped payload would have hit the
object, only that the payload stayed out of a declared region.
