# Rule transfer: activation correctness on oracle facts

**These are activation-correctness numbers on a hand-declared oracle fact fixture. They are not task-success or violation-rate results, they say nothing about perception, and they share no table with SafeLIBERO TSR/CAR or LIBERO-Safety SR.**

The library was loaded once and not edited between scenes; 11 scenes, 5 records.

## Records, loaded once

| record | rev | effect | subject | target | hash |
|---|---|---|---|---|---|
| `arm_link_obstacle_avoid` | 1 | `avoid_sphere` | `arm_checkpoints` | `obstacles` | `89305aec0bd9` |
| `eef_obstacle_avoid` | 1 | `avoid_sphere` | `eef` | `obstacles` | `bb2f3fd80e19` |
| `fragile_speed_limit` | 2 | `limit_speed` | `eef` | `objects_with_fact:IS_FRAGILE` | `6d50b465274f` |
| `overhead_exclusion` | 1 | `avoid_region_above` | `currently_grasped` | `objects_with_fact:IS_FRAGILE` | `639e2ea872b6` |
| `spillable_rotation_lock` | 1 | `limit_angular_rate` | `currently_grasped` | `—` | `15fc70097258` |

Identical hashes were printed before the first scene and after the last, so no record was re-authored per scene.

## Activation per scene

| record | Bag of sugar | Bottle | Candle | Cup | Fish Tank | Headphones | Knife | Nothing | Plate | Soup | Sponge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `arm_link_obstacle_avoid` | – | – | – | – | – | – | – | – | – | – | – |
| `eef_obstacle_avoid` | – | – | ● | ● | – | – | ● | ● | – | – | ● |
| `fragile_speed_limit` | – | – | ● | ● | · | – | ● | ● | · | · | ● |
| `overhead_exclusion` | – | – | ● | ● | · | – | ● | – | · | · | ● |
| `spillable_rotation_lock` | ● | ● | ● | ● | ● | · | · | – | · | ● | · |

● fired · bound but correctly silent – nothing to bind to in this scene

### How the robot was posed

A rule with a geometric clause has to be given a pose it can be satisfied at, or it binds and never fires and the sweep says nothing about it. The pose is derived from each record's declared roles, not from its expected answer, and is the same for every scene:

- `arm_link_obstacle_avoid`: neutral
- `eef_obstacle_avoid`: neutral
- `fragile_speed_limit`: end-effector placed …, neutral (no fragile object in scene)
- `overhead_exclusion`: neutral (no fragile object in scene), payload placed …
- `spillable_rotation_lock`: neutral

## Binding transfer

The same record binding to different objects across scenes, with no edit, is the property under test:

- `eef_obstacle_avoid` bound to `balloon` (5 scenes), `book` (5 scenes), `camera` (5 scenes), `glass_cup` (5 scenes), `laptop` (5 scenes), `paper_towel` (5 scenes)
- `fragile_speed_limit` bound to `laptop` (5 scenes)
- `overhead_exclusion` bound to `balloon` (4 scenes), `camera` (4 scenes), `glass_cup` (4 scenes), `laptop` (4 scenes)

## Scored against the benchmark annotations

| record | scenes | precision | recall | misses |
|---|---|---|---|---|
| `spillable_rotation_lock` | 11 | 1.00 | 0.86 | `Plate` |

Records without a row have no benchmark annotation to score against; inventing one would be scoring ourselves. Their activation profile is above.

## Manifest toggle

Default-loaded: `arm_link_obstacle_avoid`, `eef_obstacle_avoid`. Withheld: `fragile_speed_limit`, `overhead_exclusion`, `spillable_rotation_lock`. Switching a record on is a manifest edit; no filter code changes.
