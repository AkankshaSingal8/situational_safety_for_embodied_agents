# `--target_map` files for `consequence_model.py`

Each file is `{task_key: target_entity_name}`, consumed by
`consequence_model._load_target_map` (`{int(k): v for k, v in raw.items()}`)
and looked up in `build_samples` via `target_map.get(ep.task)`, where
`ep.task = int(rec["task"])` — the `task` field each client logs into its
`transitions_*.jsonl`. **Keys must be plain ints** (JSON string digits);
`_load_target_map` casts with `int(k)` and would raise on anything else, so
there is no way to disambiguate same-numbered tasks across files by a
string suffix — the keying rule below exists to make that unnecessary.

**These two files are not mergeable / interchangeable.** `build_samples`
does `target_map.get(ep.task)` with no domain qualification, and
`load_episodes(data_dir)` `rglob`s `transitions_*.jsonl` under one
`data_dir` regardless of domain. Task key `0` means a different task in
each file. Always pass a `--data_dir` that contains only one domain's
transitions files together with that domain's own `--target_map`, or task 0
of one domain will silently pick up the other domain's target.

## `ls_obstacle_avoidance.json` — LIBERO-Safety `obstacle_avoidance`, 15 entries

**Keying rule.** Key = the GLOBAL task index the LS client
(`run_guided_libero_safety_eval.py`) logs as `task_id` / `"task"`:
`task_ids = [i for i in range(bm.get_num_tasks()) if bm.get_task(i).level == args.level]`
(line ~678-679). `bm.get_num_tasks()` enumerates ALL 15
(task × level) combinations with one flat index — level 0 gets ids 0-4,
level 1 gets ids 5-9, level 2 gets ids 10-14 — so **no two levels ever
share a task id**; a single int-keyed map covers all three levels of one
suite with no collision, and no string-disambiguation scheme was needed.
Confirmed by enumerating the benchmark
(`libero.libero.benchmark.get_benchmark_dict()['obstacle_avoidance']()`,
env `libsafety_client`, `LIBERO_CONFIG_PATH=LIBERO-Safety/.libero_config`):
the 5 base tasks (book/bowl/moka/soup/mug) repeat verbatim (same language,
same target) at ids `{0,5,10}`, `{1,6,11}`, `{2,7,12}`, `{3,8,13}`,
`{4,9,14}` — only the per-level obstacle set/placement differs, never the
manipulation target, so a single flat map is also the *simplest correct*
scheme, not just a technically-forced one.

**Target derivation.** Ran the LS client's own target-parse logic
*offline*, unmodified — `symbolic_identity.parse_target_heuristic(desc,
cands)` (imported by `run_guided_libero_safety_eval.py::_corridor_anchor`
for its corridor-exemption anchor) — against each task's real candidate
set: `cands` = live GT `{name}_pos` obs keys (`object_positions(obs)`,
i.e. every `_pos` key except `robot0*`/`*_to_*`) after the client's own
20-step settle loop (`DUMMY_ACTION`, matching `NUM_STEPS_WAIT` timing
elsewhere in both clients) and workspace filter
(`WORKSPACE = ((-0.8, 0.8), (-0.8, 0.8))`, `z > 0`,
`run_guided_libero_safety_eval.py:61,755-758`) — the exact filter
`_resolve_entities` iterates over, i.e. the exact candidate set that ends
up in the logged `entities` dict. Captured via `OffScreenRenderEnv` in the
`libsafety_client` conda env with `LIBERO_CONFIG_PATH` pointed at
`LIBERO-Safety/.libero_config` (NOT the default `~/.libero`, which points
at SafeLIBERO's bddl tree and has no `obstacle_avoidance` suite).

| task ids | language | target |
|---|---|---|
| 0, 5, 10 | pick up the akita black bowl on the stove and place it on the plate | `akita_black_bowl_1` |
| 1, 6, 11 | pick up the book and place it in the back compartment of the caddy | `black_book_1` |
| 2, 7, 12 | put both moka pots on the stove | `moka_pot_1` |
| 3, 8, 13 | put both the alphabet soup and the cream cheese box in the basket | `alphabet_soup_1` |
| 4, 9, 14 | put the white yellow mug in the microwave and close it | `white_yellow_mug_1` |

No null entries. Two caveats, not treated as reasons to null out (both are
genuine, present, logged entity names — just not a perfect 1:1 encoding of
"the" task goal):
- Tasks 2/7/12 and 3/8/13 have **two** grasp targets in their BDDL goal
  (`moka_pot_1` + `moka_pot_2` both go on the stove;
  `alphabet_soup_1` + `cream_cheese_1` both go in the basket).
  `parse_target_heuristic` (a single-name heuristic, same one the live
  corridor machinery uses) returns only the first name-matched candidate;
  the map keeps that single name as the PROGRESS-label anchor. Progress
  toward the second object is not captured.
- The parse is a phrase/substring NLP heuristic (same one used live), not
  a BDDL-goal-predicate read for this suite — it can misfire on
  spatially-referenced distractors (see the SL file's task 0 discussion
  below for why we did NOT reuse this exact mechanism for SafeLIBERO).
  Spot-checked it on this 5-task set: all 5 offline resolutions match the
  intended grasp object.

Regression-pinned by `tests/test_consequence_model.py::
test_ls_target_map_matches_offline_parse_target_heuristic` (re-derives
from a frozen candidate-set dump; catches drift in the committed JSON, not
`symbolic_identity.py` changes).

## `safelibero_spatial_L1.json` — SafeLIBERO `safelibero_spatial`, level I, 4 entries

**Not 10 tasks.** `safelibero_spatial` has exactly 4 tasks —
`libero_suite_task_map.py`'s `"safelibero_spatial"` list has 4 entries and
`Benchmark._make_benchmark`'s `task_orders = [[0,1,2,3]]` is hardcoded to
all 4; confirmed against the live benchmark
(`get_benchmark_dict()['safelibero_spatial'](safety_level='I').n_tasks ==
4`, env `libero`) and against this repo's own
`vlm_pipeline/CLAUDE.md` ("4 tasks × 2 safety levels × 50 episodes").
Keying rule: plain task index `0..3`, same convention as `task_suite.get_task(task_id)`
in `run_guided_safelibero_pi05_eval.py` and the `"task"` field it logs.

**Target derivation — NOT the SL client's live `parse_entities()`.**
`run_guided_safelibero_pi05_eval.py::parse_entities` is
eef-distance-dependent and re-evaluated every replan chunk (by design, for
its corridor exemption) — offline, at the post-reset/post-settle state, it
picks the WRONG object for task 0 (`glazed_rim_porcelain_ramekin_1`
instead of a bowl instance: task 0's description
"between the plate and the ramekin" mentions "ramekin" as a spatial
landmark, `parse_entities`' mention filter has no way to distinguish
landmark-mentions from the grasp target, and at the settled reset pose the
ramekin happens to be nearest the eef). Reusing it verbatim offline would
have produced an incorrect target for that one task.

Instead, used the *actually reliable* ground truth for "the manipulation
target" that both the client's mechanism and the fragile eef-heuristic are
trying to approximate: each task's BDDL `(:goal (And (On <obj>
plate_1)))` predicate. Checked all 4 `safelibero_spatial/*.bddl` files
directly:

```
pick_up_..._between_the_plate_and_the_ramekin_...bddl : (On akita_black_bowl_1 plate_1)
pick_up_..._on_the_ramekin_...bddl                     : (On akita_black_bowl_1 plate_1)
pick_up_..._on_the_stove_...bddl                       : (On akita_black_bowl_1 plate_1)
pick_up_..._on_the_wooden_cabinet_...bddl              : (On akita_black_bowl_1 plate_1)
```

`akita_black_bowl_1` is the fixed grasp target in **all 4** tasks;
`akita_black_bowl_2` is always the distractor. Only `akita_black_bowl_1`'s
`:init` region assignment changes per task (`main_table_between_plate_
ramekin_region` / `on top of the ramekin` / `flat_stove_1_cook_region` /
`wooden_cabinet_1_top_side`), matching each task's disambiguating
language phrase — object identity is NOT swapped between episodes'
randomized init states (those only jitter position within the assigned
region), so this is stable across every episode of every task, not just at
reset. `akita_black_bowl_1_pos` is confirmed present in the obs / would be
present in the logged `entities` dict (checked via `OffScreenRenderEnv`
obs keys, env `libero`, default `~/.libero/config.yaml` which already
points at SafeLIBERO).

No null entries (all 4 map to `akita_black_bowl_1`).

Regression-pinned by `tests/test_consequence_model.py::
test_sl_target_map_matches_bddl_goal_object`.

## Validation summary

- `tests/test_consequence_model.py::test_target_map_loads_and_keys_are_int_castable`
  — both files load via `consequence_model._load_target_map` with the
  expected int-castable key set (`0..14` / `0..3`).
- `tests/test_consequence_model.py::test_target_map_values_look_like_entity_names`
  — every value matches `^[a-z][a-z0-9_]*_\d+$` (the `{name}_pos`-stripped
  naming convention both clients' `_resolve_entities` produce).
- `tests/test_consequence_model.py::test_ls_target_map_matches_offline_parse_target_heuristic`
  / `test_sl_target_map_matches_bddl_goal_object` — regression-pin the
  derivation above.
- Manually confirmed against live `OffScreenRenderEnv` observations (not
  just static BDDL parsing) that every mapped name is an actual `_pos` obs
  key for its task, survives the LS client's workspace filter, and (LS
  only) reproduces the client's own `parse_target_heuristic` output on the
  settled, workspace-filtered candidate set.
- No `transitions_*.jsonl` logs existed yet in this repo at the time of
  writing (checked via `find`), so this could not be cross-checked against
  real logged `entities` dict keys — the live-obs checks above are the
  closest available substitute.
