# Caveat 1: reconciling LIBERO-Safety violation rates across the two projects

Checked 2026-08-18 against source, not docs.

## Scene set: COMPATIBLE (no problem)
`run_guided_libero_safety_eval.py` loads
`bm.get_task_bddl_file_path_by_level_id(...)` ->
`bddl_files/<suite>/L<k>/<task>.bddl`, the benchmark's single shipped file.
knowledge_action_gap's `source_bddl` is documented as "the UNTOUCHED
LIBERO-Safety file" (`e3_paper_reproduction_suites_initstates.py:22`); its
`safe_bddl` is a KAG-*generated* twin (`build_obstacle_avoidance_twins.py`).
=> The corrected KAG protocol and the flow-guidance board run the SAME scenes.
The 25 cm `obs_init_region` defect does not affect the board.

## Metric: NOT the explanation (checked, and it points the wrong way)
`bddl/parsing.py:package_predicates` **strips the `and`** and appends each
conjunct as a SEPARATE entry in `parsed_problem['constraints']`. So
`(And (CheckRobotContact H) (CheckContact OBJ H))` becomes two constraints, and
`run_guided_libero_safety_eval.py:1940` does `any(v > 0 for v in cost.values())`
-- a DISJUNCTION, not the conjunction the code comment assumes.

  - human_safety: constraint is `(And (CheckRobotContact left_hand_1))` -- a
    single conjunct. The board's criterion is therefore IDENTICAL to KAG's
    `hazard_contact` (both reduce to `check_robot_contact`, which scans geoms
    whose name contains "gripper" or "robot" at geom_group==0;
    `bddl_base_domain.py:1027`, mirrored at `e3_kill_experiment_obstacle.py:91`).
  - obstacle_avoidance*: the board's criterion is STRICTLY LOOSER than KAG's
    (robot-contact OR object-contact vs robot-contact alone).

=> On both suites the board should report >= KAG's rate.

## The discrepancy, stated
|                | KAG (hazard_contact) | Board (benchmark constraints) |
|----------------|----------------------|-------------------------------|
| obstacle_avoidance       | 1.3%  | 4.7-5.0% (consistent) |
| obstacle_avoidance_human | 16.0% | 5.3% (BACKWARDS)      |
| human_safety             | 32.0% | 0%   (BACKWARDS, identical criterion) |

`oa` is consistent with the looser board criterion. `oah` and `hs` are backwards
from what the criteria predict. Unexplained; needs an empirical replay of
baseline episodes under both scorers (GPU hours we do not have before Aug 20).

Remaining differences not yet excluded: step budget/horizon, init-state pool
(KAG uses `.pruned_init` indices 0-9; board uses the full pool), task subset per
level, and checkpoint build.

## Two real defects found on the way (disclosable)
1. `_check_constraint` writes `cost[constraint[0]]`, keyed by PREDICATE NAME. A
   task with two `CheckContact` conjuncts (e.g. `put_both_moka_pots_on_the_stove`)
   silently keeps only the last one. `bddl_base_domain.py:931`.
2. `_check_constraint` returns `{}` when `done` is True, so a violation occurring
   on the success step is never counted. `bddl_base_domain.py:928`.

## Decision for the ReS AI paper
Do NOT present the two rates as commensurable. E2/E3 motivate the paper; they do
not need to share a y-axis with the board. Concretely:
  - state KAG's criterion inline ("any robot-hazard geom contact");
  - state the board's criterion inline ("the benchmark's own `:constraints`
    predicates, any firing");
  - one footnote saying the two criteria differ and rates are not comparable
    across projects;
  - never write the 16% number in the same sentence/table as a board row.
