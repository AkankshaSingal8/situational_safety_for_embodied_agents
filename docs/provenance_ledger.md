# Provenance Ledger

Maps every stored result file on `main` to its episode count and headline metrics, and
records which reported numbers are reproducible from a commit. Generated at tag
`pre-bugfix-2026-09-09`. Every later task cites this file when claiming a column is
unchanged.

## Inventory

| Directory | Files | Episodes/file | TSR range | CAR range |
|---|---|---|---|---|
| `baseline_benchmark` | 1 | 40 | 0.450 | 0.050 |
| `baseline_benchmark_L2` | 1 | 40 | 0.250 | 0.100 |
| `baseline_goal_L1` | 1 | 40 | 0.150 | 0.125 |
| `baseline_long_L1` | 1 | 40 | 0.000 | 0.025 |
| `baseline_object_L1` | 1 | 40 | 0.000 | 0.225 |
| `baseline_og10_L1` | 1 | 200 | 0.400 | 0.125 |
| `baseline_og10_L2` | 1 | 200 | 0.370 | 0.060 |
| `cosmos_benchmark` | 13 | 200,4,40 | 0.000–0.250 | 0.000–1.000 |
| `cosmos_benchmark_libsafety` | 4 | 45 | 0.022–0.200 | 0.800–1.000 |
| `fastwam_benchmark` | 8 | 200,40 | - | - |
| `fastwam_benchmark_50ep` | 1 | 200 | - | - |
| `fastwam_benchmark_libsafety` | 1 | 1 | - | - |
| `fastwam_benchmark_libsafety_full` | 60 | 3 | - | - |
| `fol_benchmark` | 5 | 40 | 0.100–0.550 | 0.050–0.700 |
| `fol_benchmark_L2` | 1 | 40 | 0.700 | 0.500 |
| `fol_benchmark_v5` | 1 | 40 | 0.500 | 0.275 |
| `fol_goal_L1` | 2 | 40 | 0.000–0.050 | 0.550–0.725 |
| `fol_goal_L1_n50` | 1 | 200 | 0.130 | 0.595 |
| `fol_goal_L1_v5` | 1 | 40 | 0.100 | 0.125 |
| `fol_goal_L2_n50` | 1 | 200 | 0.075 | 0.560 |
| `fol_long_L1` | 2 | 40 | 0.000 | 0.675–0.850 |
| `fol_long_L1_n50` | 1 | 200 | 0.015 | 0.720 |
| `fol_long_L2_n50` | 1 | 200 | 0.210 | 0.720 |
| `fol_object_L1` | 1 | 40 | 0.000 | 0.175 |
| `fol_object_L1_n50` | 1 | 200 | 0.195 | 0.070 |
| `fol_object_L1_v4` | 1 | 40 | 0.200 | 0.100 |
| `fol_object_L1_v5` | 1 | 40 | 0.000 | 0.100 |
| `fol_object_L2_n50` | 1 | 200 | 0.490 | 0.330 |
| `fol_spatial_L1_n50` | 1 | 200 | 0.480 | 0.595 |
| `fol_spatial_L2_n50` | 1 | 200 | 0.680 | 0.560 |
| `fol_v18_spatial_L1_n50` | 1 | 200 | 0.345 | 0.215 |
| `fol_v18_spatial_L2_n50` | 1 | 200 | 0.345 | 0.170 |
| `fol_v19_spatial_L1_n50` | 1 | 200 | 0.370 | 0.205 |
| `fol_v19_spatial_L2_n50` | 1 | 200 | 0.370 | 0.145 |
| `fol_v19e_spatial_L1_n50` | 1 | 200 | 0.345 | 0.195 |
| `fol_v19e_spatial_L2_n50` | 1 | 200 | 0.320 | 0.165 |
| `fol_v19f_spatial_L1_n50` | 1 | 200 | 0.365 | 0.210 |
| `fol_v19f_spatial_L2_n50` | 1 | 200 | 0.340 | 0.140 |
| `fol_v20_spatial_L1_n50` | 1 | 200 | 0.385 | 0.195 |
| `fol_v20_spatial_L2_n50` | 1 | 200 | 0.340 | 0.140 |
| `fol_v21d2_spatial_L1_n50` | 1 | 200 | 0.285 | 0.220 |
| `fol_v21d2_spatial_L2_n50` | 1 | 200 | 0.330 | 0.195 |
| `fol_v22_spatial_L1_n50` | 1 | 200 | 0.300 | 0.230 |
| `fol_v22_spatial_L2_n50` | 1 | 200 | 0.355 | 0.180 |
| `fol_v23b_taskbias_spatial_L1_n50` | 1 | 200 | 0.215 | 0.175 |
| `fol_v23c_replan_spatial_L1_n50` | 1 | 200 | 0.300 | 0.240 |
| `openvla_benchmark` | 8 | 200 | 0.000–0.425 | 0.015–0.260 |
| `pi05_benchmark` | 8 | 200 | - | - |

## Empirical confirmation of finding 1 (LIBERO-Safety collision channel)

The stored `cosmos_benchmark_libsafety` results confirm the `:constraints` audit directly,
with no re-run required:

| Suite | CAR in stored results | Explanation |
|---|---|---|
| `affordance` | **1.0000** | 0/15 BDDL files have a `:constraints` block. `info['cost'] == {}`, so `any(...)` is False **by construction** — vacuous, no measurement occurs. |
| `human_safety` | **1.0000** | Only `CheckRobotContact` present (12 instances). The predicate is evaluated every step but is structurally always False. |
| `obstacle_avoidance` | 0.9111 | Live `CheckContact` channel (31 instances) — measures **object–hazard** contact only. |
| `obstacle_avoidance_human` | 0.8000 | Live `CheckContact` channel (21 instances) — same qualification. |

The two 1.0000 values must be reported as `--` / not measured, never as perfect safety.

## Protocol gap, confirmed from the data

Every LIBERO-Safety result file holds **45 episodes** = 15 tasks x 3 trials x 1 seed.
The paper specifies 15 x 10 trials x 3 seeds = **450 per suite**. Current results are at
1/10 of the paper's sample size.

No stored result file contains a per-episode `collision` field, confirming finding 10:
`aggregate_table3_replication.py:63`'s `r.get("collision", False)` silently reads 0.0 for
every OpenVLA row.

## UNRECOVERABLE: fol_*_n50 campaign (2026-06-21/23)

The `slurm/eval_fol_*_n50.slurm` jobs ran `filter.py` from an uncommitted working-tree
state in `.worktrees/fol-safety-filter`. The committed version of that date (`0165ed4d`)
is a different 601-line QP/`_simple_repulsion` implementation, and neither it nor current
HEAD matches the line numbers cited in the run logs (`filter.py:256`, `filter.py:445`).
These numbers CANNOT be reproduced from any commit and MUST be re-run before publication.

Affected: `fol_{spatial,object,goal,long}_{L1,L2}_n50` (8 conditions, 200 episodes each).

**Additional confound — no same-checkpoint n=50 spatial baseline.** FOL n=50 uses
`moojink/openvla-7b-oft-finetuned-libero-spatial`; the only n=50 spatial baseline
(`baseline_og10_L1/L2`, 200 episodes) uses `...-libero-spatial-object-goal-10`. The
same-checkpoint baseline (`baseline_benchmark`) is 40 episodes. Any delta reported across
these two is a checkpoint confound, not a filter effect.

## Long-suite horizon divergence

`run_safelibero_fastwam_eval.py:93-97` and `run_safelibero_dreamzero_eval.py:61-65` use
600 steps for `safelibero_long`; every other driver and the SafeLIBERO paper use 550.
Existing Fast-WAM and DreamZero long-suite numbers are therefore not comparable to the
other rows. Corrected in Task 1.6; results produced before that change are marked here.
