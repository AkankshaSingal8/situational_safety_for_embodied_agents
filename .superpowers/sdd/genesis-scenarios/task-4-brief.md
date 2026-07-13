# Task 4 Brief: Harden scenario 3a-ii (open bag of Skittles) to production

## Context
`genesis_bakeoff/scenario_3a_ii.py` (163 lines) is a bake-off-era pilot, not yet a production scenario. It uses Genesis's rigid-body solver (one entity per candy, 20 candies, tightly packed and overlapping at init so first-step contact resolution "bursts" them apart, mimicking a torn-open bag) as a sibling to 3a-i's MPM-based rice pour — a different granular regime (discrete near-rigid pieces vs. continuous flow).

It predates `genesis_bakeoff/standard_rig.py` (built and retrofit across 1a-i/1b/2c/3a-i/4a in the immediately-preceding task, commit `f43ec78`) and predates `genesis_bakeoff/scenario_builder.py`'s init-state save/load pipeline. It has no metrics module, no safe/unsafe scripted-motion variants, and no episode loop — it is a single scene-build/burst demo, not a benchmark scenario.

Known open issue from earlier planning notes: "3a-ii's energy-injection bug needs fixing before this counts as scenario-complete, not just pilot-complete" — investigate what this refers to (likely: the overlap-burst technique may inject unrealistic/inconsistent energy depending on solver substep count or initial overlap depth, producing either an unconvincing "candies barely move" result or an unphysical "candies fly off table" result). Diagnose by rendering/inspecting actual candy trajectories, don't assume — the issue may already be resolved or may not reproduce under current Genesis version.

## Task
Bring 3a-ii to the same production standard as 1a-i/1b/3b/4a:

1. **Retrofit to `standard_rig.py`**: import `set_robot_ready_pose`, `add_standard_agentview_camera`, `render_eye_in_hand` — same convention as every other scenario. Note: check whether this scenario needs a Franka arm at all (3a-i's retrofit report noted 3a-i omits robot-pose/wrist-camera since it has no Franka entity — determine whether 3a-ii is the same "no manipulator, pure physics-hazard" scenario, or whether it should have a Franka nudging/knocking the bag per the taxonomy's actual definition of 3a-ii — check the source taxonomy PDF context via `plan-how-to-create-cuddly-mist.md`'s Section 2 table if the exact scenario definition is ambiguous).

2. **Diagnose and fix the energy-injection issue.** Render/inspect candy trajectories across several seeds. If the burst produces implausible results (candies clip through the table, fly off-scene, or barely separate), fix the root cause — likely candidates: initial overlap depth vs. solver `substeps`/`dt`, contact stiffness/restitution defaults, or candy radius vs. spacing ratio. Document the diagnosis and fix in the report, the same way the 4a task documented its mug-mesh root cause.

3. **Define the safety-relevant metric.** This scenario's hazard is presumably "spilled/scattered candies beyond a safe workspace boundary" or similar (check the taxonomy definition). Write `genesis_bakeoff/metrics_3a_ii.py` — reuse patterns from `metrics_3b.py`/`metrics_4a.py` where applicable (footprint/boundary-crossing checks), with unit tests following the same convention (20+ regression tests, matching 3b/4a's precedent).

4. **Safe vs. unsafe variant.** Determine what distinguishes a "safe" episode from an "unsafe" one for this scenario (e.g., controlled/contained release vs. uncontained burst; or a policy correctly containing the scatter vs. not) — check the taxonomy definition first; if genuinely ambiguous, use engineering judgment consistent with 3a-i's sibling scenario and document the choice explicitly in the report.

5. **Init-state variation + episode loop.** Use `scenario_builder.py`'s `generate_init_states()`/`save_init_states()`/`load_init_states()` pattern (per 1a-i's precedent) to generate N (10, matching 3b/4a precedent) episode variations, run both variants, save `metrics_summary.json` to `genesis_bakeoff/scenario_3a_ii_output/`.

6. **SLURM job**, confirmed working allocation pattern from prior tasks (GPU-shared, v100-32:1, 4 CPUs, 60GB — check `slurm/pilot_4a_genesis.slurm` for the exact template). Use a single-shot `Bash run_in_background` wait with an `until squeue` loop when monitoring your own jobs, NOT repeated polling — this project's owner has explicitly asked to minimize monitoring token usage.

## What "done" looks like
- 3a-ii renders from the same standardized camera rig as every other scenario (or explicitly documents why it can't/shouldn't, matching 3a-i's precedent).
- The energy-injection issue is diagnosed and either fixed or confirmed already fine (with evidence).
- `metrics_3a_ii.py` exists with a real safety metric + passing unit tests.
- A full 10-episode run shows a clean safe/unsafe metric contrast (or, if genuinely not separable for this hazard type, an honest writeup of why not — do not fudge numbers to force a clean split).
- Report written to `.superpowers/sdd/genesis-scenarios/task-4-report.md`.

## Committing
Commit only the files this task touches, by exact path (not `git add -A`): `genesis_bakeoff/scenario_3a_ii.py`, `genesis_bakeoff/metrics_3a_ii.py`, `genesis_bakeoff/scenario_3a_ii_output/`, any new `slurm/*.slurm` file for this scenario. Do not touch other scenario files or `standard_rig.py`.

## Report file
Write to `.superpowers/sdd/genesis-scenarios/task-4-report.md`. Final message to controller: status, root-cause/fix summary if applicable, final metric numbers, SLURM job IDs, commit hash, concerns.
