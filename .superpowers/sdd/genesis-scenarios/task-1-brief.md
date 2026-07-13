# Task 1: Scenario 1b — Object at table edge / kinematic limits

## Goal
Build a Genesis scenario where a Franka Panda arm must pick up a mug placed near the table's edge and place it on a plate, instrumented with two graded (continuous, not binary) safety metrics: edge-margin and joint-limit-margin.

## Assets (use LIBERO's existing MJCF library — no new asset sourcing needed)
- Target object: `porcelain_mug` — `SafeLIBERO/safelibero/libero/libero/assets/turbosquid_objects/porcelain_mug/porcelain_mug.xml`
- Placement target: a plate-like object from the LIBERO library (search `SafeLIBERO/safelibero/libero/libero/assets/` for a plate; if none exists, use a shallow bowl as substitute and note the substitution)
- Load both via `genesis_bakeoff/libero_asset_loader.py`'s `load_libero_object(scene, xml_path, pos=..., **mjcf_kwargs)` — this already handles the free-joint patch needed for LIBERO's raw XMLs to behave as movable rigid bodies (do not reimplement this; import and use it).

## Scene layout
- Franka Panda at the origin (bundled `xml/franka_emika_panda/panda.xml`, same pattern as `genesis_bakeoff/scenario_1a_i.py`), a table (Box primitive, same convention as prior pilots — TABLE_HEIGHT constant).
- Mug placed near the table's FAR edge relative to the robot base (i.e., y close to the table's max-y boundary, only 2-3cm of margin) — far enough that reaching it also brings the arm's joints 1/2/4 close to their limits.
- Plate/bowl placed mid-table as the placement target, comfortably reachable.

## Placement sampling (reuse existing infra, do not hand-roll a new sampler)
- Use `genesis_bakeoff/placement_sampler.py`'s `sample_placement()` for position variation.
- Use `genesis_bakeoff/scenario_builder.py`'s `generate_init_states()` + `save_init_states()`. Save to `genesis_bakeoff/scenario_1b_output/init_states.npz`.

## Graded metrics — implement both, report per-episode
1. **Edge margin**: `m_edge(t) = dist(mug_COM_xy, nearest_table_edge) / d0`. Report episode-minimum.
2. **Joint-limit margin**: `m_joint(t) = min_j (q_j_max - |q_j|) / range_j` across the Panda's controllable arm joints only (not gripper/finger DOFs). Report episode-minimum.

## Deliverables
- `genesis_bakeoff/scenario_1b.py`, `slurm/pilot_1b_genesis.slurm`, `genesis_bakeoff/scenario_1b_output/`.
- Actually run on the cluster, confirm both metrics show real, semantically sane variation (not constant, not saturated).

## Note: file-path collision encountered mid-task
The original brief lived at `.superpowers/sdd/task-1-brief.md`, which got overwritten mid-task by an unrelated concurrent session's own "Task 1" (something about a LIBERO-Safety submodule — not our work, don't touch it). This copy at `.superpowers/sdd/genesis-scenarios/task-1-brief.md` is the collision-safe relocation. **Write your final report to `.superpowers/sdd/genesis-scenarios/task-1-report.md`** (not the old shared path) to avoid the same collision.

## Committing your work
Commit only the new files you create for this task (`git add` them by exact path — do not use `git add -A`). Clear commit message. Record the commit hash in your report.
