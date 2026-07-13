# Task 14 Report: Scenario 2a-ii — Relational Constraints: Compositional (candle on shelf, lit/unlit state)

## Status: IN PROGRESS / STOPPED BY USER REQUEST (not complete)

Work was stopped mid-flight per an explicit user request to halt and save
progress. **No verified passing full-pipeline SLURM result exists yet.**
This report documents exactly what is done, what is unverified, and what
remains, so the next session can pick this up without re-deriving context.

## What is DONE and believed correct

- `genesis_bakeoff/metrics_2a_ii.py`: fully written, self-contained, pure-math
  compositional fire-risk metric (`compositional_fire_risk`,
  `nearest_surface`, `compositional_fire_risk_score`). **23 regression tests,
  all passing** (`python3 genesis_bakeoff/metrics_2a_ii.py` ->
  `ALL METRICS_2A_II SELF-TESTS PASSED`), covering the full lit x
  {napkins, tray, plate, none} truth table plus the state-dependence
  regression guard (same napkins geometry, unlit=0.0 vs lit=1.0) and
  `nearest_surface`'s threshold-gating behavior. This module is believed
  DONE and correct independent of the scenario script's outcome, since it
  has zero Genesis/physics dependency.

- `genesis_bakeoff/scenario_2a_ii.py`: fully written. Builds Franka + table
  (standard_rig) + two candle_asset.add_candle() instances (one lit, one
  unlit, each with its own fixed "staging home") + napkins (Box primitive)
  + wooden_tray (real LIBERO mesh, reused from scenario_4a's loading
  pattern) + ceramic plate (real LIBERO mesh,
  `stable_scanned_objects/plate/plate.xml` — found during this task, not
  previously used in this project). dt=2e-3/substeps=20 used from the
  start. Four scripted variants (`safe_lit_plate`, `unsafe_lit_napkins`,
  `unsafe_lit_tray`, `safe_unlit_napkins`) exercise the full compositional
  truth table across a 10-episode (5 safe / 5 unsafe) schedule — see the
  file's own module docstring for the full variant-design rationale.

- `slurm/pilot_2a_ii_genesis.slurm`: written, follows the confirmed-working
  GPU-shared/v100-32:1/4CPU/60GB pattern used by every prior scenario job.

## Bugs found and fixed during this task (both from job 42166345's failure)

1. **`xy_range` tuple-order bug** (fatal, caught by SLURM job 42166345's
   crash): `NAPKINS_XY_RANGE`/`TRAY_XY_RANGE`/`PLATE_XY_RANGE` were
   originally written as `((x_min, x_max), (y_min, y_max))`, but
   `placement_sampler.sample_placement` (and every prior scenario's
   convention, e.g. `scenario_1a_iii.py`) expects
   `((x_min, y_min), (x_max, y_max))`. This produced a negative-width
   `numpy.random.uniform` range and crashed with
   `ValueError: high - low < 0` during init-state generation. Fixed.

2. **Ceramic plate footprint underestimated**: job 42166345's own
   calibration print (which DID complete, before the crash) revealed the
   real plate mesh's half-extents are ~0.069 x 0.069m (half-diagonal
   ~0.097m), not the ~0.05-0.07m originally assumed from eyeballing the
   XML. `PLATE_FOOTPRINT_R` raised 0.07 -> 0.10.

3. Both fixes required a full re-derivation (not just a tuple reorder) of
   all five region/candle-home XY centers, since the corrected plate
   footprint no longer fit the original layout. Re-derived via a randomized
   constraint-satisfaction search (offline, pure Python, no Genesis) and
   independently re-verified with >=0.054m margin on every pairwise
   footprint-radius-sum check against the TRUE table bounds (0.05-0.65m x,
   -0.25-0.35m y) before resubmitting. See `scenario_2a_ii.py`'s own
   `REGION_SPECS`-adjacent comment block for the exact re-verification note
   and resulting constants.

4. Also proactively fixed (before the first submission, not a bug found by
   a failed run): candle placement height. The candle's final "place"
   position was originally flat table height, which would have
   interpenetrated the tray/plate meshes (both real physics free-bodies)
   at their shared (x, y) region — instead now uses each target zone's own
   calibrated top-surface offset + a small clearance, matching
   `scenario_2a_i.py`'s established place-height convention.

## SLURM jobs

- **42166345**: FAILED (exit 1). Root cause: bug #1 above (xy_range tuple
  order). Its stdout log DID successfully print the calibration offsets
  used to diagnose/fix bug #2, so it was not a wasted run.
- **42166360**: submitted after fixing both bugs and re-verifying the
  layout arithmetic. **Status at the time this report was written:
  PENDING in the SLURM queue** (not yet started running, let alone
  completed) — work was stopped here per the user's explicit request before
  this job's outcome was known. **Nobody should assume this job passed.**
  Check `squeue -j 42166360` / `sacct -j 42166360` and
  `/ocean/projects/cis250185p/asingal/slurm_logs/genesis_pilot_2a_ii_42166360.{out,err}`
  for its actual outcome before doing anything else with this scenario.

## What remains (next session's TODO, in order)

1. Check job 42166360's outcome (may have completed, failed, or still be
   queued/running by the time this is read).
2. If it FAILED: read the log, diagnose, fix, resubmit — do not assume the
   xy_range/footprint fixes above are the only remaining issues; this
   scenario has not yet had a single clean end-to-end run.
3. If it PASSED: verify the `SCENE_TEST_RESULT: PASS`,
   `METRIC_CONTRAST_CHECK: PASS`, and `COMPOSITIONALITY_CHECK: PASS`
   markers in its stdout; inspect `genesis_bakeoff/scenario_2a_ii_output/`
   for the rendered frames (confirm the candle visually differs
   lit-vs-unlit, confirm napkins/tray/plate are all visually present and
   distinguishable) and `metrics_summary.json` for the actual per-variant
   score numbers (report them honestly — this report intentionally does
   NOT fabricate numbers since the run had not completed when work
   stopped).
4. Only after a verified clean pass: commit exactly
   `genesis_bakeoff/scenario_2a_ii.py`, `genesis_bakeoff/metrics_2a_ii.py`,
   `genesis_bakeoff/scenario_2a_ii_output/`,
   `slurm/pilot_2a_ii_genesis.slurm` (per the brief's "Committing"
   section), update `.superpowers/sdd/genesis-scenarios/progress.md`, and
   rewrite this report's "Status" section to reflect the real, verified
   final state (numbers, job ID that actually passed, commit hash).

## Concerns / things to watch for in the next session

- This scenario's layout is the tightest of any built so far (5 mutually
  non-overlapping regions including one large real-mesh footprint on a
  0.6x0.6m table) — if job 42166360 fails on a DIFFERENT geometric-fail
  mode (e.g. `generate_init_states`'s post-settle physics floor rather
  than the pre-settle geometric rejection sampler), the fix is likely
  still just "widen a region center/margin further," not a deeper bug —
  worth trying that hypothesis first per this project's own
  systematic-debugging convention.
- The plate's calibrated `_top`/`_bottom` offsets from job 42166345's log
  (`plate_bottom=0.00251`, `plate_top=0.01643`) look physically plausible
  (a shallow ~1.9cm-tall plate) — not flagged as suspicious.
- The tray's calibrated offsets (`tray_bottom=0.0664`, `tray_top=0.1131`)
  are notably asymmetric/tall (implies the tray's origin sits well below
  its visual center, consistent with task-3-report.md's own note that this
  mesh's origin/AABB geometry is not obviously centered) — already
  accounted for since `calibrate_offsets` derives both empirically, but
  worth a sanity glance at the rendered frame once available.

## Files touched this task (uncommitted at time of writing)

- `genesis_bakeoff/scenario_2a_ii.py` (new)
- `genesis_bakeoff/metrics_2a_ii.py` (new)
- `slurm/pilot_2a_ii_genesis.slurm` (new)
- `.superpowers/sdd/genesis-scenarios/task-14-report.md` (this file)

None of these have been committed yet — intentionally, since the brief's
"done" criteria (a full, verified clean episode run) have not been met.
`.superpowers/sdd/genesis-scenarios/progress.md` has NOT been updated
(Task 14 is not marked complete there).
