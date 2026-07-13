# Task 12 Report: Scenario 5b — Safe Response to Unsafe Commands (push cup off table edge)

## Status: NOT DONE — stopped mid-debug per explicit user request, no committed work

This task is **incomplete**. Per the user's request (relayed by the coordinator)
to stop immediately and not start further iterations, this report documents
the current state honestly rather than claiming completion. Nothing has been
committed. All files below are present on disk, uncommitted.

## What exists on disk (uncommitted)

- `genesis_bakeoff/scenario_5b.py` — scene build + scripted push episode
  runner. Compiles cleanly (`python3 -m py_compile` passes). Follows the
  brief: reuses `standard_rig.py`, `libero_asset_loader.load_libero_object()`
  for the real `porcelain_mug.xml`, `placement_sampler.py` +
  `scenario_builder.py` for init-state variation, and `scenario_1b.py`'s
  table-edge geometry convention (`TABLE_CENTER_XY=(0.35,0.05)`,
  `TABLE_SIZE=(0.6,0.6,TABLE_HEIGHT)`, far edge at y=0.35). Built at
  `dt=2e-3, substeps=20` from the outset per the task-7 lesson.
- `genesis_bakeoff/metrics_5b.py` — pure-math graded metric
  `push_off_edge_violation_score()` (reuses `metrics_1b.nearest_edge_distance`),
  with **20 passing regression tests** (verified standalone via
  `python3 metrics_5b.py` → `ALL METRICS_5B SELF-TESTS PASSED`). This part
  is solid and not in question — all instability below is in the Genesis
  physics/scripted-motion layer, not the metric math.
- `slurm/pilot_5b_genesis.slurm` — SLURM job script, GPU-shared/v100-32:1/
  4 CPU/60GB pattern (copied from the working `pilot_2d_genesis.slurm`
  template).
- `genesis_bakeoff/scenario_5b_output/` — contains `init_states.npz`
  (50 valid init states, generated cleanly every run: 0 geometric_fail,
  0 physics_bad) and `metrics_summary.json`/rendered frames from the most
  recent COMPLETED run (job 42166326) — **not** from the in-flight job
  below, since that job never ran.

## SLURM jobs run today (all on the confirmed GPU-shared/v100-32:1/4CPU/60GB pattern)

| Job | Result | Finding |
|---|---|---|
| 42166191 | COMPLETED, exit 0 | First full pipeline run. `SAFE_UNSAFE_SEPARATION_CHECK: PASS` on paper (safe max 2.69 < unsafe min 2.71), but this was a **false pass**: 2/5 SAFE episodes actually launched the cup clear off the table (final y up to 1.03, z dropping to ~0.05-0.12m) despite the push target being only y=0.17 — i.e. the "safe" script was frequently producing unsafe outcomes. Root cause hypothesis: unconstrained-orientation IK (`quat=None, rot_mask=[False]*3`) let the wrist spin/drift during the lateral push, matching a failure class `scenario_2b.py` had already diagnosed and fixed for a different scenario. |
| 42166225 | COMPLETED, exit 0 | Applied the orientation fix (capture ready-pose quat once, hold fixed via `rot_mask=[True]*3` for every phase). This eliminated the launch bug entirely — all 5 SAFE episodes now score exactly 0.0 — but overcorrected: 3/5 UNSAFE episodes now show near-zero cup displacement (push essentially didn't happen). `SAFE_UNSAFE_SEPARATION_CHECK: FAIL` (safe mean 0.0, unsafe mean 0.55, overlapping at 0.0). Hypothesis: with orientation now rigidly downward, the actual push contact point is the fingertips, which sit ~2.8cm below the hand-link IK target (per `scenario_2b.py`'s own documented measurement) — `PUSH_Z=TABLE_HEIGHT+0.05` put fingertips only ~2cm above the table, too low to reliably strike the cup's body. |
| 42166326 | COMPLETED, exit 0 | Raised `PUSH_Z` to `TABLE_HEIGHT+0.08` (fingertips ~5cm above table, closer to cup mid-body). Result got WORSE in a different way: now overlap in BOTH directions simultaneously — 1/5 SAFE episodes still launched off the table (score 2.70), while several UNSAFE episodes still barely moved the cup. `SAFE_UNSAFE_SEPARATION_CHECK: FAIL`. This inconsistency (not a one-directional bias) pointed to genuine contact-velocity instability rather than a static height-calibration error: `free_phase()`'s pattern of re-solving IK toward the SAME fixed far-away target for the whole phase means the controller drives at high, variable speed toward a goal that can be 0.15-0.5m away, so the actual contact velocity when the gripper first touches the cup is essentially uncontrolled. |
| 42166366 | **Submitted, never ran (still PENDING in queue at stop time)** | Refactored the push motion to a quasi-static, linearly-interpolated hand target (`push_phase()`, new function in `scenario_5b.py`) instead of a single fixed far target, mirroring the lerp-based contact-motion pattern already proven elsewhere in this codebase (`carry_phase()` in `scenario_2d.py`/`scenario_3b.py`/`scenario_4a.py`). This is a plausible fix (bounds instantaneous position error, and thus commanded contact velocity, to a small roughly-constant value throughout the push) but is **UNVERIFIED** — the job had not started running when the stop instruction arrived. Do not treat this fix as confirmed.

## Honest assessment

The metric (`metrics_5b.py`) is done and correct. The scenario script's
scripted-push mechanic is not yet reliable: across three completed runs, the
SAFE/UNSAFE contrast never cleanly separated, and the failure mode changed
character between iterations (launch-on-safe → under-push-on-unsafe →
inconsistent-both-ways), which is the signature of a genuine contact-dynamics
sensitivity in this "closed-gripper-as-paddle" push mechanic, not a single
one-line bug. The quasi-static interpolated-target fix in the uncommitted
`scenario_5b.py` is a principled next step (it addresses the mechanism most
consistent with all three failure patterns: uncontrolled contact velocity)
but has not been empirically verified by a completed SLURM run.

## What's needed to finish this task

1. Run job 42166366 (or resubmit `slurm/pilot_5b_genesis.slurm` fresh, since
   that exact job may still be sitting in queue) to completion and check
   `SAFE_UNSAFE_SEPARATION_CHECK`.
2. If still not clean, likely next levers (in order of suspicion): (a) reduce
   `STEPS_PER_PHASE["push"]` variance sensitivity by making push speed
   explicitly bounded/uniform across variants rather than a fixed step
   budget for very different total distances; (b) reduce `GRIPPER_CLOSED`
   contact stiffness/friction sensitivity by widening `CONTACT_GAP`; (c)
   re-verify `PUSH_Z` once contact velocity is controlled, since the earlier
   height tuning was confounded by the velocity-instability bug.
3. Once a clean, non-overlapping separation is confirmed, capture the
   SAFE/UNSAFE agentview + eye-in-hand frames (the run script already saves
   these to `genesis_bakeoff/scenario_5b_output/`), then commit exactly:
   `genesis_bakeoff/scenario_5b.py`, `genesis_bakeoff/metrics_5b.py`,
   `genesis_bakeoff/scenario_5b_output/`, `slurm/pilot_5b_genesis.slurm`,
   and update `progress.md` + this report.

## Commit status

**No commit was made.** All files listed above remain uncommitted on disk in
the working tree, exactly as left when the stop instruction arrived, so no
work is lost.

## SLURM job IDs (for reference)

42166191 (completed, buggy), 42166225 (completed, buggy), 42166326 (completed,
buggy), 42166366 (submitted, pending/never ran at stop time — may still be in
queue or may need resubmission).
