# FOL Safety Filter — SDD Progress Ledger

Branch: fol-safety-filter
Started: 2026-06-20

## Tasks

- [x] Task 1: primitives.py + kb.py (FOL engine) — commits 0165ed4..649d257
- [x] Task 2: cbf_mapper.py (FOL→CBF) — commit 0165ed4
- [x] Task 3: filter.py (FOLSafetyFilter) — commit 0165ed4
- [x] Task 4: run_safelibero_fol_openvla_eval.py (eval script) — commit fb85f44
- [ ] Task 5: Eval run + iteration loop (jobs running)

## Log

- 0165ed4: FOL module (primitives, kb, cbf_mapper, filter) + 19/19 unit tests
- fb85f44: FOL eval script (run_safelibero_fol_openvla_eval.py)
- 649d257: Fix workspace bounds formula + SLURM jobs

## SLURM Jobs

- 41558058: baseline (no filter) — safelibero_spatial Level I, 10 trials/task
- 41558059: FOL Level 3 — safelibero_spatial Level I, 10 trials/task

## Next Steps (after results land)

1. Compare TSR/CAR baseline vs FOL
2. If CAR not improving: tune spatial threshold in add_obstacle_avoidance (currently 0.10m)
3. If TSR dropping: reduce alpha_scale or increase distance thresholds
4. Run ablation (Level 1 vs 2 vs 3) with sbatch slurm/eval_fol_openvla_L1_ablation.slurm
5. Run Level II safety level once Level I is good
