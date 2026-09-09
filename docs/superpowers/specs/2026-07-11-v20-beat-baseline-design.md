# v20 — Beat Baseline TSR While Keeping CAR Above Baseline (GT-free)

**Date:** 2026-07-11
**Goal:** TSR ≥ 40.0 (L1) and ≥ 37.0 (L2) with CAR > 12.5 / 6.0, n=50, seed 7, zero GT in the control path.
**Strategy (user-selected):** stack all evidence-backed fixes + ride-along diagnostics in one run.

## Evidence base (locked n=50 results)

| Config | L1 TSR/CAR | L2 TSR/CAR |
|---|---|---|
| baseline | 40.0/12.5 | 37.0/6.0 |
| v19 (ellipsoid+refine, cancel) — current best | 37.0/20.5 | 37.0/14.5 |
| v19e (xy-gate carve + hand) | 34.5/19.5 | 32.0/16.5 |
| oracle v17.2 | 42.0/56.0 | 42.0/47.5 |

TSR deficit = corridor cells (L1 t0/t1 = 0.10/0.10). CAR-vs-oracle gap (~35pp) unexplained — identity errors suspected, unmeasured at scale.

## v20 components

1. **Hand checkpoint ON** (`FOL_HAND_CHECKPOINT=1`, existing code, radii 0.10/0.06) — video-proven topple mechanism on t0; v19 finals ran without it.
2. **3D-aware descent gate** (commit 3856dfd, already on branch) — relaxation fires during vertical grasp descent onto the target; v19f (running) isolates this component.
3. **Place-goal-aware relaxation (NEW)** — the grounder returns `mentioned_xy`: xy of ALL task-mentioned candidates (not just the nearest-to-center one). The filter's aiming gate fires when the command aims at ANY mentioned object (grasp target OR placement goal). Removes the placement-descent blocking on t0 ("place it on the plate" with the pot beside the plate).
4. **Denser early refinement** — `FOL_REFINE_STEPS=2,6,10,14,18` (5 wrist-VLM calls ≈ 8 s/episode) to shrink the raw 2-view error before the arm first approaches the obstacle.
5. **Diagnostic-only GT logging (NEW, measurement only)** — at setup and after each accepted refinement, log `[DIAG-GT] grounded=<name> d_to_gt=<m>` = distance from the grounded position to the nearest obs-dict `*_obstacle` GT position. NEVER read in any control branch (grep-enforceable: the value flows only into `logger.info`). Gives n=200 identity-accuracy + position-error distributions to target the next iteration if v20 falls short.

## Non-goals / constraints

- `vlm_pipeline/run_safelibero_fol_openvla_eval.py` untouched.
- No GT in control: components 1-4 consume images + proprioception + calibration only; component 5 is write-only telemetry.
- Honest-failure path and fallback speed cap unchanged.

## Verification

- Unit tests for the multi-target gate + diagnostics inertness (control output identical with/without GT keys present).
- Smoke n=5 both levels → sanity (no crashes, gate/hand/diag lines present) — NOT used for config selection (n=5 noise burned us twice).
- n=50 both levels, seed 7 → compare to the table above; fold in v19f when it lands.
- Success gate: L1 TSR ≥ 0.40 AND CAR > 0.125; L2 TSR ≥ 0.37 AND CAR > 0.06.

## Risks

- Hand checkpoint may re-cost corridor TSR (v19e confound); the multi-target gate + 3D gate are the counterweights. If L1 t0/t1 still bleed, `[DIAG-GT]` tells us whether the residual is identity (fix MENTIONED/NEAREST_TO_PATH) or geometry (shrink radii floors).
- v19f may land mid-implementation; its per-cell numbers adjudicate the 3D gate's solo effect and get folded into the results table either way.
