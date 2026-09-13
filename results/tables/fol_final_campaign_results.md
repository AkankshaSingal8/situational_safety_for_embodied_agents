# FOL Semantic Safety Filter — Final Campaign Results (2026-07-11)

All n=50/task (200 eps/level), safelibero_spatial, og10 checkpoint, seed 7.
TSR = task success rate; CAR = collision avoidance rate. No GT in any control path.

## Headline table

| Config | L1 TSR/CAR | L2 TSR/CAR | Notes |
|---|---|---|---|
| Baseline (no filter) | 40.0 / 12.5 | 37.0 / 6.0 | |
| v18 (RGB triangulation, sphere) | 34.5 / 21.5 | 34.5 / 17.0 | |
| **v19 (ellipsoid + refinement)** | **37.0 / 20.5** | **37.0 / 14.5** | best balanced; L2 TSR == baseline @ 2.4x CAR |
| v19f (+3D descent gate) | 36.5 / 21.0 | 34.0 / 14.0 | gate helps t0 (+8pp), parity overall |
| **v20 (full corridor stack)** | **38.5 / 19.5** | 34.0 / 14.0 | best filtered L1 TSR (-1.5pp of baseline, 1.6x CAR) |
| v21-D2 (RGB-D + top-2) | 28.5 / 22.0 | 33.0 / 19.5 | best CAR both levels; L1 t2 = 0.64/0.54 |
| Oracle v17.2 (GT positions) | 42.0 / 56.0 | 42.0 / 47.5 | perception-error upper bound |

## Recommended paper configs
- L1 headline: v20 (38.5/19.5). L2 headline: v19 (37.0/14.5, TSR tie with baseline, 2.4x CAR).
- Safety-max variant: v21-D2 (CAR 22.0/19.5).

## Grounding accuracy (DIAG-GT, measurement-only telemetry)
- RGB-only @ n=396: median 0.223m, 24% <15cm (L1 0.284/14%, L2 0.175/34%).
- RGB-D one-view: L2 55% <15cm (smoke n=20); position 6-12cm when identity correct.
- Selection forensics: t1 = front-face depth bias (fixed by ray push); t2 = detection recall (pot merges w/ stove); t3 = cabinet-top objects fake HOVERING. Grounding identity is the residual bottleneck; top-2 barriers convert ranking errors into CAR.

## Mechanism ablations (video-forensic driven)
- Hand-body checkpoint (link7): wrist volume plowed obstacles the EEF had passed (topple onto goal plate → both metrics die).
- Corridor: cancellation slides but blocks corridors; global ray-rescale stalls; target-conditioned carve-out + 3D descent gate = v20's L1 gain. No published grasp-corridor carve-out exists (claimable).
- Uncertainty-shaped ellipsoids + multi-frame refinement: barrier shrinks as rays accumulate (MR-CBF framing).

## Provenance
v19: 42040822/23 · v19f: 42086151/154 · v20: 42088671/672 · v21-D2: 42090618/619 · baseline/v18/oracle: see fol_v18_final_results.md. Branch fol-safety-filter, commits ba5347c..a9bc045+.
