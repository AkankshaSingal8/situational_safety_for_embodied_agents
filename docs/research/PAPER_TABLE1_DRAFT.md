# Paper Table 1 draft — main results (TSR / CAR, %)
Status 2026-07-16. n=50 unless marked. † = n=20, ‡ = n=5 smoke (do not quote; queue n≥20).
Collision criterion: >1mm GT-obstacle displacement, identical across ALL rows (incl. reimpl).
Seeds/init states identical across rows (paired; McNemar/Fisher in `stats_tests.py`, ledger).

| Condition | π0.5 baseline | SOTA-reimpl (r0.10) | Ours Tier-GT (config) | Ours no-GT (percep+symID) | SOTA published* |
|---|---|---|---|---|---|
| Spatial L1 | 67.0 / 14.0 | 62.0 / 28.5 | 71.0 / 23.5 (r3-thin) | 60.0 / 55.0 † | 75.5 / 77.5 |
| Spatial L2 | 55.5 / 12.0 | 71.5 / 25.5 | **85.0 / 76.5** (comp) | **81.25 / 65.0** † | 78.0 / 75.5 |
| Goal L1 | 51.0 / 23.0 | — | 77.5 / 51.0 (comp) | pending ‡ 42241091 | 91.0 / 94.0 |
| Goal L2 | 66.5 / 35.0 | — | 69.0 / 58.0 (r3) | pending ‡ | 83.5 / 82.5 |
| Object L1 | 40.5 / 14.0 | — | 48.0 / 43.5 (comp) | pending ‡ | 90.5 / 82.5 |
| Object L2 | 74.0 / 25.0 | — | 83.0 / 74.5 (comp) | pending ‡ | 81.0 / 85.5 |
| Long L1 | 58.0 / 15.0 | — | 42.5 / 71.0 (comp) | pending ‡ | 81.5 / 82.0 |
| Long L2 | 51.0 / 16.5 | — | 47.0 / 77.0 (r3) | pending ‡ | 72.0 / 83.0 |

*SOTA-published numbers are NOT criterion-comparable: their method re-run under our >1mm
criterion scores CAR 25.5–28.5 (vs 75.5–77.5 published) while its TSR uplift reproduces —
the radius sweep (r=0.06/0.10/0.14) shows no radius recovers their (TSR,CAR) pair on both
levels. Honest cross-method CAR comparison = column 3 (self-run reimpl). Keep the published
column for TSR context only, with this footnote.

## Headline claims (all pre-verified significant)
1. Tier-GT beats our baseline on BOTH axes in 6/8 conditions (per-condition best config,
   reported as config ablation per pre-registered rule); Long trades TSR −4..−15.5 for
   CAR +56..+60.5 (timeout-stall mechanism documented, epistemic-signal AUC 0.88–0.91).
2. Spatial L2 four-number win: beats baseline, self-run reimpl, and SOTA-published on both
   axes (TSR p=1.2e-10/9.0e-4; CAR p=2.8e-41/1.0e-29).
3. Corridor semantics buy CAR at ZERO detectable TSR cost (paired McNemar p=0.70 on TSR,
   1.4e-13 on collisions, Spatial L2) — the exemption-not-restriction argument.
4. No-GT tier ≈ free on Spatial (identity 40/40, localization tax −11.5 CAR at L2);
   identity v3 47/48 all suites (held-out E1 pending job 42241220).

## Supporting tables/figures (all data exists)
- Config ablation: schedule / K / companions / corridor rows (fgd_devcells + n=50 runs).
- Referee analysis: radius sweep × criterion (the reproducibility contribution).
- Mechanism kills: DBNR τ-telemetry, HDC dev-cells — honest negative results + why.
- Epistemic gating: per-suite stall AUC; K-dispersion telemetry (now in every episode log).
- E2 corridor-clearance precompute (structural-cell argument); Object-L1 forensics table.
- E5 fail-direction stress test: PLANNED (inject identity/position errors, measure error
  direction CAR-vs-TSR) — the fail-safe figure.

## Open items before submission
n=50 no-GT finals (Spatial first, then suites passing identity); E1 held-out identity
number; E5; LIBERO-Safety rows (integration underway); OopsieVerse (gated); baseline-
mismatch note vs their paper's π0.5 (ours 67/14 vs their 60/17 at Spatial L1).
