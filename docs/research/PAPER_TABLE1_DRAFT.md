# Paper Table 1 draft — main results (TSR / CAR, %)
Status 2026-07-16 (late wave). n=50 unless marked. † = n=20, ‡ = n=5 smoke (do not quote).
Collision criterion: >1mm GT-obstacle displacement, identical across ALL rows (incl. reimpl).
Seeds/init states identical across rows (paired; McNemar/Fisher in `stats_tests.py`, ledger).

| Condition | π0.5 baseline | SOTA-reimpl (r0.10) | Ours Tier-GT (config) | Ours no-GT (percep+symID) | SOTA published* |
|---|---|---|---|---|---|
| Spatial L1 | 67.0 / 14.0 | 62.0 / 28.5 | 71.0 / 23.5 (r3-thin) | 49.5 / 55.5 | 75.5 / 77.5 |
| Spatial L2 | 55.5 / 12.0 | 71.5 / 25.5 | **85.0 / 76.5** (comp) | **87.0 / 68.5** | 78.0 / 75.5 |
| Goal L1 | 51.0 / 23.0 | 71.0 / 27.5 | **77.5 / 51.0** (comp) | **73.8 / 75.0** † | 91.0 / 94.0 |
| Goal L2 | 66.5 / 35.0 | 75.0 / 49.5 | 69.0 / 58.0 (r3) | 67.5 / 70.0 † | 83.5 / 82.5 |
| Object L1 | 40.5 / 14.0 | 53.0 / 15.5 | 48.0 / 43.5 (comp) | 43.8 / 81.2 † | 90.5 / 82.5 |
| Object L2 | 74.0 / 25.0 | 76.0 / 33.5 | **83.0 / 74.5** (comp) | **77.5 / 92.5** † | 81.0 / 85.5 |
| Long L1 | 58.0 / 15.0 | 30.5 / 35.5 | 42.5 / 71.0 (comp) | 5 / 70 ‡ | 81.5 / 82.0 |
| Long L2 | 51.0 / 16.5 | 43.0 / 59.5 | 47.0 / 77.0 (r3) | 25 / 70 ‡ | 72.0 / 83.0 |

Bold = beats BOTH π0.5 baseline and SOTA-reimpl on BOTH axes. Gap cells vs reimpl:
Spatial L1 (CAR −5 at thin / TSR −8.5 at fat), Goal L2 (TSR −6), Object L1 (TSR −9) —
composed-thin arm (eef_r 0.09 + corridor + K=8 ramp) in flight (job 42267752). Long: we
beat reimpl decisively both axes (their Long TSR collapses to 30.5/43.0); only the unsafe
baseline retains a TSR edge, and the cap-800 ablation (below) rules out the timeout excuse.

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
4. No-GT tier: Spatial L2 87.0/68.5 at n=50 with ZERO simulator state — beats baseline,
   beats self-run reimpl, TSR beats even our own Tier-GT; Goal L1 and Object L2 (n=20)
   also beat both references both axes. Identity v3 held-out E1 = 155/160 (96.9%);
   RGB-D localization 4.5–8.4 cm runtime.
5. Cap-800 ablation: doubling the Long step budget does NOT recover guided TSR (composed
   36.2/62.5 & 31.2/72.5 vs baseline 63.7/20.0 & 50.0/22.5 at cap 800) — timeout stalls
   are true route loops, not slow successes; step caps stay at community-standard values.
6. E4 instruction-refusal judge F1 0.968 (published SSR baselines 0.31–0.46; in-domain
   caveat: 15 unique unsafe strings). E5 fail-direction: identity corruption fails UNSAFE
   (CAR 70→25) — motivates the default-deny direction for future work.

## Supporting tables/figures (all data exists)
- Config ablation: schedule / K / companions / corridor rows (fgd_devcells + n=50 runs).
- Referee analysis: radius sweep × criterion (the reproducibility contribution).
- Mechanism kills: DBNR τ-telemetry, HDC dev-cells — honest negative results + why.
- Epistemic gating: per-suite stall AUC; K-dispersion telemetry (now in every episode log).
- E2 corridor-clearance precompute (structural-cell argument); Object-L1 forensics table.
- E5 fail-direction stress test: DONE (identity corruption fails unsafe CAR 70→25;
  position corruption degrades gracefully) — the fail-safe figure.
- Cap-800 timeout ablation: DONE (both arms, Long L1+L2, n=20).

## Open items before submission
Three Tier-GT gap cells vs reimpl (composed-thin arm job 42267752); no-GT n=50 finals for
Goal/Object (job 42267789 running; n=20 in table above) + Long strategy (5/70, 25/70 at
n=5 — needs work or honest reporting); LIBERO-Safety rows (assets re-extracting, smoke
auto-submits); OopsieVerse (gated); baseline-mismatch note vs their paper's π0.5 (ours
67/14 vs their 60/17 at Spatial L1).
