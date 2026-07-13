# Tier-GT In-Denoising DCBF Flow Guidance on pi0.5 — Results (2026-07-12)

Worktree: .worktrees/flow-guidance-tier-gt (branch flow-guidance-tier-gt)
Config r2: gamma=0.9, d_safe=0.01, eef_radius=0.09, inflation_slope=0.004/step, GT obstacle pos, sphere barriers.
Collision criterion: >1mm obstacle displacement (identical to stored baselines). Seed 7.

## Spatial L1 n=50/task (job 42162199, 200 episodes) — THE METRIC RUN
| | TSR | CAR | ETS | activation |
|---|---|---|---|---|
| pi0.5 baseline (stored) | 67.0 | 14.0 | 188 | — |
| **guided (r2 config)** | **71.0** | **23.5** | 182 | 40.4% |
| SOTA reference (2607.01378, their protocol) | 75.5 | 77.5 | — | — |

Per-task guided (TSR/CAR vs baseline TSR/CAR):
- t0 (corridor): **72/4** vs 38/0 — TSR +34pp, collisions near-universal (task forces operation inside margin)
- t1: **30/34** vs 58/32 — TSR −28pp REGRESSION (only weak cell)
- t2: **98/26** vs 92/8
- t3: **84/30** vs 80/16

Pre-registered gate: TSR≥62 ✅ (71.0) · CAR≥40 ❌ (23.5). PARTIAL GO.

## Smoke history (n=5/task)
| Round | Config | L1 TSR/CAR | L2 TSR/CAR | Verdict |
|---|---|---|---|---|
| r1 | eef_r=0.05, no inflation | 65/10 | 65/15 | mechanism ✅ margins ❌ |
| r2 | eef_r=0.09, +4mm/step | 75/30 | 75/25 | gate ✅ both levels |
| disabled sanity | enabled=0 | 50/0 (n=2, t0) | — | passthrough ✅ |

## Findings
1. ARCHITECTURAL CLAIM CONFIRMED AT SCALE: in-denoising guidance preserves/raises TSR
   (+4pp overall, +34pp corridor cell) at 40% intervention — no post-hoc filter ever did this
   (July 11 pilot: −17pp TSR at 26% intervention; FOL oracle tier: TSR == baseline).
2. CAR bottleneck diagnosed: min_clearance −0.013 with residual 0.009 → collisions occur while
   the EEF-point model believes constraints hold. Unmodeled geometry: hand/wrist above EEF
   (v19 forensics: hand plowed moka), fingers/carried bowl below, OSC tracking error.
   t0: r_eff=0.18 vs task geometry forces B_0<0 → DCBF escape mode, not protection.
3. t1 TSR regression (30 vs 58) is the cell to watch — likely over-constraint during
   bowl-on-ramekin grasp; candidate fix is per-phase margin or target-corridor exemption (FOL layer).
4. Next lever (server-only, ~12 lines): vertical companion spheres (hand +0.08, fingertip/payload
   −0.06) in the rollout — full-system geometry, the CAVF Layer-3 direction, not blind inflation.

## FINAL: n=50 both levels, r3 config (companion points), job 42165451, 2026-07-12
| Level | Config | TSR | CAR | act | vs baseline | vs SOTA (their protocol) |
|---|---|---|---|---|---|---|
| L1 | r2 (EEF-only, fat) | 71.0 | 23.5 | 40% | +4.0 / +9.5pp | 75.5/77.5 |
| L1 | r3 (+companions) | 58.5 | 44.5 | 48% | −8.5 / +30.5pp | 75.5/77.5 |
| L2 | r3 (+companions) | **86.5** | **49.0** | 43% | **+31.0 / +37.0pp** | 78.0/75.5 — **TSR BEATEN** |

Per-task r3 n=50 (TSR/CAR):
- L1: t0 68/20 (base 38/0), t1 20/46 (base 58/32), t2 98/58 (base 92/8), t3 48/54 (base 80/16)
- L2: t0 100/12, t1 56/20, t2 96/72, t3 94/92 (baselines: 66/0, 14/0, 62/0, 80/48)

## Gate evaluation (pre-registered: CAR>=40 AND TSR>=62)
- **L2: PASSED** — 86.5/49.0. TSR exceeds the SOTA's published L2-Spatial TSR (78.0) already at Tier-GT.
- L1: SPLIT — r2 config passes TSR only (71.0/23.5); r3 config passes CAR only (58.5/44.5).
  L1 traded rather than stacked: fat envelopes fight in-margin task phases (t1 bowl-on-ramekin
  TSR 20, t3 cabinet 48 vs 80) while t0 still leaks (CAR 20). L1 = obstacle NEAR TARGET by
  construction — the in-margin regime where uniform geometry cannot separate "approach the
  target" from "approach the obstacle".

## Interpretation
Tier-GT with uniform geometric margins has SATURATED on L1: the (TSR, CAR) frontier moved
along itself between configs. Breaking the tradeoff requires exactly the planned next layers:
(1) semantic allowed-contact/target-corridor exemption (relax payload companion when the
command aims at the grasp/place target — FOL layer decides WHICH object is target), and/or
(2) candidates+selection (sample K guided chunks, pick one that clears the obstacle AND
completes the grasp — selection escapes what uniform repair cannot).
L2 (obstacle blocks path, away from target) is the clean demonstration of the mechanism:
+31pp TSR AND 4.1x CAR simultaneously — no filter in the FOL campaign ever moved both.
