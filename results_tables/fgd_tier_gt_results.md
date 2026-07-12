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
