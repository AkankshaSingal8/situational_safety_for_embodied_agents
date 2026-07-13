# Flow-Steering Research → Adopted Upgrades (2026-07-12)

Full agent outputs: steer_theory.json / steer_robotics.json / steer_synth.json (this dir).

## Backbone verdict
Projection-per-Euler-step SURVIVES, demoted from "the method" to "the enforcement stage".
Our uniform full-strength repair at every step is the documented SafeDiffuser-RoS anti-pattern
(mode-locking; 41-100% trap rates) — exactly our L1-fat TSR signature. Fix = schedule it and
wrap in an escalation ordered by distribution shift:
  SELECT seeds first → GUIDE (soft) second → PROJECT (hard) last → CERTIFY executed prefix always.
The composed system (scheduled soft pre-shaping + late hard DCBF sweep + seed selection +
executed-prefix certificate on frozen pi0.5) is occupied by NOBODY (BayesFP lacks the hard
layer; the SLSQP SOTA lacks schedules/seeds/corridors).

## P0 stack (Phase 3/4, ~7-9 days total, all inside the existing jitted fori_loop)
1. **Denoising-time repair schedule** (1-2d): repair_weight[k] and margin_scale[k] arrays —
   zero repair at k=0-1, ramp k=2-6, full k>=7 (terminal chain certificate preserved).
   Sources: SafeDiffuser TVS, TouchGuide late-ramp, FK-Flow harmonic, guidance-interval.
   Expected: L1-fat TSR 58.5→70-75 at held CAR → run fat envelopes everywhere.
2. **Brake-then-push repair channel** (1d): shrink approach component first, radial push for
   remainder; clamp-saturated → full brake (chain feasibility becomes GUARANTEED under
   actuation limits — one-paragraph proof; max_residual→~0). Source: PC-Diffuser
   (velocity-channel corrections: 10.3% vs 74-89% residual collisions). CAR +10-20.
3. **Batched best-of-K + executed-prefix lexicographic acceptance** (2-3d): K=4-8 noises,
   kv_cache computed once + repeated (K=8 ≈ 2-3x latency), selector = feasible(executed
   5-step prefix, inflated margins) → max(min-margin, progress). CAR +10-15, TSR held.
   Sources: ITPS (sampling = least drift), FK-Steering, Ma et al. inference-time scaling.
4. **Soft hinge-gradient pre-shaping** (2d): E = Σ softplus of chain violations on the
   x1-PREDICTION rollout (grad through our analytic rollout only — zero extra NFEs);
   v ← v − η_k·(||v||/||g||)·clip(g), active steps ~2-7. Sources: Feng et al. 2502.02150
   (x1-pred gradients, softened hinge), QPILOTS-U magnitude rescale, OmniGuide composition.
   L1 TSR +5-8, CAR +3-7.
5. **Corridor = anisotropic margin exemption + soft attractor** (2-3d, = Phase 3): deficit
   scaled by (1 − c·cone(p_j; target, axis)) + attractor term in E at ~100:1 weight ratio
   (OmniGuide). Target: L1 ≥75 TSR at ≥60 CAR.

## P1 (Phase 5-6)
- FK mid-trajectory resampling w/ drift-corrected noise injection (score analytic from v for
  linear interpolant — FK-Flow 2509.01543); build ONLY if feasible-seed-starvation histogram
  (free diagnostic) shows terminal-only selection starving. Soft potentials only, harmonic
  early down-weighting, validate unguided-TSR-unchanged first.
- Gauss-Seidel 2-3-pass sweep + backward anticipation pass (1d; also the sweep-vs-joint ablation).
- Perception-confidence gating + residual-adaptive inflation (GAF gate, Feedback-WM residual)
  — the thing that makes P5 land.

## P2 / ablation-only / journal
Recurrence (TFG-style re-noise+extra step), FRS reversed-repaired informed seed,
MG-Select masked-conditioning KL progress term.

## Do NOT adopt (with reasons in steer_synth.json)
Constraint-embedded flows (retraining, convexity), learned critics/verifiers (9-wk + OOD
transfer near zero + contamination hazard), Jacobian/Fisher projection (3B backward passes),
twisted-SMC/tree samplers (built for 100s of steps), noise-space MCMC (uninformative at t≈1),
hard indicator potentials + raw-x_t gradients (documented failure modes), any pi0.5 fine-tuning.
