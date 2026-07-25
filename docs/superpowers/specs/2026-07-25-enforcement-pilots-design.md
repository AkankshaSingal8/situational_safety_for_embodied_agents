# Enforcement-Strategy Pilots — Design (2026-07-25)

User directive: do NOT finalize an enforcement strategy; run a quick but
noise-proof validation of each candidate, then pick by evidence. Constraint:
anything runnable on PSC GPUs. Validation discipline: paired designs, offline
mechanism probes before rollouts, n≥20 + McNemar/Fisher before GO/NO-GO
wording (memory: feedback-validation-not-noise).

## Candidates

**A. Adjoint noise-space optimization** — gradient-ascend the prefix-chain
margin through the frozen 10-step denoise map w.r.t. the initial noise;
actions stay exactly on the policy manifold. Unifying frame: best-of-K =
0th-order search over the same safe preimage (see theory appendix §5).
- Stage 1 (GO/NO-GO, job 42606069, `adjoint_probe.py`): compiles / no OOM /
  finite non-degenerate grad / margin increases over ≥5 ascent steps /
  grad-step walltime < ~3× a K=8 forward. Mechanism only; NO performance
  claims from this stage.
- Stage 2 (only if GO): warm-start from best-of-K winner, 3–5 ascent steps,
  dev cells n=20 paired vs current repair.

**B. FK particle steering** — K particles, soft margin potentials reweight +
resample at ~2 mid-denoising steps, hard terminal certificate retained.
- Stage 1: implement reweight/resample hook on the batched sampler; fidelity
  check FIRST: potentials off ⇒ bit-identical to current K-seed path;
  unguided TSR unchanged (n=20, one dev cell).
- Stage 2: dev cells n=20 paired.

**C. Velocity superposition repulsor** — annealed analytic repulsive velocity
added during denoising (hinge pre-shaping path). Ablation row, not a headline
bet; same n=20 paired protocol.

## Comparison protocol (pre-registered)

- Dev cells: Spatial L1 t0/t1 + Object L1 t1 (obstacle-near-grasp regime).
- All arms share: identical seeds/init states (paired), identical identity
  source (GT), identical barrier/certificate parameters; enforcement is the
  ONLY difference. Current composed CBF repair runs as the control arm in the
  same jobs.
- Metrics: TSR, CAR, prefix-certificate pass rate, correction norm / noise
  shift, latency per chunk. Verdicts by paired McNemar (TSR/CAR per episode);
  a candidate advances only if the paired delta is significant or dominance
  is uniform across cells.
- No finalization until all three stage-2 results are on the table.

## Status

- A stage 1 submitted (42606069). B and C implementation next after A's
  verdict lands (shared telemetry code).
- Theory appendix drafted: docs/theory/invariance_feasibility.md — the
  certificate is mechanism-independent (§5), so pilots compare search
  procedures under one certificate.
