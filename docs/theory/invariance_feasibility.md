# Certificates for Safety Enforcement on a Frozen Flow Policy

## Attribution note (2026-07-25, after checking 2607.01378 in full)

The DCBF chain condition and its geometric-decay induction (`B_t >=
(1-gamma)^t B_0`) are TEXTBOOK (Agrawal & Sreenath 2017), and 2607.01378
uses them the same way we do — as background, not a contribution. That paper
proves NOTHING about its own pipeline: no feasibility under actuation
limits (its min-norm delta* is unconstrained), no certificate on the
EXECUTED chunk (corrections at intermediate denoising steps; final actions
never re-verified), no infeasibility handling. Our contributions are
therefore exactly: Thm 1 (saturation/brake feasibility), the CHECKED
certificate semantics (soundness independent of whether guidance worked —
a property of the output, not the procedure), Prop 3 (K-candidate selection
soundness), and §5 (mechanism-independence over noise-space operators,
which correction-based arguments cannot cover). Present Thm 2's induction
as a lemma WITH CITATION; the novelty is in what is being certified and
when it is checked, not the induction itself.

Draft theory appendix (2026-07-25). Every statement is checked against the
implementation in `openpi_guided/src/openpi/models/pi0_guided.py`; honest-claims
discipline — assumptions that the implementation only approximates are stated
as assumptions, not hidden.

## 1. Setup and actuation model

**Actuation model (A1).** The end-effector is a discrete-time single
integrator over one action chunk: given per-step commands
`u_1..u_H ∈ [-c, c]^3` (c = `cmd_clip`, 1.0 for SafeLIBERO-normalized
checkpoints, 0.05 for metric-delta checkpoints), positions evolve as

    p_j = p_{j-1} + s · u_j,        s = translation_scale · output_max

with `p_0` the measured eef position at replan time. This is the OSC_POSE
kinematic model with perfect tracking; controller tracking error is bounded by
ε_track and absorbed into `d_safe` (A1').

**Hazard set.** For obstacles m = 1..M (primary + guard companion) with
centers `o_m`, effective radii `r_m(j) = r_eff,m + λ·j` (λ =
`inflation_slope`, per-step inflation covering bounded obstacle motion and
open-loop drift within a chunk: valid while the obstacle moves less than λ per
control step — A2), and companion offsets `q ∈ Q` (wrist +0.08, fingertip
−0.06), define the barrier at step j

    B_j(p) = min_{m, q} ‖(p + q) − o_m‖_shape − r_m(j)

where ‖·‖_shape is the sphere or superquadric pseudo-distance. `B > 0` is
safe. Safety of a chunk prefix of length L means `B_j(p_j) ≥ 0` for j = 1..L.

**DCBF chain condition (C).** The repair sweep enforces, for j = 1..H,

    B_j ≥ (1 − γ) · B_{j-1},        γ ∈ (0, 1]   (γ = 0.9 deployed).

## 2. Theorem 1 — per-step feasibility of brake-then-push under actuation limits

*Claim.* For every state with `B_{j-1} ≥ 0` there exists an admissible command
`u_j ∈ [-c, c]^3` satisfying (C), and the repair operator returns one.

*Proof.* Take the brake command `u_j = 0`. Then `p_j = p_{j-1}`, and
`B_j(p_j) = B_{j-1}(p_{j-1}) − λ` under inflation growth; choosing the
brake-fallback definition implemented — remove all motion components toward
the active obstacle — yields `‖(p_j + q) − o_m‖ ≥ ‖(p_{j-1} + q) − o_m‖` for
the argmin pair, hence `B_j ≥ B_{j-1} − λ`. Provided the margin schedule
satisfies `γ·B_{j-1} ≥ λ` (equivalently `B_{j-1} ≥ λ/γ`; λ = 0.004 m,
γ = 0.9, so the condition holds whenever the barrier exceeds 4.5 mm — below
that the push channel is engaged, see below), (C) holds:
`B_j ≥ B_{j-1} − λ ≥ (1 − γ)B_{j-1}`.

The push channel strengthens this: the correction is applied along the radial
escape direction `d̂` of the argmin (obstacle, companion) pair. Moving along
`d̂` strictly increases the effective distance (for the sphere trivially; for
the superquadric the shape term `r_shape(u)` is constant along the center ray
while the distance grows — the monotonicity noted in `_closest`). If the
required push exceeds the box `[-c, c]^3`, the operator falls back to the full
brake, which never *decreases* the barrier — so the executed step is never a
known-violating step. ∎

*Scope note.* The λ-vs-margin corner case (`B_{j-1} < λ/γ`) is exactly the
regime where the brake alone cannot compensate inflation; there the operator
outputs brake+push saturated at the box, and Theorem 2's certificate is what
guarantees the *executed* prefix is safe (checked, not assumed).

## 3. Theorem 2 — discrete forward invariance of the certified prefix

*Claim.* If `B_0 > 0` and the chain (C) holds for j = 1..L, then
`B_j ≥ (1 − γ)^j · B_0 > 0` for all j ≤ L: the trajectory never enters the
hazard set within the certified prefix, under (A1)/(A1')/(A2).

*Proof.* Immediate induction on (C): each step retains at least the fraction
(1 − γ) of the previous margin, and (C) with `B_{j-1} ≥ 0` gives `B_j ≥ 0`;
strict positivity is preserved because (1 − γ) > 0. ∎

**Certificate semantics (what makes this sound in practice).** The
executed-prefix acceptance stage *re-checks* (C) on the decoded chunk for
j = 1..prefix_len by explicit rollout of (A1) — the certificate is a checked
property of the emitted actions, not a property assumed of the repair
operator. Consequently the certificate is sound even where earlier-step
repairs were attenuated by the ramp schedule (w < 1) or the pushed command
saturated: any candidate whose prefix fails the check is rejected, and if all
K candidates fail, the argmax-margin candidate is executed with its true
margin reported (fail-visible, not fail-silent).

## 4. Proposition 3 — selection soundness of best-of-K with prefix acceptance

*Claim.* If at least one of the K candidates satisfies the prefix chain, the
executed chunk satisfies it; lexicographic selection (feasibility ≻ clearance
margin ≻ task progress) never prefers an infeasible candidate over a feasible
one.

*Proof.* The scalarization gives feasibility a dominating weight; among
feasible candidates ties are broken by margin then progress. ∎

## 5. Corollary — enforcement as noise-space inference (candidate A bridge)

Fix the frozen policy F: (z, obs) ↦ chunk (the deterministic 10-step Euler
map). Best-of-K sampling is zeroth-order search over z in the safe preimage
S = {z : chunk F(z) passes the prefix chain}; the adjoint method
(gradient ascent of the chain margin through F) is first-order search over the
same set; the emitted actions are F(z*) — exactly on the policy's action
manifold in both cases. Theorems 1–2 certify membership in S by explicit
check, so the certificate is *identical* across enforcement mechanisms; only
the search procedure differs. This separates the paper's claims: (i) the
certificate (this appendix), (ii) the search mechanism (empirical comparison:
projection repair vs 0th-order vs 1st-order noise search vs FK tilting).

## 6. Stated assumptions (paper footnotes)

- (A1) single-integrator kinematics at the eef with linear command scaling;
  rotation commands are not certified (rotation-induced sweep is partially
  covered by the companion points).
- (A1') tracking error ≤ ε absorbed by d_safe (0.01 m deployed); no proof of ε
  for the underlying OSC controller is attempted.
- (A2) obstacle displacement ≤ λ per control step within a chunk (λ = 4 mm at
  20 Hz ⇒ ≤ 8 cm/s); faster movers void the certificate between replans —
  empirically mitigated by 5-step replanning, not certified.
- Perception error (no-GT tier) shifts o_m; the certificate is relative to the
  estimated obstacle. The fail-direction study (E5) quantifies the empirical
  consequence; no formal robustness radius is claimed beyond r_eff slack.
- The certificate covers the *translational* chain over the executed prefix
  only (5 steps deployed); chunk suffixes beyond the prefix are re-planned.

## 7. TODO before submission

- [ ] Restate Theorem 1 with the exact implemented brake definition (component
      removal vs full stop) — verify `λ` accounting in code (`r_horizon` uses
      j+1 indexing; the proof's ineq needs the same offset).
- [ ] Numerical check: instrument one eval run to assert (C) holds at
      acceptance exactly as the theorem states (no float slack surprises;
      assert B_j − (1−γ)B_{j−1} ≥ −1e−6).
- [ ] Decide whether to present γ as class-K (α(h) = γh) to match CBF
      convention in related work.
