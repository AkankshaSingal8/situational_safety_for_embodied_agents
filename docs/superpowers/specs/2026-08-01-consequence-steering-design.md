# Consequence-Constrained Steering with FOL-Grounded Predicates — Design + GO/NO-GO Protocol

Status: DESIGN FROZEN 2026-08-01 (timestamp for novelty priority). Prototype
gates pre-registered below. Owner: task #43. Companion plan:
`~/.claude/plans/refactored-sprouting-puffin.md` (user-approved 2026-08-01).

## 1. Problem statement

Current method: CBF-constrained repair inside the flow-matching denoising loop
of π0.5 — minimum-displacement correction of the action chunk subject to
analytic barrier constraints evaluated on a hand-calibrated single-integrator
rollout of the end-effector. Two structural weaknesses, both observed:

1. **The actuation model is hand-specified and fragile.** The 2026-08-01
   exec-parity forensics found our rollout model wrong by 2.2–2.5× (robosuite
   OSC gains mismatch) — undetected until it caused mass timeouts. A learned
   consequence model would absorb controller response from data.
2. **Task completion is only an implicit objective.** Repair minimizes
   deviation from the policy's proposal; nothing *requires* progress. Dominant
   failure mode is timeout paralysis, not collision — motivating heuristics
   (stall-retreat, progress-select, corridor exemptions) that a
   task-completion-constrained optimization would subsume.

## 2. Proposed method

At each replan, the guided server denoises K candidate action chunks (existing
best-of-K, K=8). Replace/augment barrier-based scoring with:

    select  a* = argmax_{a in K}  TaskProgress_hat(a)
    subject to                    P_hat(violation | a) <= eps      (eps = 0.10)
                                  pessimism: + lambda * ensemble_disagreement

where both TaskProgress_hat and P_hat come from ONE learned consequence model
conditioned on (current state, action chunk). If the feasible set is empty →
fall back to the existing CBF repair machinery unchanged (certified backstop).

### 2.1 FOL-grounded consequence representation

The consequence model predicts future truth probabilities of GROUNDABLE FOL
PREDICATES rather than (only) raw trajectories:

- Safety: P(CONTACT(eef, hazard_i) within horizon H), per hazard.
- Manipulation state: P(HOLDING(·) at chunk end).
- Task progress: predicted satisfaction of the task's goal conjuncts.
  On LIBERO this is exact: BDDL goals are already conjunctions of groundable
  predicates (On/In/...). v0 progress surrogate: predicted reduction in
  distance-to-target. This unifies the pipeline: FOL is simultaneously the
  constraint compiler's output (VLM→FOL→barrier), the consequence language,
  and the optimizer's interface.
- Metric head (predicate-closure catch-all): H-step eef path + predicted
  min-distance-to-hazard. Guards against hazard modes outside the ontology
  and supplies the barrier evaluation with a better rollout than the
  single-integrator.

Labels are FREE in sim: every predicate is auto-computed from logged GT state;
binary targets are far more data-efficient than trajectory regression.

### 2.2 Three safe-action mechanisms (v1 = A)

A. **Sampling-based selection (v1, OOD-safe by construction).** Optimize only
   over the K policy-sampled chunks. Every action is in-distribution; the
   policy is the sampling prior; selection is the optimizer.
B. **Gradient repair through the predictor (v2).** If no candidate is
   feasible, differentiate P_hat(violation) w.r.t. the chunk and repair inside
   the remaining denoising steps — identical machinery to today's ∇h push, with
   a learned function supplying the gradient. Deferred until A is validated
   (predicate probabilities saturate → flat gradients; needs calibration work).
C. **CBF terminal backstop (always on).** The learned model sees H steps;
   safety is indefinite-horizon. The existing barrier repair remains as the
   fallback and terminal constraint, giving recursive feasibility a certified
   anchor. This hybrid is also the novelty wedge (§4).

### 2.3 OOD defenses (why the optimizer cannot exploit its way off-manifold)

1. Selection over policy samples only (mechanism A) — cannot leave support.
2. In-denoising repair self-corrects: perturbations at intermediate noise are
   re-projected onto the data manifold by the remaining denoising steps
   (existing ramp schedule exploits this).
3. Proximal bound on any repair (existing displacement caps / cmd_clip).
4. Ensemble pessimism (3 heads): disagreement penalty stops the optimizer
   steering into states the predictor cannot predict (model-exploitation
   guard — the state-OOD face of the problem, distinct from action-OOD).
5. Prefix acceptance (existing): only the chunk suffix is ever modified.

## 3. Architecture & data

- Model: small MLP/GRU trunk; inputs (eef pos, gripper qpos, hazard/object
  positions relative to eef, H×3 translation chunk, domain tag); heads:
  metric (H-step path), predicate probs; 3-model ensemble. Milliseconds per
  K=8 batch — fits the replan budget.
- Data: `--log_transitions` (default-off) in both eval clients dumps per-step
  (eef_pos, gripper qpos, action, GT hazard/object positions). Collection:
  LS obstacle_avoidance L0–2 (baseline n=10/task + gt_dual n=4/task) +
  SafeLIBERO Spatial L1 (baseline n=20 + guided n=10) ≈ 250 episodes ≈ 1e5
  transitions. All future GPU runs also log (free accumulation).

## 4. Prior art and positioning (sweep 2026-08-01, agent-verified abstracts)

**Exact combination unpublished.** Nearest neighbors and the wedge:

| Paper | Mechanism | Delta from ours |
|---|---|---|
| SafeDojo (2606.20698, 6/26) | VLA samples + interactive world model + Lagrangian success-vs-safety-cost | TRAINING-TIME constrained GRPO fine-tuning; not a runtime filter; no certified backstop |
| Pre-VLA (2605.22446, 5/26) | amortized safety+advantage chunk filter + resampling | no world model, heuristic thresholds not constrained opt, no backstop |
| Neuro-symbolic constrained flow matching (2607.01378, 7/26; our SOTA baseline) | min-norm symbolic-constraint repair in denoising | analytic constraints, min-deviation objective — no learned model, no task objective |
| CheckVLA (2607.26789, 7/26) | world-model verification + conformal anomaly triggers + suffix rewrite | anomaly detection vs expectation, not constrained task optimization |
| Latent Safety Filters / UNISafe / Latent-CBF (2502.00935, 2505.00779, 2511.18606) | learned world model as safety certificate (HJ/CBF, latent) | least-restrictive switching; no task objective; not over VLA chunk samples |

Related-only clusters: RoboMonkey/V-GPS (sample-and-score, no safety);
Cosmos Policy (value-ranked futures, no safety); PPGuide (performance gradient
guidance, no safety); PiJEPA (policy-prior MPPI, navigation); constrained
diffusion planners (analytic constraints); SafeVLA (training-time CMDP);
SafeDec (STL decoding, assumed dynamics).

**Claimable novelty:** (i) task-completion-CONSTRAINED objective at inference
(all existing runtime filters are min-deviation or least-restrictive);
(ii) one learned consequence model supplying both sides of the optimization
over flow-VLA samples; (iii) certified CBF backstop for a learned-model
filter (recursive feasibility); (iv) FOL predicates as the consequence
representation (found nowhere).

**Due diligence before any novelty claim in print:** full-text read of
SafeDojo + Pre-VLA appendices (inference-time ablations?); verify "Kairos"
(2606.16533) mechanism. **Head-to-head that sells it:** SafeDojo-style
finetuned policies are exactly our LS baseline column — a runtime filter
beating/holding parity with the finetuned policy, plus wins where it violates,
without any retraining.

## 5. Pre-registered gates (user-locked 2026-08-01: obstacle_avoidance;
Spatial L1 must beat ALL arms incl. ours-current; GT tier only)

- **G1 (offline, held-out episodes; fail → NO-GO, stop all GPU spend):**
  metric-head median terminal error ≤ 0.5× calibrated single-integrator
  (within 0.25 m of hazards); CONTACT AUC ≥ 0.90; progress-predicate
  Spearman ≥ 0.6 vs realized progress.
- **E1 (LIBERO-Safety obstacle_avoidance, L0–2, n=10/task, finetuned ckpt,
  exec-parity + promoted gating config [default gt_par_eng30] +
  consequence_select):** GO iff pooled SR ≥ baseline 40.0 AND violations ≤
  baseline's. Paired per-task; McNemar.
- **E2 (SafeLIBERO Spatial L1, n=50, GT):** GO iff TSR AND CAR ≥ each of
  {π0.5 baseline, AEGIS post-hoc, SOTA-reimpl, ours-current-guided}.
- **Verdict:** GO = G1∧E1∧E2 → full 4-suite LS campaign + no-GT tier + #37
  flagship. PARTIAL = G1 ∧ exactly one of E1/E2 → one diagnosis+revision
  round, rerun failing eval only. NO-GO = ¬G1 or (¬E1∧¬E2) → park with
  forensics; current CBF method remains the paper's method.

## 6. Execution notes

Phases A–E per the approved plan. No API/VLM calls anywhere. Default-off
flags, byte-identity CPU tests, subagent-driven implementation, one blocking
monitor per GPU job, every gate verdict ledgered in
`results_tables/fgd_tier_gt_results.md` and committed. Does not wait on the
parity dev gate (42899554) except to inherit its promoted config for E1.
