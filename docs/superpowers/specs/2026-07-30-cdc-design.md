# Certificate-Distillation Critic (CDC) — design & pre-registered gates

Date: 2026-07-30. Owner: #37 (novel steering method). Status: gates FROZEN
before D0 submission; changes after D0 submits require a declared amendment
(one per phase, logged here), per the house validation rules.

## Mechanism

The geometric safety pipeline (GT or no-GT tier: RGB-D localization + FOL
hazard compiler + FK barrier certificate) becomes a TEACHER. A small
language-conditioned critic g_phi(image, hazard_text, candidate_chunk) is
trained to reproduce the teacher's per-candidate certificate labels
(feasibility, min prefix margin) from raw perception. At runtime the critic
replaces the geometric certificate in best-of-K SELECTION — the one
mechanism that separated in our ablations — requiring no FK, no metric
calibration, and no obstacle geometry at deployment.

Claim shape (paper): "symbolic certificates are distillable: a 2-5M-param
critic trained on ~10^5 certificate-labeled candidates matches geometric
selection in-domain, transfers zero-shot to an unseen benchmark, and yields
a no-geometry safety tier."

## Differentiation (S4 survey, ledger 2026-07-30)

- Pre-VLA / RoboMonkey: generic learned verifiers/rankers, no certificate
  teacher, no safety semantics, no language-hazard conditioning.
- Trained latent CBFs (2507.13871, 2511.18606): hand-labeled or
  self-supervised safety, offline, not distilled from a symbolic
  certificate, not used for K-candidate selection of a frozen VLA.
- Latent Policy Barrier: robustness (OOD recovery), not semantic safety.
- DynaGuide / WorldGym: world-model rollouts scored by VLM — expensive at
  runtime; CDC is amortized (single forward pass per candidate).
- Unclaimed combination: symbolic-certificate teacher + language hazard
  conditioning + frozen-VLA best-of-K actuation + geometric-tier referee.

## Phases and frozen gates

- D0 dataset: instrumented production no-GT rollouts, 4 suites × L1/L2,
  n=10/task. G0: >= 40k replan records AND >= 8k certificate-infeasible
  candidate examples AND >= 60 collision episodes. Fail → one widening
  (n=20 on under-represented cells), then hard stop.
- D1 training: risk-only critic (DINOv2-S img + frozen text embed + chunk
  → FiLM MLP → infeasibility + margin heads). G1: (a) random-split held-out
  feasibility ROC-AUC >= 0.85; (b) top-1 selection agreement with the
  certificate >= 70%; (c) leave-one-suite-out AUC within 0.10 of random
  split. Fail (a)/(b) → one retry adding proprio-state features; fail (c)
  → NO-GO (scene memorization; claim dead).
- D2 runtime swap, dev cells (Spatial L1, Object L1), paired n=20:
  swap arm (critic selection, repair unchanged) and headline arm (critic
  selection, repair off). G2: swap within 5.0 pp of production on TSR and
  CAR; headline CAR >= baseline+15 with TSR >= baseline-5.
- D3 paper campaign (auto on G2): n=200 Spatial L1+L2 / Object L1+L2 no-GT
  headline vs production; LS obstacle_avoidance L1 zero-shot transfer n=50.
  Four-number box + McNemar/Fisher per house rules.

Gate policy: fully automatic chaining (user decision 2026-07-30) — each
phase's epilogue runs its gate script and submits the next phase iff GO.
Gate scripts write JSON verdicts only; the controller session writes ledger
entries.

## v1 exclusions

Progress head, critic-gradient guidance inside denoising, HJ reachability,
finetuned-LS-checkpoint deployment, any policy retraining.
