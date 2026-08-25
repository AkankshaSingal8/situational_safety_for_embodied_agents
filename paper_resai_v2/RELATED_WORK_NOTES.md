# Domain notes for the ReS AI draft (gathered 2026-08-18)

## Framing vocabulary — Hsu, Hu & Fisac, "The Safety Filter: A Unified View"
(Annual Review of Control, Robotics & Autonomous Systems 2024; arXiv:2309.05837)
A safety filter is factored into four modules:
  1. safety certificate / value function (defines the safe set)
  2. fallback policy pi
  3. safety monitor Delta (does the candidate action preserve future safety?)
  4. intervention scheme phi -- either SWITCHING / least-restrictive, or
     OPTIMIZATION-based min-norm (CBF-QP)
Taxonomy axes: monitoring (value-based vs rollout-based); intervention
(switching vs min-norm); fallback synthesis (model-based / data-driven /
hybrid); guarantee class (robust / probabilistic / none).

**The hook for our paper**: all four modules presuppose the safe set is GIVEN.
In open-world semantic settings the binding problem is upstream --
*which entity is the hazard*. That is the neuro-symbolic question, and it is
the one this workshop is about.

## Bajcsy line (CMU) -- latent / uncertainty-aware safety filters
- Nakamura, Seo, Bajcsy, "Uncertainty-aware Latent Safety Filters for Avoiding
  OOD Failures" (arXiv:2505.00779, RSS 2025 line): HJ reachability inside a
  generative world model's latent space; epistemic uncertainty as a proxy for
  unseen hazards; conformal calibration of the OOD threshold.
- Nakamura & Bajcsy, "How to Train Your Latent Control Barrier Function"
  (Nov 2025): smooth latent CBF filtering under hard-to-model constraints.
- Bajcsy NSF CAREER 2025, "Formalizing Open World Safety for Interactive Robots".
**Position against**: they learn the safe set from data (latent CBF / HJ value)
and treat identification implicitly. We keep the barrier textbook-simple and put
the intelligence in an inspectable symbolic election rule. Complementary, not
competing -- say so.

## The direct predecessor -- also neuro-symbolic, also on SafeLIBERO
English, Zheng & Ewetz, "Neuro-Symbolic Safety Guidance for VLA Models via
Constrained Flow Matching" (arXiv:2607.01378). Min-norm constrained optimization
interleaved INTO flow-matching denoising; "interleaving of symbolic constraint
satisfaction with neural trajectory generation"; predictive (uses the decoded
trajectory) rather than reactive. Reports 82.8% collision avoidance / 81.6% task
success on SafeLIBERO.
**This is the paper the ReS AI audience knows.** Our sharpest contrast: they
locate the symbolic content in the PROJECTION; we show (a) the projection
carries a measured, paired success tax, and (b) the symbolic content that pays
is in IDENTIFICATION. Same benchmark, same "neuro-symbolic" label, opposite
conclusion about where the symbols belong.

## Convergent independent evidence for the repair tax
PACS, "From Demonstrations to Safe Deployment: Path-Consistent Safety Filtering
for Diffusion Policies" (arXiv:2511.06385): safety mechanisms "alter actions in
ways unseen during training, causing unpredictable behavior and performance
degradation." Credit this explicitly -- it is our repair-tax claim reached from
the opposite direction, and citing it converts a reviewer's "did you cherry-pick
your control?" into corroboration.

## Noise space is occupied for PERFORMANCE, not safety
- Flow Reversal Steering (arXiv:2606.13675): invert good actions to their latent
  noise, map to nearby generalist modes; large TSR boosts.
- VLS, "Steering Pretrained Robot Policies via VLMs" (arXiv:2602.03973):
  VLM reward gradients + FK resampling in frozen pi0.5 denoising; performance.
=> Our claim "no prior SAFETY enforcement in noise space" survives; state it as
scoped to safety and cite both as the performance-side precedent.

## Other current landscape
- "Your Model Already Knows: Attention-Guided Safety Filter for VLA"
  (arXiv:2606.09749): reads VLA attention heads to localize the intended object;
  training-free identification. THE competitor to our identification chapter --
  read-only neural identification vs our symbolic election. Note their own
  framing that VLM-based obstacle identification "is too slow to run in the
  control loop and can only be invoked at episode initialization, leaving the
  filter unable to track moving obstacles" -- our MOVING literal, re-grounded at
  every replan, is a direct answer to that objection.
- "Contextual Safety Reasoning and Grounding for Open-World Robots"
  (arXiv:2602.19983).
- VLA safety survey (arXiv:2604.23775): taxonomy by attack/defense x
  training/inference time. Its open-problem #4 is that "safety benchmark
  development has not kept pace with model capability advances" -- the direct
  citation hook for our CheckRobotContact defect finding.

## Writing style to imitate
From "Safety Pretraining" (arXiv:2504.16980): abstract = problem, why the
prevailing fix is brittle, "In this work we present...", enumerated (i)-(iv)
components, one headline number. Short declarative sentences; numbers early.
From OGPO (arXiv:2605.03065): lead with the mechanism, name the stabilizing
ingredients explicitly, report what was ablated.
From the Fisac/Bajcsy line: be explicit about what is and is not guaranteed;
name the assumptions inline; state non-claims.
Common to all three: contribution list is concrete and falsifiable; no
throat-clearing; the limitation section is real, not decorative.
