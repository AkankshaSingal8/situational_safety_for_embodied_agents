# Faculty decision brief: learning to execute reusable safety skills

**Updated 18 September 2026 · Proposed experiments; no results claimed**

**Recommendation:** test whether a shared language-conditioned flow adapter can learn to execute reusable safety skills. Keep the official `pi05_libero` checkpoint parameters fixed, add a small trainable residual during action generation, and condition it on skill language, numeric requirements, and current grounded geometry. Use the existing LIBERO stack and two scoped requirements: protected-column exclusion and bounded payload tilt.

**Memory and learning have distinct roles.** A versioned `SKILL.md` stores an instruction, applicability conditions, and a supported contract. Runtime grounding binds its roles to current objects. One shared adapter learns how those inputs should influence continuous actions. Loading an approved new wording or parameter value changes inference context without retraining. An unsupported new safety primitive remains outside scope.

**The decisive comparison is before external repair.** Compare the original policy, the policy with geometric repair, a structured-conditioned adapter, the language-conditioned adapter alone, and the same language adapter with repair. Keep the checker as teacher and evaluation instrument. If only the shielded model succeeds, we have not demonstrated learned compliance. If structured conditioning matches language, prefer that simpler implementation while retaining readable skills for users.

**Data and transfer:** target 200 accepted teacher/nominal training trajectories and 50 validation trajectories across five task/context cells, with an explicit 500-attempt collection cap. Withhold T2+A and both A+B task cells from optimization and selection. Withhold numeric midpoint requirements and paraphrases as separate interpolation/generalization tests. The same pretrained tasks and assets are used; this is not an entirely unseen-task or unseen-object claim.

**Feasibility:** the local OpenPI implementation exposes a concrete flow-velocity hook. The proposed two-block adapter changes six pose channels while preserving the base gripper output. Required gates cover frozen-weight integrity, initial sampler identity, correct flow-time convention, teacher/evaluator correctness, successful training-only overfit, and save/reload parity. Grounding is oracle-assisted initially; estimated geometry/applicability are separate ablations.

**Scale:** start with an 80-episode T1 mechanism demo after training. A one-seed final pilot has 1,600 primary episodes plus 120 memory diagnostics. Three training seeds for learned methods yield 3,520 primary episodes plus 120 diagnostics. Optional edit and perception panels are budgeted separately. No full campaign launches before measured throughput and an allocation cap are recorded.

**Resources:** profile one H100 for training memory, update throughput, and simulator/model serving. Charge teacher collection, training, startup, failures, and evaluation. [PSC accounting](https://www.psc.edu/resources/bridges-2/user-guide/) uses allocated GPU wall time. A small trainable module still requires forward passes through the base model; savings must be measured and training amortization reported.

**Novelty is conditional.** Existing skill-memory, executable-safeguard, and command-conditioned policy work already covers much of the architecture. The proposed result must establish supported fixed-weight edits, useful transfer, language's contribution relative to structured conditioning, or reduced repeated correction/authoring cost. Occam's razor motivates the small shared adapter and strong simpler controls; the Bitter Lesson motivates testing whether more experience improves performance through the same interface.

**Decision requested:** agree on task families and withheld combinations, the minimum useful raw-policy improvement, and a bounded profiling/training budget. Expand into richer grounding, action-expert adaptation, or new predicates only after diagnosing the initial result.

[Complete v2.0 step-by-step protocol](2026-09-18-language-conditioned-safety-adapter-poc-protocol.md) · [Configuration specification](2026-09-18-language-conditioned-safety-adapter-poc-config.yaml) · [Historical literature review](2026-09-17-reusable-safety-skills-review.md).
