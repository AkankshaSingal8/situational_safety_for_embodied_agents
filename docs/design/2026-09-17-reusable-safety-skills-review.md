# Reusable Safety Skills for Frozen Robot Policies

**Research review and revised proof-of-concept design · Version 0.2 · 17 September 2026**

**Design update, 18 September 2026:** the current experiment trains a **shared language-conditioned flow adapter**, keeping the pretrained checkpoint fixed. The [v2.0 protocol](2026-09-18-language-conditioned-safety-adapter-poc-protocol.md) supersedes this review's training-free recommendation, provisional baselines, and episode counts. The literature analysis below is retained as historical context; checking/repair now supplies teaching data, a comparator, and an optional fallback. Raw learned-policy performance is evaluated separately.

**Status:** literature review and proposed experiments. No new robot experiments, performance results, or allocation measurements are claimed. Recommendations combine primary-source reading with source inspection of this repository.

## 1. Historical decision for the initial faculty discussion

**Proceed with a bounded experiment, but strengthen the memory experiment and narrow the novelty claim.** Keep an existing LIBERO-compatible policy frozen. Store safety knowledge as reusable, versioned skill records. Bind those records to scene entities and use a tested controller interface to constrain execution. The strongest first demonstration is a safety correction retained across process restarts and transferred to a new task, with a paired memory-deletion control.

The original plan makes sensible choices about avoiding foundation-model training, a new simulator, and learned video prediction. Its main weaknesses are scientific: the literature is substantially closer than its reference list suggests, and some proposed comparisons cannot establish a memory contribution.

**Recommended research question:** At fixed feedback and execution budgets, can a validated, reusable safety-skill representation preserve a correction across deployments and apply it selectively to new manipulation tasks, more reliably or cheaply than episodic text retrieval, fresh constraint generation, and a static semantic rule table?

The ambitious direction is a deployable safety knowledge layer whose records remain useful when tasks, objects, and eventually policies change. The feasible first step is two grounded geometric requirements on one frozen policy. The bridge between these scales should be experimentally demonstrated reuse, revision, and execution fidelity.

| Decision | Recommendation |
|---|---|
| Initial environment | The existing LIBERO-family stack with the best reproduced nominal task; prefer standard LIBERO for calibration, then controlled safety variants. |
| Main policy | Official OpenPI `pi05_libero`, exact checkpoint and configuration pinned. |
| Immediate fallback | An already integrated OpenVLA-OFT checkpoint, if it performs the selected tasks; this still tests external safety memory. |
| Smaller flow-model option | Official `lerobot/smolvla_libero`, after an independent reproduction check. |
| Memory | Two substantive records initially; expand to 6–10 only for a meaningful retrieval experiment. |
| Executable representation | Human-readable intent plus a typed contract with explicit applicability, bindings, units, reference frames, evidence, and validity conditions. |
| Initial backend | Short-horizon nominal evaluation, bounded transport repair, and per-step monitoring. |
| Essential new experiment | Acquire a correction once; restart; transfer without reauthoring; delete the memory and repeat. |
| Paper-strength extension | Compare contract-only memory with contracts plus reusable repair procedures under matched compute. |
| Training | None for the first result. Consider a small residual or LoRA adapter only after diagnosing a control bottleneck. |

## 2. What the literature permits us to claim

This is a targeted design review, not an exhaustive systematic review or proof that no prior work exists. Sources were checked on 17 September 2026. Close memory and guidance papers were inspected beyond their abstracts where available; supporting papers and implementation resources were checked at the abstract, documentation, or code level. Reported results below remain authors’ claims, not independent replications. Search covered reusable robot skills, language corrections, safety memory, executable safeguards, flow guidance, simulation, and policy deployment; references in the repository’s earlier September memo supplied additional leads that were rechecked.

### 2.1 The closest memory precedents

| Work | Established mechanism | Consequence for this project |
|---|---|---|
| [DROC: Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections](https://arxiv.org/abs/2311.10678) | Retains distilled language corrections and retrieves them across new tasks and objects. | Reusing a correction and reducing repeated human intervention are established ideas. Include an episodic/distilled-text comparator. |
| [MEMO: From Local Corrections to Generalized Skills](https://arxiv.org/html/2603.04560v1) | Builds a retrieval-augmented skillbook from feedback and successes; clusters entries into generalized text and code templates. | A persistent human-correctable skillbook is already a direct robotics contribution elsewhere. Our execution target and representation must matter. |
| [ARCHITECT](https://arxiv.org/html/2607.23784v1) | Synthesizes robot programs from corrections and accumulates reusable language/code skills; evaluates transfer and human effort. | The coding-agent skill analogy is already explicit in robotics. Markdown packaging does not differentiate us. |
| [RoboSafe](https://arxiv.org/html/2512.21220v2) | Uses long- and short-term safety memory to generate executable contextual and temporal predicates, blocking actions or triggering replanning. | “Memory plus executable safety logic around a fixed agent” is already occupied. Distinguish measured continuous trajectory repair and unchanged contract reuse. |
| [JIT-Memory](https://arxiv.org/html/2607.16247v1) | Maintains safety-relevant state and factual memory; procedural Meta-Skills guide mitigation and evolve through a Test–Verify–Write loop. Evaluates sequential acquisition with frozen-memory tests. | This is a central safety-memory comparator. Procedural mitigation, memory improvement curves, and avoiding unproductive stopping are not new by themselves. |
| [LiSA](https://arxiv.org/html/2605.14454v1) | Updates structured safety-policy memory from sparse feedback, with local refinement and confidence-based reuse. | Evidence-gated memory and exception handling have precedents, although its setting concerns tool-agent guardrails. |
| [CEDAR](https://arxiv.org/abs/2608.27797) | Represents learned skills and specifications as automata, using counterexamples and composition. | Reusable executable specifications and composition are established. Continuous manipulation requires a different grounding/control argument. |
| [Context-Aware Human Preferences for Multi-Objective Robot Navigation](https://arxiv.org/abs/2603.17510) | Persistent contextual rules become numerical preferences for a pretrained navigation policy. | Even persistent language rules influencing low-level pretrained control are not broadly novel. |

Earlier ancestry includes [Voyager](https://arxiv.org/abs/2305.16291), which accumulates executable skills in Minecraft, and [Lifelong Robot Library Learning](https://arxiv.org/abs/2406.18746), which grows composable manipulation skills. These support the architectural idea, but weaken any first-skill-library claim.

[MAP-VLA](https://arxiv.org/abs/2511.09516) retrieves demonstration-derived, learned soft prompts into a frozen VLA; freezing the backbone does not make that method training-free. [MemoryVLA](https://arxiv.org/abs/2508.19236) learns perceptual/cognitive temporal memory for an action expert. Both are useful distinctions: our proposed persistent safety knowledge is external to weights and separable from within-episode state memory.

### 2.2 Language, constraints, and action adaptation

| Design choice | Primary precedents | Assessment |
|---|---|---|
| Translate language into robot behavior | [Code as Policies](https://arxiv.org/abs/2209.07753); [VoxPoser](https://voxposer.github.io/); [ReKep](https://arxiv.org/abs/2409.01652) | Established. Constrained generation is an engineering choice, not the central novelty. |
| Semantic overhead/orientation safeguards | [Semantically Safe Robot Manipulation](https://arxiv.org/abs/2410.15185) | Particularly close: water over electronics and cup rotation are motivating examples, with semantic safety filtering of learned policies. Use those hazards as diagnostics, not novel applications. |
| Plug-in VLA action filter | [VLSA/AEGIS](https://arxiv.org/abs/2512.11891) | Established on SafeLIBERO. A basic geometric repair wrapper is a baseline. |
| Update obstacle relevance online | [Your Model Already Knows](https://arxiv.org/abs/2606.09749) | Attention and tracking can support dynamic target/obstacle selection. Semantic selectivity must exceed merely identifying the target. |
| Contextual reasoning into safe sets | [CORE](https://arxiv.org/abs/2602.19983) | Contextual safeguards with an uncertainty analysis already exist. Oracle metadata does not establish comparable perception robustness. |
| Guide a frozen diffusion/flow policy | [VLS](https://arxiv.org/html/2602.03973v1) | Established; includes programmatic stage rewards, gradient steering, and particle resampling. A selection baseline does not reproduce its full method. |
| Correct during denoising | [Neuro-Symbolic Safety Guidance via Constrained Flow Matching](https://arxiv.org/abs/2607.01378) | Direct prior art for predictive constraint correction of flow VLAs. Adding a skill file does not make this a new guidance algorithm. |
| Preserve the nominal path during intervention | [PACS](https://arxiv.org/abs/2511.06385) | Explains why arbitrary action projection can damage policy performance; path-consistent braking is a useful comparator, but cannot route around a permanent blockage by itself. |
| Typed safety contracts / formal plan guards | [SafeGate](https://arxiv.org/abs/2604.05427); [RoboGuard](https://arxiv.org/abs/2503.07885) | Contracts, deterministic gates, and temporal plan repair have precedents. A validated syntax is not proof that a physical specification is correct. |
| Train safety into weights | [SafeVLA](https://arxiv.org/abs/2503.03480) | Constrained policy learning is a legitimate later comparison; it changes the data and compute question. |
| Learn safety with imagined rollouts | [SafeDojo](https://arxiv.org/abs/2606.20698) | Interactive world-model predictions support safety RL. This is much more machinery than two geometric invariants require. |

The classical foundation is the [safety-filter literature](https://arxiv.org/abs/2309.05837): nominal performance and safety enforcement can be separated, but guarantees depend on the model, safe set, initialization, uncertainty, and feasibility assumptions.

### 2.3 Novelty verdict

**Weak claim:** “We store safety rules in `SKILL.md`, retrieve them, and guide π0.5.” This is a useful integration demonstration, but likely too incremental on its own.

**Defensible experimental claim, if supported:** “A validated safety-skill interface allows a frozen manipulation policy to retain and selectively transfer deployment corrections across held-out task–object combinations, with measured execution fidelity and reduced repeated specification effort under matched control budgets.”

**Stronger research direction:** a record stores both what must remain true and a reusable, geometry-parameterized procedure for restoring task progress. Test whether that procedure transfers across new route geometries and reduces online search, planner takeover, and repeated feedback. Existing procedural memory remains a close precedent; the evidence must establish value at the continuous-action interface.

An additional controller, GPU, or simulator is not automatically additional novelty. A clean result that identifies when reusable contracts outperform regenerated constraints is more valuable than a broad stack with confounded comparisons.

## 3. Important corrections to Version 0.1

| Original assumption or choice | Revision and reason |
|---|---|
| Safe success means no *active* skill violation | Score against independently specified ground-truth requirements. Otherwise missed retrieval makes unsafe behavior appear safe. |
| B2 is always-on avoidance | Retain it, but add a static **semantic** rule table with the same applicability, bindings, and backend. It is a much stronger control. |
| Oracle semantic retrieval demonstrates understanding | It demonstrates the consequences of supplied semantic facts. Test visual inference separately. |
| Two requirements plus 6–10 files demonstrate rich memory | Two records are enough for mechanics. Additional records need real distinctions or controlled distractor roles. |
| Skill A+B is enough to prove composition | Ensure each requirement changes a feasible trajectory and neither is redundant. An already upright policy makes B vacuous. |
| Same prompt for all baselines | Keep the original instruction as a fixed prefix. Prompt-injection B1 necessarily changes its input; also give it the facts needed to interpret hidden semantic labels. |
| No learned world model means geometry predicts action outcomes | Forward kinematics maps joints to poses; it does not predict how an OSC controller, contacts, and inertia realize a command. Calibrate a short-horizon predictor. |
| A feasible waypoint path makes a feasible robot trajectory | Check whole-arm clearance, grasp stability, controller tracking, joint limits, and return to the policy. |
| Hold/stop is always safe | A zero command does not cancel velocity or ensure a stable grasp. Define and test a bounded stopping procedure. |
| CPU-only scene tests | Non-rendering geometry tests can be CPU-only. Headless EGL rendering may use GPU resources. |
| GPU-seconds of inference determine SUs | SUs accrue on allocated GPU wall time, including simulation stalls and model startup. |
| LoRA issue establishes supported training | A feature proposal is not a merged implementation. Verify the exact JAX/PyTorch path and checkpoint. |

## 4. What the existing repository changes

Root source inspected at `75ca90d4`; `LIBERO-Safety` at `ef0f79b`; `vlsa-aegis` at `e348187`. Existing submodule modifications were present. This review adds documents only and does not repair or rerun these implementations.

| Local evidence | Implication |
|---|---|
| [OpenPI evaluation driver](../../run_libsafety_eval_openpi.py) and [serving script](../../setup/libsafety/libsafety_serve_openpi.sh) | A server/client integration already exists. Reproduction and provenance checking should precede rebuilding it. |
| [Checkpoint download script](../../setup/libsafety/libsafety_download_checkpoints.sh) | Downloads official OpenPI assets into a directory under LIBERO-Safety. Directory location does not mean safety-fine-tuned weights. |
| [Local OpenPI configuration](../../vlsa-aegis/openpi/src/openpi/training/config.py) | Inspected `pi05_libero` uses horizon 10, `discrete_state_input=False`, and no extra delta transform; these fields match the upstream configuration retrieved for this review. Hashes and the complete runtime pipeline still need verification. |
| [Knowledge base](../../fol_safety_filter/kb.py) | Already provides a predicate registry and expression parsing. Unknown predicates/errors can become `False`; this is unsuitable for reliable unknown-state handling, especially under negation. |
| [Filter](../../fol_safety_filter/filter.py) | `setup_episode()` clears the KB; the inspected action path describes direct geometric projection rather than a QP. Persistent skill storage and formal guarantees cannot be inferred from the README. |
| [Existing audit](../code_review_findings.md) and [benchmark domain code](../../LIBERO-Safety/libero/libero/envs/bddl_base_domain.py) | Source inspection confirms integer geom IDs are passed into name-based contact membership checks; costs are keyed by predicate name and omitted when `done`. Use an independent scorer. |
| [September 13 research memo](2026-09-13-grounded-safety-memory-research.md) | Already identifies grounded rule learning as a broader direction. The present design is a bounded execution-and-reuse study, not evidence that those broader learning questions are solved. |

Do not cite the existing collision columns as validation for the new method. Test intentional robot–hazard contact, carried-object contact, duplicate constraints, terminal-step violations, and a known safe trajectory. Distinguish intended grasp/support contacts from prohibited contacts. The scorer must monitor geometry at sufficiently fine time resolution to avoid missing brief contacts or crossings.

## 5. What counts as memory here

A library of human-written rules **is external semantic memory**, even if it does not learn. However, calling it memory does not establish a scientific benefit over a configuration file.

Separate four functions:

| Component | Contents | Lifetime |
|---|---|---|
| Reusable skill | A relation over roles, applicability, invariants, optional repair procedure, assumptions | Across tasks and process restarts |
| Evidence record | Original correction, supporting/contradicting cases, version history, observed outcomes | Across deployments |
| Scene state | Current entity bindings, poses, semantic facts, timestamps, grasp relation | Valid only while observations justify it |
| Execution state | Active route, subgoal, hysteresis, recovery status | Current episode or skill invocation |

Do not persist a laptop pose as part of a general overhead-protection rule. Do not reactivate a cached entity ID after a reset without rebinding it. Retaining a rule through occlusion and retaining a possibly stale pose are different operations.

The decisive memory test is a **same-current-input, different-history experiment**. Create two otherwise identical deployments with different previously supplied site rules. Restart each system before the same task. The correct action should depend on the stored rule. This establishes that persistent state is causally necessary for this constructed decision, while remaining honest that the site policy was supplied by a human.

Add two information regimes:

1. **Public/general rule:** every method can receive the same rule content. This tests representation and execution efficiency.
2. **Previously taught local rule:** all methods see the same acquisition feedback, but differ in what they retain and retrieve. This tests retention. A stateless baseline cannot recover an arbitrary hidden rule; that expected failure alone is not a research result.

A static table containing exactly the same contracts and applicability logic should produce the same behavior under deterministic retrieval. Treat equivalence as a useful sanity check. To show an advantage over that table, compare how knowledge is added, corrected, transferred, and validated under matched authoring budgets.

## 6. Simulator and task selection

The [original LIBERO benchmark](https://arxiv.org/abs/2306.03310) targets knowledge transfer across manipulation tasks. [SafeLIBERO, introduced with VLSA](https://arxiv.org/abs/2512.11891), and [LIBERO-Safety](https://github.com/LIBERO-SAFETY/LIBERO-Safety) are separate projects. The latter’s current documentation lists five suites, including `reasoning_safety`; the local evaluation stack focuses on four execution suites. Do not assume every reasoning item is a continuous-control task.

**Recommended ordering:** reproduce a standard LIBERO task, add a minimal controlled semantic variant, then evaluate on compatible safety-benchmark tasks. Move directly to a safety suite only if the unfiltered checkpoint already performs its nominal task reliably. This avoids spending the project repairing domain shift unrelated to memory.

| Alternative | When it is appropriate |
|---|---|
| Existing LIBERO / SafeLIBERO / LIBERO-Safety | Best first choice given existing policy interfaces and assets. Select by reproduction quality and hazard observability. |
| [ManiSkill](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/gpu_simulation.html) | Attractive for GPU-parallel simulation when rollout throughput becomes the limiting factor. Migrating a LIBERO policy introduces embodiment, observation, and controller compatibility work. |
| Existing SafeManip/RoboCasa track | Candidate later environment for richer stateful household hazards; first audit policy compatibility and the actual hazard monitor. |
| New simulator / fluid simulation | Justified only if the scientific question requires consequences unavailable in the existing environment. Not needed for projected exclusion and orientation constraints. |

**Task family A:** transport an existing graspable object to a fixed receptacle while respecting a protected projected region. Choose geometry where the nominal overhead route avoids physical collision, but violates the semantic requirement. Otherwise ordinary collision avoidance could explain the result.

**Task family B:** a different compatible transport task, preferably with a changed goal relation or transport direction. Merely moving the same target a few centimeters is pose transfer, not task transfer.

Use `avoid_above` and `maintain_orientation` initially. If ordinary transport never challenges orientation, keep B as a verified unit-level mechanism and select a task with an actual orientation tradeoff for rollout claims. Do not manufacture a policy failure by silently perturbing its actions; any deliberately perturbed diagnostic must be labeled.

Negative cases should keep geometry fixed while toggling a relevant supplied semantic property. Also change an irrelevant property, and include objects with renamed/shuffled identifiers. In an image-based condition, the relevant fact must be visibly observable or explicitly provided. A hidden `contains_liquid` flag cannot be inferred from identical pixels.

Screen three candidate task families using 20 resets each. Proposed development gates: at least 14/20 nominal successes, at least 5/20 semantic violations on otherwise feasible scenes, and a safe executed demonstration for retained configurations. These are task-selection heuristics, not statistical guarantees. Publish all screening outcomes; keep screening seeds out of final evaluation.

## 7. Policy selection and exact interface

The official [OpenPI LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) provides `gs://openpi-assets/checkpoints/pi05_libero/` and a reproduction path. Use that LIBERO-adapted checkpoint rather than `pi05_base`. A safety-fine-tuned benchmark checkpoint is a separate condition because its training may already include the target behaviors.

There is no conceptual need for a flow model if the method modifies completed action chunks externally. OpenVLA-OFT is consequently a sensible fallback in this repository. Flow-specific guidance requires a compatible generative head and internal model access; preserve this distinction in the paper title and claims.

The released [SmolVLA LIBERO checkpoint](https://huggingface.co/lerobot/smolvla_libero) makes the original backup proposal concrete. Its [documentation](https://huggingface.co/docs/lerobot/smolvla) and [paper](https://arxiv.org/abs/2506.01844) support its smaller flow-action design. Smaller parameter count does not establish a working local evaluation; verify processor files, normalization, camera keys, robot action convention, and success on the selected task before switching.

Record the complete interface: checkpoint hash, model/config revision, action horizon, executed prefix length, inference steps, image rotation/cropping, camera order, quaternion convention, state layout, normalization files, gripper sign, command clipping/scaling, controller frequency, and simulator timestep. Also record the imported module locations to catch unintended editable installs.

The local ten-action chunk and five-step execution prefix are engineering settings, not universal π0.5 properties. Keep the nominal inference settings for reproduction. In the safety experiment, monitor each controller step and replan after interventions; apply the same schedule to matched backend comparisons.

## 8. Skill representation and authoring

Use a small declarative schema. The language model chooses supported relations, roles, and parameters. A trusted implementation defines their numerical meaning. VLM drafting is optional when inputs are text and symbolic facts; an LLM is sufficient there. Use a VLM only when visual evidence is part of the authoring or retrieval task.

```text
safety_skills/<skill_id>/
  SKILL.md          # intent, applicability, limitations, examples
  contract.yaml     # executable requirements and typed parameters
  tests.yaml        # independent positive/negative/boundary cases
  metadata.json     # provenance, revision, approvals, evidence references
```

Illustrative contract, **not implemented by this review**:

```yaml
schema_version: 1
id: protected_transport
revision: 1
roles:
  payload: {type: rigid_object, binding: currently_grasped_object}
  targets:
    type: rigid_object_set
    binding: all_matching_objects
applies_when:
  all:
    - {fact: stable_grasp, subject: payload, equals: true}
    - {fact: task_phase, equals: transport}
    - {fact: overhead_sensitive, subject: targets, equals: true}
requires:
  - predicate: avoid_above
    subject: payload
    targets: targets
    quantifier: each
    frame: world
    clearance_m: 0.03
assumptions:
  grasp_model: rigid
  target_motion: static
unknown_action: defer_and_refresh
infeasible_action: validated_stop_then_abort
repair_template: lateral_detour
```

The 3 cm value is illustrative, not a justified safety margin. `repair_template` selects a tested procedure, not arbitrary generated code. The actual schema must define empty target sets, missing properties, multiple payload candidates, and unsupported parameters.

Prevent Markdown–YAML disagreement: execute one approved canonical contract; show its rendered meaning next to the prose. A prose edit creates a proposed contract revision, followed by a diff and tests. An uncompiled prose edit must not be reported as an executed behavior change. If YAML is edited directly, regenerate the explanatory rendering.

Validation needs three levels: syntax/types; grounding and physical preconditions; behavioral tests. Language-model-generated tests alone are insufficient because author and tests may share the same mistake. Preserve invalid drafts and failed tests when measuring authoring reliability.

## 9. The two primitives, precisely defined

**Overhead exclusion:** represent a protected object’s horizontal footprint, expanded by an engineering margin. Evaluate intersection against the carried object’s projected occupied volume, not just its center. Define “above” relative to the target’s top surface: a conservative prohibited set is the expanded footprint extruded upward. Check the payload’s swept occupied volume through that set. This is a projected-region rule, not ordinary collision avoidance.

For a fast approximation, use conservative bounding boxes or convex shapes and document their conservatism. Check every relevant target, not just the nearest one. Continuous sweep checks or sufficiently bounded interpolation must catch crossings between sampled poses. Keep penetration depth/duration and minimum margin as diagnostics.

**Orientation maintenance:** specify a payload-local axis `a`, world reference axis `z`, and maximum tilt `theta_max`. For normalized axes and orientation `R`, a suitable state margin is:

`h_orientation = (R a) · z − cos(theta_max)`.

This permits yaw while limiting tilt. If full attitude preservation is intended, use a rotation-distance constraint instead. State the reference attitude, tolerance units, activation phase, and behavior when already outside the allowed set. “Maintain orientation” without a reference is incomplete.

Neither primitive proves spill prevention, temperature safety, or grasp security. A `contains_liquid` label is a semantic variable unless fluid consequences are simulated. Rigid transport assumptions must be monitored and invalidated on slip or release.

Unit tests should cover coordinate transformations, object extent, boundary contact, swept crossing, multiple targets, yaw invariance, tilt violations, missing data, stale bindings, initially violated constraints, and contradictory compositions. Separately test actual controller tracking and the outcome scorer.

## 10. Retrieval, uncertainty, and composition

Start with deterministic evaluation of applicability over supplied facts. This isolates representation and action effects. Describe it as **oracle-fact retrieval**, not visual common-sense reasoning.

Use `true`, `false`, and `unknown`. Negating an unknown property must remain unknown. A positive human statement about a local rule is different evidence from a speculative visual inference. Store provenance and validity conditions for both.

Maintain active constraints while their applicability remains supported. Rebind or invalidate them after grasp change, reset, release, or semantic-state change. Brief occlusion need not erase a general rule, but cannot justify using an arbitrarily old target pose. Refresh/abstain policies should be explicit and scored for task cost.

For 2–10 records, scan all contracts. Vector retrieval can be a later candidate-generation accelerator; it should not silently discard a hard requirement. Compare all-record applicability evaluation with selected retrieval. Merely loading every record and then correctly evaluating applicability may be equivalent to retrieval, rather than an intentionally bad baseline.

Compose hard requirements by conjunction. Deduplicate identical bindings; do not resolve a hard conflict by averaging rewards. Detect obvious static contradictions, then check trajectory feasibility. If uncertain or infeasible, expose that status. A general rule becoming irrelevant is not the same as granting an exception to it.

For the larger study, use visual fact prediction while preserving oracle evaluation. Report missed applicable skills, false activations, binding errors, and abstention separately. This separates perception failures from incorrect contracts and controller failures.

## 11. Minimal action adaptation that can actually run

The architecture is:

```text
past correction + evidence -> validated skill revision -> persistent library
                                                            |
current scene facts -> applicability + binding -> requirements
                                                            |
observation + original task -> frozen policy -> nominal chunk
                                                            |
                 short-horizon prediction + constraint evaluation
                                                            |
              accept / bounded repair / validated stop + abort
                                                            |
                     per-step execution and independent scoring
```

**Prediction choice:** use a controller-aware short-horizon model, calibrated against simulator trajectories. [Robosuite’s controller documentation](https://robosuite.ai/docs/modules/controllers.html) distinguishes action commands from controller execution at a faster simulation rate. The local 1.4 code, not a newer documentation default, defines the exact adapter. Do not treat normalized OSC pose commands as Cartesian velocities or assume perfect pose tracking.

For debugging, a separately cloned simulator can provide privileged candidate rollouts. It must copy controller state, randomized scene state, velocities, and other relevant state, and must not advance the live environment. Label this as simulator-assisted planning and match access across methods. It is a useful ceiling, not evidence of deployment without a dynamics model.

The first executable method:

1. Obtain one nominal chunk and predict the payload and robot trajectory over a short horizon.
2. If all hard requirements and feasibility checks pass, execute a monitored prefix.
3. Otherwise generate a bounded number of lateral transport detours around expanded protected footprints. Keep the grasp closed and respect the orientation rule.
4. Convert candidate waypoints through the same controller adapter, re-evaluate, and choose feasible progress with the smallest useful deviation from nominal behavior.
5. Execute a short prefix, update state, and replan. Preserve the selected side/subgoal long enough to avoid oscillating between routes.
6. Return control to the nominal policy after clearing the region, and explicitly test whether it can finish from the altered state.
7. If no validated candidate is available, use the tested stopping routine and record task failure or abstention.

The repair routine must not silently perform the entire pick-and-place task. Initially restrict it to a verified transport phase. Log action provenance, correction magnitude, time under fallback control, policy handbacks, and recovery failures.

Start with nominal plus a few geometric alternatives. Add policy resampling only after measuring diversity; `K=8` incurs real cost and may yield eight similar unsafe routes. Maintain the same candidate source/budget when comparing memory representations.

A one-step QP is useful for small local corrections, but does not solve long detours or guarantee future feasibility. Likewise, a bounded candidate search is not complete: failure to find a route does not establish that none exists. No formal safety claim is warranted from empirical prediction accuracy alone.

**Later guidance experiment:** feed the same compiled requirements into an established VLS-style or constrained-flow backend. Compare memory versus regenerated contracts *within* that backend. Compare backends separately at matched wall-clock budgets. This avoids attributing a better optimizer to better memory.

## 12. The experiments that establish a memory contribution

### 12.1 Minimal causal sequence

1. Run task T1 with a defined local overhead-protection rule absent from memory.
2. Supply one correction and create an approved record with positive and negative examples.
3. Restart the policy client and memory process; run T2 with new objects/placements, using the exact stored revision.
4. Repeat paired T2 resets with the relevant record deleted, with an irrelevant record substituted, and with a raw-text correction retained instead.
5. Change the applicability condition through a documented revision; repeat positive and negative cases without execution-code changes.
6. Add orientation as a second record and test a nonredundant composition.

This does not require inducing physical harm: a planned violation can trigger the acquisition example in simulation. If a shield prevents the nominal motion, its unobserved outcome remains unknown. Do not label that nominal motion as empirically unsafe merely because it was blocked.

### 12.2 Baselines

| ID | Method | What it isolates |
|---|---|---|
| B0 | Frozen policy alone | Nominal task competence and violation opportunities. |
| B1 | Original task plus skill text and required semantic context | Prompt-based execution; include token-matched irrelevant text to diagnose prompt disruption. |
| B2a | Always-on geometric protection, shared repair backend | Cost and benefit of blanket avoidance. |
| B2b | Static semantic rule table, identical evaluator/backend | Whether file/retrieval machinery adds anything beyond equivalent logic. |
| B3 | Fresh per-episode contract generation from the same available rule/facts, shared schema/backend | Repeated synthesis versus reuse, including variance and authoring cost. |
| B4 | Retrieved raw or distilled corrections, compiled per episode, shared backend | Whether abstraction into persistent executable contracts helps beyond textual memory. |
| B5 | Safety-memory adaptation inspired by RoboSafe/JIT-Memory | Whether the proposed representation adds value over a strong prior memory design; label it an adaptation, not an exact reproduction. |
| Ours-C | Retrieved, unchanged contracts plus generic repair | Main contract-reuse method. |
| Ours-P | Same contracts plus stored, parameterized repair procedure | Optional test of procedural memory. |
| Oracle | Manually specified correct requirements and bindings, shared backend | Isolates compiler/retrieval errors; not an oracle motion planner. |

Do not run every row for every development scene. First use B0, B2b, B3, B4, and Ours-C on a compact subset. Add B1/B2a for the semantic counterfactual and B5 before making a paper-level memory claim. Add planner-only and memory-deletion controls to diagnose takeover and retention.

Keep offline information, feedback allowance, scene-state access, VLM identity, and execution backend matched where the causal claim requires it. Charge one-time library construction and validation to memory methods. A stronger model writing our library versus a weaker baseline model would confound the result.

### 12.3 Splits and scale

Distinguish development, acquisition, validation, and final test data. “Held out” refers to the new safety-memory study; standard LIBERO tasks may already be in the base policy’s training set. Claim held-out task–skill combinations, not unseen base-policy tasks, unless established.

Recommended first panel: 2 task families × 4 semantic conditions (none, A, B, A+B) × 25 reset seeds, balanced across 3–4 poses. That is **200 episodes per method**, not 25 independent episodes for every pose. Five methods require 1,000 episodes, plus screening/acquisition. Only include all four conditions if B is nonvacuous.

For publication, expand to multiple acquisition contexts, held-out object instances and combinations, and at least several independent acquisition orders. Keep learned memory frozen on the final test. Curves used repeatedly for model selection belong on validation data, with a separate final test.

Fifty resets in one scene do not establish generalization across scenes. Aggregate task/scene families explicitly and report each family. Use paired confidence intervals over matched resets; with enough independent families, use hierarchical resampling or a hierarchical model. With only two families, limit population-level transfer claims. Choose final sample sizes using pilot variance and the minimum meaningful effect, not a universal “50 episodes is enough” rule.

## 13. Scoring, auditability, and proof-of-concept artifacts

Let `G_e(t)` be the independently defined applicable requirements for episode `e`, and let `V_j(t)` be their actual-state violation indicators. Define:

`safe_success(e) = task_success(e) AND no physical-safety violation AND no V_j(t) for any j in G_e(t), over the full scored episode`.

The retrieved set does not define the evaluator’s obligations. Score terminal transitions and a specified post-placement settling interval. Report semantic requirements separately from physical contacts and feasibility events. For comparisons involving different site policies, each episode’s ground-truth policy is fixed before running any method.

Required outcomes:

- Task success, safe success, semantic violation rate, physical violation rate, and abstention/timeout rate.
- Per-primitive violation duration, minimum margin, and peak tilt or penetration proxy; do not combine unlike severities into an unexplained number.
- False activation and missed activation, binding accuracy, and unknown-state handling.
- Policy/fallback action fractions, path length, handback success, and time-to-goal.
- VLM calls, tokens, latency percentiles, candidate counts, allocated GPU time, and CPU simulation time.
- Human minutes, edits, repeated corrections, rejected drafts, and validation cases. Author-estimated effort is an engineering log, not a human-subject study.
- Memory retention after restart, contract hashes on transfer, revision regressions, and sensitivity to acquisition order.

**Figure 1:** fixed geometry and task under no requirement, A, and A+B; actual trajectories, applicability facts, constraint traces, and action provenance. Schematic paths are labeled schematic until rollouts exist.

**Figure 2:** correction on T1, stored record, process restart, held-out T2; compare retained versus deleted memory.

**Figure 3:** held-out safe success and repeated feedback cost versus acquisition budget, including text-memory and static-table controls.

Release paired reset manifests, exact contracts, raw drafts/rejections, independent scorer tests, episode JSON logs, and videos of successes **and** failures. A rendered skill file plus an attractive rollout is not enough evidence.

## 14. PSC resource and training decisions

The [Bridges-2 guide](https://www.psc.edu/resources/bridges-2/user-guide/) documents H100 80 GB, L40S 48 GB, and V100 resources, individual GPU requests through `GPU-shared`, and accounting at 2 SU/GPU-hour for H100 versus 1 for L40S/V100. Actual allocation access, availability, and queue delay were not measured here.

OpenPI’s [repository requirements](https://github.com/Physical-Intelligence/openpi) give indicative memory floors of over 8 GB for inference, over 22.5 GB for LoRA, and over 70 GB for full fine-tuning. These are repository-level estimates, not measured π0.5 memory for this simulator/batch configuration. A single H100 is a reasonable first profiling device; L40S may be more economical if its throughput is sufficient. V100 compatibility should be tested rather than assumed for a BF16-oriented stack.

Use CPU workers for physics and reserve GPU headroom for rendering and policy inference. Keep VLM drafting offline or on an existing accessible service. Do not run model inference or rollouts on a login node. Pin environment paths, bound library thread counts, and inspect the existing driver’s documented reset/subprocess issue before launching long jobs.

**Billing model:**

`SU = sum_over_jobs(GPUs_allocated × elapsed_allocation_hours × GPU_type_multiplier)`.

Kernel timing diagnoses utilization; it is not the bill. For a single sequential server/worker configuration, estimate elapsed time from model loading/compilation, simulation, inference, rendering, logging, and retries. With concurrent workers, measure the aggregate throughput rather than adding overlapping times.

**Illustrative sensitivity calculation, not a forecast:** for 1,000 episodes at an observed end-to-end 30/60/120 seconds per episode while one H100 remains allocated, runtime would be 8.3/16.7/33.3 GPU-hours and compute charge 16.7/33.3/66.7 SU, before startup and retries. A development reserve of 50–100 H100 hours (100–200 SU) is a discretionary ceiling for discussion, not a required or measured budget. Replace it after a 20-episode profile.

H100’s doubled SU rate means it is cheaper per completed episode than L40S only if its measured end-to-end throughput exceeds roughly twice L40S’s, ignoring queue and other constraints. Simulator bottlenecks can eliminate that advantage.

**Fine-tuning decision:** keep the policy frozen until retrieval/compiler errors are separated from control failures. If known-correct constraints still require prolonged repair, collect successful guided trajectories and learn a small context-conditioned residual or policy adapter. Include nominal and negative-applicability data so it learns when not to intervene. Split by entire episode/task–skill grouping, not adjacent frames. Keep the safety evaluator at deployment.

Test skill descriptors/parameters rather than only skill IDs if unseen-skill transfer is claimed. An adapter keyed to two IDs demonstrates two-condition adaptation, not general skill interpretation. Compare adapter-only, frozen-plus-filter, and adapter-plus-filter to separate latency from safety effects.

The [OpenPI PyTorch LoRA issue](https://github.com/Physical-Intelligence/openpi/issues/842) remains a feature proposal in the retrieved page. Verify supported code, trainable parameter counts, gradients, save/reload equivalence, and peak memory with a small training smoke test before forecasting a run. Do not promise full training or a fixed dataset size from hardware capacity alone.

## 15. Execution milestones and decision gates

Indicative schedule for one researcher with functioning dependencies: several weeks, dominated by controller integration and evaluation. Queue delays and new perception requirements can extend it. These are planning estimates.

| Stage | Concrete deliverable | Gate / response |
|---|---|---|
| 0: audit and reproduce | Independent scorer tests; pinned policy/config; 20+ clean baseline episodes; timing/memory profile | If baseline fails, repair the adapter or change the task before introducing memory. |
| 1: one invariant | Hand-authored overhead contract; paired geometry test; executed feasible detour; handback log | If transport cannot be repaired reliably, shorten the scope or improve the backend. |
| 2: actual retention | Acquisition correction, process restart, unchanged-record transfer, deletion/text-memory controls | If behavior is identical with memory deleted, identify hidden hard-coding or an uninformative task. |
| 3: selective composition | Relevant/irrelevant semantics, orientation challenge, A+B, unknown/conflict cases | If B is redundant, do not claim composition from it. |
| 4: authoring and revisions | Fixed drafting protocol, failures retained, human-readable diffs, regression tests | If every scene requires manual code edits, the reusable interface has failed its scope. |
| 5: comparative verdict | Matched B0–B4 subset; strong safety-memory adaptation; held-out results and costs | Extend only if memory adds measurable retention, transfer, reliability, or effort value. |

Suggested practical targets to set **before** final testing: a useful safe-success gain over the strongest affordable comparator, limited task-success loss in negative cases, and acceptable takeover/latency for the intended demo. Select numerical tolerances from development evidence and faculty priorities, then freeze them. Do not choose them after inspecting final results.

## 16. Bigger directions, ranked by usefulness and risk

| Direction | Why it could matter | Feasibility / novelty caution |
|---|---|---|
| Contract plus reusable repair procedure | Retains how to recover progress, not only a prohibition; can reduce online search. | Best next extension. Compare against JIT-style procedural memory and generic planning. |
| One skill, two policy backends | Tests whether safety knowledge survives changing the nominal policy. | Feasible after the first interface works; each policy needs its own validated adapter. |
| Grounded revision from counterexamples | Narrows overbroad rules and preserves exceptions with evidence. | Stronger learning question; feedback must identify the intended rule and avoid hidden test leakage. |
| Action-relevant memory selection | Retrieve based on which requirements would change candidate actions, beyond text similarity. | Potentially useful at scale; must preserve recall of hard rules and beat scanning the small library. |
| Temporal/stateful safety | Retain a fact such as “this object was heated” until evidence invalidates it. | Makes working memory necessary, but needs a faithful state variable and observable evidence; not a thermal-safety result from a label alone. |
| Differentiable guidance | May produce less disruptive repairs than post-processing. | Established approach; isolate backend benefit from memory benefit. |
| Learned dynamics or hazard evaluator | Needed for consequences beyond rigid geometry. | High data/validation burden; introduce only for a specific failure of the simpler model. |
| Full VLA fine-tuning or a new simulator | Could eventually support broader behavior. | Weak first use of resources for a memory hypothesis. |

A useful theoretical artifact would be a precise contract semantics and a conditional statement about monitor correctness under bounded geometry/prediction error. Do not elevate that to full physical safety unless stopping, inter-sample behavior, initial safety, and future feasibility are addressed. The representation’s scope should remain explicit.

## 17. Faculty pitch

> We propose to test reusable safety skills as persistent knowledge around a frozen manipulation policy. The robot receives a correction once, stores a validated contract and its applicability, and later binds that same record to new objects and tasks. We will measure whether retention and executable reuse improve safe task completion or reduce repeated specification effort compared with text memory, regenerated constraints, and static semantic rules. The initial system uses the existing LIBERO stack, official π0.5 weights, two geometric invariants, and a bounded transport-repair backend. Closely related work already demonstrates skillbooks, safety memory, and policy guidance; our experiment asks whether a stable reusable contract makes a measurable difference at the continuous-action interface.

**Decision requested:** agree on the two development tasks and held-out pairings; choose the minimum meaningful benefit and acceptable intervention cost; allocate a small profiling block first; expand only after the memory-deletion and strong-baseline tests succeed.

**What this document supports today:** a research rationale and executable experiment specification. **What still requires evidence:** nominal reproduction, reliable correction, useful transfer, reduced authoring burden, and any publication-level novelty claim.
