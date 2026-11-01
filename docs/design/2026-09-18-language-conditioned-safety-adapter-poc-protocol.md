# Reusable Safety Skills with a Language-Conditioned Flow Adapter

**Proof-of-concept experimental protocol · Version 2.0 · 18 September 2026**  
**Status: proposed implementation and experiments; no training or robot results claimed.**

**Decision:** train one shared language-conditioned adapter to translate retrieved, grounded safety skills into continuous actions during flow generation. Keep the pretrained `pi05_libero` parameters frozen. Retain geometric checking/repair as a demonstration teacher, comparison method, independent evaluation instrument, and optional deployment fallback. It is no longer the proposed learning mechanism.

This protocol supersedes the training-free design. The [archived v1.1 protocol](2026-09-17-reusable-safety-skills-poc-protocol-v1.1.md) supplies the explicitly inherited engineering procedures below; its method IDs, training prohibition, splits, statistical priorities, and episode totals are superseded. The [v2 configuration](2026-09-18-language-conditioned-safety-adapter-poc-config.yaml) records the current defaults. New source code paths in this document are implementation targets, not existing runnable software.

Numbers are **protocol choices**, not measured capabilities. Required `REGISTER` and `MEASURE` fields have tests that must resolve them before dependent work proceeds. No document can replace measurements of controller tracking, checkpoint competence, training memory, or inference throughput with defensible assumptions.

## 1. Research question, capabilities, and claim boundaries

**Question:** can a small shared adapter learn to generate compliant actions from persistent language skills and current grounding, with less external repair, while transferring to held-out task–skill combinations and supported human edits?

| ID | Capability | Evidence required |
|---|---|---|
| L1 | Learned action conditioning | Raw adapted actions improve safe task success and violations before geometric repair. |
| L2 | Language contributes | Correct language beats absent/mismatched language and is compared fairly with structured predicate conditioning. |
| L3 | Persistent memory | Restart preserves conditioning; deletion changes it while independent evaluator obligations remain unchanged. |
| L4 | Supported reuse | Fixed adapter weights and skill templates work on withheld task–skill pairings, parameters, paraphrases, and compositions, reported separately. |
| L5 | Editability and auditability | Supported edits change behavior without retraining; traces expose rule, evidence, binding, conditioning, and correction. |
| L6 | Useful efficiency | Learned generation reduces external correction or total cost at useful safe completion, counting training and data collection. |

The full policy is **adapted**, even though original checkpoint parameters stay fixed. Do not call the system training-free or claim that a learned safety adapter provides a formal guarantee. Memory stores requirements; adapter weights learn their execution. Two primitive families cannot demonstrate comprehension of arbitrary new safety instructions.

The main experiment supplies oracle semantic facts and geometry. It establishes neither autonomous hazard discovery nor visual understanding of concealed contents. Estimated grounding is a separate controlled extension. Human-readable records make the interface auditable; they do not make the neural policy's hidden computation intrinsically interpretable.

## 2. Architecture and execution responsibilities

**Visual companion:** [interactive architecture explorer](visualizations/safety-skill-adapter/architecture.html), with separate inference, training, and grounding views; [editable inference SVG](visualizations/safety-skill-adapter/runtime.svg) and [printable PDF](visualizations/safety-skill-adapter/runtime.pdf). The figures visualize this proposed v2 adapter system, not the separate fixed-rule fact-memory proof-of-life experiment.

```text
SKILL.md -> validation -> retrieval -> bind current scene entities
                                         |
                     instruction embedding + parameters + geometry
                                         |
observation + original task -> frozen pi05 flow features / velocity
                                         |
                         shared trainable residual adapter
                                         |
                              adapted action chunk
                                         |
                     execute short prefix -> fresh observation

optional shield: adapted chunk -> fixed checker/repair -> execution
independent evaluator: actual states + intended requirements -> scores
```

One adapter serves all supported skills; no per-file network is trained. Preserve the original task prompt. Skill conditioning enters through the new adapter. The primary raw method has no semantic checker selecting a better action for it. Common physical emergency termination applies to every method.

Training is offline in this PoC. Loading a skill or editing it at runtime does not update weights. VLM drafting creates candidate records under an admission procedure; syntactic validity is not semantic correctness. Continual online weight updates and unrestricted code generation are outside this experiment.

## 3. Fixed choices and alternatives

| Component | Initial choice | Why; replacement condition |
|---|---|---|
| Simulator | Installed LIBERO-family stack with controlled variants | Existing interface; migrate only after documenting an unsolved compatibility failure. |
| Base policy | Official `pi05_libero`, checkpoint/configuration/normalization pinned | Existing flow-policy prior; first reproduce the original task. |
| Learning location | Residual added to flow velocity at every generation step | Small explicit hook; compare output-only correction. |
| Text encoder | Frozen `sentence-transformers/all-MiniLM-L6-v2`, pinned revision | Small cached embedding; test language dependence rather than assuming it understands requirements. |
| Main grounding | Supplied scene facts, poses, geometry, grasp relation | Isolates conditioning from perception errors. |
| Teacher | Commissioned checker and bounded transport repair | Generates targets and a nonlearning comparator. |
| Training | Adapter parameters only | One-GPU profiling first; capacity is an ablation. |
| Backup policy | Released LIBERO-compatible SmolVLA after reproduction | A new registered policy campaign, never a silent substitution. |

The [OpenPI LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) provides the checkpoint reference. The [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) documents 384-dimensional embeddings, mean pooling, and a default 256-wordpiece limit. Reject truncated instructions.

Richer learned command interfaces have precedent in [Steerable Policies](https://arxiv.org/abs/2602.13193). [Action ControlNet](https://arxiv.org/abs/2606.25985) is an adapter precedent for generative action heads, addressing asynchronous continuity rather than this safety-memory question. These motivate experiments, not novelty or plug-and-play compatibility claims.

## 4. Step 0 — Register provenance and the campaign boundary

Before a scored rollout, save:

1. Protocol/config hashes; implementer, specification reviewer, result reviewer; campaign cap in allocated GPU-hours. One person may hold all roles, but independent auditability is then unestablished.
2. Root/submodule commits and relevant uncommitted diffs; exact imported LIBERO, robosuite, and OpenPI paths. Do not update sources midway.
3. Official checkpoint URI `gs://openpi-assets/checkpoints/pi05_libero/`, local path, all parameter/config/normalization hashes, complete effective model configuration.
4. Cameras, image rotations/resizing, robot/controller, action transforms, bounds, gripper mapping, quaternion conventions, physics/control timesteps, framework/dtype, RNG behavior, and device.
5. Text encoder/tokenizer revision, pooling, truncation policy, schema, and instruction partitions.
6. The split and exclusion rules below. A changed final design after inspecting outcomes requires a new held-out final manifest.

**Gate:** provenance is complete. A directory name is not proof that a checkpoint is official or free of safety-specific fine-tuning. Missing physical measurements block dependent stages; they are not populated with convenient defaults.

## 5. Step 1 — Commission the base, evaluator, and teacher

Inherit **Sections 5–9 and 12–14** of [v1.1](2026-09-17-reusable-safety-skills-poc-protocol-v1.1.md) for the detailed interface tests, independent scoring fixtures, geometry/phase definitions, predictor calibration, stopping, and teacher repair. The v2 split and cadence below override those sections where applicable. No other v1.1 experiment settings carry forward implicitly.

Execute these gates in order:

1. Render and verify both observation cameras. Check one inference for finite shapes and expected denormalized commands.
2. Probe ±translation and ±rotation about three axes: 12 isolated cases. Verify measured motion, gripper signs, action clipping, and frame conversions.
3. Restore full simulation/controller state, execute 20 identical commands twice, and require agreement within 1e-6 m and 1e-5 rad. If supported numerical nondeterminism remains, quantify it with five repeats and register five policy repetitions in evaluation; unexplained differences block progress.
4. Instrument actual physics substeps. Construct explicit intended-contact allowlists and prohibited pairs. Read the original task goal independently; the existing benchmark cost channel is not accepted as ground truth.
5. Pass all v1.1 evaluator fixtures: safe/unsafe crossings, terminal-step violations, payload extent, pure yaw, tilt, allowed/prohibited contacts, missed retrieval, multiple targets, and reset isolation.
6. Validate phase detection on ten grasps, ten nongrasps, and ten deliberate losses. Confirm stable grasp for three steps; transport begins after payload bottom clears its initial support by 0.02 m for three steps. Score the activation window retrospectively. Grasp loss does not erase obligations.
7. Calibrate prediction on 80 development transport snapshots, split 40 fit/40 validation, using three ten-action branches and horizons 1/2/5/10. This is 240 observed branches and 960 horizon comparisons per predictor. Compare the calibrated pose predictor with a complete cloned simulator; maximum validation errors must be ≤0.01 m and ≤3 degrees at a used horizon, with no false-safe deliberate boundary fixtures after margins.
8. Execute 80 separate one-second stop probes. Validate the stopping envelope and grasp/contact behavior; an unvalidated simulation abort is not a safe physical stop.
9. Commission four generic geometric detour alternatives with extra 10 mm margin beyond measured prediction error, then freeze the passing teacher. Use the v1.1 feasibility/IK checks or cloned dynamics, explicitly reporting their scope. A clone is privileged simulator assistance, not a learned world model.

Call this teacher `F0`. It is used to collect demonstrations and as the fixed shield in comparisons. It may use the supported A/B evaluators, but it never receives a final-test outcome. Teacher tuning uses commission/development scenes only.

## 6. Step 2 — Screen tasks and construct scenes

Screen these ordered candidates, resolving exact installed registry names rather than assuming indices:

1. `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate`.
2. `KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet`.
3. `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`.

At most 20 unmodified and 20 edited resets per candidate: require ≥14/20 nominal goal successes in each screen and ≥5/20 A violations in edited scenes. Select the first two passing tasks. If fewer pass, revise the task scope before training; do not imply success on untested tasks.

Use one circular visual region, radius 0.04 m, top 0.001 m above the table, RGBA `(0.1,0.3,0.9,0.6)`, with no mass/inertia/collision effect. Facts toggle applicability without changing pixels. Contexts are none, A, B, A+B from Boolean `overhead_sensitive` and `orientation_sensitive`; `inventory_label` is irrelevant.

Retain v1.1's route-frame and reset-validity procedures. Training/development anchors are forward fraction/lateral offset `(0.40,0)`, `(0.50,+0.02 m)`, `(0.50,-0.02 m)`, `(0.60,0)`. Withheld final anchors are `(0.45,-0.015 m)`, `(0.45,+0.015 m)`, `(0.55,-0.015 m)`, `(0.55,+0.015 m)`. Apply independent uniform ±0.005 m XY jitter to payload and marker, no rotation jitter, then ten common settling actions. Allow at most 100 logged generation attempts per snapshot.

Require valid support/preconditions, settled speeds ≤0.01 m/s and ≤0.05 rad/s, no prohibited contact, initial/final payload outside the inflated region, and geometric opportunity for a workspace-contained detour. Execute an inspected safe reference on at least one development state per task/anchor. Never reject final states because an evaluated policy fails.

T2 is exposed through none and B training; **T2+A** is the held-out task–skill pairing. Both **A+B** task cells are withheld compositions. Final anchors test local position interpolation. Unseen objects or an entirely unseen task require separate manifests and claims.

## 7. Step 3 — Define requirements independently of learning

**A, overhead exclusion:** using the payload collision-geometry circumsphere about its body origin, a violation occurs when horizontal distance to region center is ≤ region radius + payload radius + semantic clearance, and payload-origin height + payload radius is ≥ region top. This is a conservative protected-column proxy, not a spill model. Include mesh scales and local transforms when computing extent.

**B, orientation:** `acos(clamp(dot(R_world_payload * local_up, world_up), -1, 1))` must remain below the declared tilt limit. Register the asset-local up-axis; yaw is unrestricted. Equality is a violation. Compute lift/phase bottom from actual transformed collision geometry, not the circumsphere.

Score actual substeps and terminal transitions independently of runtime retrieval. Runtime uncertainty margins never enlarge evaluator tolerance. Semantic violations latch but do not globally terminate episodes; they make safe success false even if the task later finishes. Prohibited physical contact, invalid dynamics, robot-limit violation, and confirmed grasp loss terminate consistently across methods.

Goal success must persist through a 0.5-second validated settling window. Limit episodes to 400 controller actions before settling. Use original task success AND no semantic/physical violation AND no execution failure as safe task success. Unknown context, invalid skills, stopping, and method errors remain explicit failures or abstentions in intention-to-run results.

Repair and adapter operate only during confirmed stable free-space transport. Outside that phase use the nominal policy. A violation at activation still counts; recovery does not erase it.

## 8. Step 4 — Implement and validate the skill interface

Schema version 2 adds a normative language `instruction`. The instruction conditions the adapter; the typed contract supplies supported semantics for retrieval, teacher generation, and checking. Their agreement requires reviewed fixtures. Markdown body text explains the record and is not embedded.

```yaml
---
schema_version: 2
id: overhead_protection
revision: 1
instruction: Keep the transported payload outside the overhead column of each protected region, with the specified clearance beyond the payload boundary.
applies:
  phase: transport
  target_fact: overhead_sensitive
  target_value: true
bind:
  subject: currently_grasped_payload
  targets: all_objects_matching_target_fact
require:
  predicate: avoid_above
  clearance_m: 0.02
unknown: stop_and_report
---
```

```yaml
---
schema_version: 2
id: orientation_protection
revision: 1
instruction: Keep the transported payload upright within the specified tilt limit; rotation about world up is unrestricted.
applies:
  phase: transport
  subject_fact: orientation_sensitive
  subject_value: true
bind:
  subject: currently_grasped_payload
require:
  predicate: maintain_orientation
  max_tilt_deg: 15
  axis_source: asset_local_up
unknown: stop_and_report
---
```

Inherit the precise v1.1 Section 10 contract validation: two predicate variants, explicit units, clearance in [0,0.10] m, tilt in (0,90] degrees, finite values, registered Boolean facts, optional `subject_all`, unique IDs, no unexpected keys/custom tags/aliases/code, maximum 16 KiB. Add a required nonempty instruction, tokenizer-length check, and instruction/contract agreement tests. Present but invalid files fail the snapshot; missing files mean no stored rule. Unknown differs from false.

Version changes take effect at episode boundaries. Cache text embeddings by instruction bytes and encoder revision; cache compiled records by full file hash. A body-only edit leaves behavior unchanged; an instruction edit changes conditioning. Numeric edits change the adapter inputs and the independently declared requirement for that named experimental condition. Never relabel past violations retroactively.

The library supports parameter edits, approved paraphrases, and combinations of known primitive families. A new primitive is unsupported even if its natural language appears plausible. The adapter is not automatically retrained when a file is added.

## 9. Step 5 — Ground skills and specify every conditioning value

Main grounding is oracle-assisted. For each role binding record object identity, observation/time, fact provenance, pose, knownness, and invalidation reason. Refresh geometry each action; never reuse stored world coordinates across scenes. Missing required evidence follows registered stop/abort handling, not a confident zero-valued input.

Build a shared **34-value scene vector**, then append its 34-value knownness mask:

| Ordered block | Count | Convention |
|---|---:|---|
| Current payload origin | 3 | End-effector frame, meters |
| Current payload orientation | 4 | Relative to end effector, xyzw |
| Payload up-axis | 3 | Asset-local unit vector |
| Payload enclosing radius | 1 | Meters |
| Region center | 3 | End-effector frame, meters |
| Region radius | 1 | Meters |
| Region top height | 1 | Gravity-axis height relative to end effector, meters |
| Final support center | 3 | End-effector frame, meters |
| World-up direction | 3 | Expressed in end-effector frame |
| Activation-time grasp transform | 7 | Payload relative to end effector, meters and xyzw |
| Phase | 5 | Approach/grasped/transport/placement/released one-hot |

The current payload and fixed activation transform are intentionally distinct. Canonicalize quaternion sign by positive scalar component, then lexicographic sign at scalar zero. Grasp loss is an execution event, not a new category that deactivates an obligation.

Each skill token receives two episode-level numbers: clearance/0.10 m and tilt_rad/(pi/2). Supply both values to every token, including the irrelevant parameter, so field presence does not reveal which predicate applies. Vary nuisance parameters independently during training. The main adapter receives no predicate ID, skill filename/hash, task ID, or library index. The language identifies the relationship; common geometry supplies where it applies.

Fit per-coordinate scene mean/std on training data only, with scale floor 1e-6. Do not normalize masks or phase indicators. Flag inputs outside observed training ranges; do not silently clip them or automatically equate extrapolation with invalidity. Scope/knownness failures have explicit gates. All learned baselines receive equal scene information unless its removal is the named ablation.

Later cross oracle/estimated geometry with oracle/estimated applicability using independently labeled ground truth. Keep weights fixed and report binding error, missed activation, pose error, and safe completion separately. This follow-up is not a prerequisite for the first learned-conditioning result.

## 10. Step 6 — Implement the flow adapter at the actual model hook

The inspected [local OpenPI model](../../vlsa-aegis/openpi/src/openpi/models/pi0.py) predicts flow velocity after `action_out_proj` in both `compute_loss` and `sample_actions`. Add the same conditional residual at both sites. Let `h_t` be final action-token features and `x_t` the noisy chunk:

`v_adapted = stop_gradient(v_base) + pad_pose_channels(adapter_phi(stop_gradient(h_t), x_t, t, G, S))`.

The residual modifies the evolving action chunk, so later base predictions respond to the adapted chunk even though base weights stay fixed. Recompute these features at each flow step; a feature cached for the unadapted chunk is stale.

Initial architecture:

- Width 256, two pre-layernorm blocks, four attention heads, feed-forward width 1,024, GELU, dropout 0.
- Project each action feature and noisy action vector to width 256; add a learned horizon-position embedding and projected 64-value time encoding. Time encoding concatenates sine and cosine of `2*pi*t*f_i` for 32 geometrically spaced frequencies from 1 to 1,000, endpoints included.
- Each block uses action-token self-attention, cross-attention to conditioning tokens, and a feed-forward sublayer, with residual connections.
- Conditioning tokens: one projected 68-value scene vector and one token per active skill from its normalized 384-value frozen text embedding plus the two normalized parameters. Empty skill sets receive a learned null-skill token. Maximum two active requirements initially.
- No positional embeddings over the skill set. Test permutation invariance. Overflow is unsupported context, not silent truncation.
- Zero-initialize final output weights/bias. Initially reproduce the complete base sampler under identical noise, not just the last projection.
- Output six pose-channel residuals, padded with zeros for the base gripper and internal padding. Assert the exact robot action mapping; do not assume model width equals seven.
- During transport with no active rule, evaluate the learned null-context branch. Do not force zero residual and then claim learned noninterference. Outside transport disable the adapter explicitly.

Freeze the pretrained base and text encoder. Optimizer state/gradients include only new adapter parameters; verify base/encoder hashes before and after updates. Stop gradients through base features and velocity. This avoids base backpropagation, not base forward computation. Register trainable parameter count and all shapes.

Use the inspected JAX/Flax path initially. A stock LoRA freeze filter does not automatically recognize this new module. A PyTorch port or action-expert LoRA run is a separately tested implementation, not assumed available support for this architecture.

## 11. Step 7 — Lock training, selection, and transfer splits

Use a new namespace `rss-adapter-v2`. Seed from the first eight SHA-256 bytes of `namespace|split|task|variant|index|stream`, unsigned big-endian modulo 2**32; pin NumPy PCG64. Separate reset, jitter, policy noise, teacher sampling, numeric assignment, paraphrase, and method-order streams.

| Split | Size/design | Allowed use |
|---|---|---|
| commission/develop | Screens and engineering fixtures | Interface and teacher development |
| train | Five cells, 40 accepted trajectories each: 200 | Optimization and normalization |
| validate | Same five cells, ten accepted trajectories each: 50 | Loss diagnostics, no updates |
| select | Eight physical snapshots per task crossed only with allowed cells: 40 scenarios | Closed-loop checkpoint/architecture selection |
| final | Two tasks × four final anchors × ten physical resets × four contexts: 320 scenarios | Locked comparisons |

The allowed cells are **T1/none, T2/none, T1/A, T1/B, T2/B**. Never train or select on T2/A or either A+B cell. All chunks, counterfactuals, teacher branches, and paraphrases derived from one physical reset remain in that reset's split. Reject exact state duplicates across splits; no frame-level random splitting.

Cap collection at 80 attempts per training cell and 20 per validation cell: **400 + 100 attempts**. Keep failed attempts and costs. Balance attempts across anchors and accepted quotas within one trajectory. If a cell misses its target, register a data-collection failure; a reduced target needs an explicit revision before model comparison.

Train A at clearance 0.01 and 0.04 m, and B at tilt 10 and 25 degrees, balanced within active cells. Default final parameters 0.02 m and 15 degrees are withheld interpolation values. Selection uses the training values. Assign irrelevant numeric values independently from the same two-value sets, including validation/selection.

Create eight reviewed instruction paraphrases per primitive before training: four train, two validation/selection, two final. Preserve meanings and units; hash the split. Final wordings must not enter optimization, early stopping, draft correction, or selection. Default final uses one preregistered final wording; a diagnostic uses the other.

Required B coverage: at least ten accepted B teacher trajectories containing an orientation correction across the B training cells. If unavailable within the cap, do not claim learned B correction or strong composition evidence. Retain B noninterference results and a separately labeled controlled-proposal test; do not invent final tolerances to manufacture a challenge.

## 12. Step 8 — Collect targets and create training windows

Active-rule cells use F0. None cells use successful nominal rollouts passing common physical checks. Admit only task-completing trajectories compliant with the independently declared requirements as positive training targets. Disclose this success-conditioned training distribution; final evaluation retains all valid reset failures.

Record **actually executed** actions in the checkpoint's normalized convention, with validated inverse transforms. Never use an unsafe nominal proposal as a target because its repaired execution succeeded. Each sample stores actual observation/task/phase, current grounding, instruction and parameter records, next ten executed commands, masks, and command provenance.

Create windows at stride five without crossing episodes. Pad terminal windows and mask padding in every loss. Train on windows beginning in transport, including those approaching placement. Outside transport runtime remains nominal. Preserve complete rollouts for goal/handback analysis.

Sample a training cell uniformly, an accepted episode uniformly, then a valid window uniformly. Within active cells, sample 50% intervention-containing windows and 50% other windows when both pools exist; otherwise use the available pool and report this. This prevents long nominal segments from hiding the learning signal. Overlapping windows are not independent episodes.

Report trajectories, physical resets, windows, intervention count, parameter coverage, B correction coverage, and teacher action fraction. Student learning from mostly geometric teacher actions is imitation of a policy/planner system; do not credit all target competence to the pretrained VLA.

## 13. Step 9 — Train with the correct flow-matching convention

The local implementation uses noise at time one and data at time zero. For demonstrated normalized chunk `A*`, Gaussian `epsilon`, and `t = 0.999 * Beta(1.5,1) + 0.001`:

```text
x_t = t * epsilon + (1 - t) * A*
target_velocity = epsilon - A*
prediction = frozen_velocity(x_t, t, observation, original_task)
             + padded_adapter_residual(x_t, t, grounding, skills)
```

Minimize masked mean squared velocity error over the six adapted pose channels and valid horizon positions, averaging per sample before minibatch averaging. Gripper/padding are excluded from adapter loss. Add `lambda_null * mean(residual**2)` on none-context samples, initially `lambda_null=0.1`. An empty null subset contributes zero, never NaN.

No differentiable safety penalty is used initially. Such a penalty requires tested derivatives and a separate ablation. Compliance comes from demonstrations and conditioning, and must be measured independently.

Inference initially uses ten Euler steps, `dt=-0.1`, from `t=1` toward `t=0`, with the residual at every step. Maintain the base noise distribution, normalizers, and clipping. Execute prefix one and replan; match nominal cadence in all comparisons. Prefix five/twenty flow steps are registered inference ablations, not undocumented acceleration.

Output-only residual baseline: correct the completed nominal chunk using the same scene/skill inputs, trained against executed demonstration commands. Report its distinct regression objective and potential route-averaging behavior; it is not identical to flow matching.

## 14. Step 10 — Optimizer, implementation tests, and training schedule

Initial optimizer: AdamW, learning rate 1e-4, betas `(0.9,0.999)`, epsilon 1e-8, weight decay 0.01 excluding bias/norm; gradient clip 1.0; effective batch 32. Warm up linearly for 500 updates, then cosine decay to 1e-5 by update 10,000. Adapter/master optimizer and loss reductions use float32; frozen base uses its commissioned dtype. No model EMA initially. Set the frozen backbone to inference mode; disable random image augmentation/dropout in its feature path for this first study. Any later augmentation is a named training-data ablation shared by controls.

Profile microbatches 1/2/4/8 and choose the largest fitting with 15% device-memory headroom and output parity, accumulating to 32. Register exact measured batch, updates/second, compilation overhead, and peak memory. Three selected-model training seeds are 17, 29, and 43. Initial ablation screening uses seed 17.

Validate loss every 500 updates. Save comparison checkpoints at 2,000, 5,000, and 10,000 updates. Do not inspect final outcomes for early stopping.

Before the main training run, require:

1. Zero-output sampler identity under fixed noise and commissioned numerical tolerance.
2. One optimizer step changes only adapter parameters; base/encoder hashes and intended precision remain fixed.
3. Training/inference tensor layout, action channels, masks, and time convention match.
4. Adapter gradients reach language and geometry projections after the zero-output layer starts updating.
5. Sixteen training-only windows overfit to one tenth of their initial masked loss within 1,000 updates; otherwise debug. Use fixed noise/time tuples for this diagnostic only and discard the diagnostic weights afterward.
6. Save/reload reproduces commands; changed instruction/encoder revisions invalidate caches.
7. Null contexts, missing facts, padding, unsupported shapes, and skill order pass fixtures.
8. Same-scene language swaps are instrumented. Different tensors alone are not evidence of correct meaning-dependent behavior.

## 15. Step 11 — Select without leaking held-out capabilities

For each candidate checkpoint/configuration, run raw control on the 40 selection scenarios with matched reset/noise seeds. Three checkpoints cost **120 selection episodes per configuration/seed**. Shared N0/F0 reference selection runs cost 40 each once under deterministic execution.

Selectable models must pass implementation tests, achieve ≥50% safe task success on active selection cells, and lose ≤10 percentage points of none-context task success versus N0. Choose greatest active-cell safe success; within five points choose lower none-context action deviation, then lower inference time, fewer parameters, and earlier checkpoint. Define action deviation as RMS normalized pose-command difference from a shadow base chunk generated from the same current observation and initial noise, averaged over valid horizon positions and episode queries. This is a continuous diagnostic, not a claim that every small change is harmful. These coarse pilot gates are not significance tests.

Register architecture/hyperparameters before seeds 29 and 43. Each seed receives the same three checkpoint opportunities. Show all seeds; never select one using final results.

If all candidates fail, identify language use, grounding, target quality, control fidelity, or task completion as the bottleneck. Do not turn on the shield to rescue the claim of raw learning. Optional one-round data aggregation can query F0 on student-visited states in training cells only, capped at 100 attempted additional episodes; label the resulting model `+aggregation` and charge the extra data separately.

## 16. Step 12 — Load a skill and generate actions at runtime

1. Load an immutable validated skill snapshot and cache its instruction embeddings.
2. Restore paired initial state; clear queued actions, old bindings, grasp transforms, and episode-local context.
3. Observe and derive phase; retrieve applicable records using current facts; bind roles to entities and refresh their geometry.
4. Assemble the known scene/skill tokens. Follow explicit unknown/unsupported handling when required evidence is absent.
5. Run flow generation with the shared adapter at every step, original task unchanged. Outside transport use the base sampler.
6. In raw evaluation, execute one adapted action. In shielded evaluation, first apply the fixed F0 checks/repair. Preserve the pre-shield proposal either way.
7. Score actual substeps independently and obtain fresh observations before replanning.
8. Save traces; final evaluation never updates adapter weights or approved skill semantics.

Simulation pauses during synchronous inference, so latency measures throughput rather than real-time safety in a moving world. All methods share physical emergency criteria and action cadence. The main model supports one protected region and at most two active requirements; multiple-target binding/check fixtures do not establish multi-obstacle learned routing.

## 17. Step 13 — Define fair baselines and memory controls

| ID | Method | Question |
|---|---|---|
| N0 | Original frozen policy | Nominal competence and unsafe headroom |
| F0 | Frozen policy + fixed checker/repair | Nonlearning teacher quality and cost |
| G0 | Flow adapter with structured predicate embeddings/parameters and identical geometry | Does language add value beyond symbolic conditioning? |
| L0 | Language-conditioned flow adapter, no semantic repair | **Primary learned capability** |
| L1 | Same trained L0 plus fixed checker/repair | Remaining fallback dependence and deployment behavior |

G0 replaces the text representation with learned predicate embeddings while matching projection width and remaining architecture/data/budget. Report parameter differences. If G0 wins, the evidence favors structured conditioning, with readable skills retained as the user interface.

Development controls: unconditioned adapter with identical geometry; language without explicit geometry; output-only residual adapter; prompt injection with the original policy; and wrong-language swaps under matched geometry. Do not let language methods receive richer scene facts accidentally. In the unconditioned control, replace all skill tokens by one fixed null token regardless of applicability and omit skill parameters, leaving only the identical scene vector and base features; otherwise token count could leak the number of active rules. Wrong-language interventions deliberately swap conditioning embeddings after normal validation, holding bindings/parameters and evaluator obligations fixed. Label this diagnostic corruption; do not pass an inconsistent skill file through ordinary admission and claim it was validated.

Fixed-weight memory diagnostics: restart, delete the relevant record, supply equivalent instruction directly each episode without persistent memory, add irrelevant records, and edit wording/parameters. Equivalent directly supplied tokens should give equivalent actions. The library can save authoring effort without improving numerical accuracy over identical context.

Evaluator requirements remain unchanged under deletion, wrong retrieval, and wrong-language tests. A skill's absence never deletes the obligation used to score it. Parameter-edit experiments are separate intended-policy conditions with their own evaluator parameters.

## 18. Step 14 — Run bounded ablations

Use only train/validate/select cells, identical teacher datasets and optimization budgets unless data/compute is the variable. Run sequentially, not as a full Cartesian sweep. Every extra training run gets three equal checkpoint-selection opportunities.

| ID | Comparison | Budget/decision |
|---|---|---|
| A1 | Language+geometry, structured+geometry, unconditioned+geometry | Required three runs, ≤10k updates each; determine which conditioning earns its complexity. |
| A2 | Language+geometry vs language only | One additional run; quantify benefit of explicit spatial grounding. |
| A3 | Flow residual vs output-only residual | One additional run; compare compliance, completion, discontinuity, and runtime. If output-only correction is within five points and cheaper, recommend it for the next implementation. Keep the named flow hypothesis explicit; changing the primary method requires a pre-final protocol revision. |
| A4 | Width 128 vs 256, both two blocks/four heads | One extra smaller-model run. Choose 128 if within five points and ≥20% faster; otherwise retain 256. |
| A5 | 10/20/40 trajectories per training cell | Nested episode-level subsets, two additional runs; tests learning from more experience. |
| A6 | Null penalty 0/0.1/1 | Default 0.1; at most two additional runs if none-context drift motivates them. |
| A7 | Correct/swapped/deleted/paraphrased language, same geometry | Replay plus 40-scenario selection panels; no final wording used for tuning. |
| A8 | Ten/twenty flow steps × prefix one/five | At most four inference-only selection configurations; match cadence in baselines. Reject prefixes beyond the shield's validated prediction horizon. Refresh N0/F0 references for each tested inference configuration. |
| A9 | Oracle/estimated geometry × oracle/estimated applicability | Optional independent grounding study with separate labels/manifests. |
| A10 | Single-skill data vs added composition demonstrations | Separate expanded-data model; its A+B result is supervised composition, not zero-shot composition. |

At most ten initial training configurations arise from A1–A6 if all optional branches run: three A1 + one A2 + one A3 + one A4 + two A5 + two A6. Reserve one additional seed-17 G0 run if selected width/null-penalty settings differ from its original run, so the final language/structured comparison stays matched. Add four runs for seeds 29/43 of selected L0/G0. L1 reuses L0 weights. This is up to **15 runs**, excluding explicitly optional aggregation/composition/LoRA extensions. All runs have at most 10,000 updates and 120 selection episodes, yielding at most 150,000 updates and 1,800 learned checkpoint-selection episodes. Shared N0/F0 references cost 80 episodes at the initial inference setting, up to 320 if all four A8 settings are tested. A7/A8 inference-only diagnostic runs and the overfit smoke test have their own ledger entries. Stop failed branches at their gates.

If learned language embeddings fail controlled semantic fixtures, consider a larger frozen encoder or action-expert adaptation under a new budget. Do not silently replace a failed small model or claim universal language ability from two learned instruction classes.

## 19. Step 15 — Lock and execute the final panel

Final geometry contains 80 physical snapshots: two tasks × four anchors × ten reset IDs. Four contexts make **320 scenarios per method**. Use the withheld midpoint parameters and locked final instruction wording. Report familiar cells, T2/A transfer, T1/A+B composition, T2/A+B composition, and none-context behavior separately.

N0 and F0 each run once: 640 episodes. G0, L0, and L1 run for three training seeds: `3 × 3 × 320 = 2,880`. Total **3,520 primary episodes**. A one-seed pilot uses `5 × 320 = 1,600`; it does not establish training-seed robustness. Reuse deterministic N0/F0 trajectories across paired analyses rather than pretending repeated copies are independent runs.

If commissioning requires five policy repetitions, primary totals become 17,600 or 8,000. Policy repetition and training seed are distinct axes. Forecast before scheduling.

For memory diagnostics, choose 20 preregistered physical snapshots, ten per task with anchor quotas 3/3/2/2, and none/A contexts: 40 episodes per diagnostic. Relevant-record deletion, equivalent direct instruction supply, and the second held-out paraphrase add **120 episodes** for seed 17. Primary plus diagnostics is **3,640 episodes**, or **1,720** for one training seed; five policy repetitions multiply the complete chosen panel, including diagnostics. Restart and irrelevant-record tests are replay fixtures.

Initial mechanism demo: ten fresh T1 development snapshots × none/A × N0/F0/L0/L1 = **80 episodes** after a trained checkpoint exists. This pre-lock demo shows conditioning, not task transfer. Opening T2/A or A+B before the lock would contaminate their held-out status; defer those demonstrations until final evaluation.

Balance method order with seeded hashes, reuse initial snapshots and corresponding policy-noise streams, and hash initial camera/state tensors. A policy service request has a 60-second timeout and one retry only for an identified transport/service error before action execution, preserving request ID/noise. Infrastructure episode failures may be rerun once; method-caused failures stay in the denominator. Preserve all attempts.

## 20. Step 16 — Metrics and analysis before results

Primary comparison is **raw L0 versus N0 on A-only scenes**, 80 paired physical resets per training seed. Language comparison is L0 versus G0 plus the matched-language interventions. Shield comparisons are L0/L1 and L1/F0, recording raw proposal rejection and replaced actions. Analyze B and A+B separately, without diluting them into many no-rule successes.

Required metrics: original task success; safe task success; semantic/physical violations and severity; grasp loss; abort/unknown/invalid-skill rates; raw predicted versus actual margins; adapter residual norm; repair fraction among all and transport actions; completion after handback; latency; inference calls; authoring/edit effort; training and total allocated cost per safe completion. A stop without violation is not completed work. Shadow base chunks and hypothetical-trajectory checks never influence raw L0 action selection. Log their instrumentation cost separately; measure serving latency with optional shadow diagnostics disabled, while charging all actual evaluation costs to the campaign.

Bootstrap 10,000 replicates with seed 20260918, resampling physical reset IDs within task/anchor while preserving methods, contexts, and training-seed records together. Average training seeds for the primary scene-level estimate; also show each seed. Report percentile 95% intervals, raw counts, and per-stratum outcomes. These intervals are conditional on the tested tasks/seeds, not a population guarantee from three training seeds. Degenerate intervals do not prove equivalence.

Prespecified pilot targets:

- L0 improves A-only safe success over N0 by ≥20 percentage points as a point estimate, with task success ≥50% and paired uncertainty shown.
- None-context task success loss is ≤10 points versus N0.
- T2/A and each A+B cell reach ≥50% safe success for their respective transfer claims; composition also needs meaningful A/B demands.
- Restart/deletion/edit fixtures pass. Correct-language tests must support semantic use; changed outputs alone are insufficient.
- L1 replaces at least 25% fewer transport actions than F0 while safe success differs by no worse than five points for an efficiency indication. Compute this from each method's replaced-action fraction over executed transport actions; also report raw action counts and task completion. If F0's replaced fraction is zero, the relative reduction is undefined. This is a pilot engineering gate, not a powered noninferiority claim.

Report unsuccessful or unresolved capabilities. Do not claim learned enforcement when only L1 succeeds, or a language advantage when structured conditioning performs equally. A full paper requires power/sample-size planning informed by pilot discordances and variance.

## 21. Step 17 — Acquire skills, test edits, and audit grounding

Start with reviewed reference files to separate learning from drafting errors. Then ask a pinned VLM to draft records from the same natural-language requirements and schema. Allow three generation attempts, followed by at most one human revision session of ten active minutes. Preserve prompts, responses, rejected instructions, latency/tokens, and human effort. No per-action VLM call is required.

Runtime loading of a newly drafted file occurs only after schema, instruction/contract agreement, supported-scope, and binding fixtures pass. A new wording or supported parameter is conditioning, not training. Autonomous admission of unreviewed safety knowledge needs its own study; execution of generated syntax alone is insufficient validation.

With weights fixed, test A clearance 0.01/0.02/0.04 m and B tilt 10/15/25 degrees. The midpoint values remain withheld from optimization/selection. Evaluate paraphrases, irrelevant facts, supported scope conjunctions, and numerical edits separately. The conjunction changes retrieval logic; do not attribute it to language understanding by the adapter.

For a bounded post-lock edit panel on seed 17, use ten preregistered final physical snapshots, five per task. Three A values in A-only contexts cost 30 episodes; three B values in B-only contexts cost 30. These **60 optional edit episodes** are additional to 3,640/1,720. No retraining or model selection follows them. Retain existing midpoint runs in the ledger and reuse them only when the exact episode manifest matches.

Audit 12 matched case pairs, two each for applicability, binding, units, excessive tolerance, unsupported predicate, and correct rule/controller failure. Present one version as skill+trace and another as raw correction+trace; 24 distinct cases per reviewer, maximum five minutes each. Counterbalance versions; score diagnosis, edit correctness against fixtures, and time. Author-only work is a walkthrough, not independent human-usability evidence.

## 22. Step 18 — PSC profiling and resource decision

Begin with one available H100, separate simulator/policy processes if dependencies require it, and no simultaneous drafting VLM on that GPU. Verify allocation access, JAX/CUDA support, rendering, and the actual base+adapter footprint. A two-hour profiling template is `GPU-shared`, `gpu:h100-80:1`, one node/task, eight CPU cores, 64 GB host memory. It is not an asserted training capacity or a submitted job.

Profile 200 training updates after compilation/warmup. Measure microbatch limits, update seconds, examples/second, device memory, host memory, and compilation overhead. Profile 20 development rollout episodes, five each for N0/F0/L0/L1. Simulator stalls count. The [PSC guide](https://www.psc.edu/resources/bridges-2/user-guide/) lists H100 at 2 SU per allocated GPU-hour and L40S at 1; verify project access/accounting before scheduling.

```text
training_hours = sum(seconds_per_update * updates_per_run) / 3600
rollout_hours  = sum(episodes_per_method / episodes_per_allocated_hour)
SU            = sum(allocated_GPUs * allocated_elapsed_hours * multiplier)
```

Forecast separately: screening up to 120 episodes; teacher engineering probes; collection up to 500 attempts; up to 15 training runs and 1,800 learned checkpoint-selection episodes plus 80–320 reference selection episodes; inference-only A7/A8 panels; the overfit test; the 80-episode mechanism demo; 3,640 primary/diagnostic episodes; optional 60 edit episodes; and any stochastic repetition multiplier. Add 25% planning headroom and keep startup/failure costs visible. These are caps/targets, not one mandatory sweep or a promise of a fixed SU total.

Report total cost per safe completion, not apparent savings from rapid failure. If serving is cheaper, compute break-even deployment count as incremental training/data cost divided by per-episode serving saving, in one consistent unit. If serving savings are nonpositive, no cost break-even exists.

Cache frozen instruction embeddings. Base action features depend on noisy action and flow time; one feature per observation is not an equivalent cache. Fixed `(observation,x_t,t)` training caches are an optional approximation whose restricted noise/time coverage must be disclosed. Do not claim that frozen weights eliminate forward inference cost.

## 23. Step 19 — Save reproducible artifacts and failures

Proposed layout:

```text
experiments/reusable_safety_skills/
  protocol_lock.json
  commissioning.json
  schemas/skill_v2.schema.json
  skills/{overhead_protection,orientation_protection}/SKILL.md
  src/{skill_loader,grounder,text_encoder,flow_adapter,policy_adapter}.py
  src/{teacher,evaluator,collect_dataset,train_adapter,run_episode,analyze}.py
  tests/
  manifests/{commission,develop,train,validate,select,final}.jsonl
  datasets/{episodes,windows,normalizers}/
  checkpoints/<training_seed>/<step>/
  records/{drafts,revisions,ablation_decisions}/
  runs/<method>/<training_seed>/<episode_id>/
```

Checkpoints contain adapter/optimizer state, base/encoder hashes, architecture, normalizers, schema, training-manifest hash, seed/update count, and selection evidence. No mutable final-test cache is embedded.

Episodes contain actual states/observations, skill and instruction hashes, grounded roles/provenance, base/adapted proposals, residuals, RNG state, predicted checks, executed actions and source `base/adapter/repair/stop`, actual substep events, goal/phase, latency, and failure reason. Save initial-state hashes and videos at 20 frames per simulated second. Atomic completed markers distinguish partial runs; preserve all attempts and deduplicate exact IDs in aggregation.

Separate outcome from termination reason. Outcomes: safe success, unsafe goal reached, unsafe without goal, task failure without observed violation, not executed. Reasons: goal settled, physical violation, grasp loss, invalid dynamics, validated stop, invalid skill, unknown context, no valid action, timeout, method exception, infrastructure failure. An abort without observed violation is not automatically a validated safe stop.

## 24. Milestones and stop conditions

| Milestone | Deliverable | Gate |
|---|---|---|
| 0. Commission | Provenance, base videos, evaluator/phase fixtures | Task competence and correct interface |
| 1. Establish teacher | Useful corrected trajectories, predictor/stop evidence | Valid labels and executable corrections |
| 2. Collect | 200 train + 50 validation accepted trajectories, full attempt ledger | Coverage and leakage checks; B nonvacuity assessed |
| 3. Integrate | Sampler identity, frozen-weight, gradient, overfit, reload tests | Correct learning path and measured resource use |
| 4. Train/select | Raw checkpoints and bounded ablations | Learned success and none-context retention |
| 5. Lock/test | Raw/shielded/structured comparisons and held-out tests | Claims independent of teacher rescue |
| 6. Faculty verdict | Counts, uncertainty, failure traces, actual cost | Decide next investment from the failed/successful layer |

Stop or revise if task competence fails, the teacher cannot produce sufficient useful targets, learned outputs ignore conditioning, all improvement requires external repair, or transfer fails. A negative language result may favor G0. Optional stronger encoders, LoRA, learned grounding, or richer primitives require a new stated hypothesis and budget rather than an unbounded search for a positive result.

## 25. Novelty and the simplicity/scaling argument

The [earlier research review](2026-09-17-reusable-safety-skills-review.md) remains historical literature analysis. DROC, MEMO, ARCHITECT, RoboSafe, and JIT-Memory already cover correction memory, executable safety logic, and procedural reuse. VLS already steers pretrained flow/diffusion policies. Learned command interfaces and adapters also have precedent. Neither Markdown nor language conditioning alone is the novelty.

The proposed contribution is a controlled study of **persistent, editable safety skills interpreted by a shared learned continuous-action interface**, with grounded inputs, fixed-weight edits, withheld pairings/compositions, strong symbolic controls, and evaluation before shield repair. Publication significance depends on results and an updated literature comparison.

Occam's razor motivates the small shared adapter, two primitives, simple file memory, and structured-only control. Count teacher code, asset annotations, human review, and training as complexity. If language adds no useful benefit, use structured conditioning while retaining readable records for users.

The [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) motivates testing whether more experience improves behavior through the same general learning interface. The data-size curve addresses that question. Hand-built evaluators still limit the vocabulary and teacher imitation may inherit planner errors. Additional handwritten exceptions are not evidence of scalable learning.

## 26. Completion checklist and faculty demonstration

Before training, register all source/model/encoder revisions, skill schema/paraphrase partitions, action mapping, evaluator/teacher evidence, task manifests, split hashes, normalizers, trainable parameter list, optimizer, acquisition effort, and allocation cap. Before final evaluation, register selected checkpoints for every seed, untouched final geometry/wording, interpolation values, method information access, analysis code, exclusion rules, and measured budget forecast.

The five-minute demonstration allocates 30 seconds to task/skill/grounding; 60 to paired N0/raw L0/L1 behavior; 60 to fixed-weight edits and restart; 45 to held-out pairing/composition; 45 to pre-shield violations and a failure; and 60 to aggregate counts, repair dependence, and cost. Choose the lowest episode ID showing each required behavior plus a representative failure, not visually attractive clips. If a capability fails, show that failure.

The document is an implementation and experimental specification. Adapter code, training, teacher collection, hardware measurements, and evaluation remain to be performed. The archived training-free protocol is an engineering reference for the teacher/evaluator, not the main research claim.
