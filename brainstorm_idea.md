# Context-Aware Viability Flow Safety Filter

## Primary Architecture Without Continual RL

**Status:** Research and implementation design  
**Working name:** Context-Aware Viability Flow (CAVF)  
**Intended venue:** ICRA  
**Initial backbones:** OpenVLA and pi0.5  
**Initial benchmark:** SafeLIBERO Spatial Levels I and II  

## 1. Executive summary

CAVF is an external, context-aware safety layer for vision-language-action
(VLA) and world-action models. A frozen base policy proposes a nominal action
chunk. CAVF then:

1. Builds an object-centric representation of the scene, instruction, task
   phase, recent history, and uncertainty.
2. Compiles semantic facts and safety rules into quantitative constraints.
3. Determines whether the nominal action is safe, useful, and recoverable.
4. When necessary, generates several task-preserving trajectory repairs with a
   residual rectified-flow model.
5. Predicts the consequences of the nominal and repaired trajectories.
6. Rejects candidates that violate hard physical or semantic constraints.
7. Selects the candidate with the highest probability of safe task completion.
8. Executes only a short prefix and repeats the process with new observations.

The base VLA remains frozen. The primary architecture contains no continual
learning, online reinforcement learning, or policy-gradient update. All learned
safety components are trained offline and frozen during benchmark evaluation.

The system is designed around the following objective:

> Maximize the probability of completing the requested task without a
> contextual safety violation.

This differs from collision-only filtering. A collision-free trajectory can
still be unsafe because it contacts the wrong object, mishandles a fragile or
hot object, spills liquid, enters a dead end, interferes with a human, or
completes an unsafe task.

## 2. Scope

### 2.1 In scope

- Static, offline-trained contextual safety filtering.
- Frozen OpenVLA, pi0.5, and future VLA/WAM backbones.
- Nominal action acceptance, repair, redirection, recovery, and refusal.
- Geometric, semantic, human-interaction, mechanical, thermal, and fluid risk.
- Full-arm, gripper, and carried-object trajectory verification.
- Multimodal task-preserving repairs such as left, right, above, retreat, and
  reorientation.
- Receding-horizon execution and route consistency across action chunks.
- FOL/STL-based specification and verification.
- A learned safe-success viability critic.
- A residual rectified-flow action-chunk repair model.

### 2.2 Explicitly out of scope for the primary architecture

- Continual or lifelong learning.
- Online reinforcement learning.
- Updating the model from benchmark evaluation episodes.
- Fine-tuning the base VLA.
- Training a large video world model.
- End-to-end foundation-model pretraining.
- Unconditional formal guarantees under incorrect perception.
- Embodiment-free control without an action and kinematics adapter.

Continual safety learning can be added later as an outer loop, but it is not
required to implement or evaluate this primary architecture.

## 3. Design principles

1. **Preserve task competence.** Repair the VLA trajectory rather than replace
   the VLA with an independently trained task policy.
2. **Optimize safe success.** Avoiding a collision while causing task failure,
   deadlock, or timeout is not a successful intervention.
3. **Separate learned guidance from hard enforcement.** Learned models generate
   and rank candidates; hard constraints make the final admissibility decision.
4. **Reason about context, not only geometry.** The same motion can be safe or
   unsafe depending on object roles, task phase, material, contents, humans, and
   instruction semantics.
5. **Generate multiple routes.** A single minimum-norm correction is often
   locally safe but globally unhelpful.
6. **Verify the complete manipulated system.** The robot arm, gripper, payload,
   controller tracking error, and predicted dynamic obstacles must be included.
7. **Remain backbone-agnostic.** The primary interface is observations,
   language, robot state, and an action chunk, not model-specific hidden states.
8. **Fail safely under uncertainty.** Reobserve, slow, retract, or stop when the
   system cannot establish a sufficiently safe and useful action.

## 4. Runtime system diagram

~~~text
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
│ Camera observations, robot state, instruction, recent       │
│ history, previous route mode, and controller state          │
└────────────────────────────┬────────────────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│ CONTEXT INTERPRETER      │   │ FROZEN VLA OR WAM           │
│                          │   │ OpenVLA, pi0.5, or another   │
│ Detection and tracking   │   │ compatible foundation policy│
│ Cross-attention          │   └──────────────┬───────────────┘
│ Task-phase estimation    │                  │
│ Object role assignment   │                  ▼
│ Material/risk attributes │   ┌──────────────────────────────┐
│ Perception uncertainty   │   │ BACKBONE ACTION ADAPTER      │
└─────────────┬────────────┘   │ Convert native policy output │
              │                │ into a canonical action chunk│
              ▼                └──────────────┬───────────────┘
┌──────────────────────────┐                  │
│ OBJECT-CENTRIC CONTEXT   │                  │ Nominal A0
│ GRAPH                    │                  │
│                          │                  │
│ Target and destination   │                  │
│ Obstacles and humans     │                  │
│ Allowed contacts         │                  │
│ Held object and contents │                  │
│ Fragility/heat/liquid    │                  │
│ Task phase and history   │                  │
└─────────────┬────────────┘                  │
              │                               │
              ▼                               │
┌──────────────────────────┐                  │
│ CONSTRAINT COMPILER      │                  │
│                          │                  │
│ FOL facts and rules      │                  │
│        ↓                 │                  │
│ Numeric STL constraints, │                  │
│ costs, masks, and margins│                  │
└─────────────┬────────────┘                  │
              └───────────────┬───────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ NOMINAL VIABILITY CHECK │
                 │                          │
                 │ Is A0 physically safe?   │
                 │ Is it semantically valid?│
                 │ Does it make progress?   │
                 │ Is the future recoverable│
                 │ and sufficiently certain?│
                 └────────────┬─────────────┘
                              │
          ┌───────────────────┼─────────────────────┐
          │                   │                     │
          ▼                   ▼                     ▼
   Safe and useful     Unsafe but repairable   Forbidden, highly
          │                   │                uncertain, or no
          │                   │                viable completion
          │                   ▼                     │
          │       ┌──────────────────────────┐      │
          │       │ RESIDUAL RECTIFIED FLOW  │      │
          │       │                          │      │
          │       │ Generate K corrections:  │      │
          │       │ left, right, above,       │      │
          │       │ retreat, reorient, etc.   │      │
          │       └────────────┬─────────────┘      │
          │                    │                    │
          │                    ▼                    │
          │       ┌──────────────────────────┐      │
          │       │ HYBRID PREDICTIVE ROLLOUT│      │
          │       │                          │      │
          │       │ Robot forward kinematics │      │
          │       │ Payload motion           │      │
          │       │ Tracked dynamic objects  │      │
          │       │ Short-horizon interaction│      │
          │       └────────────┬─────────────┘      │
          │                    │                    │
          │                    ▼                    │
          │       ┌──────────────────────────┐      │
          │       │ HARD VERIFIER            │      │
          │       │                          │      │
          │       │ Full-body swept volume   │      │
          │       │ Human and obstacle margin│      │
          │       │ Joint/workspace limits   │      │
          │       │ Allowed contact rules    │      │
          │       │ Force/orientation limits │      │
          │       │ Mechanical/thermal/fluid │      │
          │       │ safety conditions        │      │
          │       └────────────┬─────────────┘      │
          │                    │                    │
          │                    ▼                    │
          │       ┌──────────────────────────┐      │
          │       │ VIABILITY RANKER         │      │
          │       │ Select the hard-feasible │      │
          │       │ candidate most likely to │      │
          │       │ achieve safe task success│      │
          │       └────────────┬─────────────┘      │
          │                    │                    │
          ▼                    ▼                    ▼
      Execute A0       Execute best repair    Reobserve, slow,
                                             retract, redirect,
                                             stop, or refuse
          │                    │                    │
          └────────────────────┴────────────────────┘
                               ▼
                    Execute a short prefix
                               │
                               ▼
                    Observe and repeat the loop
~~~

## 5. Runtime decision logic

The primary router is deterministic and auditable. It uses the hard verifier,
critic scores, and calibrated uncertainty rather than an online RL policy.

~~~text
IF the task or requested outcome violates a hard semantic rule:
    REFUSE or REDIRECT

ELSE IF perception or prediction uncertainty exceeds the allowed threshold:
    REOBSERVE, SLOW, or STOP

ELSE IF the nominal action passes hard verification
        AND its safe-success score exceeds the execution threshold:
    EXECUTE the nominal action

ELSE:
    GENERATE multiple residual-flow repairs
    VERIFY every candidate

    IF one or more candidates are hard-feasible:
        EXECUTE the candidate with the best safe-success score
    ELSE IF a certified recovery action is available:
        RETRACT or RECOVER
    ELSE:
        STOP or REFUSE
~~~

Only a short prefix of the selected chunk is executed. The complete pipeline is
recomputed after receiving the next observation.

## 6. Component specification

### 6.1 Backbone interface and action adapter

Each supported VLA or WAM implements a small adapter:

~~~text
policy observations + instruction
        → native policy action chunk
        → canonical action chunk A0
~~~

The canonical action representation contains:

- A fixed action horizon and control frequency.
- Cartesian or joint-space motion commands.
- Orientation commands.
- Gripper commands.
- Validity masks.
- Timing and embodiment metadata.
- Normalization statistics that remain fixed during training and evaluation.

If the policy produces Cartesian end-effector commands, the verification layer
expands them into full-arm trajectories using the embodiment kinematics and
inverse-kinematics/controller model.

The primary safety model must not require access to internal VLA attention.
Model attention can be evaluated as an optional perception feature, but it
cannot be required for backbone compatibility.

### 6.2 Context interpreter

The context interpreter combines:

- Current camera observations.
- A short visual and action history.
- Robot proprioception.
- Language instruction.
- Object detections, masks, and tracks.
- Optional depth and simulator state during supervised training.

Cross-attention identifies which scene entities are relevant to each part of
the instruction. Attention is used for grounding and role assignment; it is not
treated as a safety controller.

The interpreter estimates:

- Current task phase.
- Manipulation target.
- Destination or goal region.
- Allowed and prohibited contacts.
- Humans and dynamically moving objects.
- The held object and payload geometry.
- Fragility, temperature, material, and liquid contents when observable.
- Per-entity and global uncertainty.

### 6.3 Object-centric context graph

A conceptual graph record is:

~~~text
ContextGraph
├── robot
│   ├── joint state and link geometry
│   ├── end-effector and gripper state
│   └── controller tracking uncertainty
├── task
│   ├── instruction
│   ├── phase and completed subgoals
│   └── target and destination
├── entities[]
│   ├── identity/class and visual features
│   ├── pose, geometry, velocity, and uncertainty
│   ├── role: target/destination/obstacle/human/payload
│   ├── contact permissions
│   └── material, fragility, temperature, and contents
└── history
    ├── recent observations and actions
    ├── previous intervention
    └── persistent route mode
~~~

The graph prevents a purely geometric filter from treating the target,
destination, unrelated objects, and humans as equivalent obstacles.

### 6.4 Semantic and temporal constraint compiler

FOL remains part of the architecture as a semantic specification language. It
does not directly modify continuous robot actions.

Example facts:

~~~text
target(cup)
destination(shelf)
fragile(cup)
contains_liquid(cup)
human(person)
near(person, robot)
allowed_contact(gripper, cup)
forbidden_contact(arm, person)
~~~

These facts compile into quantitative constraints such as:

- Maintain a minimum human distance.
- Permit gripper contact with the target.
- Prohibit arm contact with the human and unrelated fragile objects.
- Maintain the payload within an upright orientation interval.
- Limit acceleration, impact, and grasp force.
- Avoid thermal exposure above a threshold.
- Prevent a liquid container from exceeding its spill angle.
- Require continued progress or a recoverable state within a time horizon.

STL-style robustness values provide continuous margins that can condition the
flow generator and critic. Boolean hard violations remain final vetoes.

### 6.5 Safe-success viability critic

For a context x and candidate action chunk A, the critic predicts:

- Probability of safe task success.
- Probability and severity of collision.
- Probability of semantic violation.
- Human-interaction risk.
- Mechanical, thermal, and fluid risk.
- Expected task progress.
- Deadlock or timeout risk.
- Recoverability after the chunk.
- Epistemic or predictive uncertainty.

The critic explicitly distinguishes:

1. Safe task success.
2. Unsafe task success.
3. Safe failure or deadlock.
4. Unsafe failure.

This distinction prevents the system from equating stopping forever with good
safety or equating successful completion with safe behavior.

The nominal trajectory is scored first. If it is hard-feasible and has a
sufficient safe-success score, it passes through unchanged.

### 6.6 Residual rectified-flow repair generator

The repair generator learns a distribution over corrections rather than
regenerating the entire task trajectory:

~~~text
Ak = A0 + delta_Ak
~~~

where delta_Ak is conditioned on:

- Context graph and task phase.
- Nominal action chunk A0.
- Compiled constraint embedding.
- Current payload and robot state.
- A route-mode latent.
- Previous route mode and intervention history.

The route latent represents different solution modes such as:

- Pass left.
- Pass right.
- Move above.
- Retreat before approaching.
- Change wrist or payload orientation.
- Slow down while preserving the path.

The route state persists across control cycles. This reduces left/right
oscillation and inconsistent chunk-to-chunk corrections.

#### Rectified-flow training target

For a paired nominal and repaired trajectory:

~~~text
target residual = A_safe - A0
noise z ~ Normal(0, I)
interpolated residual at time t = (1 - t) z + t target_residual
target velocity = target_residual - z
~~~

The model learns the vector field that transports noise to the safe,
task-preserving residual. At inference, several route-conditioned residuals are
sampled using a small number of integration steps.

Useful auxiliary losses include:

- Action smoothness.
- Minimal unnecessary deviation from A0.
- Endpoint and subgoal consistency.
- Route-mode consistency.
- Gripper and orientation consistency.
- Penalties for candidates rejected by differentiable soft constraints.

The hard verifier remains responsible for final enforcement.

### 6.7 Hybrid predictive rollout

The initial system should not train or run a large video world model. It uses a
hybrid short-horizon predictor:

1. Robot link motion from forward kinematics and the controller model.
2. Payload geometry rigidly or compliantly attached to the gripper.
3. Constant-velocity or tracked predictions for dynamic obstacles and humans.
4. Physics simulation when privileged simulation is available.
5. An optional compact learned residual model for contact and object motion.

The predictor must return trajectories and uncertainty for:

- Every robot link.
- Gripper fingers.
- Held objects.
- Relevant scene objects.
- Humans or dynamic obstacles.

Prediction uncertainty expands safety envelopes rather than being ignored.

### 6.8 Hard verifier

The verifier masks any candidate violating non-negotiable constraints. It
checks:

- Full-arm and gripper swept-volume collision.
- Payload swept volume.
- Permitted versus prohibited contacts.
- Human separation.
- Joint position, velocity, and acceleration limits.
- Workspace and self-collision limits.
- Controller tracking-error tubes.
- Payload orientation.
- Force, impact, compression, or tension limits when available.
- Thermal exposure.
- Liquid spill and fluid-damage conditions.
- Task-specific semantic invariants.

The verifier provides conditional assurance:

> Given the estimated scene state, bounded prediction error, embodiment model,
> and controller assumptions, the selected trajectory satisfies the encoded
> constraints over the checked horizon.

It must not be described as an unconditional real-world guarantee when
perception, material properties, or human motion are uncertain.

### 6.9 Candidate ranking and selection

Let C contain the nominal action and K repaired candidates. Hard verification
defines:

~~~text
C_feasible = {A in C such that every hard constraint is satisfied}
~~~

The selected action is:

~~~text
A_selected = argmax over A in C_feasible of
             safe_success_score(A)
             - uncertainty_penalty(A)
             - unnecessary_intervention_penalty(A)
~~~

The unnecessary-intervention penalty favors the nominal policy when two
actions are equally safe and useful. It does not allow a hard violation.

### 6.10 Safe fallback controller

If no task-progressing candidate is feasible, the system must not force a
repair. Its fallback hierarchy is:

1. Reobserve from the current pose.
2. Reduce speed or shorten the execution horizon.
3. Open or close the gripper only if verified safe.
4. Retract along a verified recent path.
5. Move to a predefined safe holding configuration.
6. Stop.
7. Refuse or request assistance when the task itself is unsafe or ambiguous.

## 7. Formal problem statement

At control cycle t, define:

- x_t: observation, instruction, robot state, context graph, and history.
- pi_b: frozen base VLA/WAM policy.
- A0 = pi_b(x_t): nominal action chunk.
- G_theta: residual rectified-flow generator.
- V_psi(x_t, A): safe-success viability critic.
- H(x_t, A): hard verifier.

The generator produces:

~~~text
Ak = A0 + G_theta(x_t, A0, route_k, noise_k)
~~~

for k from 1 to K. The feasible candidate set is:

~~~text
F_t = {A0, A1, ..., AK} intersect {A such that H(x_t, A) = true}
~~~

If F_t is nonempty, select:

~~~text
A* = argmax over A in F_t of V_psi(x_t, A)
~~~

where V_psi represents calibrated safe-task-success utility with uncertainty
and intervention regularization. If F_t is empty, invoke the safe fallback
controller.

## 8. Offline training-data pipeline

~~~text
Safe demonstrations
        +
Frozen VLA nominal rollouts
        +
Unsafe/failing action chunks
        +
Planner, MPC, or optimizer repairs from the same state
        +
Semantic safe/unsafe twins
        +
DamageSim mechanical/thermal/fluid outcomes
        │
        ▼
Unified paired trajectory records
        │
        ├── Train context and risk labels
        ├── Train safe-success viability critic
        ├── Train residual rectified-flow repairs
        ├── Train pairwise candidate ranking
        └── Calibrate risk and uncertainty thresholds
        │
        ▼
Freeze all safety-model parameters
        │
        ▼
Run benchmark evaluation with no updates
~~~

### 8.1 Data sources and roles

| Source | Primary role |
|---|---|
| LIBERO and LIBERO-Safety demonstrations | Safe task behavior and contextual positive examples |
| Frozen OpenVLA/pi0.5 rollouts | Realistic nominal failures, deadlocks, and unsafe actions |
| SafeLIBERO training variants | Collision-risk contexts and trajectory-repair generation |
| Planner/MPC/trajectory optimizer | Task-preserving corrected trajectories from matched states |
| HazardArena-style twins | Semantic execute/refuse and context contrast |
| OopsieVerse and DamageSim | Mechanical, thermal, fluid, and health-degradation labels |

SafeLIBERO official evaluation initialization states must not be used to train
the filter, repair generator, critic, thresholds, or route selector.

### 8.2 Required unified record

~~~text
SafetyRecord
├── observations and recent history
├── language instruction
├── robot and controller state
├── context graph and uncertainty
├── base-policy identifier
├── nominal action chunk A0
├── repaired action chunk A_safe, when available
├── route-mode label
├── compiled constraints and margins
├── risk vector
│   ├── collision
│   ├── semantic
│   ├── human interaction
│   ├── mechanical
│   ├── thermal
│   └── fluid
├── task progress and recoverability
└── final outcome
~~~

The essential training example is a same-state pair:

~~~text
(context, unsafe or failing nominal action, safe task-preserving repair)
~~~

Safe demonstrations alone do not provide this supervision.

## 9. Offline training stages

### Stage A: Context and critic training

- Train object roles, task phase, and risk attributes.
- Train multi-risk outcome heads.
- Train task-progress, deadlock, and recoverability heads.
- Train pairwise ranking so safe successful candidates outrank unsafe or
  unsuccessful candidates.
- Calibrate probabilities on held-out scenes and tasks.

### Stage B: Residual-flow training

- Construct matched nominal/repaired action chunks.
- Cluster or label repair routes.
- Train conditional residual rectified flow.
- Add smoothness, minimal-deviation, and route-consistency losses.
- Train with multiple frozen VLA sources so the corrector does not overfit one
  policy's characteristic errors.

### Stage C: Frozen-system integration

- Generate multiple candidates for every repair-triggering state.
- Apply the actual hard verifier.
- Train or tune the ranker using only training and validation scenarios.
- Freeze weights and thresholds before benchmark evaluation.

No stage uses policy gradients, online reward optimization, or continual model
updates.

## 10. Backbone independence

CAVF is independent of the policy architecture when all of the following are
available:

- A compatible observation and instruction interface.
- A canonical action adapter.
- Robot kinematics and collision geometry.
- A controller model or conservative tracking-error bound.
- Payload attachment and geometry estimates.

The same filter weights should be evaluated with OpenVLA and pi0.5. Small
embodiment adapters may differ, but the context encoder, critic, repair
generator, and semantic compiler should be shared.

For pi0.5, internal flow-guidance can be evaluated as a white-box ablation.
The main result must use the external action-chunk interface so that OpenVLA
and non-flow policies remain supported.

## 11. Safety hierarchy

The architecture uses four layers:

| Layer | Responsibility | Learned? |
|---|---|---|
| Context interpreter | Understand objects, roles, task phase, hazards, and uncertainty | Partly |
| Viability critic | Predict safe task completion and rank candidates | Yes, offline |
| Rectified-flow repair | Generate task-preserving alternative trajectories | Yes, offline |
| Hard verifier and fallback | Enforce encoded constraints and stop when necessary | No online learning |

The critic may be wrong. A high critic score cannot override a hard verifier
failure.

The verifier may be incomplete because perception or specifications are
incomplete. Uncertainty handling and explicit limitations are therefore part
of the system, not optional presentation details.

## 12. Initial SafeLIBERO pilot

### 12.1 Evaluation scope

- SafeLIBERO Spatial Level I.
- SafeLIBERO Spatial Level II.
- All four tasks per level.
- All 50 official initialization states per task.
- OpenVLA and pi0.5.
- Existing unfiltered baseline results are reused.
- Only filtered evaluations are run.

This corresponds to 400 filtered episodes per backbone and 800 filtered
episodes in total.

### 12.2 Repository baselines

The currently recorded baselines are:

| Backbone | Level | TSR | CAR |
|---|---:|---:|---:|
| OpenVLA | I | 42.0% | 12.0% |
| OpenVLA | II | 35.0% | 11.5% |
| pi0.5 | I | 67.0% | 14.0% |
| pi0.5 | II | 55.5% | 12.0% |

These values should be verified against the committed result artifacts before
being used in a paper table.

### 12.3 Published SafeLIBERO targets

The constrained-flow paper reports the following relevant pi0.5 Spatial
results:

| Method | Level | TSR | CAR |
|---|---:|---:|---:|
| Constrained flow guidance | I | 75.5% | 77.5% |
| Constrained flow guidance | II | 78.0% | 75.5% |
| Best published value among reported methods | I | 75.5% | 77.5% |
| Best published value among reported methods | II | 81.0% | 77.5% |

The first pilot is a technical validation against the corresponding repository
baseline. The eventual SafeLIBERO claim requires comparison against the best
published value for each metric and level under a matched protocol.

### 12.4 Pilot go/no-go criteria

The architecture continues beyond the pilot only if:

1. Filtered TSR exceeds the corresponding unfiltered TSR.
2. Filtered CAR exceeds the corresponding unfiltered CAR.
3. No individual task suffers a severe collapse hidden by the aggregate.
4. Improvements hold on paired initialization states with uncertainty
   intervals or a paired bootstrap analysis.
5. Intervention rate and latency are operationally reasonable.
6. Qualitative inspection confirms task-preserving rerouting rather than
   accidental stopping or timeout exploitation.

An architecture that improves CAR by stopping more often while reducing TSR is
not sufficient.

## 13. Full benchmark evaluation

### 13.1 SafeLIBERO

Tests:

- Predictive collision avoidance.
- Long-horizon trajectory repair.
- Task-preserving detours.
- Full-arm and payload-aware verification.

Primary metrics:

- Task Success Rate (TSR).
- Collision Avoidance Rate (CAR).
- Safe task success.
- Intervention rate.
- Mean correction magnitude.
- Deadlock and timeout rate.
- Runtime latency.

### 13.2 LIBERO-Safety

Tests:

- Affordance-aware manipulation.
- Human-robot interaction.
- Static and dynamic obstacle avoidance.
- Semantic safety reasoning.
- Deadlock and temporal-overflow behavior.

Additional metrics:

- Semantic violation rate.
- Human safety violations.
- False refusal and false intervention.
- Phase and target-role accuracy.

### 13.3 OopsieVerse

Tests:

- Mechanical damage.
- Thermal damage.
- Fluid damage.
- Safe task completion rather than collision alone.

Additional metrics:

- Damage-free success.
- Health loss.
- Damage severity by type.
- Undetected damage rate.

## 14. Required ablations

The paper should isolate the contribution of each component:

1. Frozen base VLA without a filter.
2. Existing local/projective filter.
3. Hard verifier without learned repair.
4. Residual flow without a viability critic.
5. Viability critic with a single optimizer repair.
6. Single-route versus multimodal route generation.
7. No route memory versus persistent route state.
8. End-effector-only versus full-arm/gripper/payload verification.
9. Geometry-only versus context graph and semantic compiler.
10. Collision-only critic versus safe-success multi-risk critic.
11. Rectified flow versus diffusion repair at matched parameter and sampling
    budgets.
12. Oracle simulator state versus learned perception.
13. Single-backbone training versus mixed OpenVLA/pi0.5 training.

## 15. Key failure modes and mitigations

| Failure mode | Required mitigation |
|---|---|
| Local detour causes task failure | Safe-success critic and task-progress prediction |
| Left/right oscillation | Persistent route latent and hysteresis |
| End effector clears but arm collides | Full-link swept-volume verification |
| Held object collides or spills | Payload geometry and orientation constraints |
| Target treated as obstacle | Contextual role and allowed-contact graph |
| Unsafe task completed successfully | Semantic refusal or redirection |
| All repairs are invalid | Verified recovery or stop |
| Perception is uncertain | Expanded margins, reobservation, slow mode, or stop |
| Critic exploits stopping | Separate safe failure from safe success |
| Repair leaves policy distribution | Residual generation and minimal-deviation loss |
| Cross-backbone mismatch | Canonical action interface and mixed-policy training |
| Long inference latency | Nominal fast path, compact predictor, few flow steps |

## 16. Training and compute assumption

The intended model scale is moderate:

| Component | Approximate trainable scale |
|---|---:|
| Context fusion and task-phase model | 50M–100M |
| Viability critic and risk heads | 30M–80M |
| Residual rectified-flow corrector | 80M–200M |
| Total | 160M–380M |

Large visual encoders may be frozen. Nominal VLA actions and visual embeddings
should be cached where possible.

The complete pilot fits on one H100-80GB. Two to four H100s are preferable for
faster candidate generation, ablations, and multiple seeds. A cached-feature
configuration can be trained on V100-32GB GPUs with smaller batches.

## 17. Implementation sequence

This document defines the intended architecture; it does not authorize code
changes by itself. The recommended implementation order is:

1. Define the canonical observation, action, context-graph, constraint, and
   candidate interfaces.
2. Implement full-arm, gripper, and payload rollout verification.
3. Build matched nominal/repaired SafeLIBERO training records on disjoint
   training scenarios.
4. Train and calibrate the safe-success viability critic.
5. Implement the deterministic execute/repair/refuse router.
6. Train the multimodal residual rectified-flow generator.
7. Add route memory and receding-horizon integration.
8. Integrate OpenVLA and pi0.5 through separate action adapters.
9. Run unit, replay, and counterexample tests.
10. Run the frozen SafeLIBERO Spatial Level I and II pilot.
11. Continue to LIBERO-Safety and OopsieVerse only after passing the pilot gate.

## 18. Research hypotheses

The primary experiments should test:

1. A critic that predicts safe task completion will preserve TSR better than a
   collision-only or minimum-norm safety projector.
2. Multimodal residual-flow repairs will find useful routes that a single local
   correction cannot.
3. Full-arm and payload verification will reduce violations missed by
   end-effector-only methods.
4. An explicit context graph and semantic compiler will transfer safety
   behavior across geometric, semantic, and damage benchmarks.
5. A shared external safety model can improve more than one frozen VLA without
   model-specific retraining.
6. FOL/STL is most effective as a specification and verification layer, not as
   the continuous action generator.

## 19. Intended contribution

The intended contribution is not merely another collision-avoidance filter.
It is a VLA-agnostic safety architecture that combines:

- Object-centric contextual reasoning.
- Symbolic and temporal safety specifications.
- Multimodal residual rectified-flow trajectory repair.
- Safe-task-success viability prediction.
- Full-system physical and damage verification.
- Execute, repair, recover, redirect, and refuse behavior.

The central empirical claim must be demonstrated rather than assumed:

> CAVF improves both task success and contextual safety over frozen base
> policies and existing safety filters across SafeLIBERO, LIBERO-Safety, and
> OopsieVerse.

## 20. Starting references

- Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via
  Constrained Flow Matching: https://arxiv.org/abs/2607.01378
- SafeDojo: https://arxiv.org/abs/2606.20698
- Your Model Already Knows: Attention-Guided Safety Filter for VLA Models:
  https://arxiv.org/abs/2606.09749
- LIBERO-Safety: https://arxiv.org/abs/2606.23686
- OopsieVerse: https://arxiv.org/abs/2606.31993
- Safe Learning in Robotics:
  https://arxiv.org/abs/2108.06266

