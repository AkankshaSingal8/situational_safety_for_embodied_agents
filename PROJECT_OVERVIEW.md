# Research Project Brief: Situational Safety for Embodied Agents (Manipulation)

**Target venue:** CoRL 2026
**Scope:** Tabletop / kitchen-style manipulation with robotic arms. Simulation-first; hardware experiments deferred to a later stage of the project.
**Base stack:** OpenVLA and π0.5 VLAs (LIBERO-trained), SafeLIBERO as the safety benchmark, MuJoCo for rollouts, external VLMs (GPT-family, Qwen-family) for semantic predicate generation, CBF-QP as the safety filter.

This document is the canonical project context. It describes (1) what the project is, (2) what the safety taxonomy looks like, and (3) exactly how the safety pipeline is structured so that implementation choices stay aligned with the research contributions.

---

## 1. Problem Statement and Contributions

Modern generalist VLAs (OpenVLA, π0.5, etc.) map multi-view RGB observations and a natural-language instruction to a low-level action, but they are trained to *complete* tasks, not to *remain safe while completing them*. "Safety" in existing manipulation benchmarks is also narrow — typically just end-effector collision with obstacles. Real deployment-time safety is a much richer, **situational** property that depends on geometry, physical state, semantics, task context, interaction with humans, and the agent's own meta-awareness of its limits.

The project makes two contributions:

1. **A six-level safety taxonomy** for embodied manipulation that explicitly separates geometric, state-based, semantic, task-contextual, interactive, and meta-level safety concerns, with sub-categories at each level.
2. **A generalist, VLM-grounded safety filter** that sits on top of a pretrained VLA and is designed to address violations at every level of the taxonomy. It uses a VLM to propose *open-vocabulary semantic predicates* from observations, grounds those predicates into parametric spatial/kinematic constraints, assembles them into a control barrier function (CBF), and filters the VLA's nominal action through a CBF-QP to produce a safe action.

The combination is the research claim: a *single, predicate-driven* pipeline can identify and enforce safety violations across all six levels of the taxonomy without per-task hand-engineering.

---

## 2. Safety Taxonomy

The taxonomy is hierarchical. Higher levels subsume lower ones in the sense that a scene can violate multiple levels simultaneously, and the pipeline must be able to surface and constrain each.

- **Level 1 — Geometric / Collision Safety.** Hard physical-contact constraints. Sub-categories:
  - *Robot–Obstacle Collision:* prevent contact between any robot link and static or dynamic obstacles.
  - *Robot–Human Collision:* maintain minimum separation distances and/or limit contact forces.
  - *Self-Collision:* prevent contact between different links of the robot's own kinematic chain.
  - *Joint / Velocity / Torque Limits:* respect actuator bounds, max joint velocities, and force/torque constraints.
  - *Workspace Boundaries:* confine end-effector motion to a defined operational volume.
- **Level 2 — Process / State Safety.** (Sub-categories TBD.) Safety of the *physical state* of manipulated objects and the world: spills, tipping, dropping, breaking, thermal/liquid/sharp-state hazards.
- **Level 3 — Semantic Safety.** (Sub-categories as drafted.)
  - *3a: Spatial-Relational* — relationships like "above," "behind," "inside," "clear of."
  - *3b: Behavioral* — *how* an action is executed (gentle vs. forceful, slow vs. fast near fragile items).
  - *3c: Pose / Orientation* — carrying a cup upright, not rotating a knife toward a person.
- **Level 4 — Task-Contextual Safety.** (Sub-categories TBD.) Constraints that are only safety-relevant given the instruction or scene, e.g. "don't pour near the laptop," "don't stack anything on the plated food."
- **Level 5 — Interactive / Relational Safety.** (Sub-categories TBD.) Safety in the presence of humans or other agents: intent inference, handover etiquette, social-spatial norms.
- **Level 6 — Meta-Safety.** (Sub-categories TBD.) The agent's awareness of *its own* safety state: recognizing infeasibility, constraint conflicts, degraded perception, out-of-distribution scenes, and communicating these upstream.

Sub-categories at Levels 2–6 are still to be finalized; the schema should be extensible so new sub-categories can be added without breaking the predicate/constraint representation.

---

## 3. Method: VLM-Grounded Predicate CBF-QP Safety Filter

### 3.1 Pipeline architecture (simulation)

At each control step (or each action-chunk boundary, configurable):

1. **Perception.** Collect multi-view RGB observations from the MuJoCo cameras and the natural-language task instruction.
2. **Nominal policy.** Pass observations + instruction to the VLA (OpenVLA or π0.5) to obtain a nominal action (or action chunk) **a_nom**.
3. **Predicate proposal (VLM).** Pass the same observations + instruction to a VLM (GPT-family, Qwen-family) under a multi-prompt strategy. The VLM returns a structured set of *semantic predicates* describing what must hold for the agent to be safe in the current situation, along with a short natural-language explanation for each.
4. **Predicate grounding.** Convert each predicate into a parametric constraint in the robot's state space:
   - Spatial predicates (e.g. `above(X, Y)`, `around(X, r)`, `below(X, Y)`, `inside(region)`) → superquadric / ellipsoid / half-space regions in (x, y, z).
   - Kinematic predicates (e.g. `slow_near(X)`, `no_fast_rotation_of(X)`) → bounds on Cartesian / joint velocities or angular rates.
   - Orientation predicates (e.g. `upright(cup)`) → constraints on end-effector / held-object rotation.
   - STL-structured predicates (optional alternative form) for temporal requirements such as "remain above the table surface until the grasp is released."
5. **CBF assembly and QP filter.** Treat each grounded constraint as a barrier h_i(x) ≥ 0 and assemble a CBF-QP of the form

   ```
   a* = argmin_a  ‖a − a_nom‖²
        s.t.     ḣ_i(x, a) + α_i(h_i(x)) ≥ 0   ∀ i
                 a ∈ A_admissible
   ```

   The executed action is **a\*** (the safe action closest to the VLA's nominal action).

### 3.2 Open-vocabulary, VLM-authored predicates

A key design choice: **the predicate vocabulary is not hand-coded.** The VLM proposes predicates, names them, and supplies a usage definition that the grounding layer consumes. In current experiments the VLM outputs names like `above`, `around`, `below`, but the pipeline should accept arbitrary VLM-authored predicates — including STL-flavored temporal predicates — as long as the VLM also returns enough structured metadata (arguments, reference object, geometric hint, activation condition) for the grounding layer to instantiate a constraint region.

Multi-prompt strategy: distinct prompts for (a) scene description, (b) safety-relevant object/region enumeration, (c) predicate proposal with definitions, (d) predicate-to-geometry hints. The prompts and their outputs are saved per rollout so the VLM evaluation in §4 can analyze them.

### 3.3 Update cadence

Predicates and constraints are re-queried either every step or every **N** steps (action-chunk boundary). This is a configurable knob with an explicit tradeoff: per-step updates catch scene changes fastest; per-chunk updates are cheaper and amortize VLM latency. Implementation should expose both modes cleanly and log the chosen cadence per run.

### 3.4 Infeasibility detection and meta-safety feedback (Level 6)

The CBF-QP can become infeasible when the VLM proposes an over-constrained set of predicates relative to the current state and task. The pipeline must:

1. **Detect** infeasibility (QP solver status, persistent constraint violation, or no action satisfying all barriers).
2. **Classify** the failure mode: (a) strictly infeasible given fixed physical constraints, (b) over-specified by the VLM and relaxable, (c) conflict between competing semantic predicates.
3. **Respond** by (i) relaxing *soft* semantic predicates via slack with hierarchical priorities (Level-1 geometric/physical constraints never relax; higher-level semantic ones can), (ii) re-querying the VLM with the infeasibility explanation to produce a revised predicate set, or (iii) aborting the rollout with a structured "task infeasible under current safety constraints" message.

This feedback loop is what connects the runtime pipeline back to Level 6 of the taxonomy.

---

## 4. VLM Evaluation

Because the pipeline's safety behavior is only as good as the predicates the VLM produces, VLM evaluation is a first-class deliverable, not an afterthought. Two evaluation axes are needed:

1. **Prompt quality / predicate quality (per VLM).** Given a SafeLIBERO scene and a ground-truth hazard annotation, score the VLM's predicate set on: coverage (were all relevant hazards named?), correctness (were the predicate arguments and reference objects right?), groundability (did the grounding layer successfully instantiate a constraint from the predicate?), and over-constraint (did the VLM invent predicates that rendered the task infeasible without cause?).
2. **Cross-VLM comparison.** Same metrics, head-to-head across GPT-family and Qwen-family models, logged per scene so per-category strengths are visible (e.g., which VLMs are better at orientation predicates vs. spatial-relational predicates).

Both should be saved as structured JSON per rollout and aggregated into comparison tables.

---

## 5. System-Level Metrics

Every policy variant (vanilla VLA baseline vs. VLA + safety filter) is evaluated on SafeLIBERO along the following axes:

- **Collision avoidance rate** (primary Level-1 metric).
- **Task success rate** (to confirm the filter does not destroy task performance).
- **Execution time steps** (efficiency / over-cautiousness proxy).
- **Per-taxonomy-level violation counts** where SafeLIBERO annotations support it.
- **Infeasibility rate** (fraction of rollouts where the pipeline triggers meta-safety abort).
- **VLM latency / QP solve time** (system overhead).

The headline comparison is **baseline VLA** vs. **VLA + VLM-CBF-QP filter** on the same seeds / scenes. Ablations should isolate (a) predicate source (fixed hand-designed vs. VLM-proposed), (b) update cadence, (c) VLM choice, (d) with/without infeasibility feedback loop.

---

## 6. Roadmap

- **Now (paper scope for CoRL 2026):** Full simulation pipeline on SafeLIBERO with OpenVLA and π0.5, GPT and Qwen as predicate-producing VLMs, all six taxonomy levels addressed via the predicate-CBF-QP mechanism, with infeasibility feedback and VLM evaluation.
- **Next:** Hardware experiments on a physical manipulator to validate transfer of the predicate-grounding and CBF-QP layers beyond MuJoCo.

---

## 7. Implementation Guidance for Claude Code

When writing or modifying code for this project, keep the following architectural contracts intact:

- **Predicate schema is the interface between the VLM and the rest of the pipeline.** All VLM outputs must be parsed into a structured predicate object (name, arguments, reference object(s), geometric hint, activation condition, priority/level). Do not hardcode predicate names downstream — route them through the grounding layer by type.
- **Level-1 barriers are never soft.** Slack/relaxation logic must not be able to violate geometric, self-collision, joint-limit, or workspace-boundary constraints. Softness applies only to semantic (Level 3+) predicates.
- **Every rollout must persist:** VLA action trace, VLM prompts and raw responses, parsed predicate set, grounded constraints, QP status per step, final safe actions, and per-metric results. This is the substrate for both the VLM evaluation and the paper's ablation tables.
- **VLA backbones and VLM backbones are swappable.** Keep OpenVLA/π0.5 behind a policy interface and GPT/Qwen behind a VLM client interface; do not let prompt-engineering details leak into the core loop.
- **Update cadence is a flag, not an assumption.** Code paths must support both per-step and per-action-chunk VLM/CBF updates without duplication.
- **Infeasibility handling is a first-class code path,** not an exception to swallow. The meta-safety classifier, VLM re-query, and abort-with-explanation branches should each be explicit functions with their own logs.