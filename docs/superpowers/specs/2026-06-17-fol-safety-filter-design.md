# FOL-Grounded Generalized Safety Filter for Robot Manipulation

**Date**: 2026-06-17  
**Status**: Design / Pre-implementation  
**Scope**: SafeLIBERO sim evaluation (OpenVLA + FastWAM) + hardware WAM taxonomy testing

---

## 1. Problem Statement

State-of-the-art VLA and world-model policies fail catastrophically on the SafeLIBERO safety benchmark. FastWAM achieves only ~5% CAR (Collision Avoidance Rate) on SafeLIBERO Spatial Level I despite 45% task success — it completes tasks by bulldozing through obstacles. OpenVLA fares similarly.

Existing safety filter approaches have two failure modes:
1. **Geometric-only CBFs**: Fast but semantically blind — can't distinguish a full glass of water from an empty one, or a lit candle from an unlit one.
2. **Fixed-vocabulary VLM filters** (e.g. Brunke et al.): Hard-code predicate types to `{spatial, behavioral, pose}` — cannot compose new constraints, cannot handle novel objects or scenarios, cannot generalize across the 5-level safety taxonomy.

**Goal**: A single, policy-agnostic safety filter grounded in First-Order Logic (FOL) that:
- Covers all 5 safety levels in the Robot Safety Taxonomy (geometric → semantic → process → task-contextual → interactive)
- Works with any policy (VLA, world model, diffusion policy) through a unified interface
- Composes new safety rules at runtime from a primitive predicate vocabulary
- Can be evaluated on SafeLIBERO (simulation) and all 17 hardware taxonomy scenarios

---

## 2. Safety Taxonomy Coverage

The filter is designed to handle all scenarios from `Robot_Safety_Taxonomy_for_FOL.pdf`:

### Level 1 — Geometric / Collision Safety
| Scenario | Description | Key Constraint |
|---|---|---|
| 1a-i: Robot-Obstacle Collision | Arm path through occupied space | Signed-distance CBF per object |
| 1a-ii: Robot-Human Collision | Human hand in workspace | Speed + force limiting near body parts |
| 1a-iii: Multi-Obstacle Corridor | Paper between two lit candles | Simultaneous clearance from N obstacles |
| 1b: Kinematic Limits | Target at workspace boundary | Joint position + velocity bounds |

### Level 2 — Semantic Safety
| Scenario | Description | Key Constraint |
|---|---|---|
| 2a-i: Material Incompatibility | Wet sponge over stack of paper | Block approach if WET ∧ ABSORBENT ∧ ABOVE |
| 2a-ii: Compositional (candle) | Lit/unlit candle near napkins | State-dependent rule activation |
| 2b: State-Dependent Force | Picking up a fragile egg | Grip force limit based on fragility |
| 2c: Content-Dependent Tilt | Moving full glass over laptop | Rotation lock if spillable container |
| 2d: Part-Functional | Handing a knife | Grasp region restricted to handle |

### Level 3 — Process / State Safety
| Scenario | Description | Key Constraint |
|---|---|---|
| 3a-i: Containment/Spill (pour) | Pouring rice bowl into cup | Tilt angle + pour-velocity CBF |
| 3a-ii: Containment/Spill (open) | Picking up open bag of skittles | Orientation constraint + tilt limit |
| 3b: Structural Integrity | Placing big cube on small cube | Topple-stability CBF (center-of-mass) |

### Level 4 — Task-Contextual Safety
| Scenario | Description | Key Constraint |
|---|---|---|
| 4a: Sequencing | Clear table: glass on plate on placemat | Partial-order precondition gate |
| 4b: Goal Object-Specific | "Hand me something to stir with" | Task-goal compatibility check |

### Level 5 — Interactive / Relational Safety
| Scenario | Description | Key Constraint |
|---|---|---|
| 5a: Act/Ask | Ambiguous items on table | Confidence < threshold → query human |
| 5b: Safe Response | "Push cup to table edge" | Refuse + propose safe alternative |
| 5c: Runtime NL Constraint | "Keep 15cm from the glass" | Parse NL → FOL rule → CBF |

---

## 3. Architecture

### 3.1 System Diagram

```
Observation (RGB, depth, EEF pos/quat, joint state)
       │
       ├─── [ONCE at episode start, ~3s]
       │           │
       │    ┌──────▼────────────────────────────────────────┐
       │    │           FOL Knowledge Base Setup             │
       │    │                                                │
       │    │  1. Object detection (GroundingDINO / GT)      │
       │    │  2. Level 1 VLM binary queries (IS_WET, etc.) │
       │    │  3. Level 2 VLM composition → FOL rules       │
       │    │  4. Level 3 novel predicate invention (if any)│
       │    └──────┬────────────────────────────────────────┘
       │           │ FOLKnowledgeBase populated
       │           │
       │    [AT 20 Hz, <1ms]
       │           │
       │    ┌──────▼────────────────────────────────────────┐
       │    │       Active Constraint Evaluation             │
       │    │                                                │
       │    │  1. Ground spatial predicates (geometry)      │
       │    │  2. Ground robot-state predicates (sensors)   │
       │    │  3. Evaluate FOL rules against cache          │
       │    │  4. Emit active CBF constraint set            │
       │    └──────┬────────────────────────────────────────┘
       │           │
Policy action ─────┤
       │    ┌──────▼────────────────────────────────────────┐
       │    │    SemanticSafetyFilter (existing QP solver)  │
       │    │    Dykstra projection onto CBF constraints     │
       │    └──────┬────────────────────────────────────────┘
       │           │
       └───────────▼
              Certified Safe Action → Environment
```

### 3.2 Policy-Agnostic Interface

All four policies output the same 7-DoF action format: `[dx, dy, dz, drx, dry, drz, gripper]`. The existing `SemanticSafetyFilter.certify_action()` already accepts this format. The FOL filter wraps it:

```python
class FOLSafetyFilter:
    def setup_scene(self, obs: dict, task: str, objects: List[str]) -> None:
        """Run once per episode. Populates KB with predicate truth values and FOL rules."""

    def certify(self, u_cmd: np.ndarray, ee_pos: np.ndarray,
                ee_quat: np.ndarray) -> np.ndarray:
        """Run at 20 Hz. Evaluates active rules, delegates to QP solver."""

    def update_constraint(self, nl_spec: str) -> None:
        """Level 5c: parse NL constraint at runtime, add to KB."""
```

The 3-line insertion into any existing eval script:
```python
fol_filter = FOLSafetyFilter(...)
fol_filter.setup_scene(obs, task_description, detected_objects)  # once
# ... in action loop:
action = fol_filter.certify(action, obs["robot0_eef_pos"], obs["robot0_eef_quat"])
```

---

## 4. The Three Predicate Levels

### 4.1 Level 1: Primitive Predicates (Predefined)

A typed vocabulary of groundable predicates. Each has a name, arity, and a grounding function that maps observations → truth value (bool) or continuous value (float).

**Spatial predicates** — grounded via geometry, <1ms:
```
NEAR(a, b, d)           dist(center(a), center(b)) < d
ABOVE(a, b)             pos_z(a) > pos_z(b) + 0.02
BELOW(a, b)             pos_z(a) < pos_z(b) - 0.02
SUPPORTS(a, b)          BELOW(a, b) ∧ dist_xy(a, b) < 0.05
IN_WORKSPACE(pose)      pose within robot reachable bounds
NEAR_JOINT_LIMIT(j, m)  |q_j - q_lim| < m
TILTED(obj, θ)          |tilt_angle(obj)| > θ
NEAR_EDGE(pose)         table_edge_dist(pose) < 0.05
```

**Physical state predicates** — grounded via VLM binary query (once per episode, majority vote N=3):
```
IS_FRAGILE(obj)         "Is {obj} fragile or breakable?"
IS_WET(obj)             "Is {obj} wet or contains liquid?"
IS_LIT(obj)             "Is {obj} currently lit or on fire?"
IS_FULL(container)      "Is {container} full?"
IS_OPEN(container)      "Is {container} open or uncovered?"
IS_FLAMMABLE(obj)       "Is {obj} flammable (paper, wood, fabric)?"
IS_ABSORBENT(obj)       "Is {obj} water-absorbent?"
IS_HUMAN(obj)           "Is {obj} a human body part?"
HAS_SHARP_PART(obj)     "Does {obj} have a dangerous sharp region?"
IS_HEAVY(obj)           "Is {obj} significantly heavy (>2kg)?"
```

**Robot state predicates** — grounded via sensor readings, <1ms:
```
GRIPPING(eef, obj)      gripper_state=CLOSED ∧ NEAR(eef, obj, 0.05)
HOLDING(eef, obj)       GRIPPING ∧ pos_z(obj) > pos_z(obj_start) + 0.02
MOVING_FAST(eef, v)     ||v_eef|| > v
```

**Composed Level 1 shorthands** (boolean combinations, no VLM needed):
```
IS_SPILLABLE(c)         IS_FULL(c) ∧ IS_OPEN(c)
IN_COLLISION(a, b)      NEAR(a, b, 0.0) — distance ≤ 0
```

### 4.2 Level 2: Composed Predicates (VLM-constructed at runtime)

At episode start, the VLM receives the RGB image + task instruction and outputs FOL rules by composing Level 1 predicates. This extends the Brunke et al. multi-prompt approach to open-ended FOL formulas rather than fixed-type queries.

**Prompt structure**: The VLM is given:
1. The RGB scene image
2. The task instruction
3. The full list of Level 1 primitive predicates
4. A structured JSON output schema

The VLM outputs rules like:
```json
{
  "rules": [
    {
      "name": "FIRE_HAZARD",
      "formula": "IS_LIT(candle) AND IS_FLAMMABLE(napkin) AND NEAR(eef, candle, 0.3)",
      "cbf_type": "spatial",
      "cbf_params": {"margin": 0.3, "alpha_scale": 0.5},
      "violation_action": "avoid"
    },
    {
      "name": "SPILL_RISK",
      "formula": "IS_SPILLABLE(glass) AND HOLDING(eef, glass)",
      "cbf_type": "parametric",
      "cbf_params": {"max_angular_vel": 0.05},
      "violation_action": "slow"
    }
  ]
}
```

Rules are validated against the primitive vocabulary schema before use. Malformed rules are logged and discarded.

**Multi-prompt voting**: Each rule is independently generated N=3 times; consensus rules (appearing ≥2/3 times) are accepted.

### 4.3 Level 3: Novel Predicates (VLM-invented)

When the scene contains objects that none of the Level 1 predicates can describe, the VLM proposes new predicates with a name, a grounding query (evaluated as a VLM binary question), and a safety effect:

```json
{
  "novel_predicates": [
    {
      "name": "IS_SENTIMENTAL",
      "grounding_query": "Is {obj} an irreplaceable or sentimental item?",
      "cbf_type": "parametric",
      "effects": {"speed_scale": 0.3, "force_scale": 0.3},
      "used_in_rule": "IS_SENTIMENTAL(obj) AND GRIPPING(eef, obj) → EXTREME_CAUTION"
    }
  ]
}
```

Novel predicates are:
- Evaluated lazily (VLM query on first use, cached thereafter)
- Stored in a `NovelPredicateRegistry` per episode
- Used exactly like Level 1 predicates in rule evaluation

**Level 5c** (runtime NL constraint specification) follows the same path: natural-language input → VLM parses → new FOL rule → added to KB live during execution.

---

## 5. CBF Construction from FOL Rules

The FOL engine produces an active constraint set each timestep. Each constraint maps to one of the CBF types already supported by `SemanticSafetyFilter`:

| Rule type | CBF form | Existing code hook |
|---|---|---|
| Spatial avoidance (NEAR, ABOVE) | Superquadric: `h(x) = sq_implicit(ee, env) - 1` | `_cbf_sem()`, `build_cbf_envelope()` |
| Rotation lock (IS_SPILLABLE, IS_FULL) | Angular velocity: `‖ωr‖ ≤ ω_max` | `_cbf_rot()` |
| Velocity limit (IS_HUMAN, IS_FRAGILE) | Linear velocity: `‖v_eef‖ ≤ v_max` | `_cbf_ws()` (extended) |
| Force/grip limit (IS_FRAGILE) | `grip_force ≤ f_max` — new parametric constraint | New: `_cbf_force()` |
| Hard block (NEAR_EDGE, NEAR_JOINT_LIMIT) | Direct velocity component zeroing | New: `_cbf_hard()` |
| Sequencing gate (SUPPORTS) | Pre-condition check → block action if unmet | New: `_cbf_precond()` |
| Act/Ask (Level 5a) | Confidence < threshold → emit `ASK` signal (not a CBF) | New: `HRIInterface` |

The existing Dykstra QP solver in `SemanticSafetyFilter._cbf_sem()` handles all spatial constraints unchanged. The new parametric/velocity constraints are light additions.

**Alpha (class-K∞) function scaling**:
- Default: `α(h) = 1.0 · h`
- Caution (IS_HUMAN, IS_FRAGILE): `α(h) = 0.25 · h`
- Extreme caution (IS_SENTIMENTAL novel): `α(h) = 0.1 · h`

---

## 6. What's Novel vs. Literature

| Gap in existing work | This system |
|---|---|
| Brunke et al. (RA-L 2025): multi-prompt for fixed predicate types | Multi-prompt for **open-ended FOL formula generation** |
| All CBF-based safety filters: geometric only | **Semantic predicates** (IS_WET, IS_LIT, IS_FRAGILE) compose with geometry |
| No runtime rule composition | VLM **builds new FOL rules** from primitive vocabulary at episode start |
| No runtime predicate invention | VLM **proposes novel predicates** with VLM grounding queries (Level 3) |
| All safety filters are policy-specific | **Single interface** for VLA, diffusion model, world model |
| Safety = collision avoidance only (Level 1) | **5-level taxonomy** from geometric through interactive safety |
| No act/ask integration in safety filter | **Level 5a**: low confidence → emit human query before acting |
| No NL runtime constraint parsing | **Level 5c**: "keep 15cm from the glass" → new FOL rule → active CBF |
| No formal predicate representation | **Typed FOL vocabulary** with verifiable grounding functions |

The key novelty is the combination of: (1) a formal FOL backbone with grounded primitives, (2) runtime VLM composition of new rules from primitives, and (3) policy-agnostic application across the full safety taxonomy.

---

## 7. Implementation Roadmap

### Phase 1 — FOL Predicate Engine
**Files**: `fol_safety_filter/primitives.py`, `fol_safety_filter/kb.py`

- `GroundingFn` protocol: `(obs, objects, args) → bool | float`
- Register all Level 1 predicates with their grounding functions
- `FOLRule` dataclass: formula string + parsed AST + CBF type + params
- `FOLKnowledgeBase`: predicate cache, rule list, 20 Hz update loop
- Simple recursive-descent FOL parser (AND/OR/NOT + quantifier stubs)

### Phase 2 — VLM Composition Engine
**Files**: `fol_safety_filter/vlm_composer.py`, `fol_safety_filter/vlm_novel.py`

- Level 2: structured prompt → JSON → `FOLRule` objects; JSON schema validation
- Level 3: novel predicate invention + lazy VLM grounding on first use
- Multi-prompt wrapper (N=3 majority vote) — reuse `MultiPromptVLMAnalyzer` pattern from `semantic_cbf/multiprompt_pipeline.py`
- Level 5c: NL → FOL rule parser (VLM call on demand during execution)

### Phase 3 — CBF Mapper
**File**: `fol_safety_filter/cbf_mapper.py`

- Map `FOLRule.cbf_type` → constraint for `SemanticSafetyFilter`
- New: `_cbf_force()` for grip force limits (IS_FRAGILE)
- New: `_cbf_hard()` for hard blocks (NEAR_EDGE)
- New: `_cbf_precond()` for sequencing gates (SUPPORTS)
- Reuse: `build_cbf_envelope()`, `sq_implicit()`, `sq_grad()` from `vlm_pipeline/semantic_cbf_filter.py`

### Phase 4 — Policy-Agnostic Filter Interface
**File**: `fol_safety_filter/filter.py`

- `FOLSafetyFilter` wrapping the full pipeline
- Episode setup: VLM binary queries + composition + novel invention
- Runtime loop: geometric grounding + rule evaluation + QP solve
- HRI interface stub for Level 5a/5b

### Phase 5 — SafeLIBERO Evaluation
**File**: `vlm_pipeline/run_safelibero_fol_eval.py`

- `--policy [openvla|fastwam|cosmos|dream]` flag
- `--fol_level [1|2|3|all]` ablation flag
- Extended metrics: TSR, CAR, ETS + FAR (filter activation rate) + predicate-level breakdown
- Modeled on `run_safelibero_fastwam_eval.py` action loop

### Phase 6 — Hardware Taxonomy Testing
- WAM robot adapter (Cartesian velocity interface)
- Scenario test harness for all 17 PDF scenarios
- Ground-truth predicate annotation for each scenario (for VLM accuracy eval)

---

## 8. File Structure

```
fol_safety_filter/
  __init__.py
  primitives.py          # Level 1: GroundingFn protocol + all primitive definitions
  kb.py                  # FOLKnowledgeBase, FOLRule, NovelPredicate dataclasses
  vlm_composer.py        # Level 2: VLM → structured FOL rules
  vlm_novel.py           # Level 3: novel predicate invention + registry
  cbf_mapper.py          # FOL active constraint → SemanticSafetyFilter input
  filter.py              # FOLSafetyFilter — main public interface

vlm_pipeline/
  run_safelibero_fol_eval.py   # New evaluation script

docs/superpowers/specs/
  2026-06-17-fol-safety-filter-design.md   # This document
```

**Existing files reused without modification**:
- `vlm_pipeline/semantic_cbf_filter.py` — `SemanticSafetyFilter`, `build_cbf_envelope()`, `sq_implicit()`, `sq_grad()`
- `semantic_cbf/multiprompt_pipeline.py` — multi-prompt VLM pattern
- `vlm_pipeline/safelibero_utils.py` — env setup, image utilities

---

## 9. Grounding Strategy Summary

The simplest robust grounding approach:

**Spatial predicates** → direct geometric computation from object poses (MuJoCo GT in sim; DepthAnything V2 + SAM2 on hardware). No VLM needed. <1ms.

**Physical state predicates** → one VLM binary question per (predicate, object) pair at episode start. 5–10 objects × 5 predicates × N=3 votes ≈ 75–150 VLM calls. At ~100ms/call (batched Qwen): ~3–8s setup overhead. Cached for entire episode.

**Robot state predicates** → direct sensor readings (joint encoders, EEF pose, F/T sensor on hardware). <1ms.

**Novel predicates (Level 3)** → VLM binary question at first reference. Cached thereafter.

This is deliberately conservative on VLM usage: most of the safety-critical computation at 20 Hz is fast geometric + boolean logic with zero VLM calls.

---

## 10. Metrics

| Metric | Definition | Target (vs. baseline) |
|---|---|---|
| TSR | Task Success Rate | Maintain ≥90% of baseline |
| CAR | Collision Avoidance Rate | >90% (baseline: ~5%) |
| ETS | Mean execution timesteps | ≤10% increase |
| FAR | Filter Activation Rate | Diagnostic — % timesteps where CBF modifies action |
| PAR | Predicate Accuracy Rate | VLM binary query vs. ground truth annotation |
| Level Coverage | % taxonomy scenarios where filter correctly activates | Target: >85% across L1–L4 |

---

## 11. Open Questions / Risks

1. **VLM setup latency**: 3–8s at episode start may be acceptable for sim eval but is borderline for real-time hardware. Mitigation: precompute on static scenes, parallelize queries.
2. **FOL formula parsing correctness**: The recursive descent parser needs careful testing for edge cases (nested parentheses, quantifiers).
3. **Level 5 (interactive) completeness**: Act/Ask and refuse/propose require a human-in-the-loop interface that is out of scope for sim eval — design HRI stub for hardware only.
4. **SafeLIBERO task specificity**: The 4 task suites (spatial, goal, object, long) primarily test geometric collision avoidance (Level 1) — Levels 2–5 scenarios are separate hardware tests.
