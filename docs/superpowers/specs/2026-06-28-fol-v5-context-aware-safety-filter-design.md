# FOL Safety Filter v5 — General Context-Aware Safety Design

**Date:** 2026-06-28  
**Status:** Implementation in progress  
**Goal:** CAR > 12% AND TSR > 42% (both beat OpenVLA baseline) on all SafeLIBERO suites  
**Key constraint:** No dependency on GT object naming ("_obstacle" substring)

---

## 1. Motivation

### Why v1–v4 Fall Short

| Version | Obstacle Detection | Key Failure |
|---------|-------------------|-------------|
| v1/v2 | `"obstacle" in name.lower()` | 100% GT-dependent |
| v3 | Same heuristic | Same GT dependency |
| v4 | VLM `/infer_fol` endpoint | Task 0: VLM confused target (TSR=2%) |

**Root causes of v4 failures:**
1. **Target confusion**: VLM wasn't told which object was the pickup target → identified `akita_black_bowl` (the target) as the obstacle
2. **IS_FRAGILE over-constraint**: Added 0.30m avoidance radius for fragile objects → over-blocked legitimate motion → CAR reduced
3. **Goal task regression**: Filter blocked final approach to goal receptacle (stove, drawer) → TSR penalty on Goal suites
4. **Single obstacle only**: Can't handle scenes with multiple hazards

### Paper Alignment

From Robey et al. (Sci. Robot. 2026): The three axes for layered safety:
1. **Declarative**: AI constitution rules ("never handle weapons") — our FOL rule vocabulary
2. **Architectural**: Separate planning from actuation; VLM as external grounder — our VLM→KB→CBF pipeline
3. **Algorithmic**: CBFs as last line of defense — our geometric projection filter

v5 advances all three axes:
- Richer declarative vocabulary (per-object properties, spatial context)
- Better architectural grounding (structured single-call VLM output, explicit target separation)
- Smarter algorithmic filter (phase-aware CBF, multi-obstacle, property-specific radii)

---

## 2. v5 Architecture

```
Episode Start:
  [agentview, eye_in_hand, back_view (opt)] + task_description + candidates
                    │
              ┌─────▼──────────────────────────┐
              │  VLMObstacleGrounder v5         │
              │  Single call to /infer_fol_v5   │
              │  Returns structured JSON:        │
              │    task_target + goal_location   │
              │    + obstacles[] w/ properties   │
              │    + avoidance_radius per obs     │
              └─────────────────────────────────┘
                    │
              ┌─────▼──────────────────────────┐
              │  RuleComposer v5                │
              │  Multi-obstacle rules           │
              │  Property-specific radii        │
              │  (IS_FRAGILE → slow, not wide)  │
              └─────────────────────────────────┘
                    │
              ┌─────▼──────────────────────────┐
              │  FOLKnowledgeBase              │
              │  Workspace bounds rules        │
              │  Obstacle avoidance rules      │
              │  Property-derived rules        │
              └─────────────────────────────────┘

Per-Timestep (20 Hz):
  robot_state (ee_pos, gripper_width, object_poses)
                    │
              ┌─────▼──────────────────────────┐
              │  Phase Detector                │
              │  approaching / carrying /      │
              │  placing (based on gripper+z)  │
              └─────────────────────────────────┘
                    │
              ┌─────▼──────────────────────────┐
              │  CBF Point Filter              │
              │  Per-obstacle, per-phase radii │
              │  Goal relaxation in placing    │
              │  Arm checkpoint protection     │
              └─────────────────────────────────┘
                    │
              u_safe (certified action)
```

---

## 3. VLM Prompt Design (v5)

### Core Principle
One structured JSON call per episode. The prompt:
1. Explicitly names the task target (parsed from description before calling VLM)
2. Tells VLM not to flag the target as an obstacle
3. Asks for structured JSON with task_target + goal_location + obstacles[] + all_properties
4. Asks VLM to suggest per-obstacle avoidance_radius (context-aware, not hardcoded)

### Prompt Template

```
You are a robot safety monitor. Analyze the robot workspace image(s).

ROBOT TASK: {task_description}
CANDIDATE OBJECTS IN SCENE:
{candidates_list}

{target_hint}

Identify safety-relevant information. Return ONLY valid JSON, no other text:
{
  "task_target": "<exact name from candidates above, or null>",
  "goal_location": "<where robot places/puts the object, exact name or null>",
  "obstacles": [
    {"name": "<exact name from candidates>", "properties": ["IS_HEAVY"|"IS_HOT"|"IS_FRAGILE"|"IS_SPILLABLE"|"IS_SHARP"], "avoidance_radius": <0.10 to 0.30>}
  ],
  "all_properties": {"<name>": ["PROP", ...], ...},
  "confidence": <0.0 to 1.0>
}

Property definitions:
- IS_FRAGILE: ceramic/glass/egg — easily broken; avoidance_radius 0.12-0.15
- IS_SPILLABLE: full open container with liquid; avoidance_radius 0.15-0.20
- IS_HOT: stove/candle/heater — burn hazard; avoidance_radius 0.20-0.28
- IS_SHARP: knife/scissors — cut hazard; avoidance_radius 0.12-0.18
- IS_HEAVY: over 2kg cast/iron; avoidance_radius 0.15-0.22

Rules:
- task_target and goal_location MUST NOT be in obstacles
- obstacles are objects the robot should avoid, NOT the pickup target
- If no hazardous obstacles, use "obstacles": []
- avoidance_radius: 0.10=minimal, 0.20=standard, 0.30=very dangerous
```

Where `{target_hint}` is:
- When target parsed: `"TASK TARGET (robot must pick this up): {parsed_target}\nDo NOT list the task target as an obstacle."`
- Otherwise: empty

### Why This Fixes Task 0

v4 prompt: "Which ONE object should the robot avoid?"  
→ VLM saw `akita_black_bowl` and voted it the obstacle (confidence=1.0)

v5 prompt: "TASK TARGET: black bowl. Do NOT list the task target as an obstacle."  
→ VLM correctly identifies moka_pot or ramekin as the obstacle

### API Credit Minimization

| Metric | v4 | v5 |
|--------|-----|-----|
| Calls per episode | 6 (3 obstacle + 3 props) | 1 structured call |
| Tokens per episode | ~3× | 1× |
| Inference time | ~46s | ~15-20s |

For remote APIs (Claude, GPT-4V): use cheapest capable model (Haiku, gpt-4o-mini).
For local Qwen: one call per episode, cached for full rollout.

---

## 4. Phase-Aware CBF

### Phase Detection

```
approaching: gripper_width >= 0.025 (open)
grasping:    gripper_width < 0.025 AND target_z ≤ 0.86
carrying:    gripper_width < 0.025 AND target_z > 0.86
placing:     gripper_width < 0.025 AND dist_to_goal < 0.30
```

### Radius Scaling by Phase

| Phase | Warning radius | Hard radius | Rationale |
|-------|---------------|-------------|-----------|
| approaching | base_r (VLM) | base_r × 0.50 | Full avoidance |
| carrying | base_r × 0.70 | base_r × 0.35 | Tighter — less room |
| placing | base_r × 0.50 | base_r × 0.25 | Allow reaching goal |

Goal relaxation: when in "placing" phase AND dist_to_goal < 0.30m → multiply all obstacle radii by 0.50.

### Property-Specific Rules (v5 vs v4)

| Property | v4 behavior | v5 behavior |
|----------|-------------|-------------|
| IS_FRAGILE | FRAGILE_WIDER: radius 0.30m → over-constraint | FRAGILE_SLOW: velocity limit 0.10 m/s only |
| IS_SPILLABLE | rotation lock ✓ | rotation lock ✓ |
| IS_HOT | hot zone 0.25m ✓ | hot zone, capped 0.28m ✓ |
| IS_SHARP | not implemented | spatial avoidance ✓ |

The IS_FRAGILE change is critical — the 0.30m fragile radius was causing CAR reduction.

---

## 5. Real-World Franka Compatibility

The filter requires only:
- `ee_pos` (3D): from robot state estimation
- `gripper_width` (float): from finger position sensors
- `{object}_pos` (3D each): from any perception system (FoundationPose, SAM2+depth, etc.)
- Camera images: standard RGB camera(s)

No MuJoCo, no simulation-specific APIs. The VLM inference is completely decoupled (HTTP call to any server).

### Pluggable VLM Backends

```python
# Local Qwen (HPC/lab)
export QWEN_SERVER_URL=http://localhost:5001

# Remote Claude API (any machine)
export CLAUDE_API_KEY=sk-ant-...
export USE_CLAUDE_API=1  # future feature

# Remote GPT-4V (any machine)
export OPENAI_API_KEY=sk-...
export USE_OPENAI_API=1  # future feature
```

All backends return the same structured JSON schema.

---

## 6. Predicate Vocabulary (v5)

### Physical-State Predicates (VLM-grounded, cached per episode)
- `IS_FRAGILE(obj)` — ceramic, glass, egg
- `IS_SPILLABLE(obj)` — full open container
- `IS_HOT(obj)` — stove, candle, heater
- `IS_SHARP(obj)` — knife, scissors, blade
- `IS_HEAVY(obj)` — cast iron, heavy box
- `IS_WET(obj)` — wet sponge, damp cloth
- `IS_LIT(obj)` — actively burning
- `IS_FLAMMABLE(obj)` — paper, fabric
- `IS_HUMAN(obj)` — human body part

### Spatial Predicates (geometric, 20 Hz)
- `NEAR(a, b, r)` — dist(a,b) < r
- `ABOVE(a, b)` — z_a > z_b
- `BELOW(a, b)` — z_a < z_b
- `SUPPORTS(a, b)` — a supports b
- `IN_WORKSPACE()` — EEF in bounds
- `NEAR_EDGE()` — EEF near boundary

### Robot-State Predicates (sensor-derived, 20 Hz)
- `GRIPPING(eef, obj)` — gripper closed + near obj
- `HOLDING(eef, obj)` — gripping + lifted
- `MOVING_FAST()` — EEF velocity > threshold
- `TILTED(obj)` — object tilted > threshold
- `EEF_TILTED()` — EEF tilt when holding spillable

### Composed Rules (runtime)
VLM output generates: `NEAR(eef, obstacle, r)` rules per obstacle  
Heuristic level-2 rules: FIRE_HAZARD, WATER_DAMAGE, SPILL_RISK, FRAGILE_GRASP, etc.

---

## 7. Expected Performance

Based on root cause analysis:

| Task | Expected v5 TSR | Rationale |
|------|-----------------|-----------|
| Spatial T0 | ~40% (up from 2%) | Target exclusion fix |
| Spatial T1 | ~50% (up from 44%) | Better obstacle ID |
| Spatial T2 | ~90% | Already good |
| Spatial T3 | ~50% | Similar |
| **Overall spatial** | **~57%** vs 48.5% v3 | +8.5pp |

CAR target: ~55% (similar to v3, or higher if more obstacles detected).

---

## 8. Files Modified

| File | Change |
|------|--------|
| `vlm_pipeline/qwen_vlm_server.py` | + `/infer_fol_v5` endpoint |
| `v5/fol_safety_filter/filter.py` | Phase-aware, multi-obstacle filter |
| `v5/fol_safety_filter/vlm_grounder.py` | Structured single-call grounder |
| `v5/fol_safety_filter/rule_composer.py` | Multi-obstacle + property radii |
| `v5/fol_safety_filter/primitives.py` | Copy from v4 (unchanged) |
| `v5/fol_safety_filter/kb.py` | Copy from v4 (unchanged) |
| `v5/fol_safety_filter/cbf_mapper.py` | Copy from v4 (unchanged) |
| `fol_safety_filter/filter.py` | Deployed v5 (overwritten) |
| `fol_safety_filter/vlm_grounder.py` | Deployed v5 (overwritten) |
| `fol_safety_filter/rule_composer.py` | Deployed v5 (overwritten) |
| `slurm/eval_fol_v5_smoke.slurm` | NEW: spatial L1, n=2 |
| `slurm/eval_fol_v5_spatial_L1_n50.slurm` | NEW: spatial L1, n=50 |

`run_safelibero_fol_openvla_eval.py` — NOT MODIFIED (hard constraint)

---

## 9. Iteration Plan

After initial v5 results:
1. If TSR < 42%: Check VLM logs for target confusion; tighten prompt
2. If CAR < 12%: Increase base radii; check phase detection fires
3. Prompt iteration: If VLM accuracy < 80%: add few-shot examples to prompt
4. If Goal tasks still regress: Tune goal relaxation threshold (0.30m → 0.40m)
5. If time allows: Test on all 8 suite×level conditions
