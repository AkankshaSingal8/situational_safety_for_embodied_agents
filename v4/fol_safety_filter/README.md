# FOL Safety Filter — V4 (VLM-Grounded Obstacle Detection + Predicate Composition)

## What's New vs V3

### Change 1: VLMObstacleGrounder (`vlm_grounder.py`)

Replaces the eval script's joint-name scan (`"obstacle" in joint_name`) with an
in-filter detection step that runs at episode start from obs dict keys.

**Two modes:**
- `use_vlm_obstacle=False` (default): scans obs keys for `"obstacle"` substring —
  functionally equivalent to v3's heuristic, but runs inside the filter rather than
  requiring the eval script to identify the obstacle externally.
- `use_vlm_obstacle=True`: queries Qwen2-VL with the scene RGB + task description.
  Falls back to heuristic if VLM is unavailable.

**Also extracts physical properties** (IS_SPILLABLE, IS_FRAGILE, IS_HOT) from the
object name, enabling richer rules.

### Change 2: RuleComposer (`rule_composer.py`)

Generates FOL rules from the grounded predicate dict instead of hardcoded templates:

| Predicate       | Rule generated                      | CBF effect                        |
|-----------------|-------------------------------------|-----------------------------------|
| obstacle found  | `NEAR(eef, {obs}, 0.20) → CANCEL`  | Base avoidance (same as v1–v3)    |
| IS_SPILLABLE    | `IS_SPILLABLE → ROTATION_LOCK`      | Limit wrist rotation near liquid  |
| IS_FRAGILE      | `NEAR(eef, {obs}, 0.30) → AVOID`   | Wider warning zone (30cm vs 20cm) |
| IS_HOT          | `NEAR(eef, {obs}, 0.25) → AVOID`   | Thermal exclusion zone            |

### Change 3: kb.py `__init__` accepts `rules=` list

`FOLKnowledgeBase(rules=composed_rules)` allows dynamic rule injection without
modifying the class. Backward compatible — defaults to empty list.

## Files Changed vs V3

- `filter.py`: +VLMObstacleGrounder call in `setup_episode()`, +RuleComposer,
  all CBF geometry unchanged
- `kb.py`: +`rules` param in `__init__`
- `vlm_grounder.py`: NEW
- `rule_composer.py`: NEW
- `primitives.py`, `cbf_mapper.py`, `__init__.py`: byte-for-byte identical to v3

## Fallback Behavior

Without VLM (`use_vlm_obstacle=False`, default):
- Obstacle identified by scanning obs keys for "obstacle" substring
- Physical properties inferred from object name keywords
- Rules identical to v3

With VLM (`use_vlm_obstacle=True`):
- Qwen queried at episode start (~5s overhead)
- If Qwen unavailable → falls back to heuristic silently
- IS_SPILLABLE/IS_FRAGILE/IS_HOT rules added when VLM reports them

## Expected Results

V4 in heuristic mode should match v3 exactly on all 8 conditions.
V4 in VLM mode adds property-aware rules but obstacle identification is the
same → TSR/CAR should stay comparable.

Full VLM generalization benefit appears when testing on scenes WITHOUT
"obstacle" in object names — the primary use case beyond SafeLIBERO.
