# Reusable FOL Rule Memory — Design

**Date:** 2026-09-18
**Branch:** `memory`
**Status:** approved for implementation

## 1. Problem

The live method's logic layer does not decide anything on the dominant channel.

`FOLSafetyFilter.certify` builds an active-constraint set and maps it to CBFs:

```python
active   = self.kb.evaluate(state)          # filter.py:771
cbf_set  = self.mapper.map(active, state.ee_quat)   # filter.py:784
u_safe,_ = self._apply_cbf(u_cmd, state, cbf_set, arm_checkpoints)
```

but `_apply_cbf` then ignores `cbf_set`'s spatial content and runs its own loop:

```python
for obs_name in self._obstacle_names:               # filter.py:878
    warning_r = max(0.20, self._obstacle_radii.get(obs_name, 0.20))
    self._apply_point_cbf(u, ee, obs_pos, warning_r=warning_r,
                          hard_r=0.10, push_hard=0.08, ...)
```

`CBFMapper.map` even documents the bypass explicitly: sphere-shaped `NEAR`
constraints are deliberately *not* forwarded to the QP because
`filter.py._apply_point_cbf` already handles them (`cbf_mapper.py:168-175`).
Only `velocity_limit` and `angular_limit` are read back off the mapped set.

Consequences:

- There is **one** monolithic obstacle rule, with a speed knob and a rotation
  knob attached, rather than a set of independent rules.
- Rule records in `kb.py` / `rule_composer.py` cannot change the geometry that
  runs. Deleting `OBSTACLE_AVOID_*` from the KB would not stop obstacle
  avoidance; it would only stop it being *counted* in `rule_counts`.
- Every constant that determines behaviour (`0.20`, `0.10`, `0.08`, the arm
  triples, `cancel_scale=0.6`) lives in Python, so a rule cannot be authored,
  reviewed, versioned, or transferred as data.
- Rule parameters are an open, untyped dict, so a misspelled key is silent.
  `rule_composer.py:76` and `filter.py:427,508` all author rotation rules as
  `cbf_params={"angular_limit": 0.10}` (or `0.15`), while the consumer reads
  `params.get("omega_max", 0.25)` (`cbf_mapper.py:184`). Every rotation-lock
  rule in the live path therefore runs at the 0.25 rad/s default and the
  authored value is unreachable. Nothing reports this. A closed, typed
  parameter set per effect predicate (§3.1) makes that class of defect a load
  error instead.

## 2. Goal

Decompose the safety logic into independent, individually loadable FOL rule
records held in an on-disk memory, and route the filter's decisions through
them — such that deleting a record demonstrably removes exactly one behaviour,
and a record authored once applies unchanged to new scenes and payloads.

Non-goal, explicitly: the 1,000-episode campaign in
`docs/design/2026-09-17-reusable-safety-skills-poc-protocol.md`, the pi05
policy, a LIBERO-Safety adapter, repair procedures, and runtime rule
*induction*. This work delivers the representation, the loader, the
decomposition, and an offline transfer measurement. Nothing else.

## 3. The memory

```
fol_safety_filter/safety_memory/
  MANIFEST.yaml                 # which rule ids are default-loaded vs opt-in
  <rule_id>/
    SKILL.md                    # YAML front matter = executable contract
                                # Markdown body = rationale, never executed
    tests.yaml                  # positive / negative / boundary activation cases
    metadata.json               # revision, provenance, evidence references
```

The store lives in the repository, versioned with the code, because it is
runtime-loaded data that must travel with the branch. The layout is the one
settled in `docs/design/2026-09-17-reusable-safety-skills-review.md:171`; this
is a down-scoped instantiation of that design, not a competing one.

### 3.1 Contract schema, version 1

One record = one activation condition + **one** effect. A record carrying two
effects is invalid. That restriction *is* the decomposition.

```yaml
---
schema_version: 1
id: eef_obstacle_avoid
revision: 1
description: Keep the end-effector out of each grounded obstacle's exclusion zone.
roles:
  subject: {binding: eef}
  target:  {binding: obstacles}
applies_when: "NEAR(SUBJECT, TARGET, 0.20)"
requires:
  predicate: avoid_sphere
  warning_r: 0.20
  hard_r: 0.10
  push_hard: 0.08
  shape: ellipsoid_if_available
  cancel_scale_on_target_corridor: 0.6
violation_action: avoid
unknown_action: stop_and_report
---
```

`applies_when` is a formula in the **existing** `kb.py` mini-language. No new
formula language is introduced; `parse_formula` is reused verbatim. `SUBJECT`
and `TARGET` are the only reserved role tokens and are substituted at
compile time by the binding resolver.

Role bindings, closed set:

| binding | resolves to |
|---|---|
| `eef` | `state.ee_pos` (the single symbol `eef`) |
| `arm_checkpoints` | each entry of `_get_arm_checkpoints(env)` |
| `obstacles` | each name in `self._obstacle_names`, in that order |
| `currently_grasped` | the payload name, or `None` |
| `objects_with_fact:<F>` | each object whose oracle/VLM fact `F` is true |

Effect predicates, closed set of four:

| predicate | typed params | applied by |
|---|---|---|
| `avoid_sphere` | `warning_r`, `hard_r`, `push_hard`, `shape`, `cancel_scale_on_target_corridor` | `_apply_point_cbf` |
| `limit_speed` | `v_max` | translational clamp |
| `limit_angular_rate` | `omega_max` | angular clamp |
| `avoid_region_above` | `clearance_m`, `z_max`, `cancel_scale` | new column projection |

Four effects carry five rules: `avoid_sphere` is shared by two records that
differ only in `roles.subject`. That sharing is the point — see §5.

### 3.2 Validation

Strict and all-or-nothing. An unknown key, an out-of-range parameter, an
unparseable formula, a second effect, a missing required field, a duplicate
`id`, a non-integer `revision`, or a file over 16 KiB raises `InvalidRuleError`
and **fails the whole snapshot**. A present-but-broken record never degrades to
a silent partial load.

This mirrors the failure mode `CLAUDE.md` warns about for
`vlm_pipeline/semantic_cbf_filter.py`, where a missing import silently
downgrades the filter to geometry-only and "changes every reported number".
Silent degradation is the specific thing this schema refuses.

Ranges: `warning_r ∈ (0, 0.60]`, `hard_r ∈ (0, warning_r]`,
`push_hard ∈ [0, 0.30]`, `v_max ∈ (0, 1.0]`, `omega_max ∈ (0, 5.0]`,
`clearance_m ∈ [0, 0.10]`, `cancel_scale ∈ (0, 1]`.

### 3.3 Loading

Records load in sorted `id` order at episode setup, are validated before the
first action, and are cached as a compiled representation. Changes during an
episode are ignored and logged; a revision takes effect at the next episode
boundary. Reloading from disk must reproduce a byte-identical compiled
representation — this is a test, not an aspiration.

Each record is hashed (front matter and body separately) so a rule transferred
to a new scene can be shown to be the *same bytes*, and any edit is a visible
revision rather than an invisible drift.

## 4. Execution path

`rule_memory.py` compiles each loaded record into zero or more `FOLRule`
instances by resolving `roles` against the episode scene: one record over three
obstacles yields three rules with `bindings={"TARGET": obs_i}`. This lifting is
what makes a record reusable rather than scene-specific.

`_apply_cbf` becomes a dispatcher over the `ActiveConstraint` list the KB
already produces, with one handler per effect predicate. `_apply_point_cbf`
remains the geometric primitive, unchanged.

### 4.1 Deterministic effect ordering

`_apply_cbf` mutates `u` in place, so ordering is semantics, not style. The
current nesting is obstacle-major — for each obstacle, the EEF projection then
each arm checkpoint — and the dispatcher must reproduce it exactly:

1. `avoid_sphere` and `avoid_region_above`, sorted by
   `(target_index, subject_rank)` with `subject_rank`: `eef`=0,
   `arm_checkpoints`=1, then checkpoint order.
2. task-progress bias (`FOL_TASK_BIAS`), unchanged, not a rule.
3. `limit_speed`, strictest wins.
4. `limit_angular_rate`, strictest wins.

Env-var behaviours (`FOL_ELLIPSOID`, `FOL_RAY_RESCALE`, `FOL_TASK_BIAS`,
`FOL_CHUNK_REPLAN`) keep working and keep their precedence as overrides of the
record fields.

### 4.2 Falsifiability

Deleting `arm_link_obstacle_avoid` from `MANIFEST.yaml` must remove the arm
channel from `report()["rule_counts"]` *and* change `u_safe` on a fixture where
an arm checkpoint is the only thing inside a warning radius. Today neither
happens, because the KB's output never reaches the projection. This is the
test that distinguishes a real rerouting from a new file format layered over
unchanged code.

## 5. The five rules

| # | `id` | effect | loaded | what it is |
|---|---|---|---|---|
| 1 | `eef_obstacle_avoid` | `avoid_sphere` | default | the 0.20 / 0.10 / 0.08 EEF geometry now hardcoded at `filter.py:886` |
| 2 | `arm_link_obstacle_avoid` | `avoid_sphere` | default | **same effect, `subject` rebound to `arm_checkpoints`** (0.15 / 0.08 / 0.06) — the record now owns those radii; `_get_arm_checkpoints` supplies only body positions, except that the opt-in `FOL_HAND_CHECKPOINT` bodies keep their tighter per-body override |
| 3 | `fragile_speed_limit` | `limit_speed` | default | the `cbf_set.velocity_limit` channel as a standalone record |
| 4 | `spillable_rotation_lock` | `limit_angular_rate` | default | the `cbf_set.angular_limit` channel as a standalone record |
| 5 | `overhead_exclusion` | `avoid_region_above` | **opt-in** | new: do not carry a payload through the column above a sensitive object |

Rules 1–4 are ports of behaviour that already runs and are parity-critical
(§6). Rule 5 is new, off by default, and is what the activation demo toggles.

Two parameter notes that are behaviour, not bookkeeping:

- Rule 1's `warning_r: 0.20` is a **floor**, matching
  `max(0.20, self._obstacle_radii.get(obs_name, 0.20))`. A grounded
  per-obstacle radius may expand it and may never shrink it.
- Rule 4 carries `omega_max: 0.25`, which is what the rotation channel
  *actually* runs at today (§1). Authoring `0.10` here would be a silent
  tightening dressed up as a port. The defect is recorded in the rule's
  `SKILL.md` body and left for a separate, measured change.

Rule 4 is also re-pointed semantically: `rule_composer.py:70` tested
`IS_SPILLABLE(obstacle)`, which locks the wrist because something *in the
scene* is spillable. Rule 4 tests the **carried payload** instead
(`subject: currently_grasped`), which is what the benchmark's rotation-lock
ground truth annotates. Parity is unaffected — the golden trace supplies the
angular channel directly — but the previous path stays reachable via
`FOL_RULE_SOURCE=composer` for comparison, and this change is called out
rather than buried.

Rule 2 is the in-library generalization evidence: one effect implementation,
one schema, a different `roles.subject`, and no new geometry code. Rule 1 and
rule 2 are not two implementations of avoidance; they are two bindings of one.

## 6. Acceptance bar: bit-exact parity

Every number in `fol_safety_filter/README.md` was produced by the hardcoded
loop. Parity is therefore the difference between a restructuring and a silent
change to all reported results.

`tools/fol_parity_check.py` drives `_apply_cbf` over a scripted deterministic
trajectory — EEF poses, obstacle poses, arm checkpoints, `u_cmd` sequence, no
simulator and no GPU — and writes a golden trace. The golden trace is generated
once from the `main` checkout and committed as
`fol_safety_filter/tests/data/parity_golden.npz`.

**Pass condition:** with rules 1–4 default-loaded and rule 5 absent, the
refactored filter reproduces the golden `u_safe` with `max|Δ| == 0` across the
env-var matrix `{FOL_ELLIPSOID, FOL_RAY_RESCALE ∈ {0,1,target}, FOL_TASK_BIAS}`,
for both oracle-named and `visual_`-prefixed obstacles (the latter exercises the
target-corridor relaxation).

Exact equality, not a tolerance: the refactor performs the same float
operations in the same order, so any nonzero difference is a behaviour change
and must be explained rather than absorbed.

## 7. Generalization demo

Axis: **same records, new scenes and payloads, same driver.** Chosen over
cross-benchmark or cross-policy transfer because it is fully offline and
therefore actually verifiable rather than asserted.

`prompt_tuning_benchmark_set/` supplies the fixture: 11 payload scenes
(`Cup`, `Candle`, `Knife`, `Sponge`, `Fish Tank`, `Soup`, `Plate`, `Bottle`,
`Bag of sugar`, `Headphones`, `Nothing`) with ground-truth annotations for
caution and rotation-lock, and 5 scenes with ground-truth unsafe spatial
relations. The object set is shared across the spatial scenes while the carried
payload changes, so the correct activation set changes while the rule records
do not.

`tools/fol_rule_transfer_demo.py`:

1. Loads the unmodified library and records each record's content hash.
2. For each scene, resolves bindings from an **oracle fact table**
   (`fol_safety_filter/safety_memory/fixtures/scene_facts.yaml`) — declared as
   oracle, so this measures rule generalization and not VLM perception. Facts
   are restricted to names already in `PREDICATE_REGISTRY`
   (`IS_SPILLABLE`, `IS_LIT`, `IS_FRAGILE`, `HAS_SHARP_PART`, `IS_FLAMMABLE`,
   `IS_FULL`, `IS_OPEN`, `IS_HEAVY`, `IS_WET`, `IS_ABSORBENT`); no new predicate
   is introduced, so rule 5's "overhead-sensitive" target set is expressed as
   `objects_with_fact:IS_FRAGILE`.
3. Evaluates activation per scene and compares with ground truth.
4. Reports, per rule: scenes where it fired, which object each role bound to,
   and precision / recall against the annotations.
5. Re-runs with rule 5 enabled to show manifest-level toggling.

Output: `results/safelibero/fol/rule_memory/transfer_report.{json,md}`.

Reported honestly as an activation-correctness measurement on an oracle-fact
fixture. It is not a task-success or violation-rate claim, and it does not go
in a table with TSR or CAR. Per `CLAUDE.md`, nothing here shares a column with
LIBERO-Safety SR.

## 8. Tests

| file | covers |
|---|---|
| `tests/test_rule_schema.py` | validator rejects each malformed class: unknown key, two effects, out-of-range param, bad formula, duplicate id, oversize file |
| `tests/test_rule_loader.py` | sorted-order load; reload reproduces identical compiled repr; one bad record fails the whole snapshot; mid-episode edits ignored |
| `tests/test_rule_binding.py` | one record over N obstacles yields N rules; `arm_checkpoints` binding; empty target set is a valid no-op |
| `tests/test_effect_dispatch.py` | each of the four effect handlers; effect ordering is obstacle-major; strictest-wins for the two limits |
| `tests/test_rule_falsifiability.py` | removing rule 2 from the manifest changes `rule_counts` **and** `u_safe` on an arm-only fixture |
| `tests/test_parity.py` | the §6 golden-trace comparison |
| `tests/test_rule_contracts.py` | executes every record's `tests.yaml` |

Run with `python -m pytest fol_safety_filter/tests vlm_prompt_runner/tests`,
the command `CLAUDE.md` already specifies.

## 9. Risks

**Parity failure on the ellipsoid or corridor branches.** These have the most
implicit state (`_obstacle_ellipsoids`, `_vision_targets_xy`). Mitigation: the
golden trace covers both, and the env-var matrix is part of the pass condition
rather than a follow-up.

**The loader becomes a file format over unchanged code.** Mitigation: §4.2 is a
test, not a note. If the numeric geometry still lives in `_apply_cbf`, that
test fails.

**Scope creep into the PoC protocol.** Mitigation: §2 names it out of scope and
this spec delivers no episode campaign.

## 10. Out of scope

pi05 and any second policy; the LIBERO-Safety adapter; repair procedures and
bounded transport repair; runtime rule induction and learned predicates; VLM
authoring of records (records here are human-authored); any GPU campaign.
