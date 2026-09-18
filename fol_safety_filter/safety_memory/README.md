# Safety memory — the rule store

This directory *is* the memory. Each subdirectory holds one safety rule as
data: a versioned contract the filter loads at episode setup, not code.

Design: `docs/superpowers/specs/2026-09-18-reusable-fol-rule-memory-design.md`.

## Why this exists

Before this store, the filter had one monolithic obstacle rule. `certify()`
evaluated a knowledge base and mapped it to CBFs, then `_apply_cbf` ignored the
mapped spatial content and ran its own loop over `self._obstacle_names` with
literal `0.20` / `0.10` / `0.08` geometry. Deleting a rule from the knowledge
base did not stop obstacle avoidance — it only stopped it being *counted*.

Three consequences, all of which this store is meant to remove:

1. A rule could not be authored, reviewed, versioned, or transferred as data.
2. A rule's parameters were an open untyped dict, so a misspelled key was
   silent. `rule_composer.py:76` authors `{"angular_limit": 0.10}`; the
   consumer reads `params.get("omega_max", 0.25)` (`cbf_mapper.py:184`). Every
   rotation lock in the live path runs at the default and the authored value is
   unreachable.
3. A rule could contradict its own precondition undetected. For
   `moka_pot_obstacle` the composer authors a rotation lock and the rule's
   `IS_SPILLABLE` precondition is permanently false, because
   `_props_from_name` keys on `pot` while `_heuristic_ground` sets `is_open`
   only for bowl/cup/glass/mug.

Both defects 2 and 3 are pinned by
`fol_safety_filter/tests/test_composer_channel_status_quo.py` rather than
fixed, since fixing either changes behaviour and the reported results were
produced with both present.

## Layout

```
MANIFEST.yaml            which rule ids exist and which load by default
fixtures/scene_facts.yaml  oracle facts for the offline transfer demo
<rule_id>/
  SKILL.md               YAML front matter = the contract; body = rationale
  tests.yaml             positive / negative / boundary activation cases
  metadata.json          revision, provenance, evidence
```

The Markdown body of `SKILL.md` is documentation. It is never executed and
never parsed for semantics. Front matter and body are hashed separately, so a
rule shown to transfer between scenes can be shown to be the *same bytes*.

## The contract

One record = one activation condition + **one** effect. A record carrying two
effects is invalid, and that restriction is the whole point: it is what makes
the rules separable.

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
violation_action: avoid
unknown_action: stop_and_report
---
```

`applies_when` is a formula in the existing `kb.py` mini-language — no new
language was introduced, `parse_formula` is reused. `SUBJECT` and `TARGET` are
the only reserved tokens; the binding resolver substitutes them.

### Role bindings

| binding | resolves to |
|---|---|
| `eef` | the end-effector point |
| `arm_checkpoints` | one instance per monitored arm body |
| `currently_grasped` | the carried payload, or nothing |
| `obstacles` | each grounded obstacle, in grounding order |
| `objects_with_fact:<F>` | each object whose fact `F` is true |

`objects_with_fact` uses three-valued logic. An object whose fact is *unknown*
is not selected, and is reported as an unknown rather than silently treated as
false.

### Effect predicates

| predicate | params |
|---|---|
| `avoid_sphere` | `warning_r`, `hard_r`, `push_hard`, `shape`, `cancel_scale_on_target_corridor` |
| `limit_speed` | `v_max` |
| `limit_angular_rate` | `omega_max` |
| `avoid_region_above` | `clearance_m`, `z_max`, `cancel_scale` |

Four effects carry five rules: `avoid_sphere` is shared by two records that
differ only in `roles.subject`. That is the generalization claim in miniature —
one effect implementation, a different binding, no new geometry code.

## Validation is strict and all-or-nothing

A single invalid record fails the whole snapshot. An unknown key, an
out-of-range parameter, a second effect, an unparseable formula, an unknown
predicate, a duplicate id, a file over 16 KiB, or a YAML anchor is an error —
never a quietly skipped record.

This is deliberate. `CLAUDE.md` warns that removing
`vlm_pipeline/semantic_cbf_filter.py` does not raise but "silently downgrades
the filter to geometry-only and changes every reported number". Silent
degradation is the specific failure this store refuses.

## Adding a rule

1. `mkdir <rule_id>` and write `SKILL.md`, `tests.yaml`, `metadata.json`.
2. Add `{id: <rule_id>, default_loaded: false}` to `MANIFEST.yaml`. Start
   opt-in; a record that changes a channel's selection behaviour should not
   become the default until that change has been measured.
3. Give `tests.yaml` a positive, a negative, and a boundary case. A rule that
   fires everywhere is not a safety rule.
4. Run `python -m pytest fol_safety_filter/tests -q`.

Editing a record's parameters is a `revision` bump. The parameters are
load-bearing: changing `warning_r` here changes what the robot does, and
`test_rule_falsifiability.py` exists to keep that true.

## What these rules do not establish

Every record here is a geometric or kinematic constraint over proximity, speed,
orientation rate, and occupied volume. None of them proves spill prevention,
thermal safety, or grasp security. A fact named `IS_SPILLABLE` is a semantic
label, not a fluid simulation. The rules constrain the robot's motion relative
to labelled objects, and that is the entire claim.
