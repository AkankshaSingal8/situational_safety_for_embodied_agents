---
schema_version: 1
id: spillable_rotation_lock
revision: 1
description: Limit wrist rotation rate while carrying a payload that must stay upright.
roles:
  subject: {binding: currently_grasped}
applies_when: "IS_SPILLABLE(SUBJECT) OR IS_LIT(SUBJECT)"
requires:
  predicate: limit_angular_rate
  omega_max: 0.25
violation_action: slow
unknown_action: stop_and_report
---

# Rotation lock for upright-payload transport

While the gripper holds a payload that must stay level, the commanded angular
rate is clamped. A subject-only rule: it depends on what is being carried, not
on anything else in the scene.

## The activation formula was measured, not guessed

`IS_SPILLABLE(SUBJECT) OR IS_LIT(SUBJECT)`, scored against the rotation-lock
ground truth in `prompt_tuning_benchmark_set/prompt_tuning_rotation.json` over
all 11 payload scenes:

| formula | recall | false positives |
|---|---|---|
| `IS_SPILLABLE(SUBJECT)` | 5 / 7 | 0 |
| `IS_SPILLABLE(SUBJECT) OR IS_LIT(SUBJECT)` | **6 / 7** | 0 |

The disjunct is required rather than convenient. A lit candle is honestly
`IS_FULL: false` — it is not a filled container, and its tilt hazard is the
flame, not a spill. Keying only on spillability misses it.

`Plate` is the remaining miss and the fixture was **not** adjusted to close
it. The annotation marks a plate rotation-locked because tilting slides food
off a flat surface, which needs a predicate for an unsecured load on a flat
surface; `PREDICATE_REGISTRY` has none. Asserting `IS_FULL: true` of a bare
plate to hit the annotation would buy a better number by making the fixture
describe something untrue.

## Two ported defects, deliberately not fixed here

**The authored parameter name never worked.** `rule_composer.py:76` authors
`{"angular_limit": 0.1}` while `cbf_mapper.py:184` reads
`params.get("omega_max", 0.25)`. Every rotation lock in the live path has run
at 0.25 rad/s and the authored 0.1 was unreachable. This record carries
`omega_max: 0.25` — what actually runs. Writing `0.10` here would be a silent
tightening dressed as a port. Under this schema the same misspelling is now a
load error rather than a shrug.

**The rule it replaces could never fire.** `_props_from_name` marks
`IS_SPILLABLE` on names containing pot/bottle/cup/bowl/mug/milk
(`vlm_grounder.py:650`), but `_heuristic_ground` sets `is_open` only for
bowl/cup/glass/mug (`filter.py:74`). For `moka_pot_obstacle` the composer
therefore authored `SPILLABLE_LOCK_MOKA_POT_OBSTACLE` whose own precondition
is permanently false. Measured: `IS_SPILLABLE(moka_pot_obstacle)` is False,
`angular_limit` stays None, the channel is inert. Both defects are pinned by
`tests/test_composer_channel_status_quo.py` rather than repaired, because
repairing either changes behaviour and every reported number was produced with
both present.

## Re-pointed from the obstacle to the payload

The composer tested `IS_SPILLABLE(obstacle)` — it locked the wrist because
something *in the scene* was spillable. That is the wrong subject: a wrist
rotation spills what the wrist is holding. This record tests the carried
payload, which is also what the benchmark annotates.

That makes it a semantic change, not a port, which is why it is
`default_loaded: false`. The composer's obstacle-pointed rule is not removed:
it is still authored every episode and still feeds the rotation channel
through `cbf_mapper`, so leaving this record off changes nothing. Enabling it
is a separate, measured decision — and cheap to justify, since the channel it
would join does not currently fire on any reported obstacle.

## Limitations

An angular-rate clamp is not an orientation constraint. It limits how fast the
wrist may rotate, not how far it has already rotated, so a slow steady tilt is
unconstrained — for an actual tilt bound the constraint is
`h = (R a) · z − cos(theta_max)`, which is a different effect predicate and is
not implemented here. Nothing in this rule simulates fluid, so it does not
establish that no liquid left the container. `IS_LIT` likewise buys no thermal
guarantee.
