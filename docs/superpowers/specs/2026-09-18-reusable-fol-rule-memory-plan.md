# Implementation Plan — Reusable FOL Rule Memory

Design: `2026-09-18-reusable-fol-rule-memory-design.md`
Branch: `memory`, worktree `.worktrees/memory`

Ordering rule: **T0 must complete before any edit to `filter.py`.** The golden
trace can only be captured from the unmodified implementation.

---

## T0 — Parity harness and golden trace

Files: `tools/fol_parity_trace.py`, `fol_safety_filter/tests/data/parity_golden.npz`

1. Write `tools/fol_parity_trace.py` exposing `build_scenarios()` returning a
   deterministic list of scenario dicts. Each scenario fixes: obstacle names
   (mix of oracle-named and `visual_`-prefixed), obstacle positions, optional
   ellipsoid entries, `_vision_targets_xy`, arm checkpoints, and a list of
   `(ee_pos, ee_quat, u_cmd)` steps. Use a seeded `numpy.random.default_rng(0)`
   and round all inputs to float64 literals stored in the file, so the scenario
   set is stable under any numpy version.
2. Drive `FOLSafetyFilter._apply_cbf` directly (construct the filter, set
   `_obstacle_names`, `_obstacle_radii`, `_obstacle_ellipsoids`,
   `_vision_targets_xy`, `_object_states` by hand; no simulator, no policy).
   Build `MappedCBFSet` explicitly for the velocity/rotation channels.
3. Sweep the env-var matrix: `FOL_ELLIPSOID ∈ {0,1}`,
   `FOL_RAY_RESCALE ∈ {0,1,target}`, `FOL_TASK_BIAS ∈ {0,1}`.
4. `--emit <path>` writes `u_safe` for every (scenario, step, env-combo) to npz.
   `--check <path>` recomputes and asserts `max|Δ| == 0`, printing the first
   mismatch's scenario/step/env-combo on failure.
5. Generate the golden trace against the **unmodified** implementation:
   `PYTHONPATH=<main checkout> python tools/fol_parity_trace.py --emit fol_safety_filter/tests/data/parity_golden.npz`
   run from the worktree but importing `fol_safety_filter` from the main
   checkout at `75ca90d4`.
6. Commit the harness and the npz.

**Verify:** `--check` passes against the main checkout, and the npz contains a
nonzero number of steps where `u_safe != u_cmd` for every env-combo (a golden
trace of no-ops would prove nothing). Assert this in `tests/test_parity.py`.

---

## T1 — Contract schema and validator

Files: `fol_safety_filter/rule_memory/__init__.py`,
`fol_safety_filter/rule_memory/schema.py`,
`fol_safety_filter/tests/test_rule_schema.py`

1. `RuleRecord` dataclass: `id`, `revision`, `description`, `roles`,
   `applies_when`, `requires` (an `EffectSpec`), `violation_action`,
   `unknown_action`, `source_path`, `front_matter_hash`, `body_hash`.
2. `EffectSpec`: `predicate` (closed enum of four) plus typed params per §3.1.
3. `parse_skill_md(text, path) -> RuleRecord`: split the single YAML front
   matter block from the Markdown body; `yaml.safe_load` only; reject custom
   tags, aliases, and anchors.
4. `validate(record)` enforcing every rule in design §3.2, raising
   `InvalidRuleError` with the offending field named.
5. Reuse `kb.parse_formula` to validate `applies_when` and to check that every
   referenced predicate is in `PREDICATE_REGISTRY` and that role tokens are
   limited to `SUBJECT` / `TARGET`.

**Verify:** `pytest fol_safety_filter/tests/test_rule_schema.py` — one test per
malformed class (unknown key, two effects, out-of-range param for each param,
unparseable formula, unknown predicate, duplicate id, non-integer revision,
oversize file, missing front matter, two front-matter blocks, YAML alias).

---

## T2 — Memory store loader and binding compiler

Files: `fol_safety_filter/rule_memory/store.py`,
`fol_safety_filter/rule_memory/binding.py`,
`fol_safety_filter/tests/test_rule_loader.py`,
`fol_safety_filter/tests/test_rule_binding.py`

1. `load_library(root, enabled_ids=None) -> RuleLibrary`: read
   `MANIFEST.yaml`, load every `<id>/SKILL.md` in sorted id order, validate
   all, and fail the snapshot on any single invalid record. `enabled_ids`
   overrides the manifest for tests and for the rule-5 toggle.
2. `RuleLibrary.compiled_digest()` — a stable hash of the compiled
   representation, for the reload-identity test.
3. `binding.py`: `resolve(record, scene) -> List[FOLRule]` where `scene` carries
   `obstacle_names`, `arm_checkpoints`, `grasped`, and a fact lookup. One record
   over N targets yields N `FOLRule`s with `bindings={"SUBJECT":…, "TARGET":…}`
   and `name=f"{record.id}::{target}"`. Preserve `obstacle_names` order and
   carry `target_index` and `subject_rank` onto each rule for §4.1 ordering.
4. Empty target set yields zero rules and is not an error.

**Verify:** loader tests (sorted order; reload gives an identical
`compiled_digest()`; one bad record raises and loads nothing; mid-episode file
change is ignored) and binding tests (N obstacles → N rules; arm-checkpoint
binding; empty target no-op; `target_index`/`subject_rank` correct).

---

## T3 — Author the five records

Files: `fol_safety_filter/safety_memory/MANIFEST.yaml`, and per rule
`<id>/{SKILL.md,tests.yaml,metadata.json}` for the five ids in design §5.

1. Rules 1–4 must encode exactly the constants the unmodified `_apply_cbf`
   uses; read them off `filter.py` rather than from this plan.
2. Rule 5 `overhead_exclusion` is listed in the manifest with
   `default_loaded: false`.
3. Each `tests.yaml`: at least one positive, one negative, and one boundary
   activation case, expressed as scene facts plus expected activation.
4. Each `metadata.json`: `revision`, `author: human`, `provenance` naming the
   `filter.py` line the constants came from, `evidence: none` for rule 5.
5. `SKILL.md` bodies state the rule's intent and its limitations — including
   that these are geometric proximity rules and prove nothing about spillage,
   temperature, or grasp security.

**Verify:** `pytest fol_safety_filter/tests/test_rule_contracts.py` executes
every `tests.yaml`; the manifest lists exactly five ids.

---

## T4 — Route the filter through the library

Files: `fol_safety_filter/filter.py`, `fol_safety_filter/rule_memory/effects.py`,
`fol_safety_filter/tests/test_effect_dispatch.py`

1. `effects.py`: one handler per effect predicate. `avoid_sphere` delegates to
   the existing `_apply_point_cbf` with parameters taken **from the record**.
   `avoid_region_above` is new: a vertical-column projection reusing the
   footprint logic in `cbf_mapper._above_sq`.
2. `setup_episode`: after obstacle grounding, `load_library(...)` then
   `binding.resolve(...)` for each record, and add the resulting rules to the
   KB. Keep `rule_composer.compose_rules` reachable behind
   `FOL_RULE_SOURCE=composer` so the previous path stays available for
   comparison; default is `FOL_RULE_SOURCE=memory`.
3. `_apply_cbf`: replace the hardcoded `for obs_name in self._obstacle_names`
   block with a dispatch over the active constraints, ordered per design §4.1.
   No radius, hard radius, or push constant may remain literal in `filter.py`.
4. Preserve env-var precedence exactly (`FOL_ELLIPSOID`, `FOL_RAY_RESCALE`,
   `FOL_TASK_BIAS`, `FOL_CHUNK_REPLAN`).

**Verify:**
- `python tools/fol_parity_trace.py --check fol_safety_filter/tests/data/parity_golden.npz` → `max|Δ| == 0`.
- `grep -nE "0\.20|0\.10|0\.08|0\.15|0\.06" fol_safety_filter/filter.py` returns
  no geometry literal inside `_apply_cbf`.
- `pytest fol_safety_filter/tests/test_effect_dispatch.py`.

**If parity fails, stop and report the first mismatching scenario.** Do not
adjust the golden trace.

---

## T5 — Falsifiability test

File: `fol_safety_filter/tests/test_rule_falsifiability.py`

Construct a fixture where an arm checkpoint is inside its warning radius while
the EEF is outside every radius. Assert that with `arm_link_obstacle_avoid`
enabled, `u_safe != u_cmd` and `rule_counts` contains the arm rule; and that
with it disabled via `enabled_ids`, `u_safe == u_cmd` and the arm rule is
absent. Repeat for `eef_obstacle_avoid` with an EEF-only fixture.

**Verify:** the test fails if `_apply_cbf` is reverted to the hardcoded loop —
check this by asserting on both directions, not only the enabled case.

---

## T6 — Generalization demo

Files: `tools/fol_rule_transfer_demo.py`,
`fol_safety_filter/safety_memory/fixtures/scene_facts.yaml`,
`results/safelibero/fol/rule_memory/transfer_report.{json,md}`

1. `scene_facts.yaml`: for each of the 11 `prompt_tuning_benchmark_set` scenes,
   the oracle Boolean facts for the payload and each object
   (`IS_SPILLABLE`, `IS_LIT`, `IS_FRAGILE`, `HAS_SHARP_PART`, `IS_FLAMMABLE`,
   `overhead_sensitive`). Header comment states these are oracle facts, not
   perception output.
2. The demo loads the unmodified library once, prints each record's
   `front_matter_hash`, and for every scene resolves bindings and evaluates
   activation.
3. Score against `prompt_tuning_{caution,rotation,spatial}.json`: per rule,
   scenes fired, bound objects, and precision / recall.
4. Re-run with rule 5 enabled and report the activation delta.
5. Write the report; the Markdown version states in its first line that these
   are activation-correctness numbers on an oracle-fact fixture, not TSR/CAR,
   and shares no table with LIBERO-Safety metrics.

**Verify:** the demo runs CPU-only in under a minute; every record hash printed
for scene 1 is identical to the hash printed for scene 11 (same bytes, no
per-scene edit); the report contains real counts, not placeholders.

---

## T7 — Documentation and integration

1. `fol_safety_filter/safety_memory/README.md`: schema reference, how to add a
   record, the strict-validation contract.
2. Update `fol_safety_filter/README.md`: describe the rule-memory path, state
   that the reported result table was produced by the pre-refactor geometry and
   is preserved bit-exactly by parity.
3. Update the root `CLAUDE.md` layout table with `safety_memory/`.
4. Full suite: `python -m pytest fol_safety_filter/tests vlm_prompt_runner/tests`.
5. Commit in the task order above; push `memory` to `origin`.

**Verify:** full suite green, parity check green, `git log --oneline` shows one
commit per task, `git push -u origin memory` succeeds.

---

## Stop conditions

Stop and report rather than improvising if: the golden trace cannot be
generated from the main checkout; parity fails and the cause is not an obvious
transcription error in a record; the full test suite has pre-existing failures
on `main` (record which, do not fix them here).
