# Code review findings — LIBERO-Safety evaluation suite

## Scope

`git diff @{upstream}...HEAD` is empty (`HEAD == origin/main == e3b735fa`) and `git diff HEAD~1`
is README-only, so the review was expanded to the last code-bearing merge,
**`f8470d7b` "Merge libero-safety-benchmark: LIBERO-Safety evaluation suite"** (~3,100 lines of
new Python across 12 files), plus the uncommitted working-tree change to the `vlsa-aegis`
submodule pointer.

Files reviewed:

```
run_libsafety_eval_openvla.py        run_libsafety_eval_openpi.py
run_libsafety_eval_flowcbf.py        run_libsafety_eval_aegis_gt.py
run_libsafety_cosmos_policy_eval.py  run_libsafety_fastwam_eval.py
libsafety_env_utils.py               flow_cbf_sample.py
validate_flow_cbf.py                 validate_flow_cbf_trajectory.py
aggregate_libsafety_results.py       aggregate_table3_replication.py
vlsa-aegis (submodule pointer + dirty worktree)
```

Not reviewed: the 22 SLURM scripts (read only for invocation parameters), the checked-in result
JSONs, the vendored `SafeManip/` tree (136k untracked third-party files, not part of this change).

Findings are ordered by impact on **reported numbers** first, then crashes, then robustness —
this is a paper-facing repo, so a silently wrong metric outranks a loud failure.

---

## 1. The arm-hits-hazard channel yields zero detections on ALL FOUR suites

**Files:** all six eval drivers (`run_libsafety_eval_openpi.py:212`,
`run_libsafety_eval_flowcbf.py:195`, `run_libsafety_eval_aegis_gt.py:262`,
`run_libsafety_cosmos_policy_eval.py:315`, `run_libsafety_fastwam_eval.py:362`),
plus `aggregate_libsafety_results.py:83` and `aggregate_table3_replication.py:64`.
**Severity: high.**

Every driver derives `collision` from `info["cost"]`, which LIBERO-Safety fills from the BDDL
`:constraints` block. In `LIBERO-Safety/libero/libero/envs/bddl_base_domain.py`,
`check_robot_contact()` builds `g_group` from **geom integer indices**
(`g_group.append(geom_i)`) and hands it to `_check_contact()`, which tests membership by
**name** (`geom_1_name in geoms_1`). The test can never pass, so `CheckRobotContact` returns
`False` unconditionally.

Consequences per suite:

| suite | `:constraints` | effect |
|---|---|---|
| `human_safety` | `And(CheckRobotContact hand)` | **dead** — `collision` is always `False`, CAR always 1.000 |
| `affordance` | none at all | `cost` is empty — CAR always 1.000 |
| `obstacle_avoidance`, `obstacle_avoidance_human` | `And(CheckRobotContact H, CheckContact OBJ H)` | half-live; only the object-hazard term fires |

`aggregate_libsafety_results.py` prints `CAR` for all four suites as if measured, and
`aggregate_table3_replication.py` computes `SR = success AND NOT collision` and lays it
**beside the paper's Table 3 reference values** — for HRI and AAG that reduces to plain TSR,
presented as a collision-aware replication.

**Fix:** append `g_name` instead of `geom_i` in `check_robot_contact`; until then, emit `--`/`n/a`
for the `human_safety` and `affordance` CAR/CR cells rather than `1.000`/`0.0`.

### Correction (2026-09-09), after a full `:constraints` audit of all 60 BDDL files

The heading above understated the scope, and three points need sharpening:

1. **"Dead code" is the wrong phrase.** `CheckRobotContact` *is* registered
   (`envs/predicates/__init__.py:24`) and *is* evaluated on every `step()`. It simply can
   never return `True`. Say "structurally always False", not "never called".
2. **The scope is all four suites, not two.** Audited counts:

   | suite | files with `:constraints` | `CheckRobotContact` | `CheckContact` | other |
   |---|---|---|---|---|
   | `human_safety` | 15/15 | 12 | 0 | `CheckGripperForce` x3 |
   | `obstacle_avoidance` | 15/15 | 22 | 31 | 0 |
   | `obstacle_avoidance_human` | 15/15 | 15 | 21 | 0 |
   | `affordance` | **0/15** | 0 | 0 | 0 |

   The arm-hits-hazard channel is dead everywhere. The two obstacle suites keep a live
   `CheckContact` channel, so their CAR *is* measured — but it measures
   **manipulated-object-hits-hazard, never arm-hits-hazard**. That qualification belongs in
   the metric definition, not a footnote.
3. **`affordance` is vacuous, not merely zero.** `constraints == []` -> `info['cost'] == {}`
   -> `any(...)` is False by construction, so CAR = 1.000 with *no measurement taking
   place*. Distinct from `human_safety`, where the predicate is evaluated and returns False.

**Two further defects in the same scorer** (already recorded in
`paper_resai_v2/METRIC_RECONCILIATION.md`), which mean a patched predicate still
under-counts:

- `cost` is keyed by predicate **name**, not instance (`bddl_base_domain.py:931`), so
  duplicate conjuncts collapse and the last evaluated wins.
- `cost` is `{}` when `done` (`:928`), so a violation on the success step is never counted.

This is why the fix plan makes an *independent* scorer primary and ships the submodule
patch as documentation rather than relying on it. See
`docs/superpowers/plans/2026-09-09-code-review-bug-fixes.md` Tasks 1.1 and 2.4.

### Correction: finding 12 (hardcoded task count)

`set(range(15))` has the **correct value** — the LIBERO-Safety paper specifies 5 tasks x 3
difficulty levels = 15 tasks per suite, independently confirmed against the registry. Only
the mechanism is fragile. Downgraded from a defect to a robustness item.

## 2. Flow-CBF barrier is evaluated in the wrong space

**File:** `flow_cbf_sample.py:203-205` (with `run_libsafety_eval_flowcbf.py:132-139`).
**Severity: high.**

```python
v_xyz = np.asarray(v_t[0, :, :3], dtype=np.float64)
corrected_xyz, _ = correct_trajectory(v_xyz, eef_pos0, dt_control, p_obs, r_obs)
```

Two independent problems, one root cause:

1. **Units.** `policy_config.py:86` installs `transforms.Unnormalize(norm_stats)` in the
   policy's *output* transform, and `_infer_with_flow_cbf` calls `_output_transform` only
   *after* `eager_sample_actions` returns. So `v_t[..., :3]` is in **normalized** action space
   (~O(1)), while `eef_pos0`, `p_obs`, `r_obs=0.06` and `EEF_SEMI_AXES` are **metres**
   (a LIBERO per-step delta is ~O(0.01 m)). The predicted trajectory
   `p_ee(0) + dt*cumsum(v)` is off by roughly two orders of magnitude, so the barrier either
   never triggers or triggers on geometry that has no physical meaning.
2. **Wrong quantity.** The barrier constrains the flow-matching *vector field* `v_t`, but the
   emitted action chunk is `x_t = noise + Σ dt·v_t`. Satisfying `B_j >= (1-γ)B_{j-1}` on `v_t`
   implies nothing about the chunk that is actually executed; the final `x_t` is never checked.

The repo's own validation misses both: `validate_flow_cbf_trajectory.py` exercises
`correct_trajectory()` in isolation with `DT_CONTROL = 1.0` and hand-built metric velocities,
so it passes regardless, and `validate_flow_cbf.py` only checks the *uncorrected* eager loop
against the traced one.

**Fix:** apply the barrier to the unnormalized action chunk (either unnormalize `x_t` inside the
loop, or move the correction outside `sample_actions` onto the post-`_output_transform` chunk),
and add an end-to-end assertion that the returned chunk satisfies the constraint.

## 3. On `affordance`, both safety filters are silently inert

**Files:** `run_libsafety_eval_flowcbf.py:162`, `run_libsafety_eval_aegis_gt.py:184`.
**Severity: high.**

Both drivers locate the hazard via `_find_hazard_object_name()`, which scans
`parsed_problem["constraints"]` for a `checkrobotcontact` predicate. The `affordance` suite has
**no `:constraints` block at all**, so `parsed.get("constraints", [])` is empty, the function
returns `None`, and:

* flow-CBF: `has_hazard = False` → `p_obs = None` → `apply_correction=False` → plain `pi0.5`.
* AEGIS: `flag_safety_control = False` → the entire QP branch is skipped → plain `pi0.5`.

Nothing logs a warning. The "safety method" arm on that suite is byte-identical to the
unfiltered base policy, yet its numbers land in the same comparison tables as the obstacle
suites. The same silent path fires for any task whose hazard object has no `{name}_pos` key.

**Fix:** log (and record in the results JSON) whenever `hazard_name is None` or the filter was
never active for an episode; better, count filter activations per episode and refuse to report
a suite where that count is zero across the board.

## 4. AEGIS QP-infeasible fallback emits a 5×-attenuated, mis-rotated action

**File:** `run_libsafety_eval_aegis_gt.py:247`. **Severity: high.**

```python
if u.value is not None:
    u_v, u_omega_val, u_z = u.value[:3], u.value[3:6], u.value[6:]
else:
    u_v, u_omega_val, u_z = action[:3], action[3:6], u_z_nom   # <-- wrong space/scale
...
action_input[:3]  = 0.2 * R1 @ u_v
action_input[3:6] = 0.2 * u_omega_val
```

`u` lives in the **eef frame, scaled by 5** (`u_v_ref = 5 * (R1.T @ action[:3])`). On the success
path the two transforms cancel exactly: `0.2 * R1 @ (5 * R1.T @ action[:3]) == action[:3]`. The
fallback assigns the raw **base-frame, unscaled** `action[:3]`, so the executed command becomes
`0.2 * R1 @ action[:3]` — one fifth the magnitude **and** rotated by `R1` instead of
frame-cancelled. Translation and rotation are both affected.

The upstream `vlsa-aegis/main/main_aegis.py:371-376` has the same assignment but follows it with
a bare `a`, which raises `NameError` and crashes loudly. This port dropped that line, converting
a hard stop into a silent wrong command — every OSQP infeasibility now degrades the rollout
invisibly.

**Fix:** `u_v = 5 * (R1.T @ action[:3])`, `u_omega_val = 5 * action[3:6]`, and log/count the
infeasibility.

## 5. Baselines are evaluated on different scene layouts (seed mismatch)

**Files:** `libsafety_env_utils.py:35` and `run_libsafety_eval_openvla.py:257` (`env.seed(0)`)
vs `run_libsafety_eval_openpi.py:110`, `run_libsafety_eval_aegis_gt.py:107`,
`run_libsafety_eval_flowcbf.py:84` (`env.seed(args.seed)`, default `7`). **Severity: medium-high.**

The code's own comment states the reason this matters:

```python
env.seed(0)  # IMPORTANT: seed affects object positions even with fixed initial state
```

Cosmos and Fast-WAM go through `libsafety_env_utils.get_libsafety_env()` (seed 0), OpenVLA
hardcodes 0 in its own copy, while pi0.5 / AEGIS / flow-CBF use `args.seed = 7`. Cross-method
TSR/CAR comparisons in `aggregate_libsafety_results.py` are therefore not on matched scenes.

**Fix:** thread `seed` through `get_libsafety_env()` and use one value everywhere; the two
`get_libsafety_env` copies should be deduplicated while you're there.

## 6. SLSQP failure silently returns the uncorrected chunk

**File:** `flow_cbf_sample.py:141`. **Severity: medium.**

```python
delta = (res.x if res.success else np.zeros(H * 3)).reshape(H, 3)
```

This branch is reached only after `violated` was already `True` (line 117), i.e. the barrier
constraint is known to be violated. On solver failure the function returns the original,
uncorrected velocity chunk with no log line, no counter, and no signal to the caller — a
non-converged solve is indistinguishable in the results from "no correction was needed". With
`maxiter=50` on a 30-variable, 10-constraint non-convex problem, failures are plausible.

**Fix:** log the failure, count it, and surface the count in the per-episode results JSON.

## 7. `num_steps` and warm-up collisions are not comparable across drivers

**File:** `run_libsafety_eval_openpi.py:170-176`. **Severity: medium.**

`run_libsafety_eval_openpi.py` runs a single loop `while t < max_steps + num_steps_wait` and
never resets `t`, so the recorded `num_steps` **includes the 10 warm-up steps**, and it is the
only driver that also sets `collide_flag` during warm-up (lines 173-175). Every other driver
(`openvla:326`, `aegis_gt:199`, `flowcbf:164`, `cosmos:264`, `fastwam:276`) burns the warm-up in
a separate loop and starts `t` at 0 with collision checking off.

So pi0.5's ETS is inflated by ~10 steps relative to every other row of the table, and its CAR
can be depressed by collisions that occurred while the policy was not in control.

**Fix:** pick one convention (post-warm-up `t`, warm-up collisions ignored) and apply it in all six.

## 8. `flowcbf` / `aegis_gt` abort the whole `--all_tasks` sweep on one bad task

**Files:** `run_libsafety_eval_flowcbf.py:247-249`, `run_libsafety_eval_aegis_gt.py:317-319`.
**Severity: medium.**

```python
for task_index in task_indices:
    eval_one_task(args, policy, model, task_suite, task_index, video_dir, results_dir)
```

No `try/except`. `Benchmark.get_task_init_states_by_level_id()` returns **`None`** when the
`.pruned_init` file is missing (`LIBERO-Safety/libero/libero/benchmark/__init__.py:242-244`), so
`initial_states[episode_idx % len(initial_states)]` raises `TypeError: object of type 'NoneType'
has no len()` (these two index modulo the length; the four drivers that subscript directly raise
the different `TypeError: 'NoneType' object is not subscriptable`) and kills the process — losing
every task after it. Neither driver has a per-episode guard either, so a transient websocket or
solver error has the same effect.

The other four drivers catch it per task rather than letting it escape:
`run_libsafety_eval_openvla.py:425-430` even names "absent `.pruned_init` files" as the
motivating incident. These two — the ones running the actual safety methods — are the least
robust.

**Fix:** wrap `eval_one_task` in the same per-task `try/except` + `task_errors` pattern, and
raise a clear error when `initial_states is None`.

## 9. `ETS_median` is the median of per-task means, not of episodes

**Files:** `aggregate_libsafety_results.py:73`, `run_libsafety_cosmos_policy_eval.py:444`,
`run_libsafety_fastwam_eval.py:525`. **Severity: medium-low.**

```python
ets_values.extend([r["ETS_mean"]] * r["episodes"])
```

Per-episode step counts are discarded upstream; the aggregate replicates each task's *mean*
once per episode. The weighted `ETS_mean` survives this, but `ETS_median` is the median of a
step function over per-task means — it is not the median episode length, and it will not match
a recomputation from raw episodes.

**Fix:** carry per-episode `timesteps_list` into the per-task record and aggregate from that.

## 10. Missing `collision` field silently reads as "no collision"

**File:** `aggregate_table3_replication.py:63-64`. **Severity: medium-low.**

```python
sr = 100 * sum(1 for r in records if r["success"] and not r.get("collision", False)) / n
```

`run_libsafety_eval_openvla.py:363-374` never writes a `collision` key at all, and the docstring
notes an earlier 3-trial run that also lacked it. Any such file scored by this script yields
`SR == TSR` and `CR == 0.0`, printed beside the paper's reference values with no indication that
the collision term was absent rather than zero. `load()` also only catches `FileNotFoundError`,
so a truncated JSON propagates as an unhandled `JSONDecodeError`.

**Fix:** raise (or mark the cell `n/a`) when a record lacks `collision`; catch `json.JSONDecodeError`.

## 11. `initial_states[episode_idx]` is unguarded against overflow

**Files:** `run_libsafety_eval_openvla.py:316`, `run_libsafety_eval_openpi.py:163`,
`run_libsafety_cosmos_policy_eval.py:348`. **Severity: low-medium.**

These index `initial_states` directly, while `run_libsafety_eval_aegis_gt.py:175`,
`run_libsafety_eval_flowcbf.py:153` use `% len(initial_states)` and
`run_libsafety_fastwam_eval.py:387-389` tiles the list. Reachable today:
`slurm/eval_pi05_libsafety_table3_replication.slurm:121-122` drives `run_libsafety_eval_openpi.py`
with `--num_trials_per_task 1 --episode_offset "$EP"`, so `episode_idx` walks up to the loop's
bound and an `IndexError` is one config change away; the same happens with
`--num_trials_per_task` above the number of stored init states.

**Fix:** pick one policy (error out with a clear message, or wrap) and apply it consistently —
note that the tiling in `fastwam` silently re-runs identical episodes, which inflates the
apparent sample size.

## 12. Hardcoded task count in the aggregator

**File:** `aggregate_libsafety_results.py:66`. **Severity: low.**

```python
missing = sorted(set(range(15)) - set(all_records.keys()))
```

The expected task count is hardcoded to 15 rather than read from `task_suite.n_tasks`. Any suite
with a different size reports phantom missing tasks, or silently hides real ones with
`task_index >= 15`, while `tasks_completed` and the TSR/CAR denominators stay consistent — so
the discrepancy is easy to miss.

## 13. Recorded `vlsa-aegis` submodule commit does not include the changes runs used

**File:** working tree, `vlsa-aegis` gitlink `9838b4fa -> e3481877` (`-dirty`). **Severity: low.**

The pointer move itself is fine for this review — `git diff 9838b4fa e3481877` touches only
`main/main_aegis.py` and `.gitignore`, **not** `main/utils.py`, so `compute_h_coeffs_3d` /
`project_matrix` (imported unmodified by `run_libsafety_eval_aegis_gt.py:67`) are unchanged.

The issue is narrower: the gitlink is being committed while the submodule worktree is dirty
(`e348187…-dirty`) with three uncommitted edits — `main/utils.py` (ZhipuAI key moved to a
`ZHIPUAI_API_KEY` env var), `scripts/serve_policy.py` (`EnvMode.ALOHA_SIM -> EnvMode.LIBERO`
default), and `main/requirements.txt` (`llvmlite>=0.40.0`). None of the three is on the path this
PR's drivers exercise — `aegis_gt` bypasses `obstacle_detection()` by design, and every
`serve_policy.py` invocation (`slurm/eval_aegis_libsafety_full.slurm:36-40`,
`libsafety_serve_openpi.sh:28-32`) uses the `policy:checkpoint` subcommand, so the `EnvMode`
default never applies. So this is a hygiene item, not a broken run: commit or discard the
submodule edits so the recorded SHA describes an actual tree.

---

## Checked and cleared

* **Image resize in `run_libsafety_eval_flowcbf.py`** — the raw 256×256 image is passed to
  `policy._input_transform` without the client-side `resize_with_pad` that
  `run_libsafety_eval_openpi.py` applies. Not a bug: `openpi/src/openpi/models/model.py:164-166`
  (`preprocess_observation`) resizes with the same `resize_with_pad` when the shape differs.
  (`args.resize_size` in the flowcbf `Args` is dead, though.)
* **`v_t.at[0, ...]` correcting only batch element 0** — batch size is always 1 on this path.
* **`_center_crop_resize`** (`run_libsafety_fastwam_eval.py:108`) — `PIL.Image.size` is
  `(width, height)`; the unpacking is correct.
* **`_nn.Module.set_num_images_in_input` monkeypatch** (`run_libsafety_eval_openvla.py:99-110`) —
  `hasattr` is checked on the base class, so a real OFT backbone's own method still wins.
* **Style-only items not written up:** `json.load(open(path))` fd leak
  (`aggregate_libsafety_results.py:56`), `validate_flow_cbf.py` running at import with a mid-file
  `import jax.numpy`, unused `sys` / `R` / `tqdm` imports in the flowcbf and aegis drivers,
  `if done: break` before `t += 1`.

---

## Suggested order of work

1. Fix `check_robot_contact` (finding 1) — every collision number in the repo depends on it.
2. Fix the flow-CBF action space (finding 2) and add an end-to-end constraint assertion.
3. Make an inactive filter loud (finding 3) and fix the AEGIS fallback (finding 4).
4. Unify seeding (5), warm-up accounting (7), and the init-state indexing (11) across all six
   drivers — ideally by extracting the shared episode loop rather than patching six copies.
5. The aggregator issues (9, 10, 12) once the per-episode data they need is being recorded.
