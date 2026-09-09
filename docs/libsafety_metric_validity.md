# LIBERO-Safety — validity of the collision signal, per suite

Machine-checkable record behind the metric caveats in
`docs/experimental_setup_appendix.md` §2.3. That section carries the interpretation;
this file carries the command and its output so the numbers can be re-verified.

## Reproduce

```bash
BD=LIBERO-Safety/libero/libero/bddl_files
for s in human_safety obstacle_avoidance_human obstacle_avoidance affordance; do
  files=$(find "$BD" -path "*${s}*" -name '*.bddl' | grep -E "/${s}/" | sort)
  echo "$s: files=$(echo "$files" | grep -c .)" \
       "with_constraints=$(echo "$files" | xargs grep -l ':constraints' | wc -l)" \
       "CheckRobotContact=$(echo "$files" | xargs grep -ho 'CheckRobotContact' | wc -l)" \
       "CheckContact=$(echo "$files" | xargs grep -ho 'CheckContact' | wc -l)" \
       "CheckGripperForce=$(echo "$files" | xargs grep -ho 'CheckGripperForce' | wc -l)"
done
```

## Verified output (2026-09-09)

| Suite | .bddl files | with `:constraints` | `CheckRobotContact` | `CheckContact`¹ | `CheckGripperForce` |
|---|---|---|---|---|---|
| `human_safety` | 15 | 15 | 12 | 0 | 3 |
| `obstacle_avoidance` | 15 | 15 | 22 | 31 | 0 |
| `obstacle_avoidance_human` | 15 | 15 | 15 | 21 | 0 |
| `affordance` | 15 | **0** | 0 | 0 | 0 |

¹ The `CheckContact` column counts the literal substring, which also occurs inside
`CheckRobotContact`. The figures above are the paren-balanced per-file counts of genuine
`CheckContact` conjuncts and agree with an independent audit.

## What each suite can and cannot measure

| Suite | Benchmark's own collision signal | Why |
|---|---|---|
| `affordance` | **None — vacuous.** CAR ≡ 1.000 with no measurement occurring. | `constraints == []` → `info['cost'] == {}` → `any(...)` is False by construction. |
| `human_safety` | **None — predicate dead.** CAR ≡ 1.000. | Only `CheckRobotContact` present; it is registered and evaluated every step but is structurally always False (`bddl_base_domain.py:1062,1064` build an index list fed to a name-based membership test at `:1016,1019-1023`). |
| `obstacle_avoidance` | **Partial.** Live via `CheckContact`. | Measures **manipulated-object-hits-hazard**, never arm-hits-hazard. |
| `obstacle_avoidance_human` | **Partial.** Live via `CheckContact`. | Same qualification. |

**The arm-hits-hazard channel yields zero detections in all four suites.**

## Empirical corroboration in stored results

`cosmos_benchmark_libsafety` (45 episodes per suite) shows exactly the predicted pattern,
with no re-run required:

| Suite | Stored CAR | Consistent with |
|---|---|---|
| `affordance` | 1.0000 | vacuous — no constraints block |
| `human_safety` | 1.0000 | dead predicate |
| `obstacle_avoidance` | 0.9111 | live `CheckContact` channel |
| `obstacle_avoidance_human` | 0.8000 | live `CheckContact` channel |

## Consequence for reporting

- Emit `--` / `n/a` for the `affordance` and `human_safety` collision columns. Reporting
  `1.000` claims perfect safety where no measurement took place.
- Qualify the two obstacle suites' CAR as object–hazard contact in the metric definition,
  not in a footnote.
- Use `libsafety_contact.any_robot_hazard_contact()` for a genuine robot-contact metric.
  A patched benchmark predicate is still insufficient: `_check_constraint` keys `cost` by
  predicate *name* (duplicate conjuncts collapse, `:931`) and returns `{}` when `done`
  (a violation on the success step is uncounted, `:928`).
