# Safety Memory — Proof-of-Life Results

**Run 19 September 2026.** Protocol: [`docs/design/2026-09-19-safety-memory-proof-of-life-protocol.md`](../../docs/design/2026-09-19-safety-memory-proof-of-life-protocol.md).
Predicates and rules are FIXED and human-written. Nothing is induced. The question is
only whether persisting a fact across episodes changes safety outcomes.

## The claim being tested


With fixed rules, memory can only affect safety when the rule's antecedent is not
recoverable from the current observation. So the endpoint is an **interaction**, not a
main effect: memory should be decisive when the hazard is hidden and a no-op when it is
visible.

## Setup


One rule, written by hand, identical in every arm:

    MARKED(x) AND RTE(y) -> keep_away(x, y, margin)

Memory stores facts, never rules: `MARKED(eef)=true, since=<episode,t>, evidence=<contact>`.
JSON-backed, so persistence across a process restart is testable rather than assumed.

Three arms share one enforcement mechanism (a discrete control-barrier projection) and
differ only in the belief source:

| arm | believes MARKED(x) when |
|---|---|
| memoryless | the mark is visible in the current observation |
| memory | it was written during a previous episode, or is visible now |
| oracle | the simulator's ledger says so |

The memoryless arm gets a **perfect** current-frame grounder, which is deliberately the
strongest possible baseline.

Four cells, each an exposure episode followed by a full reset and a scored episode. Object
poses are re-randomised between them; object instance identity persists.

| cell | exposure | required behaviour |
|---|---|---|
| H_POS | touched the marked source | constraint active |
| H_NEG | touched a visually identical clean source | **no** constraint |
| H_RESET | touched marked source, then washed | constraint released |
| O_POS | touched marked source, mark visible in the scored frame | memory redundant |

## Leak check (passed, by construction)


Within a pair, H_POS and H_NEG share the scored-episode seed. Their scored observations
were measured to be **bit-identical** (max absolute feature difference 0.0 over 30 pairs).
No current-frame grounder can separate them, so the hidden condition is genuinely hidden
rather than merely hard. Classifier cross-validation is reported as a consistency check
only; identical features with opposite labels force at-or-below-chance accuracy.

**The same check passes in physics, and more strongly.** Across the paired
SafeLIBERO episodes, every observable scored-scene quantity is identical between the
marked and unmarked twin (maximum difference 0.000), and the memoryless arm's
trajectories are bit-identical across twins in 6 of 6 pairs. So in the physics board the
no-memory arm literally cannot behave differently depending on whether the tool is
marked. Any difference in violation rate therefore comes from the hidden state alone,
which only memory can reach.

## Result 1 — kinematic testbed (n = 40 pairs, 640 episodes, 4 arms)


| cell | arm | viol | task | safe | interv | minDist | belief |
|---|---|---|---|---|---|---|---|
| H_POS | memoryless | 1.000 | 1.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| H_POS | memory | **0.000** | 1.000 | **1.000** | 0.204 | 0.090 | 1.000 |
| H_POS | memory_deleted | 1.000 | 1.000 | 0.000 | 0.000 | 0.045 | 0.000 |
| H_POS | oracle | 0.000 | 1.000 | 1.000 | 0.204 | 0.090 | 1.000 |
| H_NEG | memoryless | 0.000 | 1.000 | 1.000 | 0.000 | 0.041 | 0.000 |
| H_NEG | memory | 0.000 | 1.000 | 1.000 | **0.000** | 0.041 | **0.000** |
| H_NEG | oracle | 0.000 | 1.000 | 1.000 | 0.000 | 0.041 | 0.000 |
| H_RESET | memoryless | 0.000 | 1.000 | 1.000 | 0.000 | 0.041 | 0.000 |
| H_RESET | memory | 0.000 | 1.000 | 1.000 | **0.000** | 0.041 | **0.000** |
| H_RESET | oracle | 0.000 | 1.000 | 1.000 | 0.000 | 0.041 | 0.000 |
| O_POS | memoryless | 0.000 | 1.000 | 1.000 | 0.204 | 0.090 | 1.000 |
| O_POS | memory | 0.000 | 1.000 | 1.000 | 0.204 | 0.090 | 1.000 |
| O_POS | oracle | 0.000 | 1.000 | 1.000 | 0.204 | 0.090 | 1.000 |

**Primary endpoint.** Violation gap (memoryless minus memory): hidden +1.000,
visible +0.000. **DiD = +1.000, 95% CI [+1.000, +1.000], n = 40 pairs.**
Paired McNemar on H_POS: b = 40, c = 0, p < 0.0001.

**Conservatism guards hold.** In H_NEG and H_RESET the memory arm believes nothing
(belief rate 0.000), intervenes never (0.000), and completes the task as often as the
memoryless arm (1.000). The H_POS gain is therefore not blanket avoidance.

**Deletion panel (D2).** Deleting the stored entry, with every other code path
identical, restores memoryless behaviour exactly (H_POS violation 1.000 vs memory
0.000). The stored fact is the cause of the safety gain, not the pipeline around it.
In O_POS the deleted arm still avoids, because there the mark is visible and memory was
never load-bearing, which is the same interaction seen from the other side.

**Memory equals oracle** in every cell. Once the fact is correctly persisted there is
nothing left for privileged simulator state to add.

**Reproducible.** Two independent runs of this board agree exactly. Getting there
required removing a dependence on Python's `hash()`, which is randomised per process and
was silently making campaigns irreproducible; it is now a stable CRC.

These numbers are saturated at 0 and 1. That is a property of a clean testbed, not
evidence of robustness. It validates the machinery; the physics board is the real test.

## Result 2 — how good must the write be?


Write errors are injected into the memory path only; ground truth is untouched.

Missed writes (false negatives), H_POS:

| missed-write rate | violation | task success |
|---|---|---|
| 0.00 | 0.000 | 1.000 |
| 0.25 | 0.292 | 1.000 |
| 0.50 | 0.458 | 1.000 |
| 0.75 | 0.792 | 1.000 |
| 1.00 | 1.000 | 1.000 |

Spurious writes (false positives), H_NEG:

| spurious-write rate | violation | task success | intervention rate |
|---|---|---|---|
| 0.00 | 0.000 | 1.000 | 0.000 |
| 0.25 | 0.000 | 1.000 | 0.121 |
| 0.50 | 0.000 | 1.000 | 0.175 |
| 1.00 | 0.000 | 1.000 | 0.283 |

The two error modes are **asymmetric**: a missed write costs safety roughly in proportion
to the miss rate, while a spurious write costs intervention effort and leaves task success
intact in this geometry. That asymmetry is what would justify a fail-safe (over-inclusive)
write policy in a later system, and it is measured here rather than assumed.

## Result 2b — the interaction is not knife-edge


Sweeping the keep-away margin over a 1.8x range and the actuator noise over 4x
(n = 24 pairs per setting):

| margin | noise | H_POS memoryless viol | H_POS memory viol | H_NEG memory task | H_NEG memory interv |
|---|---|---|---|---|---|
| 0.09 | 0.0015 | 1.000 | 0.000 | 1.000 | 0.000 |
| 0.09 | 0.0060 | 0.958 | 0.000 | 1.000 | 0.000 |
| 0.11 | 0.0015 | 1.000 | 0.000 | 1.000 | 0.000 |
| 0.11 | 0.0060 | 0.958 | 0.000 | 1.000 | 0.000 |
| 0.13 | 0.0015 | 1.000 | 0.000 | 1.000 | 0.000 |
| 0.13 | 0.0060 | 0.958 | 0.000 | 1.000 | 0.000 |
| 0.16 | 0.0015 | 1.000 | 0.000 | 1.000 | 0.000 |
| 0.16 | 0.0060 | 0.958 | 0.000 | 1.000 | 0.000 |

The separation does not depend on a tuned margin, and the conservatism guard holds
everywhere: in the lookalike cell the memory arm never intervenes and never loses the
task.

## Result 3 — SafeLIBERO physics: commissioning


Physics runs use the SafeLIBERO fork of LIBERO, `safelibero_spatial` task 0, a Panda
under OSC_POSE, no renderer (physics only). Commissioning measurements, none of which
could be assumed:

| quantity | assumed | **measured** |
|---|---|---|
| delta-EEF action scale | 0.05 m/step | **0.0113 m/step** |
| physics step cost | — | 44 ms at control_freq 20 |
| reset determinism (EEF) | — | 0.0 across identical seeds |
| object pose spread across seeds | — | 1.6–11 cm |
| env construction | — | 21.8 s on a GPU node |

Two environment facts cost real time and are recorded so they are not rediscovered:

- **The login node hangs** on environment construction under both `egl` and `osmesa`,
  blocked at roughly zero CPU. Even physics-only MuJoCo must be submitted through the
  batch scheduler here.
- The installed `libero` **is** the SafeLIBERO fork. Only `safelibero_*` suites exist;
  `libero_spatial` raises `KeyError`.

### Reachability governs the experiment geometry

The first physics runs showed every arm failing the task. The cause was not the filter:
the **unfiltered** arm also failed. A targeted probe measured which transit endpoints the
arm can hold at transit height:

| target side | best distance achieved | sag |
|---|---|---|
| −x of the protected object (L = 0.12, 0.18, 0.24) | **0.000, 0.000, 0.001** | none, holds z = 1.010 |
| +x of the protected object (L = 0.12, 0.18, 0.24) | 0.051, 0.102, 0.147 | sinks to z ≈ 0.94 |

The arm cannot hold targets beyond about x = 0.13 at that height and sags toward the
table instead. The original transit put its goal in exactly that unreachable region, so
the task was impossible for every arm. A coarse reachability map then fixed the final
geometry: the **ramekin** is the protected object, because it is the one with reachable
space on both sides at a separation larger than the keep-away margin.

Two further defects were found and fixed by inspecting logs rather than by assumption:

1. **Loose intermediate-waypoint tolerance.** A 0.045 m tolerance on the descend waypoint
   let the arm begin translating before finishing its descent, so its true clearance over
   the protected object was up to 0.045 m larger than intended and the nominal path was
   not reliably unsafe (closest approach 0.081 m against a 0.08 m threshold). Tightened
   to 0.012 m, after which the nominal approach is a consistent 0.044–0.049 m.
2. **Entanglement between exposure targets.** Driving straight from one object to the next
   leaves the gripper pressed into the first; in some pairs the arm never travelled to the
   wash object at all (final distance 0.57 m) and the reset cell silently degraded. Fixed
   by lifting to disengage before traversing.

### The scene is not at rest after reset, and that invalidates naive scoring

Switching the violation signal to object displacement produced a result that looked
impossible: the protected object moved 0.23 m in **every** arm, including the ones whose
filter was active, while no robot checkpoint ever came within 0.20 m of it.

A control run settles it. With the robot **commanded to hold still** for 80 steps:

| seed | ramekin | plate | bowl 1 | bowl 2 | cookies |
|---|---|---|---|---|---|
| 2000 | 0.112 | 0.069 | 0.073 | 0.072 | 0.204 |
| 2003 | 0.226 | 0.072 | 0.074 | **1.646** | 0.061 |
| 2007 | 0.071 | 0.069 | 0.077 | 0.072 | 0.446 |

Objects drift up to 0.23 m on their own while physics resolves the initial placement, and
at seed 2003 a bowl leaves the table entirely. Motion stops by roughly step 5–10.

Two consequences, both now implemented:

1. **Every measurement starts after a settle phase.** Zero actions are stepped until
   per-step motion falls below 4 mm for three consecutive steps, and the protected
   object's reference pose is captured *after* that, not at reset.
2. **Unsettled or runaway scenes are excluded, not scored.** A declared rule drops any
   episode where a role object drifts more than 0.30 m during settling or ends below the
   table. This is a property of the simulator's initial placement, independent of arm and
   of memory, so it applies identically everywhere and the count is reported.

   The rule is **cell-aware**, and that matters more than it sounds. At one seed a single
   lookalike bowl falls off the table. Applied to all role objects, that one bowl
   invalidated 16 of 30 episodes, including hidden-positive episodes that never touch it.
   Scored against only the objects each cell actually uses, exclusions drop to 4 of 44
   with no loss of rigour: an episode is discarded exactly when something it depends on
   is unusable.

This is a defect in how the scene is used, not in the benchmark, but it would silently
corrupt any displacement-based safety metric computed from the reset pose. It is worth
stating plainly because displacement from the initial pose is the standard collision
signal in this benchmark family.

### An end-effector-only barrier does not describe the robot

Before the drift was understood, the arm checkpoints were added for an independent and
still-valid reason: a barrier on the end-effector point alone leaves the forearm and wrist
unguarded. The filter now enforces every constraint at the gripper, both fingers, wrist,
and two arm links, with a per-checkpoint margin scale, which is the same structure the
repo's existing FOL filter uses.

### Safety interaction reproduces in physics, with the interval excluding zero

Two physics boards are reported because each answers a different half of the question.
Board A has the statistical power on the interaction. Board B has liveness.

#### Board A — the interaction, n = 24 pairs, 382 episodes, complete

| cell | arm | viol | interv | minDist | belief |
|---|---|---|---|---|---|
| H_POS | memoryless | 0.542 | 0.000 | 0.103 | 0.000 |
| H_POS | **memory** | **0.042** | 0.540 | 0.126 | 0.917 |
| H_POS | memory_deleted | 0.542 | 0.000 | 0.103 | 0.000 |
| H_POS | oracle | 0.042 | 0.540 | 0.126 | 0.917 |
| H_NEG | memoryless | 0.000 | 0.000 | 0.103 | 0.000 |
| H_NEG | **memory** | 0.000 | **0.000** | 0.103 | **0.000** |
| H_RESET | memoryless | 0.125 | 0.000 | 0.103 | 0.000 |
| H_RESET | memory | 0.042 | 0.124 | 0.109 | 0.292 |
| O_POS | memoryless | 0.042 | 0.540 | 0.126 | 0.917 |
| O_POS | memory | 0.042 | 0.540 | 0.126 | 0.917 |

- **Hidden-positive violations: 0.542 to 0.042.** Paired difference +0.500,
  95% CI [+0.292, +0.708], McNemar b = 12, c = 0, **p = 0.0005**.
- **Primary endpoint. DiD = +0.500, 95% CI [+0.292, +0.708], n = 24 pairs.**
  The interval excludes zero. Memory reduces violations specifically where the hazard is
  not recoverable from the current observation, and does nothing where it is.
- **Deletion restores memoryless behaviour in every cell**: 0.542 against memory's 0.042
  in hidden-positive, 0.125 against 0.042 in reset, identical in the other two.
- **Lookalike cell: zero violations, zero interventions, zero belief, in all four arms.**
- **Visible cell: all four arms identical at 0.042.** Memory is exactly a no-op there.

Task success is 0 for every arm on this board because of the reachability defect
documented above, so no conservatism claim rests on it. That is what Board B supplies.

#### Board B — liveness, settle-corrected, pooled over two seed ranges, complete

640 episodes run, 128 excluded by the cell-aware rule, 512 scored, 20 paired.

| cell | arm | viol | task | safe | interv | belief |
|---|---|---|---|---|---|---|
| H_POS | memoryless | 0.750 | 0.875 | 0.156 | 0.000 | 0.000 |
| H_POS | **memory** | **0.281** | 0.719 | **0.500** | 0.257 | 0.844 |
| H_POS | memory_deleted | 0.750 | 0.875 | 0.156 | 0.000 | 0.000 |
| H_POS | oracle | 0.281 | 0.719 | 0.500 | 0.257 | 0.844 |
| H_NEG | all four arms | 0.000 | 0.906 | 0.906 | 0.000 | 0.000 |
| H_RESET | memoryless | 0.062 | 0.875 | 0.812 | 0.000 | 0.000 |
| H_RESET | memory | 0.031 | 0.875 | 0.844 | 0.005 | 0.062 |
| O_POS | all four arms | 0.281 | 0.719 | 0.500 | 0.257 | 0.844 |

- **Hidden-positive violations 0.750 to 0.300** on the paired subset. Difference +0.450,
  95% CI [+0.250, +0.650], McNemar b = 9, c = 0, **p = 0.0039**.
- **DiD = +0.450, 95% CI [+0.250, +0.650], n = 20 pairs.** Excludes zero, independently
  of Board A.
- **Liveness is preserved.** Task success 0.800 for memory against 0.950 for memoryless
  on the paired subset, a difference of 0.150 that is not significant (p = 0.25), while
  safe success rises from 0.156 to 0.500. The gain is not the robot stopping.
- **Lookalike cell is exactly flat**: all four arms identical at 0.000 violations and
  0.906 task success, with memory intervening zero times.
- **Visible cell: all four arms identical at 0.281.** Memory is a no-op there.
- **Deletion restores memoryless behaviour**: 0.750 against memory's 0.281.

#### Memory recovers the hidden fact perfectly; the residual is the filter's

Belief against the ledger's truth, complete data:

| cell | n | truly marked | memory believed | agreement |
|---|---|---|---|---|
| H_POS | 32 | 0.844 | 0.844 | **32/32** |
| H_NEG | 32 | 0.000 | 0.000 | **32/32** |
| H_RESET | 32 | 0.062 | 0.062 | **32/32** |
| O_POS | 32 | 0.844 | 0.844 | **32/32** |
| **total** | **128** | | | **128/128** |

The memoryless arm agrees on 5 of 32 in the hidden-positive cell, which is what being
blind looks like. Memory never misses a mark, never invents one on the visually identical
lookalike, and always releases after the wash.

Restricting to episodes where the tool was genuinely marked (n = 27):

| arm | violation | task success | safe success |
|---|---|---|---|
| memoryless | 0.889 | 0.852 | 0.000 |
| **memory** | **0.333** | 0.667 | **0.407** |
| oracle | **0.333** | 0.667 | **0.407** |
| memory_deleted | 0.889 | 0.852 | 0.000 |

Memory and oracle are **identical on every column**. The information problem is solved
completely, and the violations that remain are an enforcement limitation of the barrier
against this arm and scene, not a memory limitation. A better filter would improve both
rows equally and would not change the memory conclusion.

## Scope and honest limits


- The write and the evaluator's truth both derive from the same proximity event. So the
  supported claim is **"enforcement on persistent facts works"**, not "retrieval is
  solved". Instance re-identification is a separate problem, deliberately deferred.
- The nominal policy is scripted, not a VLA. That removes sampling variance from the
  measurement and is appropriate for proof of life; a learned policy is step 3.
- Observability is modelled as whether the property appears in the observation. No VLM is
  in the loop, by design.
- The hazard is a marker-transfer analogue implemented as contact bookkeeping. No fluid,
  thermal, or biological claim is made.

