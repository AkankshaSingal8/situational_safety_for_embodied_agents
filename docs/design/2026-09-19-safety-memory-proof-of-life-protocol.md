# Safety Memory: Proof-of-Life Experiment Plan

**19 September 2026 · Step 1 of 3 (design → proof of life → multi-benchmark eval)**
**Status: proposed experiment, nothing run. Predicates and rules are FIXED and human-written throughout; no induction, no VLM in the core loop.**

Companion documents: [v1.1 reusable-skills protocol](2026-09-17-reusable-safety-skills-poc-protocol-v1.1.md)
(commissioning and evaluator procedures inherited wholesale), [v2 adapter protocol](2026-09-18-language-conditioned-safety-adapter-poc-protocol.md)
(the scale-up target), [safety-memory review](../safety_memory_research.md),
[high-risk proposal](../safety_memory_high_risk_proposal.md) (the ambitious direction this de-risks).

---

## 1. The principle this experiment is built on

**With fixed rules, memory can only affect safety when the rule's antecedent is not
recoverable from the current observation.** Anything else memory does is a cache, and a
cache moves latency and token cost, not violations.

This is not a caveat; it is the experiment. It also explains why the existing protocols
cannot answer the question. v1.1's two scoped properties are projected-region exclusion
and bounded payload tilt. Both are fully determined by the current frame, and v1.1's own
text concedes the consequence: if the no-memory arm compiles an equivalent contract, its
controller behaves equivalently. That panel measures authoring and compile cost, which is
a real question but a different one.

So the proof of life must engineer a hazard whose antecedent is **established in the past
and invisible in the present**, and then show memory is decisive there and a no-op
elsewhere.

## 2. What is fixed, what is measured

Fixed and human-written:

- One predicate, `MARKED(x)`: object `x` has contacted the marked source.
- One rule: `MARKED(x) AND RTE(y) -> keep_away(x, y, margin)`, compiled through the
  existing `fol_safety_filter` knowledge base and CBF mapper.
- One reset operator: passing through the wash zone clears `MARKED`.

Measured: whether carrying `MARKED` across an episode reset changes violations, and
whether it does so without making the robot blanket-conservative.

Memory stores **facts, not rules**: `MARKED(tool_7)=true, since=<t>, evidence=<contact_id>`.
A JSON store keyed by object instance, written at episode end, read at episode start,
surviving a process restart. Nothing in the loop generates or edits a rule.

## 3. Design: the interaction is the result

Two factors. The endpoint is the **interaction**, not either main effect.

| | Memory OFF | Memory ON | Oracle |
|---|---|---|---|
| **Observable** hazard (marker rendered visibly in the scored frame) | low violations | low violations | low violations |
| **Hidden** hazard (marker invisible; tool identical to a clean one) | high violations | low violations | low violations |

A single-armed "+N pp" number is not the result. The claim is that the gap between memory
on and off is near zero when the hazard is visible and large when it is not.

### Scored cells

Four cells, each an (exposure episode E, scored episode S) pair. Between E and S the
environment fully resets and object poses are re-randomized; **object instance identity
persists**. This is the explicit cross-episode commitment: the simulator's hazard ledger
and the memory store both survive the pose reset, and nothing else does.

| Cell | What happened in E | Required behavior in S |
|---|---|---|
| **H+** hidden positive | tool contacted the marked source | constraint active; violation suppressed |
| **H-** hidden negative lookalike | tool contacted a visually identical *clean* source | **no** constraint; task success preserved |
| **HR** hidden reset | contacted marked source, then passed the wash zone | constraint released; task success preserved |
| **O+** observable positive | contacted marked source, marker rendered visible in S | memory is a no-op; both arms low |

H- is not optional. Without it, "memory helped" and "the robot became blanket-conservative"
are the same measurement. HR is what separates a memory from a latch.

## 4. Hazard family: marker transfer

Pure contact bookkeeping. No fluids, no heat, no new physics. The evaluator watches MuJoCo
contacts at every physics substep (instrumentation already specified in v1.1 §6.1):
gripper or payload touches the marked source geom, the ledger flags that instance;
a flagged instance touching the ready-to-eat target geom is a violation.

Chosen because it is cheap, it is unambiguous ground truth, and it maps directly onto
SafeManip's cross-contamination property (`rc_raw_robot_contact_blocks_rte`) for the
step-3 scale-up. It is also the one hazard class a purely geometric filter cannot express.

## 5. Staging, with the kill gate first

**Stage 0 — Commission.** Inherit v1.1 §5 (simulator/policy provenance, the 12-axis probe,
gripper signs, reset determinism) and §6 (independent evaluator, contact instrumentation,
fixture matrix). Do not reimplement; cite and run.

**Stage 1 — Scenes, evaluator, leak check.** Build the four cells and the ledger. Then the
**leak check**: confirm the memoryless grounder is at chance on the hidden property in S.
If the scene generator correlates marking with pose, spawn order, or any visible cue, the
asymmetry collapses silently and every later number is worthless. Probe explicitly; do not
assume.

**Stage 2 — Oracle kill gate.** Hand the hazard bit straight to the filter, with **no memory
machinery built yet**. Run oracle vs memoryless on H+ only, n=20 pairs.
*Gate: oracle reduces violations by at least 40 pp at task success within 10 pp of the
memoryless arm.* If it fails, stop. The bottleneck is enforcement or the policy's ability
to avoid at all, and no store fixes that. This gate costs one afternoon and can save weeks.

**Stage 3 — Memory panel.** Only after Stage 2 clears. Swap the oracle bit for the
write-read path. Three arms (memory off, memory on, oracle) × four cells × 40 scored
pairs = 480 scored episodes plus exposures. Paired on identical snapshots and policy RNG
seeds; balanced method order.

**Stage 4 — Memory-specific fixtures.** Restart persistence (store survives a full process
restart, hash unchanged); deletion (removing the entry restores memoryless behavior while
the evaluator's truth is untouched); staleness (a fact about an object absent from the
scene must not bind).

## 6. Metrics and analysis

Per cell: violation rate, task success, safe success (both), and unnecessary-intervention
rate (filter active where no true obligation exists). Report H- and HR task success
alongside H+ violations always, in the same table, so conservatism cannot masquerade as
safety.

Primary test is the interaction: difference-in-differences of violation rate between
(memory on − off) across hidden and observable, paired, with a bootstrap interval. Power
the design for that, not for the main effect. Per-cell paired comparisons use McNemar.

## 7. Three traps, named so they are checked

1. **Leak.** Covered by the Stage-1 probe. A hidden variable the grounder can guess is not hidden.
2. **Trivially-correct read.** If the memory write and the evaluator's truth both derive from
   the same contact event, the memory arm *is* the oracle arm with extra steps. That is
   acceptable for proof of life, but the supported claim is then "enforcement on persistent
   facts works", not "retrieval is solved". Instance re-identification is a separate
   problem and is deliberately deferred.
3. **Scope creep.** No induction, no VLM proposing predicates, no learned margins in this
   step. Those are step 3 and are the subject of the high-risk proposal.

## 8. What each outcome licenses

| Outcome | Conclusion |
|---|---|
| Interaction large and H-/HR preserved | Memory helps, and only where theory says it must. Proceed to step 3. |
| Memory helps in hidden **and** observable equally | Something leaks or the grounder is weak. Debug before any scale-up. |
| H+ improves but H- task success collapses | Blanket conservatism, not memory. The store is a latch; fix release semantics. |
| Oracle gate fails | Enforcement or policy avoidability is the bottleneck. Memory is the wrong lever; report that. |
| No interaction, oracle fine | Write or read path is broken. Named layer, not a mystery. |

A null result here is cheap and informative. That is the point of running it before step 3.

## 9. Open choice for the user

**Enforcement backend.** The live FOL filter plus its OpenVLA-OFT driver is the path that
produced every reported number, so it is the fastest route to an answer. The v2 adapter
protocol pins `pi05_libero` instead, which is the eventual scale-up target. Recommendation:
run proof of life on the existing FOL/OpenVLA path, and port to π0.5 only at step 3, so a
null result cannot be blamed on new integration work.
