# Safety Memory: learning predicates and rules at runtime — literature map and design brainstorm

Date: 2026-09-09. Status: BRAINSTORM (pre-approval). No code.
Inputs: four survey reports (scratchpad `survey_memory.md`, `survey_nesy.md`,
`survey_vla_bench.md`, `survey_guarantees.md`, ~160 searches, abstract-verified),
the July 2026 survey (`2026-07-11-lit-survey-redesign-brainstorm.md`), the FOL
code (`fol_safety_filter/`), and the SafeManip port ledger.

---

## 0. The reviewer's objection, answered honestly

Brunke et al. (RA-L 2025, arXiv 2410.15185): 3D semantic map + LLM multi-prompt
voting → (object, relation) constraints → CBF. Our pipeline: VLM literals →
election rule → superquadric barrier → in-denoising repair. Same shape.

What our "FOL" layer is today, from `kb.py` / `primitives.py` / `rule_composer.py`:
- The election rule is a conjunction of three ground literals per object. No
  quantifier ranges over anything at inference time, no variable binds across
  objects, no inference beyond one forward chain. It is propositional.
- Predicates are a fixed 20-entry list with hand thresholds (`NEAR` 0.15 m,
  `TILTED` 20°, `IS_HOT` from a cached VLM lookup). Brunke's above/around/
  behavior/pose taxonomy is the same kind of object, shorter.
- The ledger already recorded the empirical verdict twice: scene graph NO-GO
  (19/19 parity with the flat scorer), FOL-COMPILE v2 NO-GO.
- The critique literature has a name for this: **reasoning shortcuts**
  (Marconato et al. NeurIPS 2023 2305.19951; JAIR 2026 2510.14538) — a symbolic
  layer that is architecturally symbolic but semantically decorative. A reviewer
  who knows logic will cite it.

So: as a *mechanism at fixed knowledge*, FOL adds nothing over Brunke. The
only argument that makes a relational representation load-bearing is that it is
a **hypothesis space for induction**: you cannot learn a rule from experience
without a language to learn it in, and you cannot carry a rule from a stove to
a kettle without a variable to bind. Theory backs this specifically: policies
generalise across object counts iff the needed features lie in the C2 fragment
of FOL (Ståhlberg/Bonet/Geffner KR 2022, AAAI 2025); lifted world models are
orders of magnitude more sample-efficient than propositional ones
(2602.17217); simpler FOL formulas transfer better to unseen structures
(2602.18956). None of that applies to a hand-written rule table. All of it
applies to an induced one.

**Conclusion: runtime learning is not an add-on to the FOL story. It is the
missing half that would justify the FOL story.** If we do not learn the rules,
we should drop the FOL branding (July verdict). If we do, FOL becomes the thing
a neural or propositional memory cannot do: transfer safety across object
identity and count.

---

## 1. What the literature has and has not done (compressed)

**Occupied (must cite and differentiate):**
- Stateless VLM → CBF semantic filters: Brunke RA-L'25; VLSA/AEGIS 2512.11891;
  CORE 2602.19983 (nav); Santos/Bajcsy ICRA'25 2409.14580 (human language updates
  an HJ failure set online — the only one touching a continuous controller);
  barrier-enhanced flow matching 2607.29569; NS-CFM 2607.01378. **None persists
  anything across episodes.**
- Safety memories at the planner level: **RoboSafe** 2512.21220 (LLM-seeded
  executable predicates, long/short-term DB, write-back criterion = "the code
  ran"); **JIT-Memory** 2607.16247 (Test-Verify-Write meta-skills on IS-Bench,
  +30 pp safe-success, discrete OmniGibson primitives); workshop episodic-
  violation memory (OpenReview KrmyJtyE6k). All discrete actions, no filter, no
  geometry, no guarantee, no held-out-class protocol.
- Predicate invention: VisualPredicator ICLR'25 2410.23156 (LLM writes VLM-calling
  predicate programs online, for *planning*); pix2pred 2501.00296; SkillWrapper
  2511.18203 (active validation, soundness theory); InterPreT RSS'24 (needs
  human language). None for safety, none feeding a filter.
- Rule/constraint induction: ILCL RA-L'25 2507.11000 (TL constraints from demos,
  offline); Popper-family ILP; LLM-as-ILP (ILP-CoT, ADVENT 2607.01585). None
  online from a robot's own violations.
- Learned CBFs from data: Yang & Sibai 2510.21560, 2503.10641 — neural,
  offline, not lifted, cannot transfer classes.
- Cross-episode memory on frozen VLAs: Retrieve-then-Steer 2605.10094 (successes
  only), Harness VLA 2607.08448 (task success). Not safety.
- Failure explanation: REFLECT, AHA, ARMOR, Code-as-Monitor — per episode,
  nothing accumulates into a constraint.
- Motivation, citable: Rethinking Safety for Generalist Robots 2609.06326
  (Majumdar/Bajcsy, 6 Sep 2026): names "reasoning–action misalignment" and says
  "incorporating memory will be critical for ensuring that robots do not repeat
  the same mistakes."

**Unoccupied (verified by negative searches in all four surveys):**
- G1 Persistent, experience-updated *geometric* constraints for a continuous-
  control safety filter.
- G2 Closing the loop from the robot's *own* violations/near-misses (not human
  language) to constraint updates in a certified filter.
- G3 Adaptive-conformal calibration of per-(object, relation) VLM predicates
  across episodes to a target violation rate.
- G4 Any semantic-safety benchmark with a multi-episode / learning-curve axis.
- G5 Soundness governance for self-written robotic safety memories (verify
  before persist).
- G6 Failure explanation → verified, retrievable constraint.
- N1 Runtime induction of *lifted* safety rules that parameterise CBFs.
- N2 Lifted safety rules learned online that transfer to unseen object classes.
- W1 VLM judging *imagined* futures for semantic hazards in manipulation.

---

## 2. Design: the Safety Memory (SM)

One sentence: **the robot keeps an experience store of predicate traces and
outcomes, induces lifted safety rules from it, verifies each rule by replaying
the failure it explains, and compiles surviving rules into the barriers of the
existing in-denoising filter; the rule vocabulary is open (predicates are
invented), the rule language is first-order (rules transfer across objects),
and the memory's violation rate is monitored with an anytime-valid statistic.**

Data flow per episode:

```
scene ──VLM ground──▶ objects + open-vocab attributes
   │                         │
   │   SM.semantic (lifted rules) ──bind variables──▶ active barriers ──▶ π0.5 in-denoising DCBF ──▶ act
   │                                                                              │
   └────────── predicate trace (grounded literals per step) + consequence events ◀┘
                                  │
                       SM.episodic (traces, outcomes)
                                  │  consolidate (offline between episodes)
              attribute ──▶ propose predicates/rules ──▶ score ──▶ replay-verify ──▶ write / prune
```

Eleven decisions follow. Each has: what, why (the failure it prevents),
alternatives considered, prior art, and whether it is new.

### D1. Learning signal = the robot's own consequences, attributed by a VLM

**What.** A violation oracle from the simulator (contact, spill, containment
loss, contamination, toppling, damage) marks *that* something went wrong and
*when*; a VLM watching the rollout window around the event names *what*
(which object, which relation, which held object). The learner never sees the
simulator's privileged predicates, only the scalar event and its own grounded
trace. Near-misses (barrier margin below δ without an event) are logged as
weak positives.

**Why.** This is the only signal that scales with compute and data (Bitter
Lesson prior) and yields a learning curve. Instructions are not the signal
(user constraint). Human corrections (Santos ICRA'25, DROC) are prior art and do
not scale to n≥20 per cell.

**Alternatives.** (a) VLM-only judgment of rollouts — cheapest, but our own
E2/E3 knowledge–action-gap result and SafetyALFRED's 92.5% QA vs 60.1%
mitigation say VLM verdicts are unreliable as sole supervision; keep as fallback
where the sim is blind. (b) World-model imagined consequences (W1) — genuinely
novel and pre-action, but the claim then rests on the world model; keep as an
*optional second source* (D11), not the primary.

**Prior art / new.** Consequence-driven constraint updates in a certified
filter: G2, unoccupied. Attribution via VLM borrows REFLECT/AHA; the new part is
that the explanation becomes a hypothesis, not a report.

**Critique.** Attribution noise breaks any version-space argument (guarantees
survey Claim D). Mitigation: attribution is itself scored by discrimination
(D4) and calibrated (D8); a rule is never written on one episode's word.

### D2. Two stores: episodic traces and semantic lifted rules, with consolidation

**What.** `SM.episodic`: per episode, the grounded literal trace (which
predicates held on which objects at each step, barrier margins, actions,
outcome events, scene embedding). `SM.semantic`: lifted rules with support,
confidence, calibration residuals, provenance (which episodes). Consolidation
runs between episodes and promotes patterns from episodic to semantic.

**Why.** Retrieval of raw episodes (k-NN scenes) is RAG for safety: it cannot
transfer to a scene that looks different but shares a hazard. Rules alone lose
the evidence needed to re-score them when new episodes arrive. The split
mirrors complementary-learning-systems and Generative Agents' reflection, and
it is what lets D4 re-run induction as data grows.

**Alternatives.** Rules only (RoboSafe) — cannot prune or re-calibrate. Episodic
only (Retrieve-then-Steer, JIT-Memory's example buffers) — no transfer, and it
is the ablation row reviewers will ask for.

**New.** The pair is standard in agent memory; applying it to constraints that
compile into barriers is G1.

### D3. Predicate invention: open vocabulary, grounded two ways

**What.** A predicate is a named unary/binary program with two groundings:
(i) a VLM query program (VisualPredicator-style: "is this object hot / does it
contain liquid / is it sharp") and (ii) a prototype in VLM/CLIP embedding space
built from the object crops that co-occurred with violations. New predicates
are proposed by an LLM from attribution text ("the pot was hot") and kept only
if they *discriminate*: the predicate must separate violation traces from
non-violation traces on held-out episodes (the discrimination property that
2604.23377 shows is necessary against reasoning shortcuts). The fixed list in
`primitives.py` becomes the seed vocabulary, not the ceiling.

**Why.** The paper's central claim is that the constraint vocabulary is open
and grows with experience. Brunke's and our current taxonomy are closed.
Prototype grounding is what lets a never-seen kettle be classified as hot
without a VLM having been asked about kettles (scaling with data rather than
with rules).

**Alternatives.** VLM-query only — every new object costs a VLM call and the
VLM can miss the cue (this is the 0.998-AUC-but-0%-verbalization pathology).
Embedding-only — un-auditable, no name, cannot be composed into a rule a human
can read; and reasoning-shortcut risk is worst here.

**Prior art / new.** Mechanism borrowed from VisualPredicator (ICLR'25) and
pix2pred; SkillWrapper gives the soundness/completeness framing for
FM-generated predicates. Applying invention to *safety* predicates supervised
by *violations* is N1. Dual grounding with a discrimination gate is new in this
setting.

### D4. Rule induction: LLM-proposed, ILP-scored, lifted

**What.** Hypothesis class H = {UNSAFE(t) ← R(eef_or_held, x, t) ∧ P₁(x) ∧ … ∧
Pₖ(x)} plus geometric template τ (superquadric family, margin) and class-K
gain, with x a *variable*. An LLM proposes candidate clauses from attribution
text; each is scored on `SM.episodic` by coverage (violations explained),
precision (non-violation episodes where it would fire), and a
minimum-description-length prior. This is Popper-style generate–test with an
LLM as the generator (ILP-CoT, ADVENT do this for toy domains). Candidates that
tie are kept as a small version space, not argmax (guard-set lesson: fail-safe
sets beat fail-unsafe argmax).

**Why.** A lifted clause is invariant to object identity and count; that is the
whole transfer claim (§0). Search over H scales with compute; hand rules do not.

**Alternatives.** Neural CBF from violation data (2510.21560) — not lifted,
cannot name why, cannot transfer classes, and cannot be pruned. Propositional
per-object rules (Brunke, RoboSafe) — no transfer. Pure LLM rule writing
without scoring (ASIMOV-style constitutions) — no grounding in the robot's
data, and it is what reviewers will call "prompting".

**Prior art / new.** N1/N2 unoccupied. Nearest: ILCL (offline, demos, TL) and
2605.26828 (offline ILP from demos, unseen objects).

**Critique.** H must be small enough that a handful of violations identify a
rule (Xie et al. 2407.04216 give a certified bound on the number of feedbacks
for a finite class). Keep H to relation × ≤2 predicates × template; the
misspecification alarm (D9) catches hazards outside H.

### D5. Write policy: Test–Verify–Write with counterfactual replay

**What.** A candidate rule is written to `SM.semantic` only if all three hold:
1. **Discriminates** on held-out traces (D3/D4 scores above threshold).
2. **Replay-verified**: the failing episode is replayed on CPU with the
   candidate barrier active (the repo already has a bit-exact CPU replay of
   per-step logs); the rule is accepted only if the barrier would have engaged
   *before* the event (h crosses zero ahead of the violation) — i.e. the rule
   explains the failure counterfactually, not just correlationally.
3. **Feasible**: chain feasibility (brake-then-push bound) and QP feasibility
   with the rule added to the current active set on the replayed scene.

**Why.** RoboSafe writes back predicates that merely execute; JIT-Memory tests
meta-skills for task success. Neither checks that a stored constraint is
*sound* for the controller. A false rule is a liveness cost with no safety
benefit (guarantees survey §4: false constraints degrade task success without
bound and void the no-trade-off theorem); an unverified rule can also make the
QP infeasible mid-episode.

**New.** G5 unoccupied. Counterfactual replay as the admission test is, to our
knowledge, unused in any safety-memory work, and the infrastructure exists.

### D6. Runtime activation: bind variables, do not retrieve scenes

**What.** At episode start the existing election stack grounds objects and
open-vocab attributes (Tier-Percep or GT). For each rule in `SM.semantic`, the
variable x is bound to every object whose predicates ground true (VLM query
or prototype match above a calibrated similarity); each binding instantiates a
barrier with the rule's template and margin. The corridor exemption, guard-set,
and companion points are unchanged.

**Why.** Binding is what makes the memory transfer: a rule learned on a stove
in one kitchen fires on a kettle in another because the *predicate* matches,
not because the *scene* matches. k-NN scene retrieval (episodic RAG) is the
ablation, not the method.

**Critique.** Over-binding is the liveness failure mode (every object "might
be hot"). Mitigations: calibrated similarity threshold (AnySafe-style conformal
threshold on prototype similarity), the corridor exemption for the task target,
and pruning (D10).

### D7. Enforcement unchanged: in-denoising DCBF on frozen π0.5

**What.** The memory changes only the *constraint set* handed to the existing
server-side repair (schedule, best-of-K, companions, corridor). No new
enforcement mechanism.

**Why.** 2607.01378 owns the in-denoising mechanism; our surviving claim is
election. Memory strengthens election, so the contribution stays coherent
(method-novelty-first: mechanism at fixed compute is the claim, K stays fixed).
It also keeps the frozen-policy story (no fine-tuning, unlike SafeBranch,
SafeVLA, SafeDojo).

### D8. Calibration: per-rule adaptive margins with abstention

**What.** Each rule carries a margin inflation updated by adaptive conformal
inference on its own violation/near-miss stream (Gibbs & Candès; ACoFi L4DC'26
for the filter-threshold variant), targeting a per-rule violation rate ε.
Prototype similarity thresholds are conformally calibrated (AnySafe). When a
rule's grounder base error exceeds ε the system cannot certify it and must
*abstain* — here: fall back to the guard-set / speed-cap regime for that
object rather than pretend.

**Why.** G3 unoccupied: nobody calibrates VLM per-(object, relation) predicates
across episodes. And Kotte 2606.29054 proves that if base error μ > target α,
any distribution-free method must abstain on ≥ (μ−α)/(M−α) of inputs; an honest
system exposes that.

**Critique.** ACI guarantees are long-run averages and burst-violable; say so.
Filter blocking shifts the input distribution (Ray 2607.22868), so offline
predicate accuracy is never quoted as a closed-loop guarantee.

### D9. Guarantee ladder and the misspecification alarm

State exactly these, in this order (from the guarantees survey §6):
- **A. Conditional invariance.** Given the stored rules and correct grounding,
  the filter keeps the stored safe set invariant; appending a rule mid-episode
  is safe iff the state lies inside the new set (belief-CBF discrete-update
  condition, 2607.15016), else backup.
- **B. Calibrated grounding** on the *closed* seed vocabulary with an
  abstention arm.
- **C. Anytime-valid violation-rate control** over the *evolving* memory via an
  e-process or ACI statistic (Conformal Selective Acting 2605.20270) — the only
  guarantee shape that survives adaptive updates.
- **D. Misspecification alarm.** If the version space in H empties for a
  recurring violation, the system declares "the rule language cannot express
  this hazard" and escalates (Xie et al. certified misspecification). This turns
  unknown-unknowns from silent failure into a logged event.
- **E. Composite budget** P[hazard] ≤ P[uncovered] + P[mis-grounded] +
  P[filter fails], with the first term *reported empirically*, never proved.

Do not claim: deterministic invariance w.r.t. the true hazard set; any
no-trade-off result for a conservative memory; transfer of offline VLM accuracy
to closed loop.

### D10. Forgetting and pruning

**What.** Rules whose recent activations produce filter interventions but no
violations or near-misses on the *unfiltered* counterfactual (replay with the
rule removed) lose confidence and are demoted to episodic; rules contradicted
by new traces are re-induced. Bounded per-rule evidence buffers (JIT-Memory
pattern).

**Why.** The literature has no bound on the liveness cost of false rules; the
only defence is to remove them. Also addresses drift/injection risk of
self-updating memories (SSGM 2603.11768).

### D11. Optional second signal: imagined consequences

**What.** Before acting, roll K candidate chunks through the world model
already in the repo (cosmos-policy / dream-policy), have the VLM judge the
imagined video for semantic hazards, and feed judged-unsafe imagined outcomes
into `SM.episodic` as weak positives.

**Why.** W1 is open in manipulation; it lets the robot learn a rule without
ever suffering the consequence, and it sidesteps the evaluation-contamination
worry from the July continual-learning verdict. Kept optional because the
claim then depends on world-model fidelity; run it as an arm, not the spine.

---

## 3. Benchmark and protocol

**Requirement.** Recurring hazard *classes* across many object *instances*,
a native consequence signal, public VLA checkpoints, sequential episodes.

| Candidate | Consequence signal | Classes / diversity | Ckpts | Days | Verdict |
|---|---|---|---|---|---|
| **ManiGuard** 2608.17386 (OmniGibson, 18 Aug 2026, Apache-2.0) | LTLf DFA per-step labels: containment, toppling, dropping, contact, articulation, liquid particles | 6 families × ID + 4 OOD = 1,000 scenarios; 26 liquid tasks | π0, π0.5, GR00T N1.6, SmolVLA fine-tuned on HF | 5–10 (Isaac Sim on Bridges2 is the risk) | **Primary.** Natural held-out-family split. Add heat via OmniGibson temperature states if a thermal class is wanted. |
| **SafeManip** (RoboCasa, already in repo, port ledger through F8) | LTLf: forbidden contact, grasp/release, cross-contamination, containment, enclosure | 50 tasks, dozens of candidate hazards per kitchen | π0.5, GR00T | 3–5 remaining | **Secondary / fallback.** Cross-contamination is a hazard a geometric filter cannot express — the semantic-memory showcase. No fluids/heat. |
| OopsieVerse-OG | health incl. thermal + fluid | ~20 categories | none released | 7–12 | Best consequence model, no policy. Check whether ManiGuard OG ckpts transfer. |
| SafeLIBERO / LIBERO-Safety | displacement / collision+spill | one hazard class | π0.5 | 0–2 | Physical-safety ablation only; cannot show class transfer. |

**Protocol.** Hold out 1–2 hazard families (ManiGuard: Jar + Dusty, plus
Clutter-liquid). Run sequential episodes on the remaining families with the
memory live; freeze; test zero-shot on held-out families and OOD perturbations.
Report (i) violation rate vs episode index (learning curve), (ii) held-out-class
transfer, (iii) TSR alongside CAR (false-rule liveness cost is a first-class
number), (iv) memory size and abstention rate. Paired on identical scenes,
n≥20 per cell, McNemar/Fisher before any GO.

**Baselines / ablations.** Stateless VLM→CBF (Brunke/AEGIS shape, our own
stack with memory off); episodic k-NN retrieval (RAG row); propositional rules
(no variables: the FOL-earns-its-keep row); neural CBF from the same violation
data (2510.21560 shape); language-conditioned latent filter 2608.00315 for the
"learned filter transfers to unseen constraints" contrast; JIT-Memory-style
planner memory where the benchmark allows; oracle rules (ceiling). Fixed K and
fixed steps everywhere; any compute-scaling arm is a matched control only.

---

## 4. Risks, honestly

- **Timeline.** ICRA 2027 deadline is ~2026-09-15. This is a new subsystem plus
  a new simulator; it is the *next* paper (RSS 2027 / CoRL 2027), not this one.
- **Isaac Sim on Bridges2.** ManiGuard needs Isaac Sim (RTX, driver ≤580).
  Unverified on this cluster. SafeManip/RoboCasa (MuJoCo) is the hedge and is
  already half-ported.
- **Reasoning shortcuts.** Invented predicates can separate traces for the
  wrong reason. The discrimination gate and replay verification reduce but do
  not eliminate this; report a predicate-audit table (VLM name vs GT attribute).
- **Statistics.** Learning curves at n=20–50 per cell are noisy; the July
  verdict (SafeDojo's +0.12 pp L2) stands as a warning. Pre-register gates.
- **Contamination.** Memory learned on a family must never see held-out
  families; keep the split in the benchmark loader, not in a script.
- **Grounder base error.** If VLM attribute error on a class exceeds ε, the
  design *must* abstain there; that is a reportable number, not a bug to hide.

---

## 5. What is genuinely new, in one table

| Decision | Nearest prior | What is new |
|---|---|---|
| D1 own-consequence signal → filter | Santos ICRA'25 (human language → HJ) | robot's own violations, no human |
| D3 invented safety predicates, dual-grounded | VisualPredicator (planning) | violation-supervised, discrimination-gated, prototype transfer |
| D4 lifted rules induced online → barriers | ILCL (offline TL), RoboSafe (propositional DB) | first-order, online, compiled to CBF params |
| D5 replay-verified admission | RoboSafe ("it ran"), JIT-Memory (task success) | counterfactual soundness before persist |
| D8/D9 anytime-valid rate control over evolving memory | ACoFi, Conformal Selective Acting (not memory) | applied to a growing constraint set + misspecification alarm |
| §3 learning-curve + held-out-class protocol | none (G4) | first multi-episode semantic-safety evaluation |
