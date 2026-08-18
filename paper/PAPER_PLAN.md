# Paper Plan — Noise-Space Safety

Paragraph-level plan for `paper/main.tex` (ICRA 2027 submission). Each entry says what the paragraph must accomplish, which result it cites, and whether that number is **frozen** or **pending** a queued job. Update the draft against this plan section by section; anything marked ⏳ has a named job dependency at the bottom.

**Working title** (keep): *Noise-Space Safety: Certified Training-Free Steering of Frozen Flow Vision-Language-Action Policies.*

**One-sentence thesis**: Safety enforcement for frozen flow VLAs belongs in the policy's noise space, not its action space; identification of what to guard belongs in a three-literal first-order rule with VLM-grounded predicates, not in VLM authority — and both claims are established by paired, pre-registered comparisons on two benchmarks.

---

## Abstract (one paragraph, ~200 words)

Rewrite from the current draft in five sentences, one per claim, newest numbers:

1. **Sentence 1 — the problem and the repair tax**: frozen flow VLAs are unsafe; the dominant fix (CBF repair of actions, post-hoc or in-denoising) carries a paired, significant success tax (32.5 → 67.5 TSR on obstacle-near-path cells, McNemar p=0.0066).
2. **Sentence 2 — our mechanism**: noise-space operators (Select / Tilt / Ascend) + soft repulsor keep every action on the policy manifold and trace a tunable success–safety frontier under one checked, mechanism-independent prefix certificate.
3. **Sentence 3 — identification**: a three-literal FOL rule over VLM-grounded predicates matches ground-truth identity at zero measured cost (all p≥0.18), and **each literal is causally necessary** — deleting any one costs 17–29 TSR points or raises violations (ablation, all p<1e-3). ⏳ *final ablation cells.*
4. **Sentence 4 — the two benchmark boards**: SafeLIBERO five-architecture board (beats both shields where engaged); LIBERO-Safety five-arm board — beats both dedicated shields by +15–20 macro TSR (p<1e-15) at equal-or-lower violations, cuts violations 12:1 vs baseline (p=0.003) at TSR parity on 3/4 suites. ⏳ *uniform n=300 refresh.*
5. **Sentence 5 — the refusal sweep**: 100/100/80 on the safety-reasoning track vs best published 80/72/36.

---

## 1. Introduction (~6 paragraphs)

- **Para 1.1 — Hook: flow VLAs imitate superbly and unsafely.** Concrete failure numbers (86–88% collision on SafeLIBERO cells; gripper driven into a human hand in 40% of episodes). The prevailing fix is repair; thesis sentence: repair is the wrong intervention point. *(Keep current draft P1, refresh numbers.)*
- **Para 1.2 — Pillar 1: measuring the repair tax.** Paired five-arm enforcement comparison, identical seeds/identity/certificate; deleting the repair alone lifts 32.5→55.0; noise-space search reaches 67.5 (p=0.0066) at unchanged collisions. Why: off-manifold corrections produce action sequences the policy never learned to continue. **Frozen.**
- **Para 1.3 — Pillar 2: a certified, tunable frontier in noise space.** The frozen denoise map makes safety a *preimage* problem; introduce Select/Tilt/Ascend + repulsor; success and safety are separable (FK tilt carries success, repulsor carries safety and unsticks obstacle-on-grasp); one checked prefix certificate covers every operator. **Frozen.**
- **Para 1.4 — Pillar 3: identification is GT-free, and every literal earns its place.** The FOL rule; GT-free at zero cost (p≥0.18); the VLM grounds predicates, the logic decides. **Add one sentence** previewing the causal ablation: deleting any literal costs 17–29 TSR points (geom-only also *raises* violations; no-exempt self-paralyzes). ⏳
- **Para 1.5 — The assembled system and its headline boards.** SafeLIBERO board + LIBERO-Safety board + refusal sweep in one training-free stack; the LS board sentence is the architectural headline (dedicated shields halve human-safety success; we hold it at baseline). ⏳ *uniform-300 macro numbers.*
- **Para 1.6 — Contributions list and paper roadmap.** Bulleted, 4 items: repair-tax measurement; noise-space operator family + certificate; FOL identification + causal ablation; two-benchmark boards + negative-results section.

## 2. Related Work (6 paragraphs — keep current organization by intervention point)

- **Para 2.1 — Prompt/conditioning interventions**: safety prompting is behaviorally live but safety-inert (forward-ref to Negative Results). *(Keep.)*
- **Para 2.2 — Activation-level interventions**: COAST, "Your Model Already Knows" (read-only identification — a direct competitor to our identification chapter). *(Keep.)*
- **Para 2.3 — Velocity-field / in-denoising interventions**: sota2607 (direct predecessor — same DCBF background, but no feasibility proof, no re-verification, no infeasibility handling), OmniGuide, VLS (mechanically closest to Tilt/Ascend, but performance-not-safety, no certificate). *(Keep.)*
- **Para 2.4 — Intermediate-state and output-stage interventions**: Constricting BFs, SafeDiffuser boundary-trapping; AEGIS post-hoc, PACS (strongest certified output-stage work; its OOD critique of CBF push-away converges with our repair-tax measurement). *(Keep; merge current P4+P5 if space is needed.)*
- **Para 2.5 — Noise-space manipulation, the unoccupied lane**: prior noise-space work is performance-only (initial-noise opt, DSRL); no prior safety enforcement in noise space, none paired against matched-seed repair. That lane is this paper. *(Keep.)*
- **Para 2.6 — Benchmarks and semantic identification**: SafeLIBERO, LIBERO-Safety, Brunke et al. (corridor exemption transfers; superquadrics don't). **Add**: LIBERO-Safety's own baselines and the two shields we reimplement as LS board rows.

## 3. Problem Setup (2 paragraphs — keep)

- **Para 3.1 — Episode formalism and metrics.** π0.5 flow decoding, chunk execution; metrics (TSR/CAR on SafeLIBERO; benchmark contact-predicate violations + SafeSR on LIBERO-Safety). State the n per cell **here once**: SafeLIBERO n=200; LIBERO-Safety n=300/cell uniform ⏳ (fallback: frozen heterogeneous board with per-cell n noted).
- **Para 3.2 — Privilege tiers.** Tier-GT vs Tier-Percep footnote, unchanged; collision scoring always GT-anchored in both tiers.

## 4. Method (4 subsections)

### 4.1 Identification (3 paragraphs)

- **Para 4.1.1 — The three-literal hazard-election rule.** Eq. (1) as in draft. Present the general form used on LIBERO-Safety: `HAZARD(x) := PROTECTED(x) ∨ MOVING(x) ∨ (PRED(x) ∧ ¬MENTIONED(x) ∧ ¬TARGET(x))` where PRED is NearestToPath (obstacle suites) or a VLM property predicate HOT/SHARP (affordance) — one rule, suite-appropriate danger predicate, declared routing protocol table in appendix.
- **Para 4.1.2 — Predicate grounding: two interchangeable backends, plus abstention.** Hand heuristics + cached VLM priors vs VLM predicate grounding; identical decisions on 59 scenes; paraphrase competence ("deliver it to me" ⇒ hand protected). *(Keep.)* **Add**: abstention semantics — when the property predicate elects nothing, the filter guards nothing rather than force-electing (forward-ref to ablation row 4).
- **Para 4.1.3 — Geometric grounding at the no-GT tier.** Open-vocab detection + depth back-projection, error medians, conservative fallback. *(Keep.)* **Add one sentence**: part-level semantics are extracted (blade vs handle) but grounded at object level; part *perception* fails the pre-registered side-discrimination gate — forward-ref to Limitations.

### 4.2 Enforcement: noise-space operators (5 short paragraphs — keep draft structure)

- **Para 4.2.1 — Select (zeroth-order best-of-K).**
- **Para 4.2.2 — Tilt (Feynman–Kac reweighting, near-free).**
- **Para 4.2.3 — Ascend (first-order gradient through the denoise map).**
- **Para 4.2.4 — Soft repulsor field (the soft complement; unsticks obstacle-on-grasp).**
- **Para 4.2.5 — Baseline operator: composed in-denoising repair (the control arm).**
- **Possible added Para 4.2.6** ⏳: if an A/B smoke promotes, add stall-resample and/or progress-select as one paragraph ("two liveness refinements of Select") with dev-gate numbers; if not promoted, omit entirely — un-promoted mechanisms go to Negative Results only if informative.

### 4.3 Certificate (keep as-is)

- **Para 4.3.1 — Actuation model and barrier definition.**
- **Para 4.3.2 — Lemma (textbook geometric decay; no novelty claimed).**
- **Para 4.3.3 — Theorem 1: feasibility under actuation limits.**
- **Para 4.3.4 — Prop 2: checked prefix certificate (mechanism-independent).**
- **Para 4.3.5 — Prop 3 + Cor 4: selection soundness and mechanism independence; stated non-claims** (rotation, perception radius). **Frozen; proofs checked against implementation.**

### 4.4 Three-regime routing (keep)

- **Para 4.4.1 — The three regimes: Avoid / Duality-disengage / Refuse.** Refusal judge F1 with in-domain caveat. **Add one sentence**: routing decisions come from the same predicate layer as identification (PROTECTED ∧ REFERENCED ⇒ duality) — one rule system, not three systems.

## 5. Experiments I — Paired Enforcement Comparison (keep structure, close TODOs)

- **Para 5.1 — Pre-registered protocol.** Paired seeds, identical identity and certificate; enforcement is the only difference. *(Keep.)*
- **Table 1 + Para 5.2 — Round 1: five enforcement arms, and the repair tax attributed.** Projection is the success killer; Tilt/Ascend beat the control; repulsor alone moves the safety axis. **Frozen.**
- **Para 5.3 — Round 2: the Pareto frontier and the eta sweep.** Close the standing TODO with the n=200 finalization verdict (job 42640949): state which config landed in the pre-registered acceptance box, or that none did and composed repair remains production with the noise-space family reported as the tunable frontier. **Check ledger and write whichever is true.**
- **Para 5.4 — Renoise killed by its own gate; compute is the differentiator.** *(Keep.)* Close the latency-table TODO with server telemetry (repair / select / FK / ascend / repulsor per-chunk wall-clock; ascend 57 ms/step measured).

## 6. Experiments II — Identification (keep, close TODOs)

- **Table 2 + Para 6.1 — Identity-source swap: GT-free identity is indistinguishable from ground truth.** All p≥0.18, trends favor the FOL rule; the residual no-GT tax is geometric, not semantic. **Frozen.**
- **Para 6.2 — Fail-direction study: identification errors fail unsafe.** Forced second-best identity costs CAR (70→25), not TSR → motivates the top-2 guard-set mode (19/19 recall). Close the pending-extension TODO: report the n=50 + Goal/Object power extension if run, else delete the TODO and scope the claim to the cells reported.

## 7. Experiments III — SafeLIBERO Board (keep)

- **Table 3 + Para 7.1 — Five-architecture board at n=200, one collision criterion.** Include the referee footnote on the non-reproducible published CAR. **Frozen.**
- **Para 7.2 — Finding: post-hoc ≈ degenerate in-denoising on every cell.**
- **Para 7.3 — Finding: the composed machinery is the separator** (+6–19 TSR, 2–3× CAR over both shields where engaged, with Fisher p-values).
- **Para 7.4 — Finding: the Long-horizon stall afflicts every shield; ours recovers +15–19 TSR.**
- **Para 7.5 — Finding: the GT-free column headline** (dominates both references on both axes in 4/8 cells; TSR parity + CAR 2–5× without privileged state).

## 8. Experiments IV — LIBERO-Safety Board (REWRITE — biggest gap between draft and results)

The draft's current Sec. IV (off-the-shelf checkpoint tables) was **removed by directive 2026-08-02**. Replace with the safety-finetuned-checkpoint five-arm board:

- **Para 8.1 — Setup and provenance.** Safety-finetuned π0.5 checkpoint, four physical suites, benchmark violation scoring, uniform protocol; checkpoint-provenance footnote (self-run baseline, published numbers quoted with caveat); declared per-suite routing protocol referenced.
- **Table 4 — The five-arm board.** Baseline / post-hoc shield / SOTA-reimpl / ours-GT / ours-noGT × oa, hs, oah, aff; TSR/viol/SafeSR + macro. ⏳ uniform n=300 (jobs 43515322–26); fallback = frozen board (macro: base 71.4/2.5, posthoc 52.5/1.2, reimpl 50.0/0.5, GT 69.6/1.5, noGT 67.6/0.6).
- **Para 8.2 — Headline: we beat both dedicated shields by +15–20 macro TSR at equal-or-lower violations.** hs cells p=1.8e-16 / 3.2e-19; per-suite story: shields survive static-obstacle suites but halve human_safety (37.3/32.7 vs our 86.0) — safety by freezing vs safety while moving.
- **Para 8.3 — Against the unfiltered baseline: violations cut 12:1 at TSR parity.** noGT violations 12:1 over 750 paired episodes (p=0.003), TSR parity on oa/hs/aff; GT beats baseline TSR on oa (77.8 vs 73.6, n=450, p=0.045).
- **Para 8.4 — Honest disclosure: the oah freeze tax.** oah TSR significantly below baseline (p=0.005/0.03 by tier); baseline's oah successes carry 5.3% violations vs our 1.7–1.9%. If an A/B liveness arm is promoted ⏳, report the recovered oah number here with its paired test.
- **Para 8.5 — The GT/no-GT pair doubles as the position-provenance ablation.** Full perception costs ~2 macro TSR points. One sentence, forward-ref to Sec. 9.
- **Table 5 + Para 8.6 — The refusal track sweep.** reasoning_safety suite: judge refuses 100/100/80 vs best published 80/72/36, while the unguided policy *executes* unsafe instructions — the knowing-doing figure. Separate small table (no physical filtering involved).

## 9. Experiments V — FOL Ablation (NEW SECTION)

The causal-necessity evidence for every literal of Eq. (1). This is the novelty defense against "the CBF machinery does all the work."

- **Para 9.1 — Ablation design: delete one conjunct, pair everything else.** Each arm deletes exactly one clause; n=150/cell paired 1:1 against the full method on identical initial states; exact McNemar; incumbent column recomputed on the paired subset (footnote: headline cells pool larger n).
- **Table 6 — The ablation matrix.** Rows = deleted clause × suite; columns = full / ablated TSR + viol, discordants, p:
  - `geom_only` (delete all logic; guard nearest-to-path top-k): oa 68.7→44.0 + viol 0.7→4.7 (p=1.2e-6, the only arm that *raises* violations); oah 56.0→39.3 (p=2.5e-4); hs 88.7→68.7 (p=5.3e-6); aff ⏳.
  - `no_exempt` (delete ¬MENTIONED∧¬TARGET): oa 68.7→40.0 (p=1.8e-8, self-paralysis on the task target); oah 56.0→39.3 (p=6.2e-4); aff ⏳.
  - `no_class` (delete PROTECTED∨MOVING): hs 88.7→82.0 (p=0.10, trend); oah ⏳ **key cell** — the moving-hand suite is where class predicates should bind.
  - `no_abstain` (delete abstention; force-elect on empty election): affordance collapse chain 42.7→47.3→56.0 vs 56.7 base.
  - property provenance (hand-coded predicate tokens instead of VLM): 47.3 vs 56.0 on affordance — the VLM grounding is what scales.
  - position provenance (GT positions instead of perception): = the GT vs no-GT board rows (pointer, no new table row).
  - `no_heat_gate` (delete the state-aware hot-only-when-heated gate): offline-only — zero guard-set change on cached scenes, rollout identical by construction (one sentence, not a table row).
- **Para 9.2 — Reading the matrix.** Every significant arm confirms its pre-registered prediction; geometry alone is not merely weaker but *less safe* (distance ranking evicts the true hazard from the guard set); exemptions are worth ~29 TSR points; abstention ~13. If oah/no_class lands significant ⏳, the class-predicate row completes the matrix; if not, report as trend and scope the claim to the suites where it binds.

## 10. Negative Results (keep all five, tighten)

- **Para 10.1 — Renoise repair: no gain over rejection sampling at equal compute.**
- **Para 10.2 — Safety prompting is not a safety mechanism** (four testbeds; language moves task behavior, never avoidance).
- **Para 10.3 — Scene graphs: NO-GO three times** (ties flat scorer on object hazards, collapses on relational; the fix was a 10-line class rule, not structure).
- **Para 10.4 — VLM rule composition: bounded on both sides, fails unsafe** (oracle grounding gains ≤1 scene; real critics strictly worse; empty over-restrictive conjunctions leave the true hazard unguarded).
- **Para 10.5 — Whole-scene VLM identification: 18–64% where the decomposed rule scores 100%.**
- **Para 10.6 — Shape ablations: what transferred is the corridor exemption and the protected class, not the geometry.**

## 11. Limitations and Discussion (3 paragraphs)

- **Para 11.1 — Honesty items.** Self-run reimplementation caveat; checkpoint provenance; refusal lexicon in-domain; pixel-only naming future work. **Update**: cell sizes now n=300 uniform (or state the heterogeneity if the rerun doesn't finish ⏳); remove the stale "n=50 single-seed" sentence.
- **Para 11.2 — Structural limitations, including part grounding.** Obstacle-on-grasp ceiling; oah freeze tax (quantified, disclosed in board); identity errors fail unsafe at k=1; rotation uncertified; benchmark-definition divergence (Long L2). **Add — part grounding**: VLM part semantics (blade/handle) are extracted but perception fails the side-discrimination gate (2/5 vs ≥4/5), so barriers are object-level; the semantic layer is ahead of what perception can ground; GT-part bracket (sim geoms) is the proposed test of the headroom.
- **Para 11.3 — Real-robot transfer discussion.** Close the TODO: sensing error floors vs our tuned margins (3.6–6.7 cm medians), OSC tracking for A1′, Tilt/Ascend latency at control rate, A2 mover bound (≤8 cm/s) vs natural hand speeds.

## 12. Conclusion (1 paragraph — keep draft, add one clause)

Draft conclusion stands; add the ablation clause: "…and every literal of the identification rule is causally necessary, measured by paired deletion."

## Appendix

- Proof sketches (keep).
- Declared per-suite routing protocol table (from ledger 2026-08-09 entry).
- Full ablation arm configs + episode counts.
- Referee radius-sweep detail if the main-text footnote is cut for space.

---

## Open dependencies (as of 2026-08-17)

| # | Item | Feeds | Job(s) | Fallback if not landed |
|---|---|---|---|---|
| 1 | FOL ablation arms: oah/no_class, aff/geom_only, aff/no_exempt | Sec. 9 Table 6 | 43461517 (pending) | Report completed 6 rows + cached 3; scope class-predicate claim to hs trend |
| 2 | Uniform n=300 LS board (arms 3–6) | Sec. 8 Table 4 | 43515322/23/25/26 | Frozen heterogeneous board with per-cell n column |
| 3 | A/B liveness smokes (stall_resample, progress_select, composed) | Sec. 4.2 + Sec. 8 Para 8.4 | 43523242/44/45 → promotion run if gate passes | Omit from Method; oah tax stands as disclosed |
| 4 | Affordance n-boost (winner's-curse audit) | Sec. 8 Table 4 aff cell | 43211480 (check status) | Frozen 56.0/0 cell |
| 5 | SafeLIBERO K-sweep | Sec. 5 compute paragraph | sl_ksweep_1–4 | Report K=8 only |
| 6 | E1 n=200 enforcement finalization verdict | Sec. 5 Para 5.3 TODO | 42640949 (check ledger for verdict) | State composed repair remains production |
| 7 | GT-part bracket (proposed, cheap) | Sec. 11 Para 11.2 | not yet queued | Limitation stated qualitatively |

**Write order recommendation**: Sec. 9 (new, all-but-3-cells frozen) → Sec. 8 rewrite (frozen board now, swap numbers when uniform-300 lands) → close Sec. 5/6 TODOs from ledger → Limitations refresh → abstract/intro number pass last, after all tables freeze.
