# Direction Evaluation: CAVF, SCFM, Continual Safety Learning, and Everything Else

**Date:** 2026-07-12 (evening) · Companion to `2026-07-12-context-aware-safety-filter-brainstorm.md` and `brainstorm_idea.md` (CAVF spec)
**Provenance note:** the continual-learning literature fan-out failed on an API spend limit; §4's survey is from model knowledge (solid through late 2025; post-2025 entries carried over from the earlier verified survey). Everything else is grounded in the verified deep-reads.

**User's stated objectives (priority order):** (1) TSR and CAR/damage performance on SafeLIBERO, LIBERO-Safety, OopsieVerse; (2) a *general* context-aware safety filter (backbone-agnostic, hazard-general).

---

## 1. Direction inventory — evaluation matrix

Scores: ● strong ◐ partial ○ weak. "Beat-SOTA" = probability of beating 82.8 CAR / 81.6 TSR on SafeLIBERO if executed well. "9wk" = fits ICRA deadline.

| # | Direction | Beat-SOTA | Generality | 3-bench fit | Novelty | 9wk | Verdict |
|---|---|---|---|---|---|---|---|
| D1 | **SCFM** — semantic guidance inside π0.5 denoising | ~0.5 | ○ (flow-VLAs only) | ◐ | ◐ combinational | ● | Adopt as the π0.5 **candidate generator** inside D9, and as the white-box ablation |
| D2 | **SafeSelect-N** — best-of-K chunk sampling + safety critic | ~0.35 alone | ◐ (stochastic policies only — see §3.3 OpenVLA caveat) | ◐ | ◐ | ● | Adopt as second candidate source; mechanistic ablation spine |
| D3 | **CP-Cert** — conformal per-chunk collision bounds | n/a (rigor layer) | ● | ● | ● (first on SafeLIBERO; BOKBO is std-LIBERO) | ● | Adopt; calibrated margins feed every tier |
| D4 | **FaR** — stall detection + detour/recovery | n/a (component) | ● | ● | ◐ composition claim | ● | Adopt as fallback tier + Goal-L2 retrofit row |
| D5 | Attention-guided perception (2606.09749 mechanism) | n/a | ○ (needs internals; CAVF §6.1 forbids requiring it) | ◐ | ○ (published) | ◐ | Optional perception feature only, gated on <8cm |
| D6 | Guidance-as-Teacher distillation into π0.5 | ~0.3 | ○ | ○ | ◐ | ○ | Deferred to journal (unchanged) |
| D7 | Standalone policy-agnostic flow corrector (residual repair model) | ◐ | ● | ● | ● (unclaimed for VLA safety) | ○ **data pipeline is the crux** | Core of CAVF; stage it — stretch for ICRA, core for journal (§3.2) |
| D8 | **Continual safety learning from RL/execution feedback** | ○ direct | ● in principle | ◐ | ◐–● (semantic variant possibly unclaimed) | ○ as a loop | Adopt only two offline-safe elements now; defer the loop (§4) |
| D9 | **CAVF** — the umbrella architecture (context graph → compiler → candidates → critic → hard verifier → router/fallback) | ~0.5 via D1+D2 inside it | ● | ● (designed for all three) | ● as a system | ◐ **only in staged form** | **Adopt as the paper's architecture, staged (§3)** |

**Bottom line: CAVF and SCFM are not competitors — SCFM is one candidate generator inside CAVF.** The ICRA paper should be CAVF-the-architecture with cheap candidate sources (guidance + resampling + detour primitives); the residual rectified-flow repair generator (D7) and the continual outer loop (D8) are the journal extension. This preserves the head-to-head SOTA shot (via the π0.5 guidance path) *and* the generality claim (via the OpenVLA path), without betting the deadline on training three new models.

---

## 2. Why CAVF is the right umbrella (and what it fixes in my earlier plan)

The earlier SCFM-flagship plan optimized for the SafeLIBERO leaderboard; CAVF optimizes for the actual stated goal — a **general** context-aware filter. Specific things CAVF gets right that the SCFM plan underweighted:

1. **Backbone-agnosticism as a first-class claim.** 2607.01378 works only on flow-matching VLAs. AEGIS works on anything but caps TSR. A filter that improves BOTH π0.5 and OpenVLA-OFT with shared weights/logic is a claim neither can make — and it's the claim that matches "general context-aware safety filter."
2. **Safe-success viability objective.** Explicitly separating safe-success / unsafe-success / safe-failure / unsafe-failure kills the "CAR up by stalling" pathology at the objective level, not by patching. This should be the paper's stated objective function regardless of which components ship.
3. **Multimodal repair beats min-norm correction.** The July 11 pilot (tangential projection, 25 TSR / 0 CAR) and the whole v1–v21 campaign are evidence that a single local correction is globally unhelpful. Route-conditioned candidates + a ranker is the principled fix. (Note: in-denoising guidance partially escapes this because the policy re-adapts between Euler steps — that's why D1 stays the π0.5 workhorse.)
4. **Full-system verification** (arm links, gripper, payload swept volume) attacks a gap every SafeLIBERO method shares — EEF-only geometry. Cheap to implement (we have FK + link checkpoints already) and worth real CAR on cells where the *hand/forearm* plows obstacles (our video forensics: L1 t0 moka-plow).
5. **Execute/repair/recover/redirect/refuse hierarchy** is exactly the LIBERO-Safety SSR story (safe alternative vs binary refusal — F1 headroom 0.31–0.46) and the Robey/Kolter layered-architecture axis, in one router.

### Corrections/cautions on the CAVF spec (honest review)

- **§12.3 numbers check out** against the verified deep-read (SOTA Spatial per-level CAR 77.5/75.5, TSR 75.5/78.0). ✔
- **The matched-pair data pipeline (§8.2) is the single biggest schedule risk in the whole document.** `(context, unsafe nominal, safe task-preserving repair)` pairs at scale require planner/MPC repairs from matched states (CuRobo-style infra we don't have) — realistically 3–5 weeks of engineering alone. This is why the repair generator must be staged (§3.2), not core for ICRA.
- **OpenVLA-OFT cannot be resampled** (§3.3): its OFT head is deterministic continuous regression (L1 loss, parallel decoding) — there is no temperature to turn up. Candidate diversity for OpenVLA must come from *external* sources: geometric detour primitives (FaR routes: left/right/above/retreat — matching CAVF's route modes), verifier-projected variants, or (journal) the trained repair flow. The generality claim is real but it is the *expensive* half of the paper — budget it explicitly.
- **The critic has a cold-start asset the spec doesn't mention:** the v1–v21 campaign left thousands of collision-labeled, success-labeled SafeLIBERO rollouts on disk (plus 400 saved observation sets). A safe-success critic head can be trained on *existing* logs (Bajcsy-position-paper-style negative data) before a single new rollout is collected. This de-risks Stage A substantially.
- **Compute estimate (§16) is fine; the data estimate is the problem.** 160–380M params on 1×H100 is realistic; generating and validating matched repairs is not, in 9 weeks.
- **§10's "internal flow-guidance as white-box ablation only" is strategically wrong for ICRA** — it discards our only ~0.5-probability shot at the SafeLIBERO leaderboard. Resolution (§3): guidance IS one of CAVF's candidate generators on π0.5 (architecture unchanged — candidates are policy-source-agnostic), so the main result and the leaderboard shot are the same system. The external-interface generality claim is carried by the OpenVLA arm.

---

## 3. The staged plan: CAVF-ICRA (9 weeks) vs CAVF-Journal

### 3.1 CAVF-ICRA — what ships in 9 weeks

Architecture exactly as the CAVF spec (context graph, FOL/STL compiler, hybrid rollout, hard verifier, viability ranking, router, fallback), with **candidate generation instantiated cheaply**:

| Backbone | Candidate sources (K≈8–16) | Notes |
|---|---|---|
| π0.5 | A0 + fresh-noise resamples (D2) + semantically-guided samples (D1, in-denoising) + detour-primitive repairs (D4) | Leaderboard arm; guidance gives the +TSR mechanism |
| OpenVLA-OFT | A0 + detour-primitive repairs (D4 routes: left/right/above/retreat/slow) + verifier-tube projections | Generality arm; honest expectations (base 40–42 TSR) |

- **Critic**: analytic barrier margins + small learned safe-success head trained offline on the existing v1–v21 logged rollouts + new training-scene rollouts; calibrated (D3) with conformal quantiles; frozen at eval.
- **Hard verifier**: full-arm + gripper + payload swept volume (the whole-arm white space), allowed-contact rules from the context graph, orientation/spill constraints.
- **Router**: exactly CAVF §5, including refuse/redirect for SSR.
- **Benchmarks**: SafeLIBERO (all suites, both backbones, self-run ladder) → LIBERO-Safety TSA/FSHOA + SSR refusal-F1 (released π0.5) → OopsieVerse RoboCasa (behind failure-taxonomy gate; damage-aware caution constraints, honest about grip-crush).
- **Headline claims**: (i) one context-aware filter, two backbones, three benchmarks, TSR *and* CAR/damage up; (ii) SafeLIBERO head-to-head via the π0.5 arm (contest 82.8/81.6); (iii) first method rows on LIBERO-Safety/OopsieVerse; (iv) conformal per-chunk bounds; (v) safe-success objective + full-system verification.
- **Pilot gate** = CAVF §12.4 unchanged (Spatial I/II, both backbones, paired-state analysis).

### 3.2 CAVF-Journal — deferred core

- **Residual rectified-flow repair generator (D7)**: bootstrap matched pairs from the ICRA system itself — guided-vs-unguided π0.5 chunks from identical states are free (A0, A_guided) repair pairs; add planner repairs later. Trained repair flow then replaces detour primitives on ALL backbones (this is what makes OpenVLA candidates rich) — likely the biggest generality win, and it converts D6 (distillation) into one more ablation.
- **Continual outer loop (D8)**: §4's semantic constraint refinement + critic updates across deployments, on training scenes with frozen-eval protocol.
- Thermal/fluid OopsieVerse (OmniGibson/Isaac), dynamic obstacles (Level III), real-robot validation.

### 3.3 Revised 9-week skeleton (deltas from the previous plan)

W1: canonical interfaces + full-arm/payload verifier + critic training on existing logs + protocol reconciliation + FaR detector offline validation (unchanged). W2: SOTA replication gate (unchanged) + detour-primitive candidates on OpenVLA. W3: π0.5 arm full runs (guidance + resampling under CAVF router). W4: OpenVLA arm runs; critic/ranker calibration (D3). W5: pilot gate → suites beyond Spatial; **gate: if OpenVLA arm underdelivers, narrow generality claim to "architecture supports it" + report honestly**. W6–7: LIBERO-Safety + OopsieVerse (gated as before). W8–9: ablations (CAVF §14 list, subset), seeds, writing, code release.

---

## 4. Continual learning of safety from RL/execution feedback — assessment (solo survey)

**The idea**: the safety layer improves across episodes from its own execution feedback — collisions, near-misses, filter interventions, successes. Seed: Brunke et al. 2021 survey (arXiv 2108.06266), whose taxonomy (learn dynamics safely / learn safety certificates / safe exploration) and open problems (perception-in-the-loop safety, scalable certification, benchmarks) frame exactly this space.

### 4.1 Prior-art map (model knowledge; verify cites before paper use)

**Safety critics updated from experience:** Recovery RL (Thananjeyan et al., RA-L'21 — offline-pretrained risk critic + recovery policy, both refined online; feedback = violations), SQRL (Srinivasan et al. '20 — safety Q pretraining → filter + penalty during fine-tuning), Conservative Safety Critics (Bharadhwaj et al., ICLR'21 — CQL-style pessimism bounds failure prob during exploration), SafeDreamer (ICLR'24 — Lagrangian world-model RL). *Established: critics-from-feedback works but is sample-hungry and gives statistical, not hard, safety.*

**Learned/adaptive barriers & reachability:** neural CBFs from data (Robey et al. CDC'20; Dawson & Fan survey '23), adaptive CBFs (Taylor & Ames — parameter adaptation under model uncertainty), GP-based safe learning (Berkenkamp/SafeOpt lineage — data-efficient, low-dim only), learned HJ filters updated with data (DeepReach lineage; ISAACS; Bajcsy-group parameter-conditioned value functions), latent-space safety learned from rollouts (UNISafe CoRL'25 — conformal OOD + latent HJ; Latent Policy Barrier NeurIPS'25). *Established: barriers can be learned and adapted; nobody does it for semantic constraints or VLA manipulation benchmarks.*

**Online conformal adaptation:** Adaptive Conformal Inference (Gibbs & Candès, NeurIPS'21) and successors (DtACI/AgACI) — provable coverage under distribution shift with O(1) online quantile updates. *Directly usable for margin inflation that adapts across episodes.*

**Constraint inference & refinement:** inverse constraint learning (Scobee & Sastry ICLR'20; Malik et al. ICML'21), LLM reward/constraint evolution from rollout statistics (Eureka ICLR'24, Text2Reward ICLR'24). *The template for VLM safety-rule refinement from execution evidence exists in reward space; the safety-constraint instantiation for manipulation appears unclaimed — this is the genuinely novel kernel of D8.*

**Intervention-as-signal:** HG-DAgger, EnsembleDAgger, Sirius (RSS'23 — human interventions during deployment become training data). Replace "human" with "shield/verifier" ⇒ learning from filter interventions — adjacent to safe-DAgger/shield-distillation in safe RL; our Guidance-as-Teacher (D6) is exactly this, once iterated.

**VLA-specific (verified earlier):** SafeDojo (offline RL in a world model — NOT continual; weak L2 zero-shot +0.12pp is the cautionary tale that learned safety memorizes layouts), SafeVLA, RePO-VLA (recovery fine-tuning), GAF ("critic generalization and uncertainty-aware guidance remain the central bottlenecks" — their words).

### 4.2 Two structural problems for an ICRA-2027 continual paper

1. **Evaluation contamination.** Updating the filter *during* benchmark evaluation violates the frozen-eval convention every competitor follows (and CAVF §8.1 explicitly forbids). The clean protocol — continually learn on training scenes, freeze, evaluate — is scientifically fine but then the contribution is "offline training with self-generated feedback data, iterated," i.e., D6/D7's data flywheel, not a continual-learning result per se.
2. **Statistical resolvability.** A learning curve needs enough episodes to resolve; SafeLIBERO cells are n=50. Showing "safety improved from episode 1→50" with per-cell noise ±7pp requires effect sizes that would themselves be suspicious. Cross-task/cross-suite transfer curves (learn on Spatial, improve on Object) are more resolvable but are a generalization claim, not continual learning.

### 4.3 What to adopt (now) and what to defer

**Adopt for ICRA (both are offline-safe and cheap):**
- **Adaptive-conformal margin calibration across training episodes** (ACI-style updates on realized min-distances; frozen quantiles at eval) — slots directly into D3/CP-Cert; provable; replaces hand-tuned inflation. Framing: "the filter's uncertainty margins are *learned from its own execution history*."
- **Critic-from-negative-data**: train the safe-success critic on the v1–v21 logged rollouts + new training-scene rollouts — literally "safety learned from RL-era execution feedback," offline. Cite the Brunke taxonomy + the negative-data position paper as framing.

**Defer to journal (the real continual loop):**
- **Continual semantic constraint refinement** — VLM proposes constraint hypotheses; execution evidence (violations despite compliance / interventions never near violation) confirms, refutes, or re-weights predicates and margins across deployments, Eureka-style but for safety rules. Plausibly unclaimed; needs a benchmark protocol that doesn't exist yet (deployment rounds, held-out layouts) — designing that protocol is itself a contribution, and one more reason it's a paper of its own.
- Iterated guidance-distillation (D6 as DAgger loop), online critic updates, lifelong cross-benchmark transfer.

**Verdict on D8:** wrong headline for this ICRA cycle (contaminated eval or unresolvable curves; fights the training-free selling point; SafeDojo's L2 result shows learned safety generalizing poorly), but two of its cheapest instruments strengthen the ICRA paper immediately, and the full loop is a strong, differentiated journal/next-paper direction that CAVF's architecture is explicitly designed to accept (its §2.2 says exactly this).

---

## 5. Final recommendation (supersedes §4 of the previous spec where they differ)

1. **Adopt CAVF as the paper's architecture and framing** — it is the "general context-aware safety filter" the project is actually about, and every prior workstream (SCFM, SafeSelect, CP-Cert, FaR, FOL compiler) slots into it as a component rather than competing with it.
2. **Ship CAVF-ICRA (§3.1)**: cheap candidate generation (guidance + resampling on π0.5; detour primitives on OpenVLA), critic from existing logs, full-arm verifier, conformal calibration, router with refuse/redirect. π0.5 arm carries the SOTA head-to-head; OpenVLA arm carries generality; three benchmarks as previously gated.
3. **Defer to journal**: residual rectified-flow repair generator (bootstrap its training pairs from the ICRA system), continual semantic safety learning (D8 full loop), distillation, thermal/fluid damage, dynamic obstacles, real robot.
4. **Adopt now from D8**: adaptive-conformal margins + critic-from-negative-data (offline, frozen-eval-clean).
5. **Unchanged**: self-run baseline ladder on pinned checkpoints; W2 replication gate; W5 scope gate; honest-numbers discipline (LIBERO-Safety +1–4pp SR headroom; OopsieVerse grip-crush exclusion; >1mm-displacement CAR strictness footnoted in every cross-paper comparison).
