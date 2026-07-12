# Context-Aware Safety Filter — Research Direction Brainstorm (ICRA 2027)

**Date:** 2026-07-12 · **Deadline:** ICRA 2027, expected **Sept 15, 2026** (CFP not yet posted; historical pattern) · **~9 weeks out**

Produced from deep-reads of arXiv 2607.01378 (SOTA), 2606.20698 (SafeDojo), 2606.23686 (LIBERO-Safety), 2606.31993 (OopsieVerse), 2602.19983 (CORE), two literature surveys (~40 papers), a 4-lens proposal panel, and 2 adversarial critique passes (area-chair + control-theorist personas).

---

## 1. Verdict on the current FOL post-hoc filter: RETIRE as enforcement, KEEP as constraint source

**The post-hoc CBF/QP shield cannot beat 82.8 CAR / 81.6 TSR, even with a frontier VLM and π0.5. The blocker is architectural, not perceptual.** Four independent datapoints now bound the post-hoc family's TSR:

1. **Our own campaign**: with ORACLE perception (GT positions, zero grounding error), TSR ≈ baseline (42.0 vs 40.0/37.0; never above +5pp) at CAR 4.5–8×. Three n=50 barrier-reshaping iterations (v19/v19e/v20/v21-D2) provably only *redistributed* successes. Goal suite regressed −13.5pp; corridor cells → 0.
2. **AEGIS on π0.5** (VLSA paper, the SafeLIBERO creators, vision-based perception): 77.9 CAR / **68.1 TSR** — +17.3pp TSR over base, with Long collapsing to 43.8. They name our failure mode verbatim: *"enforcing safety with AEGIS can drive the system into out-of-distribution states… where the policy may behave erratically and fail to recover toward the goal."*
3. **Attention-guided filter** (arXiv 2606.09749, per-step CBF on π0.5): SR ~67.5 — same ~68 plateau.
4. **SafeDojo's AEGIS reimplementation on OpenVLA-OFT**: TSR −19.7pp vs its own SFT baseline.

Beating 81.6 TSR requires **+30.7pp over the unguided π0.5 base (50.88)**. The best post-hoc uplift ever recorded is +17.3pp, achieved *with* GT-quality obstacle state. The missing ~13pp comes from the policy **re-planning around** constraints — definitionally absent when a frozen policy's output is clipped after generation. Better VLM ≠ fix: perception was never the binding constraint (our oracle tier proves it).

**What survives from 18 months of FOL work:**
- The **VLM→FOL→CBF constraint compiler** (symbolic obstacle grounding 100% vs VLM whole-scene 0–10%, multi-prompt majority voting, caution/rotation-lock semantic classes, superquadric geometry, pixels-only 2-view triangulation funnel) — this becomes the **constraint source** for the new enforcement mechanism. Nobody else has it.
- The **negative result itself** — now corroborated by three published papers — is citable motivation and an ablation row (we can *reverse* our own Goal L2 regression as a demonstration).
- The entire SafeLIBERO eval harness, π0.5/OpenVLA/Cosmos/Fast-WAM baselines, 400 saved episode observation sets.

---

## 2. Corrected reading of the SOTA (arXiv 2607.01378)

Deep-read corrections to what we believed:

- **Numbers**: 82.81 CAR / 81.62 TSR average. The abstract's "+6.3% / +19.8%" over AEGIS are **relative** (absolute: +4.96pp CAR, +13.49pp TSR). Per-suite (CAR/TSR): Spatial 76.50/76.75, Goal 88.25/87.25, Object 84.00/85.75, Long 82.50/76.75.
- **Base**: frozen π0.5-LIBERO, **unguided base = 50.88 TSR / 18.69 CAR** — comparable to our measured π0.5 (57.9/19.3, 7pp protocol gap to reconcile in W1). SOTA = filter uplift, not a better policy.
- **Mechanism**: after every Euler step of the 10-step flow denoising, interpret the intermediate chunk (H=10) as an EEF trajectory via single-integrator rollout of translational deltas; enforce discrete exponential-CBF condition B_j ≥ (1−γ)B_{j−1} (γ=0.9, d_safe=1cm) via min-norm SLSQP (30 vars, <1ms); the corrected chunk feeds the next denoising step so the velocity field **adapts to corrections** — that's where the TSR comes from.
- **"Neuro-symbolic" is marketing**: neural = π0.5, symbolic = hand-written geometric distance constraints. **No VLM, no logic, no semantics, no mechanism to decide what is an obstacle.**
- **Privileged state**: obstacle positions/geometry *from the simulator* ("isolating the control contribution from perception challenges"); perception-derived constraints explicitly deferred to future work.
- **Other gaps**: EEF-only (no arm links), sphere obstacles with unstated radii, **zero ablations**, only 2 baselines, +37% ETS on Object, **no code** (project page: "Code Coming Soon"), episode counts implied not stated, no seeds/error bars. Venue so far: NeuS 2026 workshop-scale conference.

**SafeLIBERO leaderboard (avg CAR/TSR, all on π0.5 unless noted):**

| Method | CAR | TSR | Notes |
|---|---|---|---|
| Flow guidance (2607.01378) | **82.8** | **81.6** | GT state, EEF-only, no code |
| AEGIS (VLSA 2512.11891) | 77.9 | 68.1 | vision-based, per-step QP; creators of benchmark |
| Attention-guided (2606.09749) | ~76.4 | ~67.5 | π0.5 attention = obstacle selector; adds dynamic Level III |
| SafeDojo (2606.20698, OpenVLA-OFT) | 69.5 | 64.5 | safe RL in video world model; 32×A800; L2 margin +0.12pp only |
| SafeVLA (in SafeDojo tables) | 74.0 | 54.3 | safe RL |
| π0.5 base | 18.7 | 50.9 | |

Nothing else evaluates on SafeLIBERO (verified via citation graph of the VLSA paper). Cross-paper caution: AEGIS scores 68 TSR on π0.5 but 32 on OpenVLA-OFT — **backbone and reimplementation noise are huge; we must self-run every baseline on one pinned checkpoint.**

---

## 3. The unoccupied intersection (novelty white space)

1. **Perception-grounded, semantically-selected constraints inside flow denoising.** SOTA: GT geometry only. AEGIS: vision but post-hoc QP. OmniGuide (2603.10052): VLM-grounded *energy* guidance in π0.5/GR00T denoising but no barrier conditions, no safety benchmark — **closest uncited threat, must cite and ideally port as a baseline**. TouchGuide proves π0.5 accepts in-loop classifier guidance (mechanism precedent — the constraint *source* is our novelty, not the mechanism).
2. **Semantic constraint families inside a generative sampler**: caution class-K slowdowns, rotation locks, per-class margins, grasped-object volume union — things spheres cannot express and no guidance paper has.
3. **Shield + recovery composition**: recovery literature (VoLo, Rewind-IL, RePO-VLA) never touches shields; shield papers document stalls without fixing them.
4. **Safety-critic best-of-N on a safety benchmark**: RoboMonkey/TACO/GAF verify task quality, not collisions; BOKBO (2605.30660) is conformal **abstention** on **standard LIBERO** (verified — not SafeLIBERO), abstains rather than completes.
5. **Cross-benchmark generality**: no method reports on SafeLIBERO + LIBERO-Safety + OopsieVerse. LIBERO-Safety's enforcement leaderboard is **empty** (their own Appendix-D chunk-level CBF, Eq. A.3, has zero quantitative numbers — we can publish its first numbers as our geometric baseline). OopsieVerse (RSS 2026, 12 days old) has no method results at all.
6. **Distribution-free per-chunk collision certificates on a manipulation benchmark** (CORE's martingale bound is navigation-only, average-rate calibrated, not conformal).
7. **Whole-arm safety**: every SafeLIBERO filter is EEF-centric.

---

## 4. Recommended plan: SCFM-A flagship + capped workstreams

### Flagship: **Semantic Constrained Flow Matching with Model-Consistent Acceptance** (merged best-of from 4 panel variants + critiques)

VLM→FOL semantic constraint compiler (reused, upgraded) → superquadric/ellipsoid barriers with SQ-CBF-stabilized gradients (2602.11049) → **min-norm correction at every Euler step of π0.5's denoising** (reimplementing 2607.01378 — first public code) → **terminal acceptance test on the fully-denoised chunk** (executed-path DCBF check; resample on failure, safe-prefix fallback; SafeFlowMatcher insight) → report **terminal-correction magnitude/frequency** so drift stays visibly small.

**Additions over SOTA:** semantic selection of *what* to constrain (no benchmark-designated obstacle), semantic constraint classes (caution class-K γ-modulation, rotation locks during liquid transport, per-class margins), grasped-object volume union (debounced grasp detection), wrist-axis capsule arm heuristic (honestly labeled), full ablation table (γ, N, H, d_safe, guidance placement — first for this family), SSR as co-headline metric.

**Three perception tiers** (the reviewer-proof ladder):
- **T0**: replication of 2607.01378 on pinned checkpoint (validates harness; W2 gate: within ~3pp).
- **T1 (apples-to-apples)**: GT obstacle state + our geometry/enforcement → contest 82.8/81.6 directly.
- **T2 (semantic)**: VLM-selected constraint set + GT positions → "first no-oracle-obstacle-set entry at SOTA level". Gate: ≥90% selection accuracy offline on prompt_tuning_benchmark_set *before* GPU commit.
- **T3 (pixels-only)**: our triangulation funnel + **conformally calibrated margin inflation** (from CP-Cert workstream) → first perception-derived in-generation entry. Honest range: 60–75 CAR / 72–78 TSR (8cm error vs 1cm d_safe closes some L2 corridors — pre-compute corridor clearance on the 400 saved episodes).

**Primary claim framing**: uplift-over-matched-base on a self-run ladder (base → AEGIS → reimplemented SOTA → ours) on one pinned checkpoint; published SOTA numbers become the footnote. This survives even if replication lands a few pp off their table.

### Capped workstreams (each feeds the flagship, none is a standalone paper)

- **SafeSelect-N** (≤25% GPU): sample K chunks from fresh noise, analytic barrier-critic selection, escalation ladder (fresh noise → partial re-noise → safe-prefix → bounded projection). Product = the **mechanistic decomposition** (selection vs correction vs stop) reviewers will ask for + OpenVLA-OFT generality arm. W1 go/no-go: fraction of blocked states with a clearing sample at K=16 (<30% ⇒ selection is inert in corridors; demote to ablation).
- **CP-Cert** (cheap, high rigor value): split-conformal calibration of chunk collision scores on realized simulator distances. **Repaired statement** (theorist): joint/selective-risk bound P(collision ∧ accepted) ≤ α (or RCPS), accepted-branch only, one-chunk-per-episode exchangeability variant, escalation frequency reported, L1-calibrate → L2-test coverage-degradation experiment front and center. Ship calibrated per-object inflation quantiles into T3 regardless.
- **FaR (stall recovery)** (gated): stall detector validated **offline on already-logged v19/v20 Goal-suite failures in W1 (zero GPU)**; geometric detour candidates, VLM veto; injected as guidance attractor with constraint-priority line-search + hysteresis. Headline row: **Goal L2 retrofit 7.5 → ≥21 TSR** (reversing our own published-quality negative result). Run geometric-only ablation first — if VLM veto adds nothing, ship as geometric stall-recovery without VLM branding.

### Breadth: cross-benchmark first-mover (behind gates)

- **LIBERO-Safety** (W6+, same robosuite/MuJoCo stack, released π0.5 weights): first method rows on **TSA/FSHOA**; quantify their own Eq. A.3 chunk-CBF as the geometric baseline. **Honest expectations**: trained π0.5 CR is only 2–6.3% on static cells ⇒ SR headroom is +1–4pp there; larger gains only on dynamic/unseen cells if chunk-boundary re-grounding works (quantify ~0.5s staleness vs scripted MANO hand speed). SSR suite: refusal-F1 protocol primary, safe-alternative execution as pre-registered qualitative+coverage row. Their p.42 future-work agenda literally describes our architecture — quote it as third-party motivation.
- **OopsieVerse** (stretch, behind W6 failure-taxonomy gate): RoboCasa/MuJoCo track only (no Isaac Sim risk). **Physics honesty (theorist)**: velocity caps bound impact/impulsive force only — NOT quasi-static grip crush (gripper dim untouched by every filter) and NOT post-grasp drops. Scope claims to approach-impact damage tasks; check GR00T failure taxonomy (impact vs crush vs drop) *before* committing the integration week. Here post-hoc-style modulation is admissible: the hazard IS the task object, so slowing near it doesn't block corridors.

### What we explicitly do NOT do (this cycle)

- **Guidance-as-Teacher distillation** (self-distill filter into π0.5 weights, zero-overhead deployment + FAR-collapse figure): genuinely attractive, but serially dependent on the flagship landing by W4, needs unexercised openpi fine-tuning infra, and its distilled-only artifact recreates SafeDojo's L2-generalization weakness (+0.12pp). **Hold for the journal/follow-up; bank the teacher rollouts.**
- World-model runtime veto (Cosmos): latency + scope creep; no claim rides on it.
- Latent-space CBF: no guarantees story reviewers reward, high risk.
- AAG affordance suite (part-level grounding beyond our funnel) — GT-mesh-region tier only if time allows.

---

## 5. Nine-week schedule (gates in bold)

| Week | Work |
|---|---|
| W1 | Protocol reconciliation (their 50.88 vs our 57.9 π0.5 base — max steps/init states/episode counts); openpi denoising-loop injection point + batched-noise latency measurement; **SafeSelect diversity go/no-go**; FaR stall-detector offline validation on logged failures; VLM selection accuracy offline (**≥90% gate for T2**); corridor-clearance vs margin-inflation precompute on 400 saved episodes. |
| W2 | **T0 replication gate (within ~3pp of published or self-run ladder becomes primary framing)**; SQ-CBF gradient integration; terminal acceptance test. |
| W3 | T1 GT-tier full runs (n=50, all 8 conditions); AEGIS self-run baseline on pinned checkpoint. |
| W4 | T2 semantic tier; SafeSelect decomposition runs; **FaR gate** (geometric vs VLM-arbitrated ablation); CP-Cert calibration. |
| W5 | T3 pixels tier with calibrated inflation; **W5 gate: flagship on track ⇒ breadth; else activate fallback framing (breadth-primary)**. |
| W6 | LIBERO-Safety TSA/FSHOA integration + Eq. A.3 baseline; **OopsieVerse failure-taxonomy gate**. |
| W7 | LIBERO-Safety runs; OopsieVerse RoboCasa runs (if gate passed); Goal L2 retrofit row. |
| W8 | Ablation table (γ/N/H/d_safe/placement); seeds for headline cells; writing. |
| W9 | Writing, figures (mechanism figure: correction magnitude across denoising steps; decomposition figure), code release prep. |

## 6. Risk register (from adversarial critiques)

| Risk | P | Mitigation |
|---|---|---|
| Replication misses published numbers (no code, unstated r_obs/episodes) | med | Self-run ladder = primary evidence; their numbers a footnote |
| Superquadric-vs-sphere CAR effect is **sign-uncertain** (tighter fit hugs closer) | med | T1 measures it directly W3; margins tunable; don't pre-claim +CAR from geometry |
| Corridor closure at 8cm perception error kills T3 | med | Calibrated inflation; honest 60–75 CAR pre-registration; T2 tier exists precisely to isolate this |
| SafeSelect diversity collapse in L2 corridors | med-high | W1 gate; demote to ablation spine |
| LIBERO-Safety SR headroom arithmetic (CR only 2–6%) | certain | Promise CR-halving + 1–4pp SR, not +5–15pp |
| OopsieVerse grip-crush unaddressable | high | Taxonomy gate; scope to impact-damage tasks |
| Schedule stack (3 benchmarks, 1 flagship) | high | Hard W5 gate + pre-committed cut order: OopsieVerse first cut, LIBERO-Safety second |
| Certificate language overclaim (single-integrator, open-loop, arm redundancy) | certain if unfixed | "Model-consistent acceptance", empirical rollout-error bound from saved episodes; repaired conformal statement |

## 7. Surgical novelty sentence

*"We present the first safety-guidance system for flow-matching VLAs whose constraints are **semantically selected and perception-grounded** rather than benchmark-designated: a VLM→FOL compiler emits CBF-style constraint families (exemption-set obstacle selection, caution class-K modulation, rotation locks, grasped-object union) enforced by min-norm corrections inside π0.5's denoising with a model-consistent terminal acceptance test — evaluated at three perception tiers on SafeLIBERO against self-run baselines, with the first quantitative safety-method entries on LIBERO-Safety [and OopsieVerse], first per-chunk conformal collision bounds on a manipulation safety benchmark, and public code."*

Must-cite differentiations: 2607.01378 (enforcement locus; GT-state, no semantics), OmniGuide (energy guidance, no barriers/benchmark), TouchGuide/VLS (mechanism precedent), SafeDiffuser/SafeFlowMatcher (CBF-in-denoising for planners), AEGIS/Brunke/CORE (VLM→CBF post-hoc / navigation), BOKBO (conformal abstention, standard LIBERO), SQ-CBF, CompliantVLA-adaptor (impedance vs barriers), Robey et al. Sci Robotics framing.
