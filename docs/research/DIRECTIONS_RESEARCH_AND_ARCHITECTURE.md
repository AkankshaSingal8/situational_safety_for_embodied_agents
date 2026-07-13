# Context-Aware Safety for Flow VLAs: Research, Architecture, Methods, Novelty
**Consolidated reference — 2026-07-13.** Companion progress log: `PROGRESS_LOG.md`. Raw survey outputs: `steer_{theory,robotics,synth}.json`, `semsafe_{ident,ground,synth}.json` (this directory). Specs: `docs/superpowers/specs/2026-07-12-*.md`. Results ledger: `.worktrees/flow-guidance-tier-gt/results_tables/fgd_tier_gt_results.md`.

## 0. Mission (locked 2026-07-13)
Beat **both** the unfiltered π0.5 baseline (ours: n=50, seed 7, >1mm-displacement collision criterion) **and** the SOTA (arXiv 2607.01378; published avg 82.8 CAR / 81.6 TSR; self-run reimplementation is the honest referee) on **both TSR and CAR**, with the best method reported at **two privilege tiers** — GT simulator state and no-GT (VLM + RGB-D). Venue: ICRA 2027 (~Sept 15). Three directions: (1) identifying contextual/semantic safety, (2) steering the flow policy, (3) [parked] learning from steering.

## 1. Why this architecture: the two governing empirical facts
1. **Post-hoc shields have an architectural TSR ceiling** (~68 TSR on π0.5) — quadruple-confirmed (our FOL campaign oracle tier; AEGIS; attention-guided filter; SafeDojo's AEGIS row). A frozen policy never sees post-hoc corrections and keeps emitting the blocked intent. In-generation guidance re-plans: corrections injected between flow-matching Euler steps become part of the trajectory the model itself completes.
2. **Uniform geometry cannot answer semantic questions.** Measured on SafeLIBERO: any single envelope either leaks collisions (thin) or blocks sanctioned task motion (fat) whenever the obstacle is near the target/grasp-point/destination. Spatial L1 traded 71.0/23.5 ↔ 58.5/44.5; Long L1 collapsed to 0% TSR at the destination. The fix must know *what the task sanctions touching* — that is Direction 1's entire reason to exist.

## 2. Current architecture (implemented; worktree `flow-guidance-tier-gt`)
```
CLIENT (eval loop, per chunk)                    SERVER (guided pi0.5, JAX)
 obstacle discovery (GT tier: sim names/pos)      GuidedPolicy.infer: pops `guidance`
 entity parsing: target + destination from        payload BEFORE input transforms
   instruction x obs keys (corridor sanction)      (policy observation path untouched)
 guidance payload: {obstacle pos/radius, eef,       │
   target_pos, dest_pos, enabled}                   ▼
 executes first 5 of H=10 actions (receding      guided_sample_actions:
 horizon); collision metric >1mm displacement     prefix (vision+language) ONCE → KV cache
                                                  broadcast cache to K candidates (axis 1)
                                                  10 Euler steps; after EACH:
                                                   DCBF repair sweep (see §3) with
                                                   denoising-time schedule + corridor-
                                                   scaled margins + companion points
                                                  final: executed-prefix acceptance →
                                                   lexicographic best-of-K selection
                                                  diagnostics: correction norm, min
                                                   clearance, residual, feasible count
```
Isolation: private openpi copy (`openpi_guided/`), one server serves guided/unguided (enabled=0 = bit-identical passthrough), 15 unit tests, per-episode JSONL, flaky nodes excluded, server JIT warmup pre-port.

## 3. Direction 2 — steering the flow policy (state: built, n=20-confirmed)
**Enforcement stage (the backbone):** per-Euler-step repair of the intermediate chunk to satisfy the discrete ECBF chain `B_j ≥ (1−γ)B_{j−1}` over a single-integrator EEF rollout (quantile denorm inside the loop; translational dims only; γ=0.9, d_safe=0.01).
Components layered on it (each a flag, each tested):
- **Denoising-time schedule** (SafeDiffuser-trap/ITPS/PC-Diffuser convergent finding): zero repair steps 0–1, ramp 2–6, full ≥7. Measured outcome-neutral alone at n=5; retained (no harm; theory says it buys headroom for fat margins).
- **Brake-then-push channel**: shrink approach component first; on per-dim clamp saturation choose max-barrier of {clamped push, full brake} → **chain feasibility degrades provably to non-worsening under actuation limits** (one-paragraph proof; residuals reported, ≈0 in practice).
- **Companion points** (hand +8cm, payload −6cm; `companion_scale` switch for the reimpl arm): full-manipulated-system geometry; the single biggest CAR mover (L1 23.5→44.5 at n=50).
- **Best-of-K (K=8) + executed-prefix certificate**: K noise seeds, shared prefix KV (2–3× latency, not 8×), selection = prefix-feasible → max margin → progress. Measured: Object L1 t1 TSR 18→80 (n=5). The certificate is *model-consistent* (single-integrator, open-loop) — never claimed as invariance.
- **Corridor/destination exemption** (Direction 1's first tooth, see §4): margins ×(1−0.6) inside cylinders around eef→target and eef→dest, with an along-segment gate (t>0.15) preventing the exemption-sphere failure. Measured: **Long L1 t0 0% → 65% TSR at 80% CAR (n=20)**.
**Methods still to try (ranked, from `steer_synth.json`):** soft hinge-gradient pre-shaping on the x₁-prediction (analytic grads, steps 2–7; L1 TSR +5–8 expected); Gauss-Seidel multi-pass sweep with backward anticipation (sweep-vs-joint ablation); FK particle resampling with velocity-derived score (P1 — only if feasible-seed starvation shows); recurrence/re-noise polish (P2 ablation); FRS reversed-repaired seed (P2); γ/d_safe/corridor-radius sweep on the composed system.
**SABRE-Flow novel components (the named-method play, +4–5 days):** (i) barrier-certified flow sampling — CBF condition on the denoising ODE with closed-form correction → forward-invariance theorem for the generation process (the schedule+brake design is its discretization); (ii) semantic Riemannian correction metric (min‖δ‖_M, M from the constraint compiler — corridor/caution/rotation as one object); (iii) persistent noise inheritance across chunks (route hysteresis in noise space; chunk-boundary coherence unaddressed in all surveyed work); (iv) dual-dynamics guidance strength (multiplier carry = integral action, replaces tuned η).
**Direction-2 novelty statement:** the composition — scheduled soft-then-hard enforcement + actuation-aware provable repair + seed selection + executed-prefix certificate + semantic anisotropy, on a frozen 3B flow VLA — is occupied by nobody (BayesFP lacks the hard layer; SLSQP-SOTA lacks schedules/seeds/corridors/companions and has no code, no ablations, GT-only). SOTA is recoverable as our degenerate ablation row (`--no-use_companions --repair_schedule uniform`, K=1, corridor off).

## 3b. Steering: related work, baselines, and the novelty ledger

### The must-cite map (full mechanism details in `steer_theory.json` / `steer_robotics.json`)
**Direct competitors (safety steering of VLAs, SafeLIBERO):**
| Paper | Mechanism | Why we win / differ |
|---|---|---|
| **2607.01378** (SOTA) | min-norm SLSQP per Euler step, GT spheres, EEF-only | our degenerate ablation row; no schedule/seeds/companions/semantics/code/ablations |
| AEGIS/VLSA 2512.11891 | post-hoc per-step CBF-QP, vision | TSR ceiling (68.1); the benchmark authors' own OOD-drift admission |
| Attention-guided 2606.09749 | post-hoc CBF, attention picks target | per-step reactive; single uncalibrated exemption; we cite for the attention-grounding idea |
| BayesFP | FK particle tilt on π0.5 | soft-only — no hard layer, no certificate; becomes our soft-only ablation |
**Enforcement-mechanism lineage (cited as foundations):** SafeDiffuser (ICLR25; CBF-in-denoising for planners + the local-trap analysis our schedule answers), SafeFlowMatcher (ICLR26; executed-path-only verification, vanishing-correction schedule), TouchGuide (classifier guidance on π0.5 — mechanism precedent), OmniGuide 2603.10052 (VLM energy guidance in π0.5/GR00T denoising — closest semantic-guidance neighbor, no barriers/no safety benchmark), VLS, QPILOTS, GAF, DynaGuide (critic/dynamics-gradient steering), PC-Diffuser (velocity-channel corrections → our brake), ITPS (sampling beats gradient guidance on drift — motivates best-of-K), Guided Flows 2311.13443 + Feng et al. 2502.02150 (FM guidance theory; x₁-prediction gradients, softened hinges), FK-Steering ICML25 / TDS / FPS / FK-Flow 2509.01543 (particle steering; harmonic early down-weighting; soft potentials only), Ma et al. CVPR25 (noise-space search), RoboMonkey/TACO/MG-Select (best-of-N with task verifiers), BOKBO 2605.30660 (conformal abstention, **standard LIBERO only** — verified), Latent Policy Barrier, PolyFlow (constraint embedding — needs retraining; cited as why inference-time constraints require our route), FRS (reverse-flow informed seeds).

### Novelty ledger (component-by-component, hostile-reviewer honest)
**Borrowed with citation (not ours):** the enforcement locus (per-step correction in denoising); the schedule *concept*; best-of-N *concept*; executed-path verification *concept*.
**Novel — unclaimed per both surveys (re-verify at submission; field moves weekly):**
1. **Actuation-aware provable repair**: brake-then-push with barrier-max fallback ⇒ chain feasibility degrades *provably* to non-worsening under per-dim |cmd|≤1 clamps. No surveyed method handles actuator limits inside its guarantee — SOTA's SLSQP can emit unexecutable corrections silently.
2. **Executed-prefix lexicographic certificate** for chunked receding-horizon VLAs (feasible→margin→progress; resample-not-abstain). BOKBO abstains and is std-LIBERO; FK methods certify nothing per-rollout.
3. **Semantic corridor/destination exemption inside guidance** — anisotropic margins derived from task semantics (sanctioned approach segments with an along-segment gate). OmniGuide has goal *attractors*; nobody modulates *safety margins* by task sanction. Empirically decisive (Long L1 0→65 TSR).
4. **Manipulated-system companion geometry in guidance** (hand/payload points) — every SafeLIBERO method is EEF-only; worth +21pp CAR at n=50.
5. **Jittable exact in-order sweep** (constraint j depends only on steps ≤j ⇒ one pass is exact) — replaces the host QP; engineering novelty enabling per-step × per-particle × terminal reuse.
6. (If built) **SABRE components**: generation-process forward-invariance theorem; semantic Riemannian metric; cross-chunk noise inheritance; dual-dynamics strength.
**The defensible headline**: not any single component but the *composition* — scheduled soft→hard enforcement + seed selection + prefix certificate + semantic anisotropy + provable actuation-aware repair on a frozen 3B flow VLA, with the SOTA as a literal ablation row and public code. **Reviewer attack & answer**: "engineering composition" → answered by the ablation table (each component earns its row), the clamp-feasibility proof, the fail-direction figure (E5), and the two-tier privilege results no competitor attempts.

## 4. Direction 1 — identifying contextual/semantic safety (state: candidates chosen, E-protocol running)
Hazard taxonomy to cover: geometric · property (hot/fragile/liquid/sharp) · relational-state (towel-ON-stove, knife-NEAR-edge) · instruction-level (refuse/redirect) · damage/force. Output contract: identification compiles into constraint classes (keep-out geometry, margins, velocity boxes, rotation locks) + router decisions.
**Candidates (from `semsafe_synth.json`; no headline commitment yet):**
- **C1 SANCTION** (p≈0.35): default-deny RGB-D occupancy; semantics only *sanctions exemptions* (target/destination/support/corridor), conformally calibrated on exemption-FPR. **Fail-safe by construction** — the unnamed/dynamic hazard (LIBERO-Safety's intruding hand) is covered with zero machinery; identification errors withhold exemptions (liveness cost) instead of deleting protection (safety cost). Both surveys name this white space verbatim. The corridor exemption now in production is C1's first component.
- **C2 FOL-COMPILE v2** (p≈0.40): generalize the validated 100%-vs-0–10% neuro-symbolic decomposition (VLM does local micro-judgments; FOL makes scene-level decisions) to all five hazard classes, with per-predicate conformal calibration (uncertainty ⇒ worst-case constraint) and **runtime rule authorship** (VLM writes new FOL rules from a grounded vocabulary — v16 machinery, built+verified). Owns three named white spaces. Fails unsafe like every enumerator → compose over C1's floor.
- **C4 GOVERNOR** (adopt as tier): instruction-channel-isolated judge routing each hazard to a response modality (refuse/redirect/constrain/modulate) with conformal gating; targets LIBERO-Safety SSR-L2 (published best 20–36% refusal); offline-evaluable.
- (C3 continuous risk fields: demoted — novelty lost to arXiv 2512.08233; folded into C2's margin classes.)
**Strategic read:** C1+C2+C4 are layers of one stack (itself the surveys' "unified stack" gap); the E1–E5 protocol picks the headline: E1 labeled ID benchmark (150–200 scenes), E2 corridor precompute (**done**: exemption restores straight-line liveness on 3/4 spatial tasks; t3 = obstacle-on-path family, <6cm from the approach line — the geometrically non-negotiable residue), E3 identification-swap (GT/C1/C2 with fixed enforcement — the arbiter), E4 SSR refusal-F1, E5 fail-direction stress test (unique publishable figure).
**Grounding trust recipe (from `semsafe_ground.json`):** verbalized VLM confidence is disqualified (overconfident; GDINO-class detectors emit high-confidence false positives); the publication-standard mechanism is split conformal prediction over sampling/self-consistency frequency (KnowNo→IntroPlan lineage); target grounding gets triple redundancy (parse × detector × policy-attention agreement).
**Direction-1 novelty statement:** first *fail-safe-by-construction* semantic safety layer (default-deny + calibrated exemption sanctioning) + first unified five-class hazard compiler with heterogeneous constraint output + runtime rule authorship; the error-DIRECTION analysis (E5) exists in no surveyed paper.

## 5. The no-GT tier (perception; required deliverable)
Replaces exactly three privileged inputs (obstacle identity/position/size — see the privilege audit in the approved plan): Qwen/API VLM naming + symbolic FOL selection (validated 21/21) + RGB-D bbox-median back-projection + bbox×depth/f size + uncertainty-inflated margins (fixed → conformal). Client-side only — the guidance payload is source-agnostic, so the entire Direction-2 stack transfers unchanged. Default-deny occupancy (C1) upgrades this tier from "identify the hazard" to "sanction the exemptions".

## 6. Assumptions (verified where possible)
1. Single-integrator EEF rollout ≈ realized path (OSC kp=150; mitigated by +4mm/step inflation, not eliminated).
2. Intermediate denoising state ≈ action chunk (same assumption as SOTA; late-step enforcement + terminal acceptance carry the weight).
3. translation_scale = 0.05 m/command — **verified from robosuite OSC_POSE config**.
4. Quantile action norm stats from the checkpoint = training stats (loaded identically to openpi's own pipeline).
5. GT tier: single benchmark-designated obstacle, static during chunk; matches SOTA privilege exactly.
6. Companion offsets in world-z assume downward gripper (LIBERO tabletop valid).
7. Collision = >1mm obstacle displacement (matches our baselines; SOTA criterion unstated → mismatch note + self-run referee).
8. Seed 7 + official init states ⇒ paired episode comparisons (McNemar/paired bootstrap apply).
9. Corridor sanction from string-parse entities (GT tier); no-GT tier replaces with detection + attention agreement.

## 7. Limitations (honest; the paper's limitations section)
1. **No invariance guarantee**: model-consistent verification + provable non-worsening under clamp + (planned) conformal per-chunk bounds; open-loop gaps between chunks uncovered; never say "guaranteed safe".
2. **3-point body model**, not swept volume — arm links unmodeled (deferred until data demands).
3. **Gripper-force dim untouched** — crush damage unaddressable at trajectory level (OopsieVerse scoped to impact).
4. **Obstacle-on-path residue** (E2's t3 family, Object L1 t1/t2): straight-line clearance geometrically impossible; curved-approach selection or contact-adjacent acceptance are the only options; SOTA's own 46-CAR cell corroborates.
5. Corridor exemption is a *deliberate, reported* safety reduction along sanctioned approaches; wrong sanction = wrong hole (mitigations: along-segment gate, agreement gating in no-GT tier, corridor contacts still scored).
6. Rotation locks / velocity boxes designed, unexercised until LIBERO-Safety/OopsieVerse.
7. Latency: K=8 ≈ 2–3×/chunk (reported); SOTA itself pays +37% ETS on Object.
8. Cross-paper numbers indicative until the reimpl arm lands; baseline-protocol mismatch note stands (their π0.5 base 60.0/17.0 vs ours 67.0/14.0).
9. Sim-only this cycle; LIBERO-family physics; n=50 resolution ±7pp aggregate.
