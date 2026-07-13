# System Design — Runtime Context-Aware Safety Steering for Flow VLAs
**The system we are building** (target state; implemented parts marked ✅, in-build 🔨, planned 📋). Each major design choice carries its reason and the rejected alternative. Companions: `DIRECTIONS_RESEARCH_AND_ARCHITECTURE.md` (research/novelty), `PROGRESS_LOG.md` (chronology), `OUTCOME_STEERING_SPEC.md` (ROCS detail).

## 0. One-paragraph thesis
A frozen foundation policy (π0.5) cannot know every scene: edge cases, subtle context, and commonsense wrongness (glass-on-laptop) lie outside its teleop training distribution. We build a runtime layer that (a) **knows what the policy doesn't** — internet-scale commonsense compiled into outcome rules, and (b) **knows when the policy doesn't know** — epistemic signals read from the sampler we control — and steers action *generation* accordingly: relocating outcomes, tightening geometry, never merely blocking. Safety enforcement lives *inside* the denoising loop because post-hoc correction provably caps task success (quadruple-confirmed TSR ceiling ~68 on π0.5).

## 1. System overview
```
                        ┌─ SLOW CONTEXT LOOP (episode start / on-change; seconds) ─┐
 scene RGB(-D) ────────►│ generative VLM (Qwen/API): objects, properties           │
 instruction ──────────►│ FOL compiler + runtime rule authorship:                  │
                        │  roles (TARGET/DEST/OBSTACLE/SUPPORT), outcome rules     │
                        │  (FORBIDDEN_ON/RELEASE/TILT), sanctioned alternatives,   │
                        │  constraint classes (margins/velocity/rotation/refuse)   │
                        └───────────────┬──────────────────────────────────────────┘
                                        │ rules + entities (conformally gated)
                        ┌─ FAST CONTEXT LOOP (per chunk; ms) ──────────────────────┐
                        │ SigLIP/CLIP region-text scores (identity/property refresh)│
                        │ DINOv2 tracking (positions under motion)                  │
                        │ RGB-D poses → relational predicates ON/ABOVE/NEAR-EDGE    │
                        │ zero-cost option: probe π0.5's own SigLIP prefix features │
                        └───────────────┬──────────────────────────────────────────┘
                                        │ guidance payload (source-agnostic)
 ┌─ GUIDED SAMPLER (per chunk, inside π0.5's 10-step flow denoising) ───────────────┐
 │ K=8 seeds, prefix KV shared ✅ → per-Euler-step scheduled DCBF repair ✅          │
 │  (brake-then-push ✅, companion points ✅, corridor/dest exemption ✅,            │
 │   conditional outcome constraints 🔨ROCS: release barrier, tilt lock)            │
 │ event parse of chunk 🔨 (release index, landing region, tilt excursions)         │
 │ executed-prefix certificate + lexicographic selection ✅                          │
 │ epistemic gate 🔨: K-dispersion + denoising-curvature → margin/prefix modulation │
 └───────────────┬──────────────────────────────────────────────────────────────────┘
                 │ selected chunk (+ diagnostics)
 ┌─ ROUTER (deterministic, auditable) 📋 ────────────────────────────────────────────┐
 │ execute | steer (above) | tighten (uncertainty) | recover/hold | refuse/redirect │
 │ (governor tier for instruction-level hazards; SSR safe-alternative execution)    │
 └───────────────────────────────────────────────────────────────────────────────────┘
 Receding horizon: execute 5 of 10 actions (3 under high uncertainty), re-observe.
```

## 2. Design choices and reasons (decision record)
**D1. Enforce inside generation, not after.** *Why:* frozen policies never see post-hoc corrections → stall/OOD drift; measured ceiling ~68 TSR (AEGIS, attention-filter, our oracle tier, our pilot). In-denoising corrections are integrated by the policy's own velocity field → re-planning (measured: TSR rises). *Rejected:* post-hoc QP shields (18 months of our own evidence), training-time safety (SafeDojo: 32×A800, +0.12pp zero-shot transfer — safety memorizes layouts).
**D2. Projection-style repair as the enforcement primitive; schedule it.** *Why:* hard constraints need enforcement, not preference (vs classifier-gradient guidance); but always-hard from pure noise is the documented SafeDiffuser trap → trust early / soft mid / exact late. *Rejected:* SMC/particle backbone (exact only in distribution, no per-rollout statement, unproven noise-injection fidelity at 10 steps — kept as gated upgrade), pure terminal projection (concentrates shear in one step).
**D3. In-order sweep, not a host QP.** *Why:* constraint j depends only on steps ≤ j ⇒ one pass is exact; pure jnp ⇒ runs per-step × per-particle × terminal at negligible cost; no host callback per Euler step. *Rejected:* SLSQP (SOTA's choice — unjittable per-step, no actuation awareness).
**D4. Actuation limits inside the guarantee (brake-then-push, barrier-max fallback).** *Why:* corrections the robot can't execute are silent safety holes; with per-dim |cmd|≤1, feasibility provably degrades to non-worsening. *Rejected:* unclamped corrections (SOTA — unstated behavior at limits).
**D5. Manipulated-system geometry (EEF + hand + payload points).** *Why:* video forensics — hand plowed obstacles the EEF-point cleared; +21pp CAR at n=50. Switchable (`companion_scale`) for the reimpl referee. *Rejected (for now):* full-arm swept volume — cost/benefit deferred until data demands.
**D6. Semantics modulate GEOMETRY (corridor/destination exemption), and semantics gate OUTCOMES (ROCS).** *Why:* uniform envelopes cannot distinguish sanctioned approach from hazard approach (measured L1 tradeoff; Long L1 collapse) — margins must be anisotropic in *task-sanction space*; and commonsense wrongness is a relation the action MAKES TRUE (glass ON laptop), only expressible as event-conditional constraints (release/tilt/activation), not positions. *Rejected:* static relational keep-outs (Brunke-style "never above" — wrongly vetoes transit), risk-field soft costs alone (fails unsafe, novelty lost to 2512.08233).
**D7. Best-of-K selection with an executed-prefix certificate; resample-not-abstain.** *Why:* selection stays on the policy manifold (ITPS: least drift per unit alignment); the executed 5 steps are what actually collide; abstention (BOKBO) fails tasks — resampling completes them. K-batch is cheap via shared prefix KV (2–3×, not 8×). *Rejected:* verifier-model ranking (training burden + near-zero OOD transfer per GAF; contamination risk on 50-episode suites).
**D8. Two-rate context.** *Why:* generative-VLM reasoning is seconds-slow; scenes change and rules' arguments (where is the laptop NOW) need chunk-rate refresh → encoder transformers (SigLIP/CLIP scoring, DINO tracking) at ms-rate; relations stay geometric (CLIP-family provably weak at relations/negation/state). Zero-cost option: π0.5's own SigLIP prefix features. *Rejected:* VLM-per-chunk (latency), encoder-for-relations (documented failure mode).
**D9. Fail-safe error direction wherever achievable.** *Why:* every surveyed identifier fails UNSAFE (missed hazard = collision). Default-deny occupancy + sanctioned exemptions (C1) inverts this: identification errors withhold exemptions (liveness cost, visible) instead of deleting protection (safety cost, invisible). Uncertainty responses are all conservative (tighten/slow/hold). *Rejected:* enumerate-the-hazards as the only layer (C2 alone) — kept as the semantic-typing layer ON TOP of the floor.
**D10. Uncertainty from the sampler we control, no training.** *Why:* K-seed dispersion (mode-clustered) ≈ deep ensemble over noise, free from the existing batch; denoising-path curvature ≈ flow-regime confidence; both calibratable offline against thousands of logged episodes. Response = re-weight trust (margins↑, prefix 5→3, velocity caps), never invent actions. *Rejected:* learned OOD critics/world-model filters (training + transfer + contamination).
**D11. Trust = split conformal over consistency/similarity scores.** *Why:* verbalized VLM confidence and detector scores are systematically overconfident (survey-verified); conformal-over-frequency is the publication standard (KnowNo lineage); per-hazard-class risk budgets; uncertainty always maps to MORE constraint. *Rejected:* raw thresholds, verbalized confidence.
**D12. Two privilege tiers as first-class deliverables.** *Why:* GT tier = apples-to-apples with SOTA (their exact privilege) isolates enforcement; no-GT tier (encoders + RGB-D + FOL into the SAME payload) is the deployable claim no competitor attempts. The payload interface is source-agnostic BY DESIGN so the steering stack is identical across tiers. *Rejected:* perception-first development (the FOL campaign proved enforcement, not perception, was the binding constraint to validate first).
**D13. One server, one code path.** *Why:* enabled=0 ⇒ bit-identical vanilla π0.5 (validated) — baselines, ablations, and guided runs share infrastructure; every flag (schedule/K/companions/corridor) is an ablation row. Honest-reporting instruments built-in: correction norms, residuals, clearance, feasible-count, per-episode JSONL.
**D14. Receding horizon with uncertainty-scaled prefix.** *Why:* open-loop error grows over the executed prefix; executing 5 of 10 bounds it; shrinking to 3 under high uncertainty re-observes exactly when prediction is least trustworthy. *Rejected:* full-chunk execution (open-loop risk), per-step re-query (latency).

## 3. Interfaces (stable contracts)
- **Guidance payload** (client→server, per chunk): `{enabled, eef_pos, obstacle[s] pos/radius, target_pos, dest_pos, [ROCS: held_object, forbidden_regions, sanctioned_regions], [epistemic: overrides]}` — metric, source-agnostic (GT sim / perception). Popped before input transforms; policy observation path untouched.
- **Constraint classes** (compiler→sampler): keep-out geometry (+companion set), margin multipliers (corridor/anisotropy), velocity boxes, rotation locks, release barriers, refuse/redirect (router-level).
- **Diagnostics** (server→client, per chunk): correction_norm, min_clearance, max_residual, feasible_count, selected_margin, [event parse verdicts], [dispersion/curvature] — everything the paper's mechanism figures and the calibration loops consume.

## 4. What the system does NOT claim (honesty contract)
No invariance guarantee (model-consistent verification + provable non-worsening under clamps + conformal bounds only); no crush-force control (gripper force unactuated at trajectory level); obstacle-on-path residue is geometrically non-negotiable (reported, not hidden — E2 quantifies it); corridor exemptions are deliberate, reported protection reductions along sanctioned approaches; encoder context inherits open-set FPR (conformally gated); sim-only this cycle.

## 5. Evaluation mapping (which component is proven where)
| Component | Proving ground |
|---|---|
| In-generation enforcement + stack | SafeLIBERO 8 conditions, self-run ladder (baseline / reimpl-SOTA / ours), both tiers |
| Corridor/destination exemption | Long L1 resurrection (0→65 TSR n=20), Spatial L1 frontier |
| ROCS outcome steering | LIBERO-Safety SSR L1/L2 (safe-alternative execution vs refusal-only 0.31–0.46 F1), HazardArena twins |
| Epistemic gating | LIBERO-Safety robustness suites (noise/layout/unseen — purpose-built OOD probes) |
| Fail-safe identification (C1/C2) | E1/E3/E5 protocol; first method rows on LIBERO-Safety TSA/FSHOA |
| Damage-aware caution | OopsieVerse RoboCasa (impact class; release barriers partially recover drops) |
