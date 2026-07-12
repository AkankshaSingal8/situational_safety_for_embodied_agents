# Deep-Research Report: Semantic/Contextual Safety Filtering — Literature Map, FOL Verdict, Redesign Recommendation

*Produced 2026-07-11 by a research subagent (~90 web searches + arXiv fetches, adversarial verification pass on the 7 claims driving the recommendation).*

## (A) Annotated literature map

### A1. Direct competitors on SafeLIBERO (must cite, must differentiate)

| Paper | What it does | Gap it leaves us |
|---|---|---|
| **VLSA/AEGIS** (arXiv 2512.11891, Dec 2025) — introduced SafeLIBERO | VLM identifies obstructing objects, open-set detection + depth → CBF-QP on VLA actions. +59% obstacle avoidance, +17% success | Uniform avoidance regardless of intent; admits residual failures from detector misidentification but **never quantifies grounding as bottleneck**; no fallback when grounding fails |
| **"Your Model Already Knows"** (arXiv 2606.09749, Jun 2026) — attention-guided safety filter, **on SafeLIBERO** | VLA attention heads pick the current target; YOLOE-11m-seg masks + tracker make everything else a CBF obstacle. Policy is **π0.5**; matches privileged-state oracle on static tasks; +43% Safe-Success on dynamic Level III vs init-only baseline | Purely geometric target/obstacle dichotomy; no semantic hazards (fragile/hot), no guarantees. **Warning: nearly closes the static geometric-avoidance gap on SafeLIBERO — pure geometric avoidance is no longer a strong headline** |
| **Neuro-Symbolic Constrained Flow Matching** (arXiv 2607.01378, Jul 2026), on SafeLIBERO | CBF correction inside flow-matching denoising. 82.8%/81.6% CA/TS vs AEGIS single-step CBF 77.9%/68.1% | Architecture-locked to flow/diffusion heads — **inapplicable to OpenVLA-OFT's L1 head**; our explicit contrast citation |
| **SafeDojo** (arXiv 2606.20698, Jun 2026), on SafeLIBERO L-I/II | Action-conditioned video world model rolls out candidate chunks + safety head; **requires world-model fine-tuning (1.5K SafeLIBERO trajs) + constrained-GRPO RL of the VLA** — not inference-time. L-I SSR 53.25%, TSR 64.5% | Not training-free; no interpretable constraints; no guarantee |
| **CORE** (arXiv 2602.19983, Feb 2026, Ravichandran/Pappas) | VLM contextual safety reasoning → spatially grounded safe sets → CBFs **with probabilistic guarantees under perceptual uncertainty**. **Navigation/mobile robots, not manipulation** | Highest-threat framing competitor. Differentiate on: manipulation, quantified grounding bottleneck, intent-conditioned response, honest-failure semantics |
| **Brunke et al.** RA-L 2025 (arXiv 2410.15185) | LLM multi-prompt voting → semantic constraints → CBF safeguard, real kitchen | Treats voted VLM output as *correct*; no uncertainty propagation into the barrier — that sentence is our gap statement |

Disambiguate in related work: **LIBERO-Safety** (arXiv 2606.23686, ECCV 2026 — a *different* benchmark from SafeLIBERO, benchmark-only) and **SafeVLA** (arXiv 2503.03480, NeurIPS 2025 — training-time constrained RL, the contrast class to filters).

### A2. Context/positioning papers
- **Robey et al., Science Robotics, Apr 29 2026** (DOI 10.1126/scirobotics.aef2191) — layered framework with **three defensive lines** (constitutions, multi-stage checkpoints, safety-informed training data); note "three-axis" is not their wording. Intro citation: a runtime semantic filter is exactly their "checkpoint" layer, unbuilt for manipulation.
- **ASIMOV constitutions** (DeepMind, arXiv 2503.08663, CoRL 2025) — judgment-only, no control loop. **RoboGuard** (RA-L 2026, arXiv 2503.07885) — plan-level LTL guardrails, not continuous control. **SafeVLA-Bench** (arXiv 2606.00773): 36–56% of *successful* rollouts violate a safety clause — motivation numbers.
- **Hsu/Fisac safety-filter survey** (Annual Reviews 2024, arXiv 2309.05837) — related-work anchor.
- **KnowSafe does not exist as a robotics paper** (only an artificial-pancreas CPS paper, arXiv 2311.07460). Don't cite; nearest robot analogue is LLM+KG safety checking (arXiv 2412.11387).

### A3. Tools for the redesign (key findings)
- **Uncertainty-aware CBFs**: MR-CBFs (Dean et al., **CoRL 2020**), CVaR/Risk CBFs (Ahmadi, Singletary, L-CSS 2022/23), belief CBFs (Vahs & Tumova 2309.06499; multimodal sample-based BCBF 2504.04097), Wasserstein DR-CBF (AIAA 2024). **Hard-searched negative result: no paper enforces a CBF over a discrete belief about WHICH object is the hazard.** Discrete-hypothesis safety exists only in MPC form (Control-Tree Optimization ICRA 2021; existence-probability planning 2002.01254) or as deferral (KnowNo). *Identity-belief CBF is an open, claimable formulation* — cite those three preemptively.
- **Conformal pattern is mature and accepted**: KnowNo (CoRL 2023 best paper), Perceive-with-Confidence (Majumdar, CoRL 2024 — **geometric detection uncertainty only, explicitly not identity**), conformal CBFs (CP-NCBF 2503.17395; "CP in the Loop" IEEE 2025), Lindemann RA-L 2023. Nobody has conformalized *VLM semantic grounding* into a barrier margin.
- **Grounding robustness**: Set-of-Mark (2310.11441) turns open-set detection into k-way selection — the cheap fix for wrong-object; probabilistic data association (Bowman ICRA 2017 best paper) is the canonical keep-k-candidates citation; Grounded-SAM-2 gives detect-once-track-forever; KineDepth (2409.19490): DepthAnythingV2 1.42m error at tabletop, Metric3Dv2 0.068m — brackets our 0.22m stereo number. Qwen2.5-VL: 3B's largest deficit vs 72B is RefCOCO+ testB (74.1 vs 83.7) — attribute-heavy referring, exactly our "which object is the hazard" case.
- **Projection-hurts-the-policy is now documented**: PACS (2511.06385: "up to 68%" sim / 37% hardware improvement over reactive CBF filters via path-consistent braking, for any chunked policy incl. deterministic); ITPS (ICRA 2025); action-aliasing analysis (2509.12833). Our "slide" and ray-rescale modes are unpublished cousins of PACS braking — cite it and position ours as intent-conditioned generalization.

## (B) Verdict on the FOL claim: decorative as currently framed — fix or drop the branding

**A reviewer who knows logic will call it FOL-washing, and they'd be right.** The layer is a propositional rule table over ground atoms fixed at extraction time (MENTIONED/HOVERING/TALL → OBSTACLE). No quantifiers over an unbounded domain at inference time, no relational generalization, no inference beyond one forward chain, no verifiable artifact. The load-bearing-logic test has three prongs:

1. **Soundness by synthesis** — monitor compiled from spec so violations impossible by construction (Alshiekh AAAI 2018; LIMP 2402.11498). We don't have this.
2. **Guaranteed coverage** — statistical wrapper converts predicate error into calibrated failure probability composing into the controller's guarantee (KnowNo; Perceive-with-Confidence; conformal LTL 2309.10092). We don't have this *yet* — the achievable fix.
3. **Compositional transfer** — rules provably recombine to novel hazard descriptions (Neuro-Symbolic Concepts 2505.06191). Untested.

Cite-worthy against ourselves: **reasoning-shortcuts** (Marconato et al., NeurIPS 2023, 2305.19951) formally characterizes symbolic layers that are architecturally symbolic but semantically decorative.

**Fixes, in order of realism:** (i) rename honestly — "interpretable symbolic hazard-selection rules," sell interpretability + auditability; (ii) make predicates *probabilistic* (calibrated confidence from VLM voting) parameterizing a belief-weighted barrier — makes the symbol layer genuinely load-bearing via prong 2; (iii) compositional-generalization experiment (novel hazard phrasings → correct OBSTACLE selection) buys prong 3 cheaply. Do (i)+(ii); (iii) if time.

## (C) Ranked redesigns (paper-strength ÷ effort)

1. **Belief-weighted multi-hypothesis CBF + conformal margins** (see D). Attacks the *measured* bottleneck (wrong-object selection; oracle CAR 56 vs 19.5 = ~3x headroom), claimable finite-sample guarantee, converts symbolic layer to load-bearing. Effort: moderate. Novelty verified by hard negative search. Ablations: k=1 argmax (≈current) vs k=3; calibrated vs fixed 2.5x inflation; oracle ceiling.
2. **Conformal calibration alone** (subset of #1): ~1 week, guarantees claim, but without multi-hypothesis doesn't fix wrong-object — margin inflation around the *wrong* object is still wrong. Only inside #1.
3. **Set-of-Mark selection + detect-once-track (SAM2)**: mark candidate masks, Qwen *selects* the hazard (k-way MC instead of box regression — targets 3B's weakest skill), track masks at video rate. High CAR payoff but reviewers see a systems tweak. Do inside #1, not headline.
4. **Chunk-level safety-scored best-of-N steering** (perturbation-sampled candidates from deterministic head, CBF-scored, PACS-style braking fallback): genuine gap (nobody steers deterministic VLA chunks with a safety scorer — SafeDojo needs RL + world-model training; flow-matching guidance needs diffusion heads), but heavier, space crowding fast. Strong *second paper*, not this deadline.
5. **LTL/temporal rules**: SafeLIBERO hazards are static; buys nothing measurable. Skip — deepens the logic-washing critique.
6. **Latent/world-model safety**: per-setup training, abandons interpretability story. Skip.

## (D) Recommended move: **Conformalized Multi-Hypothesis Semantic CBF** ("don't pick the obstacle — hedge over it")

**Pitch:** the filter maintains a categorical belief over *which* detection is the hazard (weights from VLM/rule voting) plus a conformally calibrated position-error radius per hypothesis, and certifies actions against the belief — honest failure (k=0 → workspace rules + speed cap) as the same framework's boundary case.

**Why it wins review:** (1) attacks the failure mode we *measured* — paper becomes "we quantified the grounding bottleneck in semantic safety filtering, then built the filter robust to it" (VLSA names it, never quantifies; CORE guarantees but navigation; PwC explicitly excludes identity); (2) identity-belief CBF is a novel formulation with clean prior-art differentiation (Control-Tree = MPC not filter; KnowNo = deferral not control); (3) conformal wrapper upgrades the symbolic layer to load-bearing (B prong 2) and gives a theorem-shaped claim: *P(true hazard outside all certified barriers) ≤ ε under exchangeability*.

**Implementation sketch:**
- `visual_grounder.py`: `ground()` returns `[(pos_i, Σ_i, w_i)]` for top-k detections scoring above threshold on the OBSTACLE rule (Set-of-Mark selection prompt drops in here). `refine()` updates weights (renormalize; drop hypotheses the wrist camera falsifies).
- Rules layer: OBSTACLE(x) becomes a **score** (weighted sum of MENTIONED/HOVERING/TALL/near-path evidence + VLM vote fraction) — this *is* the probabilistic-predicate upgrade.
- Calibration (~100-line script): GT obstacle positions already logged (DIAG-GT). Held-out calibration split (1 of 4 tasks, or episodes 0–14/task): nonconformity `s = ||x_gt − x̂||` per grounded episode; (1−ε) quantile `q̂` replaces hand-tuned 2.5x depth inflation (calibrate per-axis to keep anisotropy). State exchangeability caveat honestly.
- `filter.py` certify loop: iterate barriers over all k hypotheses — structurally identical to stacking hand/wrist checkpoint barriers; QP shape unchanged. Ablate: (a) **robust union** — all hypotheses w_i > τ get full barriers (simple, safest, likely best); (b) **weight-scaled** α_i(h) = w_i·α(h) (CVaR-flavored). Grasp-corridor carve-out applies per-hypothesis, only when that hypothesis is not the task-mentioned target — intent-conditioning survives unchanged.
- **Minimal validating experiment** (one HPC batch): v20 config, both levels, n=50, seed 7, four arms: {k=1 fixed-inflation (≈current), k=1 conformal, k=3 conformal-union, oracle}. Success = k=3-conformal closes a visible fraction of the 19.5→56 CAR gap without TSR < ~35. Report empirical coverage (true hazard inside a certified barrier at rate ≥ 1−ε?).

**What won't survive review (honesty):** (1) calling the rule layer FOL — rebrand per B; (2) claiming geometric avoidance as the contribution — "Your Model Already Knows" already matches a privileged oracle on static SafeLIBERO with π0.5; our story must be *semantic* hazards + quantified grounding bottleneck + guarantee + honest failure; (3) current headline numbers (CAR 19.5 vs baseline 12.5) too weak to carry a paper alone — the belief-CBF experiment must land, or pivot to benchmark/analysis framing ("the grounding bottleneck in semantic safety filtering: measurement and mitigation"), which the oracle-gap data already supports; (4) SafeLIBERO-vs-LIBERO-Safety confusion — one disambiguation sentence in related work.
