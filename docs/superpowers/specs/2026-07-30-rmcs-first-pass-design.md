# RMCS first-pass design — Route-Memory Certified Selection (2026-07-30)

Status: DESIGN + pre-registered validation chain. First pass only — a
defensible GO/NO-GO on the direction, not the paper campaign.

## 1. Problem statement (evidence-anchored)

The remaining weak cells (Long L1/L2, Goal L2) fail by TIMEOUT-STALL, not
collision, and not by lack of time:

- Cap-800 ablation (job 42243244, n=20/arm): guided Long L1 36.2/62.5 and
  L2 31.2/72.5 at 800 steps vs 42.5/71.0 and 24.5/77.0 at 550 — flat.
  Baseline rises to 63.7 TSR at 800. Guided failures are stall LOOPS.
- Video forensics (job 42743730): placement-phase deadlock — the robot
  hovers over the drop zone re-approaching from near-identical states.
- Placement-descent audit: static geometry does NOT block the final descent
  (B_relax > 0 throughout) on all four audited weak Long cells — the dither
  must come from chunk-horizon transit clipping or K-selection dynamics.
- K-dispersion telemetry: k_disp mean 0.08–0.15 m, max 0.63 m — the K
  candidates differ materially. Selection has real modes to choose among.
- Every sample-level steering operator (tilt, adjoint, FK particles,
  repulsor, HDC) failed to separate at n=200; the only actuator that ever
  separated is best-of-K selection.

Diagnosis: the per-replan argmax is MEMORYLESS. When the greedy-safe
candidate leads into the same dead-end every replan, the selector re-picks
it forever — a livelock the policy cannot escape because nothing in the
system remembers that this route already failed.

## 2. Mechanism spec

RMCS makes best-of-K selection STATEFUL across replans, at fixed compute:
same K, same denoise steps, same barrier — only the argmax changes.

### 2.1 Route signature
For candidate k at replan r, over the certified prefix (first P actions):
- side bit s_k ∈ {−1,+1}: sign of the cross product (in the table plane) of
  (obstacle − eef) × prefix displacement — which side of the active
  obstacle the prefix passes;
- coarse waypoint hash w_k: prefix endpoint quantized to a 3 cm grid in the
  corridor frame (fallback: world frame when no corridor is active).
Signature σ_k = (s_k, w_k).

### 2.2 Route ledger (per episode, client-side)
For each signature σ ever selected, accumulate:
- progress(σ): net EEF displacement toward the goal anchor realized during
  chunks executed under σ;
- repair(σ): mean applied repair magnitude under σ;
- visits(σ), last_selected(σ).

### 2.3 Scoring
Production score is min-margin over the prefix (safety). RMCS score:

  score(k) = margin_score(k) − λ_taboo · taboo(σ_k) + λ_commit · commit(σ_k)

- taboo(σ) = visits(σ) · 1[progress(σ)/visit < δ_prog] — penalize routes
  that have been tried and produced no progress (accumulated-failure
  routes);
- commit(σ) = 1[σ == σ_selected at r−1] — hysteresis bonus: don't thrash
  between modes when the current one is progressing;
- hysteresis switch rule: the commit bonus is dropped (forcing a switch to
  the best non-taboo alternative signature, when one exists among the K)
  once the accumulated progress deficit under the committed route exceeds
  Δ_stall (mirrors the existing STALL_WINDOW/STALL_EPS_M detector).

Safety is UNCHANGED: candidates failing the certificate remain ineligible;
RMCS only reorders the eligible set. Worst case = production's fallback.

### 2.4 Degenerate case
If all K candidates share one signature (no mode diversity), taboo and
commit are constants across k and RMCS reduces EXACTLY to production
selection. RMCS can only act where diversity exists — hence Phase 0
measures diversity before any method code is written.

### 2.5 Livelock-escape claim (statement only; theorem deferred)
Under a mode-coverage assumption (with probability ≥ p per replan, the K
draws contain ≥ 2 distinct signatures of which ≥ 1 is non-taboo and
feasible), the taboo accumulation rule guarantees the committed signature
is abandoned after at most ⌈Δ_stall/δ_prog⌉ no-progress replans, so any
finite signature set is exhausted in finite expected time — the selector
cannot revisit a zero-progress route indefinitely, unlike the memoryless
argmax which can re-pick it with probability 1 every replan.

## 3. Differentiation (from the S4 survey ledger entry)

| Prior work | What it does | What it lacks |
|---|---|---|
| Pre-VLA / RoboMonkey | best-of-N scored by learned VM/reward | memoryless across replans; no safety certificate |
| VLA-ATTC / DREAM-Chunk | test-time chunk adaptation | no cross-replan route memory; no liveness objective |
| DynaGuide | dynamics-model guidance in denoising | generation-side, trained artifact; our sample-level ops died at n=200 |
| Retrieve-then-Steer | retrieval memory from OTHER episodes | no within-episode failure memory |
| ITPS | user-interactive steering | human in the loop, not autonomous liveness |

Unclaimed (survey verdict 2026-07-30): cross-replan memory/liveness for
safety-filtered generative policies, and certificate distillation. RMCS
claims the first; the second is a separate design.

## 4. Pre-registered validation chain (each phase gates the next)

### Phase 0 — diversity + dither measurement (diagnostics only, ~1 GPU-hr)
Instrumented replay, production config, NO behavior change. Cells: Long L1
t1/t3, Long L2 t2/t3, Goal L2 t0; n=5 episodes each. Log per replan: K
candidate prefix displacement vectors, per-candidate min-margins, selected
index, EEF position.
Measured: stall fraction (net EEF motion < 15 mm/40 steps); mode diversity
IN STALL (fraction of stalled replans whose K prefixes span ≥ 2 signatures:
side-sign differs or displacement clusters > 3 cm apart); revisit rate
(fraction of stalled replans re-selecting the previous signature).
GATE → Phase 1 iff (a) ≥ 25% of stalled replans show ≥ 2 signatures AND
(b) ≥ 50% of stalled replans re-select the previous signature.
(a) fails ⇒ NO-GO: nothing to select; Long/Goal timeouts need
generation-side diversity, not selection memory. (b) fails ⇒ NO-GO: no
loop to break.
Verification: with RMCS absent, the instrumented run must select
bit-identical chunks to production on a fixed seed.

### Phase 1 — minimal RMCS, dev-cell smoke (paired n=20/task, 2 cells)
Client-side controller (`vlm_pipeline/rmcs_controller.py`, unit-tested on
CPU: signature stability, taboo triggering, hysteresis switch, degenerate
passthrough). Server: per-candidate prefix info in diagnostics + ONE
scalar payload knob (selection-bias vector added to candidate scores),
mirroring the hdc_scale override plumbing. Same K=8, same schedule. Cells:
the two Phase-0 cells with highest (diversity × stall) product. Arms: RMCS
vs production control, identical seeds.
GATE → Phase 2 iff TSR +≥10pp on the paired cells with CAR non-inferior
(one-sided; McNemar reported) AND mechanism telemetry moves (stall
fraction drops; route switches occur after taboo). TSR gain with CAR
collapse or without telemetry movement = NO-GO (gain ≠ claimed mechanism).

### Phase 2 — pre-registered paired n=200 (only on Phase-1 GO)
Long L1 + L2 (all tasks) + Goal L2, RMCS vs production, same seeds;
McNemar on TSR, Fisher on collisions; four-number ICRA box applies. This
arm goes in the paper — or is the clean negative that closes #37.

## 5. Out of scope (first pass)
HJ reach-avoid certificate (upgrade path after Phase-2 GO), distilled
critic, Long placement exemption (different mechanism per descent audit),
any n=200 spend before Phase 1 passes.
