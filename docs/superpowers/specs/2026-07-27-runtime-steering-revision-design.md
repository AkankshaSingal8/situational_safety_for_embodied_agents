# Runtime Steering Revision — Approaches A, B, C (design)

Date: 2026-07-27. User-approved goals, runtime-only (no training):
1. **A** — strengthen production (composed repair) on weak suites/tasks with as
   general a design as possible.
2. **B** — a new steering method that could REPLACE production at similar or
   better performance.
3. **C** — kinematic lookahead, adopted wherever it earns its ablation row.

Bars (user pre-registered): SafeLIBERO — beat baseline pi0.5 AND our SOTA
reimpl on BOTH TSR and CAR on all suites. LIBERO-Safety — SR above the frozen
baseline (stretch: above the safety-SFT model where the frozen policy has the
competence) while keeping safe-rate at the current ~100.

Diagnosis anchoring the design: the dominant failure in every weak cell is the
**stall/timeout** (episodes end at max steps parked outside the barrier), not
collision. Object t1 (obstacle-on-grasp), LS human_safety (TSR 0–26 with
safe-rate 100), LS obstacle suites. Secondary: static repulsor strength cannot
trade TSR vs CAR pointwise (four n=200 arms on a monotone frontier, none in
the box).

## Where each piece lands (all flags default-off = bit-exact legacy)

### Server (`pi0_guided.py` + `serve_policy_guided.py`)
- **A1 progress-aware selection**: candidate score (currently
  `1e3*feasible + 10*min_margin + 0.1*disp_norm`) gains a DIRECTED term:
  projection of the executed-prefix displacement onto unit(target_pos −
  eef_pos), weighted by new knob `progress_weight` (default 0). Uses the
  target already present in GuidanceParams; no new inputs.
- **C lookahead scoring**: score gains `lookahead_weight * min(post-prefix
  barrier margin over the full chunk)` (default 0). The barriers for all
  horizon steps are already computed; this is a one-line dead-end penalty —
  candidates whose tail re-enters the keep-out lose ties.
- **Payload η override**: `_make_params` accepts `repulsor_eta` from the
  per-request guidance payload (falls back to server config). This is the hook
  approach B's client-side controller uses; the server stays stateless.

### Client (`run_guided_safelibero_pi05_eval.py`, LS eval script)
- **B dual-adaptive η (`--dual_eta`)**: per-episode integral controller.
  After each replan the server already returns `selected_margin` m; update
  `lam ← clip(lam + kappa*(m_ref − m), 0, eta_max)` and send `repulsor_eta =
  lam` next request. λ≈0 on the safe bulk of steps (no TSR tax), rises only
  under genuine approach. Server runs `--repair_schedule none`; the existing
  certificate-triggered renoise (`--renoise_attempts 2`) is the
  repair-as-exception backstop. Defaults to tune on dev cells: m_ref=0.02,
  kappa=0.5/m, eta_max=0.01.
- **A2 stall recovery (`--stall_recovery`)**: client tracks EEF positions; if
  displacement < 1.5 cm over the last 40 control steps → escalation ladder:
  (1) next 3 replans request `hdc_scale > 0` (existing homotopy-diverse
  candidates — select a different route instead of repairing inside the
  blocked one); (2) if still stalled, retreat primitive: scripted lift
  (+5 cm z over ~10 steps) then resume, forcing a fresh approach. At most 2
  retreats/episode.
- **A3 speed-and-separation margins (`--ssm_margins`)**: client finite-
  differences the hazard position across replans; effective margin
  `r_eff + k_ssm * max(0, closing_speed)` with the BASE margin lowered
  toward r_min when the hazard is static and far (ISO/TS 15066 shape). For
  static suites this reduces to today's fixed margin (no SafeLIBERO risk);
  for LS human_safety it removes the standing over-conservatism that stalls
  the arm. k_ssm=0.5 s, r_min = r_eff − 0.02.

## Evaluation protocol (paired, offline-first, per feedback-validation memory)
- Dev cells: Spatial L1 t0+t1, Object t1, LS human_safety L1.
- Stage 1 smoke n=10/task, arms vs production on the same seeds; a component
  survives if it does not regress either axis and improves ≥ +5pp on its
  target cell.
- Stage 2 n=20 paired + McNemar on survivors; Stage 3 full boards n=50 only
  for (a) production+A (the strengthened config) and (b) B if it matches
  production on dev cells.
- Acceptance: A-config must hold the four-number box per suite; B replaces
  production only if ≥ production on both axes per suite. LS: TSR ≥ frozen
  baseline with safe-rate within 2pp of current.

## Risks / honesty
- A2/A3 are hand rules → each is scaffolding with a named Bitter-Lesson
  replacement (learned recovery policy / learned margin model) listed as
  future work; they must each earn an ablation row or be cut.
- B's λ dynamics could oscillate at replan frequency; the integral gain is
  clamped and the renoise backstop bounds the failure at "no worse than the
  static-η arm".
- Parallel session shares the branch: all changes are additive flags in
  existing files + new slurm scripts; no edits to production defaults.
