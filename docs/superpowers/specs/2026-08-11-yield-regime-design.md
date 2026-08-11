# Yield Regime + Guard De-election + Hazard Localizer — beat-baseline campaign design

Date: 2026-08-11. Goal: convert the FINAL BOARD (macro noGT 67.4 vs base 71.5)
into a beats-baseline table by harvesting the two named taxes — the REPAIR TAX
on obstacle_human (freeze-never-resume) and the PERCEPTION TAX on no-GT
obstacle_avoidance (−8.3 vs GT) — without touching any promoted cell's config.
Deadline: table frozen this week.

## Evidence base (ledger, closed series)

- F/F2/F3 re-perception nulls: oah freezes are NOT tracking artifacts. The
  guard is correctly placed on a hand that genuinely occupies the goal region;
  the arm freezes and never resumes (F3 exact tie 49.0/1.0 vs 49.0/1.4).
  Ledger: "Remaining oah lever: stall-triggered recovery."
- Cone gate promoted +25.3 GT / +16.7 noGT on oah, but t3≈2/10, t4=0/10,
  t9≈3/10 remain — the harvestable freeze pool (~10 TSR of oah).
- GT-hs t13 forensic: hand sits within/near the 0.14 m hard core of the goal
  region — "cone can't help." Same failure family: correct identification,
  blunt response. Yield is the non-blunt response.
- oa-noGT arm G: config shuffle p=0.64 → structural detector bias, not
  gating. Only a better position estimate moves this cell.

## Phase 1 — Yield regime (training-free, highest priority)

Fourth FOL-routed safety regime, extending Avoid/Disengage/Refuse:

    YIELD(x) := HAZARD(x) ∧ MOVING(x) ∧ OCCUPIES(x, goal_region)
                ∧ STALLED(eef)

Semantics (state machine, per episode, evaluated at replan frequency):

1. TRIGGER: a hazard elected by the active FOL theory satisfies MOVING
   (settle-window displacement, existing grounding) AND its current position
   lies within r_occ of the task goal region AND the eef is STALLED
   (net eef displacement < d_stall over the last W replans while the guard is
   active). All predicates reuse existing grounded terms; no new name tokens;
   no `_obstacle_` substring anywhere in the decision path.
2. RETREAT: command motion backwards along the episode's executed eef
   trajectory (recorded prefix) until standoff distance r_standoff from the
   hazard is reached. Every retreat action is re-verified against the CURRENT
   hazard position by the existing barrier check before execution (certificate
   integrity invariant — retreat is not exempt from certification).
3. HOLD: zero-motion (or minimal station-keep) while re-grounding
   OCCUPIES(hazard, goal_region) at every replan.
4. RESUME: when ¬OCCUPIES holds for M consecutive replans, exit Yield and
   return control to the normal guided policy (fresh replan, normal regimes).
   Hysteresis prevents chatter.
5. SAFETY: Yield can trigger at most Y_max times per episode (default 2);
   afterwards revert to current behavior (freeze) — never oscillate.

Defaults (dev values, tuned only on dev smoke, then frozen):
r_occ = goal-region radius already used by cone gate; d_stall = 0.01 m;
W = 5 replans; r_standoff = 0.20 m; M = 2; Y_max = 2.

Flag: `--yield_regime` on the guided server (+ client passthrough), DEFAULT
OFF, byte-identical off (CPU test asserts identical action streams with flag
absent vs off).

Routing: class-driven, part of the declared protocol — Yield is enabled for
the dynamic-mover class (oah) and for hs (where it can only trigger if a
mover occupies the goal; static-hand hs episodes with MOVING false are
untouched — t13's hand must satisfy MOVING within the settle window to
qualify; if it never moves, Yield correctly never fires and t13 stays as-is).
Not enabled for oa/aff (no MOVING hazards by class definition).

### Gates (pre-registered)

- G-Y1 (CPU): state-machine unit tests — trigger, retreat certification
  against moved hazard, resume hysteresis, Y_max cap, off-is-byte-identical.
- G-Y2 (dev smoke, GPU): oah dev tasks n=10/task, GT tier, promoted config
  + yield. PASS iff ≥3 previously-frozen episodes now succeed AND zero new
  violations vs paired reference. NO-GO → forensics, one revision round max.
- G-Y3 (promotion, GPU): oah full n=150 paired both tiers + hs n=150 GT
  tier. PASS iff TSR strictly up, violations not up (McNemar on paired
  cells). Then n=450 confirm on promoted cells only.

## Phase 2 — Guard de-election on oa (training-free)

Today NEAR_PATH is elected once (at episode start / election time); a hazard
whose barrier the eef has already passed keeps constraining the approach.
Change: re-evaluate NEAR_PATH(x) per replan against the CURRENT eef→target
segment; a guard whose hazard no longer satisfies its electing theory is
released (de-elected). Election remains class-driven and API-free; de-election
uses the same theory formula, evaluated later in time — no new predicates.

Flag: `--dynamic_election`, default off, byte-identical off.
Gate G-D1: oa n=150 paired, both tiers: PASS iff +≥2 TSR, violations not up.
Risk control: de-election can only RELEASE guards, never add; certificate
still checks all currently-elected guards.

## Phase 3 — Learned hazard localizer (no-GT perception tax)

Small conv head: RGB-D (agentview) → 3D position of the elected hazard.
Labels free from sim GT state during data-collection rollouts (positions
logged; no human annotation). Train/val split by TASK (held-out tasks), never
by episode. Frozen VLA untouched (user-approved small-module rule).

- G-L1 (offline, before any rollout spend): median position error < 2.5 cm on
  held-out tasks; else NO-GO and the noGT oa cell ships as-is.
- G-L2 (rollout): oa-noGT n=150 paired vs current detector. PASS iff TSR +≥3
  and violations ≤ current. Projected: closes part of the −8.3 gap.

## Eval + freeze plan

Promotion arms at n=150 paired; any promoted cell re-confirmed at n=450.
Final refreeze: new FINAL BOARD; projected noGT macro ≈ 73–74 vs 71.5 base if
Phases 1–3 land; Phase 1 alone projected oah 58.9 → ~66 (noGT) — enough for
macro ≈ 69.8; Phases 1+2 → ~70.5–71.5; Phase 3 closes the rest. Claims ≤
results; the paper takes whatever survives the gates.

Scientific-soundness guardrails (binding): develop/confirm split (fresh seeds
for promoted numbers); class-driven config verbatim (no task IDs, no
`_obstacle_` substrings in decisions); certificate integrity on retreat (hard
invariant + test); disclosed weaknesses unchanged (single frozen policy; no
true held-out benchmark — mitigated by declared protocol + SafeLIBERO).

## Implementation hooks (from code map, 2026-08-11)

CLIENT-ONLY implementation — the guided server is stateless by design
(guided_policy.py:273-276); all Yield state (stall window, hysteresis,
Y_max counter, retreat trajectory) is per-episode and lives in
`vlm_pipeline/run_guided_libero_safety_eval.py`. Zero server changes.

- Flags: add `--yield_regime` (+ `--yield_r_occ 0.12`, `--yield_d_stall 0.01`,
  `--yield_w` (in replans, default 5), `--yield_standoff 0.20`, `--yield_m 2`,
  `--yield_max 2`) in the argparse block near the revision controllers
  (~:687-706); echo all into the results dict flag block (:1190-1206).
- Trajectory recording: new `eef_traj` deque (maxlen ~600), gated on
  `--yield_regime`, appended at :1147 alongside `prev_eef`; NEVER cleared
  mid-episode (unlike `eef_hist`).
- Stall detection: replan-boundary check using chunk-boundary eef positions
  (reuse `cone_prev_eef` pattern) — net displacement < d_stall over W replans
  while a guard is engaged. Distinct from `--stall_recovery`'s 40-env-step
  window; the two flags are mutually exclusive (assert).
- OCCUPIES grounding: hazard→goal distance = ||_guard_pos(step_guard[0]) −
  corridor_target|| < r_occ, with `corridor_target` re-read per replan via
  `_corridor_anchor` (:237-246). NOTE: r_occ is its OWN declared constant
  (0.12); the cone's ENGAGE_CONE_HARD_R=0.14 is eef→hazard, not hazard→goal.
- MOVING: reuse episode-level `movers` set (detect_movers, :187-190). A
  hazard must be in `movers` for Yield to arm (t13 static-hand exclusion is
  therefore a code fact, not just intent).
- RETREAT (certified — must NOT reuse the open-loop `in_retreat` +Z lift):
  walk `eef_traj` backwards toward the first recorded point ≥ r_standoff from
  the CURRENT hazard position; each retreat step emits a metric delta in the
  `_plan_actions` convention (exec_parity raw vs ACTION_TO_CMD, :60), and is
  certified client-side before execution: reject and hold instead if the
  interpolated segment passes within the hazard hard radius of the current
  `_guard_pos` (same barrier geometry the cone gate uses). Certificate
  integrity invariant has a dedicated CPU test.
- HOLD: zero-action steps; re-evaluate OCCUPIES each replan boundary.
- RESUME: M consecutive replans of ¬OCCUPIES → clear Yield state, `plan.clear()`,
  fresh server query (normal regimes resume).
- Results: per-episode JSONL adds `yields`, `yield_steps`, `yield_resumes`
  (:1156-1169).
- CPU tests: new `vlm_pipeline/tests/test_yield_regime.py`, modeled on
  `test_fol_predicates.py` (pure import of the client module, no mujoco);
  includes the defaults-off byte-identity pattern
  (`test_argparse_default_is_none...` / `..._leaves_payload_path_untouched`).
- SLURM: dev smoke from `slurm/ls_oah_rescue.slurm` pattern; promotion from
  `slurm/ls_cone_confirm.slurm` (paired arms, oah, both tiers).

## Cost

Phase 1: CPU work + 1 dev smoke (~2 GPU-h) + promotion (~8 GPU-h) + confirm.
Phase 2: piggybacks promotion arms. Phase 3: 1 data job + 1 train + 1 eval
(~12 GPU-h). Zero API cost.
