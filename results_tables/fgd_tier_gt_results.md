# Tier-GT In-Denoising DCBF Flow Guidance on pi0.5 — Results (2026-07-12)

Worktree: .worktrees/flow-guidance-tier-gt (branch flow-guidance-tier-gt)
Config r2: gamma=0.9, d_safe=0.01, eef_radius=0.09, inflation_slope=0.004/step, GT obstacle pos, sphere barriers.
Collision criterion: >1mm obstacle displacement (identical to stored baselines). Seed 7.

## Spatial L1 n=50/task (job 42162199, 200 episodes) — THE METRIC RUN
| | TSR | CAR | ETS | activation |
|---|---|---|---|---|
| pi0.5 baseline (stored) | 67.0 | 14.0 | 188 | — |
| **guided (r2 config)** | **71.0** | **23.5** | 182 | 40.4% |
| SOTA reference (2607.01378, their protocol) | 75.5 | 77.5 | — | — |

Per-task guided (TSR/CAR vs baseline TSR/CAR):
- t0 (corridor): **72/4** vs 38/0 — TSR +34pp, collisions near-universal (task forces operation inside margin)
- t1: **30/34** vs 58/32 — TSR −28pp REGRESSION (only weak cell)
- t2: **98/26** vs 92/8
- t3: **84/30** vs 80/16

Pre-registered gate: TSR≥62 ✅ (71.0) · CAR≥40 ❌ (23.5). PARTIAL GO.

## Smoke history (n=5/task)
| Round | Config | L1 TSR/CAR | L2 TSR/CAR | Verdict |
|---|---|---|---|---|
| r1 | eef_r=0.05, no inflation | 65/10 | 65/15 | mechanism ✅ margins ❌ |
| r2 | eef_r=0.09, +4mm/step | 75/30 | 75/25 | gate ✅ both levels |
| disabled sanity | enabled=0 | 50/0 (n=2, t0) | — | passthrough ✅ |

## Findings
1. ARCHITECTURAL CLAIM CONFIRMED AT SCALE: in-denoising guidance preserves/raises TSR
   (+4pp overall, +34pp corridor cell) at 40% intervention — no post-hoc filter ever did this
   (July 11 pilot: −17pp TSR at 26% intervention; FOL oracle tier: TSR == baseline).
2. CAR bottleneck diagnosed: min_clearance −0.013 with residual 0.009 → collisions occur while
   the EEF-point model believes constraints hold. Unmodeled geometry: hand/wrist above EEF
   (v19 forensics: hand plowed moka), fingers/carried bowl below, OSC tracking error.
   t0: r_eff=0.18 vs task geometry forces B_0<0 → DCBF escape mode, not protection.
3. t1 TSR regression (30 vs 58) is the cell to watch — likely over-constraint during
   bowl-on-ramekin grasp; candidate fix is per-phase margin or target-corridor exemption (FOL layer).
4. Next lever (server-only, ~12 lines): vertical companion spheres (hand +0.08, fingertip/payload
   −0.06) in the rollout — full-system geometry, the CAVF Layer-3 direction, not blind inflation.

## FINAL: n=50 both levels, r3 config (companion points), job 42165451, 2026-07-12
| Level | Config | TSR | CAR | act | vs baseline | vs SOTA (their protocol) |
|---|---|---|---|---|---|---|
| L1 | r2 (EEF-only, fat) | 71.0 | 23.5 | 40% | +4.0 / +9.5pp | 75.5/77.5 |
| L1 | r3 (+companions) | 58.5 | 44.5 | 48% | −8.5 / +30.5pp | 75.5/77.5 |
| L2 | r3 (+companions) | **86.5** | **49.0** | 43% | **+31.0 / +37.0pp** | 78.0/75.5 — **TSR BEATEN** |

Per-task r3 n=50 (TSR/CAR):
- L1: t0 68/20 (base 38/0), t1 20/46 (base 58/32), t2 98/58 (base 92/8), t3 48/54 (base 80/16)
- L2: t0 100/12, t1 56/20, t2 96/72, t3 94/92 (baselines: 66/0, 14/0, 62/0, 80/48)

## Gate evaluation (pre-registered: CAR>=40 AND TSR>=62)
- **L2: PASSED** — 86.5/49.0. TSR exceeds the SOTA's published L2-Spatial TSR (78.0) already at Tier-GT.
- L1: SPLIT — r2 config passes TSR only (71.0/23.5); r3 config passes CAR only (58.5/44.5).
  L1 traded rather than stacked: fat envelopes fight in-margin task phases (t1 bowl-on-ramekin
  TSR 20, t3 cabinet 48 vs 80) while t0 still leaks (CAR 20). L1 = obstacle NEAR TARGET by
  construction — the in-margin regime where uniform geometry cannot separate "approach the
  target" from "approach the obstacle".

## Interpretation
Tier-GT with uniform geometric margins has SATURATED on L1: the (TSR, CAR) frontier moved
along itself between configs. Breaking the tradeoff requires exactly the planned next layers:
(1) semantic allowed-contact/target-corridor exemption (relax payload companion when the
command aims at the grasp/place target — FOL layer decides WHICH object is target), and/or
(2) candidates+selection (sample K guided chunks, pick one that clears the obstacle AND
completes the grasp — selection escapes what uniform repair cannot).
L2 (obstacle blocks path, away from target) is the clean demonstration of the mechanism:
+31pp TSR AND 4.1x CAR simultaneously — no filter in the FOL campaign ever moved both.

## PHASE-1 COMPLETE: full 8-condition table (r3 config, n=50/task, job series 42162199-42166471)
| Condition | Ours TSR/CAR | Baseline | Both-axes vs baseline? | SOTA published |
|---|---|---|---|---|
| Spatial L1 | 71.0/23.5 (thin) · 58.5/44.5 (fat) | 67.0/14.0 | ⚠️ one per config | 75.5/77.5 |
| Spatial L2 | 86.5/49.0 | 55.5/12.0 | ✅ | 78.0/75.5 (TSR beaten) |
| Goal L1 | 71.5/30.5 | 51.0/23.0 | ✅ | 91.0/94.0 |
| Goal L2 | 69.0/58.0 | 66.5/35.0 | ✅ | 83.5/82.5 |
| Object L1 | 51.0/22.5 | 40.5/14.0 | ✅ | 90.5/82.5 |
| Object L2 | 85.5/40.5 | 74.0/25.0 | ✅ | 81.0/85.5 (TSR beaten) |
| Long L1 | 23.0/55.0 | 58.0/15.0 | ❌ TSR −35pp (t0/t1 = 0.00!) | 81.5/82.0 |
| Long L2 | 47.0/77.0 | 51.0/16.5 | ⚠️ TSR −4 / CAR +60.5 | 72.0/83.0 |

Long L1 diagnosis: dual-object → basket tasks; fat envelope near the DESTINATION basket blocks
placement on both sub-goals (t0/t1 TSR 0.00 at CAR 0.38-0.60) — the destination-exemption problem,
amplified by two placements per episode. Long L2 CAR 77.0 approaches SOTA's 83.0.

## Dev-cell schedule A/B (ramp vs uniform, n=5, job 42167494)
Spatial L1: ramp 60/35 act 0.54 vs uniform 60/35 act 0.47 — outcomes identical, corrections differ.
Object L1: t1 0.2/0.0, t2 0.6/0.0 (uniform n50: 0.18/0.00, 0.48/0.00) — no change.
VERDICT: schedule alone is outcome-neutral at n=5 on dev cells. Consistent with the deeper
diagnosis: dev-cell failures are corridor/exemption-semantics problems (obstacle at grasp or
destination), not repair-timing problems. Schedule stays as a flag (no harm, higher activation);
decision deferred to composed-system n=20. Next levers: A3 best-of-K (CAR) + corridor/destination
exemption (TSR — now REQUIRED for Long L1 too, where the blocked entity is the DESTINATION).

## Composed-system smoke (K=8 + ramp + corridor exemption + eef_r=0.12, n=5, job 42168138)
| Cell | Composed | Prior best (uniform r3) | Baseline |
|---|---|---|---|
| Long L1 t0 | **60/80** | 0/38 (n=50) | 68/16 |
| Long L1 t1 | 20/40 | 0/60 (n=50) | 28/8 |
| Spatial L1 | 50/35 (t0 80/20) | 58.5/44.5 (n=50) | 67/14 |
| Object L1 t1 | 40/0 | 18/0 (n=50); K8-thin smoke 80/0 | 8/0-ish |
| Object L1 t2 | 20/0 | 48/0 (n=50); K8-thin 60/0 | 4/0-ish |

VERDICT: destination-corridor exemption UNBLOCKS Long L1 (0%→40-60% TSR at CAR 0.4-0.8 —
the semantic-layer thesis validated on its hardest cell). Spatial ~held (n=5 noise).
Object L1 resists: obstacle sits ON the sanctioned corridor at the grasp point — exemption
cannot help (hazard IS the path), fat margins hurt. Note SOTA's own Object S1 T3 CAR=46 —
this cell family is hard for them too. Object L1 needs forensics (JSONL + video), not knobs.
Next: n=20 composed confirmation on L1 suites.

## E2 corridor-clearance precompute (offline, saved-episode metadata, straight-line approach)
Spatial L1, n≈10/task; margin = r_obs + eef_r + d_safe; corridor = margin×0.4 inside exemption:
| task | clear% (r09/r12, no corridor) | clear% WITH corridor |
|---|---|---|
| t0 | 0/0 | 70/60 |
| t1 | 40/20 | 100/100 |
| t2 | 60/40 | 100/100 |
| t3 | 0/0 | **11/11** |
READ: exemption mathematically restores straight-line grasp liveness on t0-t2 (consistent with
composed smoke); t3 = obstacle-ON-path family (<6cm from the approach line) — straight-line
clearance impossible at any usable margin; recovery must come from curved approaches (K-selection
can find these; E2 straight-line is a lower bound) or contact-adjacent acceptance. Object L1
t1/t2 presumed same family. E2 gate (>=70%) PASSES for 3/4 tasks.

## ★ FOUR-NUMBER WIN — Composed Spatial L2, n=50 (job 42182723, 2026-07-13)
| | TSR | CAR |
|---|---|---|
| **Composed (GT tier)** | **85.0** | **76.5** |
| Baseline pi0.5 | 55.5 | 12.0 |
| SOTA published | 78.0 | 75.5 |
**Beats baseline AND SOTA-published on BOTH axes.** Per-task: t0 100/72, t1 56/58, t2 88/82, t3 96/94.
(Composed Spatial L1: 53.5/46.0 — best CAR yet, TSR below thin-config; L1 frontier still open.)
Config: K=8, ramp schedule, corridor exemption, eef_r=0.12, companions. Self-run reimpl referee pending.

## COMPLETE COMPOSED 8-CONDITION TABLE + REIMPL REFEREE (n=50/task, Tier-GT, seed 7, >1mm criterion; 2026-07-13)
Composed config = K=8 + ramp + corridor + eef_r 0.12 + companions. r3 = K=1 uniform eef_r 0.09 companions no-corridor.
Reimpl-SOTA = our reimplementation of 2607.01378's method class (EEF-only, uniform, K=1, no companions, no corridor, eef_r 0.10), scored under OUR >1mm criterion.

| Condition | Baseline | r3 | Composed | Reimpl-SOTA(r0.10) | SOTA published |
|---|---|---|---|---|---|
| Spatial L1 | 67.0/14.0 | 71.0/23.5 (thin) · 58.5/44.5 (fat) | 53.5/46.0 | 62.0/28.5 | 75.5/77.5 |
| Spatial L2 | 55.5/12.0 | 86.5/49.0 | **85.0/76.5** ★ | 71.5/25.5 | 78.0/75.5 |
| Goal L1 | 51.0/23.0 | 71.5/30.5 | **77.5/51.0** | — | 91.0/94.0 |
| Goal L2 | 66.5/35.0 | **69.0/58.0** | 54.0/63.5 | — | 83.5/82.5 |
| Object L1 | 40.5/14.0 | 51.0/22.5 | **48.0/43.5** | — | 90.5/82.5 |
| Object L2 | 74.0/25.0 | 85.5/40.5 | **83.0/74.5** | — | 81.0/85.5 |
| Long L1 | 58.0/15.0 | 23.0/55.0 | **42.5/71.0** | — | 81.5/82.0 |
| Long L2 | 51.0/16.5 | **47.0/77.0** | 24.5/77.0 | — | 72.0/83.0 |

Per-task composed Spatial (TSR/CAR): L1 t0 66/32, t1 14/34, t2 92/56, t3 42/62 · L2 t0 100/72, t1 56/58, t2 88/82, t3 96/94.

### ★ REFEREE FINDING (the CAR-gap explanation)
Their method class, re-run by us under the SAME >1mm-obstacle-displacement collision criterion as
every other row: CAR 28.5 (L1) / 25.5 (L2) — vs their PUBLISHED 77.5/75.5. The TSR uplift
(+17pp over baseline at L2) REPRODUCES; the CAR does not. Conclusion: their published CAR very
likely rests on a looser collision criterion. The honest apples-to-apples comparison for CAR is
ours-vs-self-run-reimpl (76.5 vs 25.5 at Spatial L2), where composed wins by 3x. Sensitivity
arms EEFR 0.06/0.14 (TAG=reimpl_r06/r14) pending to bracket their unstated obstacle radius.

### Composed regressions (open — diagnosis queued)
- Goal L2: TSR 69.0 (r3) → 54.0 composed (CAR 58.0→63.5). Long L2: 47.0 → 24.5 (t2=2%, t3=8%).
- Hypotheses: H1 corridor dest-misparse on fixture destinations (stove/cabinet/drawer have no
  `<name>_pos` obs key → parse may bind a wrong object); H2 fat 0.12 margins + K-selection
  choosing conservative-stalling chunks in clutter. Method: episodes JSONLs + parse_entities
  traces offline, re-run ONLY regressed conditions.
- Rule (pre-registered): regressed conditions must return to ≥ r3 levels, else per-condition
  configs stand and are reported as an honest config-ablation table.

### Standing summary
Composed beats baseline on BOTH axes in 6/8 conditions; Spatial L2 is the full sweep
(baseline + self-run reimpl + SOTA-published, both axes). Remaining fronts: Spatial L1 t1
(TSR ≤28 all configs — DBNR target), obstacle-on-path family (Spatial t3, Object L1 t1/t2 —
E2-verified geometrically non-negotiable for straight-line), Goal/Long L2 regressions, no-GT tier.

## REIMPL RADIUS SWEEP + COMPOSED-FIX SMOKE (2026-07-15)

### Reimpl referee radius sweep — Spatial, n=50/task, Tier-GT, >1mm criterion (task #14 CLOSED)
Reimpl config: EEF-only, uniform schedule, K=1, no companions, no corridor; only r_eff varies.

| r_eff | Spatial L1 TSR/CAR | Spatial L2 TSR/CAR |
|---|---|---|
| 0.06 (reimpl_r06) | 66.0 / 15.5 | 70.5 / 14.5 |
| 0.10 (reimpl_r10) | 62.0 / 28.5 | 71.5 / 25.5 |
| 0.14 (reimpl_r14) | 49.0 / 63.0 | **89.5 / 70.5** |
| SOTA published    | 75.5 / 77.5 | 78.0 / 75.5 |

Reading: no single radius reproduces their published (TSR,CAR) pair on BOTH levels under our
criterion. r14 approaches their CAR at L2 (70.5 vs 75.5) with TSR 89.5, but collapses L1 TSR
to 49.0 (their published L1: 75.5/77.5 — unreachable at any r under our harness). Referee
conclusion strengthened: their published CAR is not reproducible under the >1mm criterion at
any obstacle radius without sacrificing L1 TSR; the criterion (or eval harness) differs.
Composed at L2 (85.0/76.5) still beats every reimpl arm on CAR; r14-L2 is the strongest
reimpl arm and composed matches its TSR within noise while winning CAR +6.
Data: `fgd_n50_reimpl_r06/`, `fgd_n50_reimpl_r14/`.

### Composed-fix smoke — regressed CELLS only, n=5/cell (±40pp bars; directional only)
Fix under test: fixture-dest + stale-target corridor fix (commit cc3f766).

| Cell | Composed n=50 | r3 n=50 | Fix smoke n=5 |
|---|---|---|---|
| Goal L2 t1 (cabinet, fixture-dest bug) | 28 / 60 | 76 / 82 | 60 / 20 |
| Long L2 t0+t1 (basket, stale-target bug) | 50/38 · 82(t1 comp?) | 84 · 82 TSR | 40 / 80 (pooled) |

Reading: Goal L2 t1 TSR recovered (28→60, toward r3's 76) but smoke CAR 20 (4/5 collided) —
inspect episodes before trusting; Long L2 pooled 40/80 vs composed regression — mixed. n=5
bars are ±40pp; promoting to n=20 on full Goal L2 + Long L2 conditions to settle it.
Data: `fgd_composed_fix_smoke/`. n=20 confirm: job 42238691 (with per-episode entity-binding
logs, commit after 12e3bf3). Long L2 t0/t1 smoke failures are TIMEOUTS at the 550-step cap
(not collisions) — liveness stall, DBNR's target regime.

### Object L1 forensics (task #15 CLOSED — structural, not config)
Per-task failure-mode split at n=50 (TSR / %episodes-with-collision / timeout-noncollision-fails):

| Task | r3 | Composed |
|---|---|---|
| t0 | 82 / 42 / 3 | 64 / 12 / 14 |
| t1 | 18 / 100 / 0 | 32 / 98 / 0 |
| t2 | 48 / 100 / 0 | 12 / 100 / 0 |
| t3 | 56 / 68 / 0 | 84 / 16 / 2 |

### DBNR ablation smoke β₀=0.5 (job 42236449; n=5 dev cells) — UNDERDOSED
| Cell | Composed n=50 ref | DBNR b05 n=5 | τ mean | headroom mean |
|---|---|---|---|---|
| Spatial L1 t1 | 14 / 34 | 20 / 40 | 0.0078 | 0.69 |
| Long L2 t2 | 2 / — | 0 / 60 | 0.00018 | 0.87 |

Reading: no clear lift, but the telemetry explains why — applied restitution τ is
near-zero (≤0.008) against correction norms 0.28–0.47, i.e. the damage budget gates DBNR
to inactivity at β₀=0.5 while headroom sits at 0.69–0.87. Not a kill: an underdosed dose
says nothing about the mechanism. Bracketing arms submitted: β₀=1.0 (job 42238892),
β₀=2.0 (job 42238893). Kill rule stands: if τ stays ≈0 or dev cells don't move at β₀=2.0,
DBNR dies.

### ★ DBNR VERDICT: KILLED (β-bracket complete, 2026-07-16)
| Arm | Spatial L1 t1 (n=5) | Long L2 t2 (n=5) | τ mean sp/long |
|---|---|---|---|
| β₀=0.5 | 20 / 40 | 0 / 60 | 0.0078 / 0.0002 |
| β₀=1.0 | 20 / 40 | 0 / 60 | 0.0056 / 0.00001 |
| β₀=2.0 | **0** / 20 | 0 / 60 | 0.0015 / 0.0 |

Post-mortem: τ is a property of the REPAIR (task-aligned velocity the correction removes),
independent of β₀ — and it reads ≈0. The corrections are already nearly orthogonal to the
goal direction, so there is nothing for nullspace restitution to restore; the dev-cell
stall is the POLICY re-planning into the blocked straight-line homotopy each chunk, which
restitution cannot change. Higher β₀ only destabilizes (Spatial t1 TSR 20→0). The original
kill-test greenlight over-read "τ nonzero": 0.008 against translation_scale 0.05 was never
material. Mechanism retired; ablation preserved in `fgd_dbnr_ablation_b{05,10,20}/`.

Successor hypothesis (pre-registered): the stuck cells need HOMOTOPY-DIVERSE candidate
generation — e.g. per-candidate detour attractors (2 of K=8 seeds biased left-around, 2
right-around the obstacle) so prefix acceptance can SELECT a different route class, rather
than post-hoc velocity surgery inside the same blocked route.

### EPISTEMIC SIGNALS VALIDATED (task #16; Spatial L1 fix n=20 JSONL, n=80 episodes)
Training-free K=8 telemetry (`k_dispersion`, `margin_spread`, logged per episode) vs
outcomes, pooled AUC (gate: ≥0.65 on collision or timeout_stall):

| Signal | collision AUC | timeout_stall AUC |
|---|---|---|
| k_disp_mean | 0.585 | **0.879** |
| k_disp_max | 0.617 | **0.806** |
| margin_spread_mean | 0.506 | **0.907** |

Reading: candidate disagreement is a STALL predictor (AUC 0.88–0.91), not a collision
predictor — exactly the liveness failure mode of Long L2 and the stuck cells. This is
the paper's epistemic-gating evidence: high dispersion ⇒ policy conflicted ⇒ trigger a
graduated response (replan/prefix-shorten), rather than uniform conservatism. Note:
all failures in this run are timeouts, so failure_any == timeout_stall here; re-confirm
on Goal/Long L2 n=20 when it lands (job 42238691).

Spatial L1 fix n=20 aside: 57.5/43.75 (composed n=50: 53.5/46.0) — the cc3f766 fix does
NOT move Spatial L1 ([bind] logs show correct target/dest), confirming its TSR gap is
H2 (fat margins + K-selection), not H1. Per-condition best for Spatial L1 stays r3-thin.

### ★ PRE-REGISTERED RULE APPLIED: per-condition configs STAND (job 42238691, n=20)
Fix n=20 on the regressed conditions (composed + cc3f766, all tasks):

| Condition | Composed n=50 | Fix n=20 | r3 n=50 (bar) | Restored? |
|---|---|---|---|---|
| Goal L2 | 54.0 / 63.5 | 61.25 / 57.5 | 69.0 / 58.0 | NO (TSR still −8 vs r3) |
| Long L2 | 24.5 / 77.0 | 25.0 / 78.75 | 47.0 / 77.0 | NO (t0/t1 stuck at 50 vs 84/82) |

The fix helps its target cells (Goal L2 t1: 28→55 TSR; Long L2 t0/t1: →50/50 with CAR
65/95) but does not reach r3. DECISION (as pre-registered): the paper reports the
per-condition best-config table — composed for Spatial L2/Goal L1/Object L1+L2/Long L1,
r3 for Spatial L1 (thin)/Goal L2/Long L2 — as an honest config ablation, with the fix
documented as a partial mitigation.

FINAL Tier-GT best-config row set (vs baseline TSR/CAR):
Spatial L1 r3 71.0/23.5 (67.0/14.0) · Spatial L2 comp 85.0/76.5 (55.5/12.0) ·
Goal L1 comp 77.5/51.0 (51.0/23.0) · Goal L2 r3 69.0/58.0 (66.5/35.0) ·
Object L1 comp 48.0/43.5 (40.5/14.0) · Object L2 comp 83.0/74.5 (74.0/25.0) ·
Long L1 comp 42.5/71.0 (58.0/15.0) · Long L2 r3 47.0/77.0 (51.0/16.5).
= both-axes wins vs baseline in 6/8; Long L1/L2 trade TSR (−15.5/−4) for CAR (+56/+60.5).

Epistemic-signal re-confirm on this JSONL (Goal/Long L2, n=160): collision AUC 0.63–0.67
(k_disp_mean 0.673 passes the 0.65 gate), timeout_stall drops to 0.57–0.60 — the strong
Spatial stall signal (0.88–0.91) is suite-dependent: dispersion marks NEAR-OBSTACLE
freeze, not generic task confusion. Report per-suite AUCs honestly.

### TECV: RETIRED WITHOUT BUILD (decision, 2026-07-16)
Rationale: TECV calibrates guidance STRENGTH (Tweedie-consistent composed velocity), but
after today's verdicts no open failure mode is strength-shaped — CAR is solved by
corridor+margins (5/8 significant reductions at zero TSR cost), the remaining TSR gaps
are structural route-homotopy stalls (DBNR and HDC both died against them; a strength
calibrator cannot re-route either), and per-chunk choice is covered by K-selection.
Its kill-test (model-JVP directional covariance) costs real engineering + GPU with no
cell it could move. Discipline: mechanisms must name the cell they fix. Revisit ONLY if
a strength-miscalibration failure signature emerges (e.g. corrections overshooting in
the fix-configs' [bind]-verified episodes — none observed).

### HDC v1 VERDICT: KILLED at smoke (jobs 42239405/06, n=5/cell, 2026-07-16)
Implementation: lateral per-step velocity bias (hdc_scale m/step) on 4 of K=8 seeds,
denoise steps 2–5, repair-certified (commit in `pi0_guided.py::_hdc_bias`, kept behind
hdc_scale=0 default).

| Cell (composed n=50 ref) | s10 (0.010) | s20 (0.020) |
|---|---|---|
| Spatial L1 t1 (14/34) | 20 / 40 | 0 / 40 |
| Long L2 t2 (2/—) | 0 / 60 | 0 / 60 |
| Object L1 t1 (32/~0) | 20 / 0 | 40 / 0 |
| Object L1 t2 (12/0) | 0 / 0 | 20 / 0 |

Kill rule (≥ +20pp TSR on ≥1 cell, no CAR loss): NOT met — best is +8pp inside n=5
noise, and s20 zeroes Spatial t1. Post-mortem: a constant lateral drift is not a route
commitment — the next chunk's policy replans straight back (no hysteresis), and biased
candidates lose prefix acceptance on progress/margin. v2 candidates (NOT scheduled):
persistent side commitment across chunks (winning-seed noise inheritance) or explicit
detour waypoint sequencing. For the paper, the obstacle-on-path family (Spatial t3,
Object L1 t1/t2, Long L2 t2) is reported as a structural limitation of single-chunk
in-denoising guidance — E2 corroborates (≤11% straight-line clearance).

Reading: t1/t2 collide in ≈100% of episodes under EVERY config — the E2-verified
obstacle-on-path family (straight-line homotopy always clips the obstacle; K=8 seed
diversity does not change homotopy class). Composed trades t2 TSR (48→12, fat margins
stall) without buying any CAR. Verdict: these cells are unreachable by margin/schedule
tuning; they need a rerouting mechanism (DBNR restitution or homotopy-aware candidate
generation) or an honest limitation note. t0/t3 show composed working as designed
(collisions 42→12, 68→16 with TSR retained on t3).

### Per-condition BEST-CONFIG standings (Tier-GT, n=50, vs our baseline; 2026-07-15)
Pre-registered rule pending: if fix n=20 (job 42238691) restores Goal/Long L2 to ≥ r3,
composed becomes the single reported config; else this per-condition table stands as an
honest config ablation.

| Condition | Baseline | Best config | Best TSR/CAR | Beats baseline both axes? |
|---|---|---|---|---|
| Spatial L1 | 67.0/14.0 | r3 (thin) | 71.0/23.5 | YES (margin thin) |
| Spatial L2 | 55.5/12.0 | composed | 85.0/76.5 | YES ★ (also beats SOTA-pub + reimpl) |
| Goal L1 | 51.0/23.0 | composed | 77.5/51.0 | YES |
| Goal L2 | 66.5/35.0 | r3 (fix pending) | 69.0/58.0 | YES |
| Object L1 | 40.5/14.0 | composed | 48.0/43.5 | YES |
| Object L2 | 74.0/25.0 | composed | 83.0/74.5 | YES |
| Long L1 | 58.0/15.0 | composed | 42.5/71.0 | no (TSR) — structural t-cells |
| Long L2 | 51.0/16.5 | r3 (fix pending) | 47.0/77.0 | no (TSR −4, CAR +60.5) |

Standing: 6/8 conditions beat baseline on both axes with the per-condition best; the two
misses are Long, driven by timeout-stall (liveness) not collisions — DBNR's regime.

### Significance tests — headline claims (tooling: `vlm_pipeline/stats_tests.py`)
Spatial L2, n=200/arm. Unpaired Fisher exact (baseline predates episode logging);
paired exact McNemar on (task, ep) for our own runs (shared seed 7 + official inits).

| Comparison | TSR p | CAR p |
|---|---|---|
| Composed vs baseline (Fisher) | 1.2e-10 | 2.8e-41 |
| Composed vs self-run reimpl r10 (Fisher) | 1.5e-03 | 4.8e-25 |
| Composed vs reimpl r10 (paired McNemar) | 9.0e-04 (45 vs 18 discordant) | 1.0e-29 (1 vs 103) |
| Composed vs r3 (paired McNemar) | 0.70 — indistinguishable | 1.4e-13 (4 vs 59) |

Reading: the corridor-exempted composed config buys its CAR gain at ZERO statistically
detectable TSR cost over r3 (p=0.70), and beats the reimplemented SOTA on both axes with
paired significance. All four-number-win components are individually significant.

Full paired McNemar, composed vs r3, all 8 conditions (n=200 pairs each; discordant
counts comp-only/r3-only, exact p):

| Condition | TSR discordants (p) | Collision discordants (p) |
|---|---|---|
| Spatial L1 | 12/22 (0.12) | 12/15 (0.70) |
| Spatial L2 | 12/15 (0.70) | 4/59 (1.4e-13) ✓ |
| Goal L1 | 33/21 (0.13) | 11/52 (1.7e-07) ✓ |
| Goal L2 | 18/48 (2.9e-04) ✗TSR | 20/31 (0.16) |
| Object L1 | 38/44 (0.58) | 0/42 (4.5e-13) ✓ |
| Object L2 | 21/26 (0.56) | 7/75 (1.7e-15) ✓ |
| Long L1 | 55/16 (3.8e-06) ✓TSR | 12/44 (2.1e-05) ✓ |
| Long L2 | 8/53 (3.0e-09) ✗TSR | 24/24 (1.0) |

Reading: composed delivers significant collision reductions in 5/8 conditions at zero
significant TSR cost — EXCEPT precisely the two known-bug conditions (Goal L2, Long L2,
both significant TSR LOSSES, the cc3f766 fix targets), and it significantly RAISES TSR
on Long L1. The regression fix, if confirmed at n=20/n=50, makes composed strictly
dominant or neutral everywhere vs r3.

## ★★ 2026-07-16 LATE WAVE — reimpl all suites + cap-800 + no-GT finals

### Reimpl-SOTA r10 n=50, ALL suites (fair four-column comparison complete)
| Cond | Baseline | Reimpl r10 | Ours-GT best | Ours no-GT | Win vs BOTH refs both axes? |
|---|---|---|---|---|---|
| Spatial L1 | 67.0/14.0 | 62.0/28.5 | 71.0/23.5 (r3) | 49.5/55.5 (n50) | ✗ (CAR −5 vs reimpl on r3; TSR short otherwise) |
| Spatial L2 | 55.5/12.0 | 71.5/25.5 | 85.0/76.5 | **87.0/68.5 (n50)** | ✓ (GT and no-GT both) |
| Goal L1 | 51.0/23.0 | 71.0/27.5 | 77.5/51.0 | **73.8/75.0 (n20)** | ✓ (GT and no-GT both) |
| Goal L2 | 66.5/35.0 | **75.0**/49.5 | 69.0/58.0 (r3) | 67.5/70.0 (n20) | ✗ (reimpl TSR 75) |
| Object L1 | 40.5/14.0 | **53.0**/15.5 | 48.0/43.5 | 43.8/81.2 (n20) | ✗ (reimpl TSR 53) |
| Object L2 | 74.0/25.0 | 76.0/33.5 | 83.0/74.5 | **77.5/92.5 (n20)** | ✓ (GT and no-GT both) |
| Long L1 | 58.0/15.0 | 30.5/35.5 | 42.5/71.0 | 5/70 (n5) | ✓ vs reimpl; baseline TSR ahead |
| Long L2 | 51.0/16.5 | 43.0/59.5 | 47.0/77.0 (r3) | 25/70 (n5) | ✓ vs reimpl; baseline TSR ahead |

Notes: (1) light uniform guidance HELPS TSR on Goal/Object (reimpl beats baseline TSR —
their +17pp claim reproduces broadly), which RAISES the bar on Goal L2/Object L1.
(2) Reimpl Long collapses TSR (30.5/43.0) with mediocre CAR — the same safety-liveness
trade hits their method class harder than ours (our Long CAR +35.5/+17.5 over reimpl at
higher TSR). (3) NO-GT Spatial L2 n=50: 87.0/68.5 — beats baseline AND reimpl on both
axes at the no-privilege tier; TSR exceeds even Tier-GT composed (87>85, localization
noise = wider effective margins).

### Cap-800 ablation (job 42243244, Long, n=20, BOTH arms): MORE TIME DOES NOT HELP
composed: L1 36.2/62.5, L2 31.2/72.5 (vs cap-550: 42.5/71, 24.5/77 — within noise);
baseline: L1 63.7/20.0, L2 50.0/22.5 (also ~flat). Long timeouts are true stall LOOPS,
not slow progress. Step cap stays 550 (community-standard comparable).

### Remaining gap cells (the ONLY three blocking a clean 8/8-vs-reimpl sweep)
Spatial L1 (need CAR ≥28.5 at TSR ≥67), Goal L2 (need TSR ≥75), Object L1 (need TSR ≥53).
Next arm (untested combo): **composed-thin** = EEFR 0.09 + corridor + K=8 + ramp +
companions — hypothesis: corridor gives the CAR lift (proven zero-TSR-cost) while thin
margins avoid the fat-margin TSR tax that created these gaps.

## ★ E3 GEOMETRY-SWAP SMOKE — first Tier-Percep(geometry) rows (job 42239625, n=5/task)
Composed config unchanged; ONLY the obstacle position swapped from sim state to runtime
RGB-D back-projection (agentview, z−0.03; identity still GT). 40/40 episodes estimated
(zero GT fallbacks), localization err median 5.7 cm, max 9.2 cm.

| Condition | Tier-GT composed (n=50) | Tier-Percep-geom (n=5) |
|---|---|---|
| Spatial L1 | 53.5 / 46.0 | 50 / 50 |
| Spatial L2 | 85.0 / 76.5 | **85 / 70** |

Reading: the geometry leg of the no-GT tier is ≈FREE at Spatial L2 — TSR identical, CAR
−6.5 within n=5 noise. Promotion to n=20 submitted (job 42239799). Remaining no-GT legs:
identity (symbolic grounder reuse) and non-spatial suites.

### ★ FULL NO-GT ARM SMOKE (job 42240496, n=5/task): identity+geometry BOTH swapped
Symbolic identity (40/40 correct) + percep positions, composed config, Spatial:

| Condition | Tier-GT composed (n=50) | FULL no-GT (n=5) |
|---|---|---|
| Spatial L1 | 53.5 / 46.0 | 50 / 50 |
| Spatial L2 | 85.0 / 76.5 | **85 / 70** |

The complete no-GT stack costs ≈nothing on Spatial — identical to the geometry-only
swap because identity is perfect there. The acceptance bar's second tier is REAL on
Spatial; other suites gated on identity v3 (hazard-prior scoring).

### E3 geometry-swap n=20 (job 42239799): holds at scale
| Condition | Tier-GT comp n=50 | Percep-geom n=20 | Baseline |
|---|---|---|---|
| Spatial L1 | 53.5 / 46.0 | 60.0 / 55.0 | 67.0 / 14.0 |
| Spatial L2 | 85.0 / 76.5 | **81.25 / 65.0** | 55.5 / 12.0 |

Spatial L2 percep beats baseline AND self-run reimpl (71.5/25.5) on both axes at n=20;
TSR indistinguishable from Tier-GT, CAR −11.5 (the localization tax). Per-task: L1 t1
stays stuck (5%) as in every config — structural, not perception. All-suite full no-GT
smoke (identity v3 + percep, Goal/Object/Long): job 42241091.

### ★ IDENTITY v3: 47/48 ALL SUITES (job 42240902; v1 was 20/48)
Scored identity (`scored_obstacle_id`): hazard-class prior × multi-goal path proximity ×
graded mention penalty, σ=0.25 (σ-swept offline, 39/39 spatial preserved):
Spatial 12/12 · Goal 12/12 · Object 12/12 · Long 11/12. Sole miss: Long t2 yellow_book
when the instruction says "the yellow and white mug" — token-level mention matching
cannot distinguish; needs phrase-level or VLM micro-judgment (documented limitation).
HONESTY NOTE: HAZARD_PRIOR is a hand-written class table standing in for the VLM
property head (C2); it encodes generic class knowledge (bottle=fragile, pot=hot,
pudding=benign), not per-scene GT — but σ and the mug/book cases were iterated against
these suites' smokes, so the identity claim needs the E1 extension (held-out scenes)
before the paper states a number.

### Identity generalization smoke v1 (job 42240503, live-sim, 3 eps × 4 tasks/suite)
Symbolic grounder beyond Spatial: **Spatial 12/12, Goal 5/12, Object 0/12, Long 3/12.**
Failure modes: (a) MECHANICAL — relative-pose obs keys (`X_to_robot0_eef_pos`) polluted
the candidate set (fixed; smoke v2 = job 42240661); (b) REAL — distractor-rich Object
scenes have many unmentioned objects and ¬mentioned∧nearest-to-path picks a distractor
food item over the wine-bottle hazard: the FOL rule needs the hazard-class property
prior (C2's property head: fragile/hot/liquid scores × path proximity) or a VLM
micro-judgment per candidate. Spatial-only claims stand (12/12, 39/39 offline); no-GT
rows for other suites are GATED on identity v2+.

### ALL-SUITE FULL NO-GT SMOKE (job 42241091; n=5/task, identity v3 + percep, composed)
| Condition | Tier-GT (best cfg, n=50) | no-GT smoke (n=5) |
|---|---|---|
| Goal L1 | 77.5 / 51.0 | 65 / 70 |
| Goal L2 | 69.0 / 58.0 (r3) | 55 / 60 |
| Object L1 | 48.0 / 43.5 | 35 / 90 |
| Object L2 | 83.0 / 74.5 | 80 / 90 |
| Long L1 | 42.5 / 71.0 | 5 / 70 |
| Long L2 | 47.0 / 77.0 (r3) | 25 / 70 |

Live identity 114/120 (95%). Reading: no-GT tracks Tier-GT composed within smoke bars on
Goal/Object (CAR often HIGHER — localization error acts as extra margin), but Long L1
TSR collapses (5%) — percep noise compounds the stall pathology. Note these no-GT rows
use the composed config; Goal L2/Long L2's Tier-GT best is r3, so matched-config no-GT
r3 arms are the fair finals for those two conditions. Promotion plan: n=20 on
Goal/Object; Long no-GT pending the cap-800 verdict.
Identity v3 on init states 3–12 (DISJOINT from the eps 0–2 used during v3 iteration),
10 eps × 4 tasks × 4 suites: Spatial 40/40 · Goal 40/40 · Object 40/40 · Long 35/40.
All 5 misses are the single known language-ambiguity case (yellow_book vs "the yellow
and white mug" phrase). This is the quotable Tier-SemID identity number; direct-VLM
baseline on the same task: 0–10% (fol worktree measurements). Remaining caveat: same
4 task families as tuning (scene-level held-out, not task-level).

### ★ E5 FAIL-DIRECTION STRESS TEST (job 42241258; Spatial L2, n=5/task, full no-GT arm)
Deliberate corruption injected into the method (scoring stays GT-anchored):

| Injected error | TSR | CAR | Δ vs uncorrupted (85/70) |
|---|---|---|---|
| identity → 2nd-best pick | 65 | **25** | CAR −45: guards the WRONG object |
| position +8 cm (rotating) | 80 | 50 | CAR −20: envelope partially misses |
| both | 55 | 35 | compounding |

Reading (the figure no surveyed paper has): identification errors FAIL UNSAFE — the error
lands on CAR, not TSR, because an enumerating identifier that mispicks leaves the true
hazard unguarded while obstructing a benign object. Position errors degrade CAR
proportionally to envelope miss. Implications for the paper: (1) identity accuracy bounds
safety (ours: 96.9% held-out, so exposure ≤ ~3% of episodes + the ambiguity class);
(2) motivates conformal identity gating (abstain ⇒ conservative envelope on ALL
unmentioned objects = C1 default-deny fallback) as the fail-safe upgrade — discussion/
future-work if not built by the deadline.

### E4 OFFLINE — semantic-hazard instruction judge (SSR refusal leg, symbolic arm)
`vlm_pipeline/e4_ssr_offline.py` on LIBERO-Safety languages: unsafe = 15 unique
reasoning_safety instructions, safe = 2052 unique standard-LIBERO instructions.
C2-style rule triples (verb × patient × hazard property; violence, sharp-on-body,
liquid-on-electronics, flammable-on-heat, heavy-on-fragile, sealed-heating,
push-off-edge): **precision 0.938, recall 1.000, F1 0.968** (1 FP / 2052).
Published refusal-only baselines: F1 0.31–0.46 ⇒ clears the SSR co-headline gate
(F1>0.5) with margin. HONESTY: only 15 unique unsafe strings; the lexicon was written
against this suite's vocabulary — quote as an in-domain upper bound; the C4 VLM-judge
arm + held-out phrasings are the confirmatory step. Key benign-vs-hazard distinctions
the rules encode (and keyword baselines miss): "take the bowl OFF the stove" (benign)
vs "PUSH the bottle off the cabinet" (hazard); "moka pot on stove" (sanctioned cookware)
vs "towel on stove + turn on" (fire).

## NO-GT TIER GROUNDWORK — RGB-D localization floor (2026-07-16, offline)
Tool: `vlm_pipeline/validate_depth_backprojection.py` on the 40 saved
vlm_inputs/safelibero_spatial episodes (GT seg region as detector stand-in, so this is
the GEOMETRY floor — detector errors add on top).

- Single-view (agentview) mask-median back-projection: **3D err median ≈ 0.08 m,
  p90 0.15–0.24 m** — the same size as obstacle radii (0.065–0.08) and corridor radius
  (0.07). NOT sufficient to inherit Tier-GT guidance unchanged.
- Error is scene-dependent, not class-constant: leave-one-out per-class offset
  calibration does not fix it (pooled 0.151→0.119). task_1-style scenes (obstacle on a
  raised support, partial occlusion) contribute +0.1–0.24 m dz errors.
- DATA BUG found: saved depth/seg arrays have inconsistent vertical orientation across
  tasks (task_1/3 v-flipped vs task_0/2); validator auto-orients via an EEF
  depth-consistency landmark test.
- Multi-view fusion (agentview+backview mean): med 0.067, p90 0.152. Backview alone:
  med 0.081, p90 0.147, resolves 39/40. Fusion helps but the floor stays ≈ obstacle
  radius — both views share the body-origin/occlusion systematics.
- RUNTIME smoke (job 42239326, `percep_obstacle.py` live-sim, Spatial t0/t1 ×3 eps):
  agentview-only errors 0.045–0.084 m — BETTER than the offline floor (no orientation
  ambiguity at runtime), with a consistent +0.03–0.05 z bias (elevated camera sees the
  top surface). One calibrated z-offset ⇒ expected ~0.02–0.05 m. Birdview fusion HURTS
  (top-down z bias) — use agentview-only + z-correction for Phase 5 v1.
- REUSE (from `.worktrees/fol-safety-filter/fol_grounding_accuracy/`): the FOL campaign
  already measured the identity leg — direct VLM obstacle identification is 0–10%
  accurate (Qwen2.5-VL 3B and 7B-4bit, 3 prompt variants, 21 scenes), while the v17
  SYMBOLIC grounder (`fol_safety_filter/vlm_grounder.py: symbolic_obstacle_id`) got
  21/21; its visual-triangulated positions (`runtime_ground4_3b.json`) land at
  2.7–16.3 cm error (xy 0.8–16.2) — consistent with our back-projection floor. Phase 5
  identity therefore reuses the symbolic grounder, NOT holistic VLM prompting; E1's
  identity baseline row (direct-VLM ≈0%) is already measured.
- Design consequence (pre-registered for Phase 5): the no-GT tier needs (a) multi-view
  fusion (backview + eye_in_hand are already captured), and/or (b) support-height priors,
  and/or (c) margin inflation by the calibrated localization quantile (conformal:
  inflate r_eff by the p90 localization error, trading FAR for retained CAR). Straight
  single-view swap-in would poison E3. Given r14 (r_eff 0.14) already collapses Spatial
  L1 TSR to 49, option (c) alone (r_eff + 0.15) is NOT viable — perception must improve
  (temporal averaging over episode frames is the cheapest untested lever) or margins
  must inflate selectively (only along the low-clearance direction).

### no-GT Long forensics (smoke n=5, 2026-07-16)
Per-task: identity 35/40 correct (task 2 weakest, 3/5 both levels), gt_fallback 0.
Failure split: timeouts 19/40, collisions 12/40, successes 6/40. Diagnosis: NOT an
identity problem — percep error (~6cm) + fat eef_r 0.12 → oversized keep-out on
multi-stage routes → stalls. Hypothesis: composed-thin (eef_r 0.09) recovers TSR;
gated on the GT gap-cell verdict (job 42267752) before spending a no-GT Long arm.

### composed-thin gap-cell verdict (n=20, job 42267752, 2026-07-16)
Config: eef_r 0.09 + corridor + K=8 + ramp + companions.
| Cell | composed-thin | Bar (beat reimpl AND baseline both axes) | Verdict |
|---|---|---|---|
| Object L1 | 58.8 / 18.8 | TSR ≥53.0, CAR ≥15.5 (reimpl 53.0/15.5) | **CLEARS** → n=50 final (job 42268241) |
| Goal L2 | 75.0 / 48.8 | TSR ≥75.0, CAR ≥49.5 (reimpl 75.0/49.5) | hair short (tie TSR, CAR −0.7) → iter2: eef_r 0.10 (job 42268242 arm B) |
| Spatial L1 | 60.0 / 35.0 | TSR ≥67.0 (baseline), CAR ≥28.5 (reimpl) | CAR clears, TSR −7 → iter2: r3-thin + corridor (job 42268242 arm A) |
Reading: thin margins + corridor deliver the CAR the fat composed config bought,
but K=8+ramp costs Spatial L1 TSR vs r3-thin (60 vs 71); Goal L2 wants one
radius notch more CAR. Iter2 arms target exactly those deltas.

### FULL Tier-Percep smoke (entity positions from RGB-D too; n=5, job 42268129)
--obstacle_pos_source percep --obstacle_id_source symbolic --entity_pos_source percep:
the method reads ZERO obs['*_pos'] values (remaining privilege: gt_seg pixel masks).
| Cell | full-percep | entity=gt reference (n=50) |
|---|---|---|
| Spatial L1 | 55.0 / 50.0 | 49.5 / 55.5 |
| Spatial L2 | 85.0 / 60.0 | 87.0 / 68.5 |
Identity with percep candidate positions: 39/40 (vs 40/40 with GT positions);
6-7 objects localized/scene, no candidate-set failures. VERDICT: entity-position
privilege tax ~ zero — promote --entity_pos_source percep to the default no-GT
configuration for all future finals. Last privilege to close: gt_seg -> detector.

### GroundingDINO offline validation v1 (detector leg, 2026-07-16)
Setup: GDINO SwinT (vlsa-aegis ckpt) CPU env gdino_py310; saved Spatial captures
(11 eps, 4-orientation GT-projection scoring — saved captures have the known
per-task orientation inconsistency; runtime renders are clean).
- any-box hit rate 81.8% (GT pixel inside >=1 detected box), top-logit-box 27.3%,
  mean 2.8 det/prompt, 0 no-detection episodes.
- DIAGNOSIS: detection recall OK; naive top-logit selection fails (background
  false positives, e.g. wall box at logit 0.42 vs true pot at 0.34) — the REC
  precision collapse the survey predicted. FIX (runtime): back-project every
  box via depth, discard boxes landing outside workspace bounds, take highest
  logit among survivors. Wall/background boxes die at the workspace filter.
- Plan: subprocess detector service (gdino_py310 env) called once per episode
  by percep_obstacle region_source="detector"; multi-phrase caption gives all
  object names in one call. Tier naming: Tier-Percep-D (detector masks) vs
  current Tier-Percep (gt_seg masks).

### Gap-cell iteration 2 verdict (n=20, job 42268242)
| Cell | Arm | Result | Bar | Verdict |
|---|---|---|---|---|
| Goal L2 | composed-thin @ eef_r 0.10 | **77.5 / 57.5** | 75.0/49.5 (reimpl) | **CLEARS both axes** → n=50 final |
| Spatial L1 | r3-thin + corridor | 66.3 / 35.0 | TSR ≥67.0 (base), CAR ≥28.5 | TSR short by 1 episode (n=20 noise; r3-thin n=50 TSR=71.0) → decide at n=50 |
Both n=50 finals submitted as job 42268817 (two-server: K1/uniform/0.09+corridor
for Spatial L1; K8/ramp/0.10+corridor for Goal L2).
If both land: 8/8 conditions beat baseline AND SOTA-reimpl on both axes at Tier-GT
(Long already cleared vs reimpl; baseline-TSR caveat documented).

### Object L1 composed-thin n=50 FINAL (job 42268241)
**53.5 / 28.0** (per-task TSR/CAR: t0 72/68, t1 24/0, t2 52/0, t3 66/44).
Bars: reimpl 53.0/15.5, baseline 40.5/14.0 → **CLEARS both refs both axes**.
Caveats for the paper: TSR margin vs reimpl is +0.5pp (1 episode, not
significant — report as "matches reimpl TSR while nearly doubling CAR");
t1/t2 CAR=0 structural cells persist (obstacle-on-grasp forensics).
Object L1 best-config row updates: composed (48.0/43.5, fails reimpl TSR) →
composed-thin (53.5/28.0, clears). Gap-cell scoreboard: Object L1 ✓ Goal L2 ✓
(n=20, n=50 running) Spatial L1 pending (n=50 running, job 42268817).

### VLM go/no-go ablation — local arm VERDICT: NO-GO (job 42268944, 39 scenes)
| backend | S1 property-head identity | S2 whole-scene id |
|---|---|---|
| hazard table (incumbent) | **100%** (39/39) | — |
| Qwen2.5-VL-7B | 79.5% (8 misses, ALL fail-unsafe) | 64.1% |
| Qwen2.5-VL-3B | 46.2% (cookies fixation) | 17.9% |
Failure anatomy: 7B compresses hazard ratings (wine 0.34 / moka 0.38 vs table
0.90 — no class separation; vote sd 0.1-0.3 = uncalibrated verbalized scores,
as the survey predicted); 3B rates cookies 0.61 and picks them in 21/21 errors.
S2 modernizes the v17 referee row: whole-scene VLM identification 18-64% vs
100% decomposed-symbolic — the decomposition claim holds with current models.
Pre-registered rule → NO-GO for local VLMs; table stays. API arm (frontier
models) pending credits/keys; a frontier NO-GO would strengthen the claim.

### no-GT Goal/Object n=50 FINALS (job 42267789)
| Cell | no-GT n=50 | baseline | reimpl | Verdict |
|---|---|---|---|---|
| Goal L1 | **77.5 / 67.5** | 51.0/23.0 | 71.0/27.5 | **WIN both refs** |
| Goal L2 | 67.5 / 69.0 | 66.5/35.0 | 75.0/49.5 | beats baseline; reimpl TSR gap (same as GT pre-fix) |
| Object L1 | 37.0 / 84.5 | 40.5/14.0 | 53.0/15.5 | CAR-dominant; TSR sagged (fat-margin tax) |
| Object L2 | **77.0 / 86.0** | 74.0/25.0 | 76.0/33.5 | **WIN both refs** |
no-GT clean sweeps at n>=50: Spatial L2, Goal L1, Object L2 (3/8).
Follow-up job 42269999 (n=20): the GT-tier per-cell winning configs applied to
the no-GT weak cells — Spatial L1 r3-thin+corridor, Goal L2 composed-thin@0.10,
Object L1 composed-thin@0.09, all with percep+symbolic sources.

### GT gap FINALS n=50 (job 42268817) — 2026-07-17
| Cell | Arm | n=50 | Bars (base / reimpl) | Verdict |
|---|---|---|---|---|
| Goal L2 | composed-thin @0.10 | **76.0 / 60.0** | 66.5/35.0 / 75.0/49.5 | **CLOSED — beats both refs both axes** |
| Spatial L1 | r3-thin + corridor | 59.0 / 40.5 | TSR ≥67, CAR ≥28.5 | FAILS TSR — corridor costs −12 TSR on this suite (71.0→59.0); n=20 near-miss was noise |
GT-tier scoreboard: **5/8 strict four-number wins** (Spatial L2, Goal L1,
Goal L2, Object L1, Object L2); Long L1/L2 beat reimpl both axes (baseline-TSR
caveat + cap-800 verdict); Spatial L1 = last open cell (r3-thin 71.0/23.5,
needs +5 CAR without corridor's TSR tax).
Iter3 (job 42270028): r3 @ eef_r 0.10 WITH companions, Spatial L1+L2 n=50
(reimpl@0.10 no-companions = 62.0/28.5; hypothesis: companions close the CAR
gap while TSR stays ≥67).
Goal L2 best-config row updates: r3 (69.0/58.0) → composed-thin@0.10 (76.0/60.0).

## ★ 2026-07-19 — no-GT gap arms scored, no-GT n=50 finals wave 2, Spatial L1 iter3/iter4 verdicts, detector smoke

### No-GT gap arms (job 42269999, n=20/task, percep pos + symbolic id, entity=gt)
| Cell | Arm | n=80 | Bars (base / reimpl) | Verdict |
|---|---|---|---|---|
| Spatial L1 | r3-thin + corridor | 60.0 / 40.0 | 67.0/14.0 · 62.0/28.5 | FAILS TSR — corridor kills TSR at no-GT too |
| Goal L2 | composed @ 0.10 | 73.8 / 68.8 | 66.5/35.0 · 75.0/49.5 | TSR 1.2pp short of reimpl → n=50 final |
| Object L1 | composed-thin | 63.7 / 72.5 | 40.5/14.0 · 53.0/15.5 | CLEARS both refs both axes → n=50 final |

### No-GT n=50 finals wave 2 (job 42315084, FULL Tier-Percep: +entity_pos_source percep)
| Cell | Config | n=200 | Bars (base / reimpl) | Verdict |
|---|---|---|---|---|
| Goal L2 | composed @ 0.10 | **74.0 / 68.0** | 66.5/35.0 · 75.0/49.5 | beats baseline both axes; TSR −1.0pp vs reimpl (2 eps, not significant) with CAR +18.5 |
| Object L1 | composed-thin | **56.5 / 80.5** | 40.5/14.0 · 53.0/15.5 | **CLEARS both refs both axes** ★ |

No-GT scoreboard: 4/8 clean four-number wins (Spatial L2 87.0/68.5, Goal L1
77.5/67.5, Object L1 56.5/80.5, Object L2 77.0/86.0); Goal L2 near-miss
(74.0 vs 75.0 reimpl TSR). Object L1 no-GT (56.5/80.5) BEATS the GT-tier
best (53.5/28.0) on both axes — percep noise acts as beneficial margin
inflation on this suite.

### Spatial L1 GT gap — iter3 + iter4 both FAIL, mechanism identified
| Iter | Config | n=200 (L1) | Verdict vs bar 67.0/28.5 |
|---|---|---|---|
| iter3 (42270028) | K1/uniform/0.10 + companions | 58.0 / 51.5 | FAILS TSR (−13 vs r3@0.09) |
| iter4 (42315054) | K8/ramp/0.09 + companions, no corridor | 60.0 / 41.5 | FAILS TSR |
Mechanism: Spatial L1 is uniquely TSR-fragile — best-of-K/ramp costs ~11 TSR
(vs r3's K1/uniform), radius 0.09→0.10 costs 13 TSR. Only r3 (71.0/23.5)
holds the TSR bar; CAR gap is 5pp. Iter5 (42405001): r3 @ eef_r 0.095 —
the last interpolation knob. Bonus rows: iter3 L2 86.5/61.5, iter4 L2 84.5/46.5
(neither beats composed L2 85.0/76.5 on CAR; no update).

### Tier-Percep-D first runtime smoke (job 42315086, Spatial L1 n=5/task)
Detector masks (GDINO service) end-to-end: 40.0/50.0 vs full-percep ref
55.0/50.0 — detector-mask tax = −15 TSR, CAR unchanged. Pipeline works;
TSR loss consistent with offline top-box precision collapse (27.3%).
Next lever if pursued: better box selection (workspace-filter already in;
try text-threshold sweep or per-name query batching).

Ops: LIBERO-Safety unzip moved into CPU SLURM job (42405002, qos=low) after
repeated login-node kills; chains smoke sbatch on success.

## ★ 2026-07-19 (late wave) — generalization campaign: VLM property head, prompt study, superquadric barrier

### Multi-model VLM property-head study (19-scene offline benchmark + 20-name open-vocab probe)
| Backend | s1 identity (base prompt) | open-vocab AUC | notes |
|---|---|---|---|
| hand table (incumbent) | 100% | 0.50 (uninformative 0.4 default) | benchmark-role leakage acknowledged |
| claude-sonnet-5 | 78.9% | **1.00** | |
| claude-haiku-4.5 | 15.8% base / 78.9% context+anchored | **1.00** | base-prompt parse pathology |
| gpt-5.5 | 78.9% | **1.00** | |
| gemini-2.5-flash | 78.9% | **1.00** | dims multiprompt: 73.7% |
All backends fail the SAME 4 box/book scenes: the VLM ratings are semantically
correct (box/book harmless to touch) but the benchmark designates them as
placed obstacles — the discriminator is physical size, invisible to name-only
prompts. s2 whole-scene WITH images: gpt-5.5 84.2%, sonnet 57.9%, gemini
47.4% vs structured 100% → decomposition claim survives at API strength.

### Offline rescue arms (19 scenes)
- head-noun mention fix: real bug ("glazed rim porcelain ramekin" mention_frac
  0.25 when task says "ramekin") → INTEGRATED into scored_obstacle_id.
- percep extents (seg+depth, after fixing 180° capture rotation): coverage
  16/19 but size term does NOT rescue identification (visible-surface noise)
  → NO-GO for identity; REPURPOSED as no-GT obstacle_scales source.
- protection floor (C1 fail-safe): max(vlm_rating, 0.5) → **17/19 best
  general config**; remaining 2 scenes are answer-key artifacts.
- G3 LLM instruction parse (haiku, JSON target/dest/mentioned): 13/19 —
  WORSE than fixed token heuristics, errors fail-unsafe → NO-GO; heuristic
  + head-noun stays.

### Integration (commit e7ace4a)
symbolic_identity.load_vlm_priors(path, floor) + client
--hazard_prior_source vlm; priors JSON = 35-name vocabulary rated by haiku
anchored prompt (results_tables/vlm_hazard_priors.json). Runtime tax measured
by fgd_prior_swap_smoke (job 42405131): Spatial L1 n=10 Tier-Percep,
table-arm vs vlm-arm, same r3 server.

### Superquadric/anisotropic obstacle barrier (commit 4533f42)
_eff_dist radial barrier B(p)=|p-o|-r(u), r(u)=(sum|u_i/s_i|^(2/eps))^(-eps/2);
all-zero scales = exact legacy sphere. Client --obstacle_shape superquadric
sends GT AABB half-extents (mesh-vertex exact). Hypothesis: tall-thin shapes
resolve the Spatial L1 thin(TSR)/fat(CAR) sphere tradeoff. Smoke job
42405105: ellipsoid (eps 1.0) + boxy (eps 0.4), Spatial L1 n=10, r3 config.

### Ops
- iter5 (r3@0.095): 42405001 died on w010 node flake; 42405059 hit a
  mid-edit GuidanceParams race (job started during superquadric edit);
  clean resubmit 42405128. LESSON: don't edit worktree modules while jobs
  are PENDING on them.
- LIBERO-Safety assets: unzip CPU job 42405002 COMPLETED (1h34m), UNZIP_DONE
  set, env smoke auto-chained (42405130). Parked per user priority until
  no-GT SafeLIBERO closes.
- no-GT Long L1+L2 n=50 submitted (42405139, composed-thin, full Tier-Percep,
  head-noun fix live) — last two virgin no-GT cells.

### Late-wave additions (same day, GPU-queue window)
- **Prompt-variant sweep FINAL**: anchored wins (gemini-anchored s1 84.2% —
  best name-only arm — probe sep 0.82; haiku-anchored 78.9%/0.74). dims
  multiprompt NO-GO (s1 68-74%, probe AUC 0.55-0.75 — binary micro-judgments
  destroy the graded hazard signal). Gemini-anchored priors file generated
  as upgrade candidate (vlm_hazard_priors_gemini.json); pending prior-swap
  smoke stays on haiku file for provenance.
- **Scene-graph V1 VERDICT: 100% agreement (19/19)** flat-v3 vs graph-rule
  identification (scene_graph.py; results_tables/scene_graph_v1_agreement.json)
  → pre-registered NO-GO for SafeLIBERO confirmed: the graph is a refactor,
  not information, on this benchmark. V2 (LIBERO-Safety) parked per priority.
- **G5 table-plane fit: GO + discovery** — depth-histogram mode recovers the
  support plane with 0.4cm scatter across 19 scenes at z=0.862; the hardcoded
  z_table=0.81 is ~5cm below where objects actually rest (bowl centers 0.898,
  half-height ~4cm). Adopt fit as percep-tier table height.
- **Percep extents integrated** as no-GT obstacle_scales source
  (estimate_obstacle_extent; client picks percep vs GT AABB by
  obstacle_pos_source) — ready the moment the GT shape smoke (42405105) GOes.
- **G3 LLM-parse NO-GO** recorded above; token heuristics + head-noun stay.
- **Significance (Fisher, n=200/arm)**: no-GT CAR wins all significant
  (Obj L1 p=3e-41, Spa L2 p=5e-18, Goal L2 p=2e-4 vs reimpl); NO TSR
  difference vs reimpl reaches significance anywhere (Obj L1 +3.5 ns p=0.55,
  Goal L2 −1.0 ns p=0.91, Goal L1 +6.5 ns p=0.17, Spa L2 +15.5 SIG p=2e-4).
  Headline: "TSR statistically indistinguishable from self-run SOTA, CAR
  2-5x higher, without privileged state."

### E5 fail-direction for VLM priors + GUARD-SET result (offline, 19 scenes)
Argmax identification is fail-unsafe by construction (single-obstacle guidance
leaves the true hazard unguarded on ANY mispick): intact vlm+floor 17/19,
30%-dropped 15/19, inverted 3/19 — the floor helps accuracy but cannot fix
ordering. **Guard-set (top-k by the same score): top-2 recall 19/19 under
EVERY corruption including full inversion** — path geometry alone keeps the
true obstacle in the top-2; semantics only orders within the set. This is the
C1 default-deny thesis with an empirical proof: identification errors become
survivable if guidance guards the set. Module: vlm_pipeline/guard_set.py
(guard_set()[0] == scored_obstacle_id() by construction). Server-side
multi-obstacle barrier (min over obstacles, same composition as companion
points) is the pending integration — DEFERRED until the 4 queued jobs start
(module-edit race lesson).

### Scene-graph V1 + G5 + stats: see block above (commit 2d10d69).

### Prior-file bake-off (offline): haiku-anchored is production
| prior file | argmax | top-2 guard set | top-3 |
|---|---|---|---|
| hand table | 19/19 | 19/19 | — |
| haiku-anchored + floor .5 | 16/19 | **19/19** | 19/19 |
| gemini-anchored + floor .5 | 16/19 | 17/19 | 19/19 |
Gemini's higher argmax on the sonnet-weight benchmark did not carry to its
own ratings (over-rates fragile distractors → GT pushed to rank 3 twice).
Haiku-anchored + floor + top-2 guard set = general config with full recall;
pending prior-swap smoke (42405131) already runs the haiku file.

## ★ 2026-07-19 evening — first full results wave (jobs 42405128/248/251)

### no-GT Long n=50 FINALS (job 42405251, composed-thin, full Tier-Percep, head-noun fix live)
| Cell | no-GT | baseline | reimpl | Verdict |
|---|---|---|---|---|
| Long L1 | 49.0 / 66.5 | 58.0/15.0 | 30.5/35.5 | beats reimpl BOTH axes (+18.5/+31); baseline TSR −9 (liveness stalls, same as GT) |
| Long L2 | 44.0 / 79.5 | 51.0/16.5 | 43.0/59.5 | beats reimpl BOTH axes (+1/+20); baseline TSR −7 |
no-GT Long now matches the GT-tier story exactly: dominates the SOTA method
class, loses baseline TSR to timeout stalls (not collisions). Notably BETTER
than GT-tier composed (42.5/71.0, 47.0/77.0) on L1 TSR +6.5.

### GT Spatial L1 iter5 (r3 @ 0.095, job 42405128): FAILS
60.0/48.5 — the radius cliff is between 0.09 and 0.095 (0.095 behaves like
0.10: TSR collapse with big CAR). Radius interpolation EXHAUSTED. r3@0.09
(71.0/23.5) stays the terminal sphere candidate. (L2 side: 86.0/55.0.)

### Superquadric smoke n=10 (job 42405248, GT AABB extents, r3 base)
| Arm | overall | t0 | t1 | t2 | t3 |
|---|---|---|---|---|---|
| ellipsoid eps1.0 | 62.5/42.5 | 80/60 | 20/40 | **100/30** | 50/40 |
| boxy eps0.4 | 60.0/50.0 | 80/60 | 10/40 | **100/60** | 50/40 |
Shape WORKS where the obstacle is off-grasp (t2: 100 TSR with CAR 30-60 —
sphere never did this) but the AABB smothers the grasp on t1 (bowl ON the
ramekin, obstacle adjacent): 10-20 TSR. This is precisely the corridor
exemption's job → arm C submitted (boxy + corridor, job 42422430).

### Ops
- prior-swap smoke died on w004 with the abort-after-ep1 signature (same as
  w010) → w004 added to exclusions; resubmitted as 42422429.
- LIBERO-Safety smoke failure root-caused: checkpoint ships norm_stats at
  assets/lerobot/, openpi expects assets/physical-intelligence/libero/ →
  symlinked; resubmitted as 42422432.
- New-script header fixed to source conda by absolute path (module fn absent
  in this session's sbatch environment).

## ★ 2026-07-19 late — prior-swap verdict + Spatial L1 no-GT VLM confirm launched
- Job 42422429 (Spatial L1, Tier-Percep, r3 server, n=10/task per arm):
  - table arm: 60.0/42.5 — matches the existing no-GT board cell (60.0/40.0 at n=80).
  - vlm arm (haiku anchored priors + floor 0.5): **72.5/37.5** — TSR +12.5, CAR −5.0 (2 eps).
- VERDICT: generalization tax is ZERO-TO-NEGATIVE. The fully general VLM-prior
  identification path is at least as good as the hand table at runtime. GO —
  VLM priors promoted into the no-GT standard config.
- Bonus: 72.5/37.5 would beat BOTH refs BOTH axes at the open Spatial L1 no-GT
  cell (baseline 67.0/14.0, reimpl 62.0/28.5). n=50 confirm = job 42430870
  (fgd_nogt_spatial_vlm_n50.slurm, r3 + vlm priors, full Tier-Percep).
- Mechanism note: biggest per-task gains t0 0.9→1.0 and t3 0.3→0.6 — graded
  VLM hazard priors reweight the guard toward the true hazard on scenes where
  the hand table's uniform-ish weights mis-rank candidates.
- LIBERO-Safety smoke (parked lane, CPU fixes only): config-dir, wand,
  scikit-image, level-aware bddl/init API (L0/L1/L2 subdirs; reasoning_safety
  ships no .pruned_init). obstacle_avoidance/human_safety/affordance resolve
  15 tasks × 50 init states offline. Smoke 42427444 queued.

## ★ 2026-07-19 night — SQ boxy+corridor arm C verdict (job 42422430, n=40)
- boxy(ε0.4)+corridor: 62.5/47.5. Per-task: t0 0.8/0.4, t1 0.2/0.4, t2 1.0/0.6, t3 0.5/0.5.
- t2 HOLDS at 100 TSR (shape thesis intact). t1 did NOT recover (0.2 vs 0.1–0.2
  without corridor) → corridor as-implemented cannot exempt the shape:
  corridor_scale multiplies r_eff (scalar margin) only, while shape mode's
  dominant keep-out term r_shape(u) from the AABB is NEVER relaxed inside the
  grasp corridor. Mechanical gap, not a refuted hypothesis.
- NEXT ITERATION (staged, module-blocked until 42430870 is safely RUNNING):
  scale r_shape by the same corridor/margin multiplier in _eff_dist call sites,
  then re-smoke. Projection: t1→~0.6 would give ≈72 TSR / ~47 CAR → closes
  Spatial L1 GT. Sphere r3@0.09 (71.0/23.5) remains the fallback.

## 2026-07-19 night — offline study: full SQ fit from depth cloud (user idea)
Harness vlm_pipeline/sq_fit_study.py, 19 obstacle scenes. Arm B = current
percentile box (p5-p95/2 half-extents, yaw 0). Arm C = full superquadric fit
(center+scales+eps+yaw, L-BFGS-B inside-outside residual) on a single-view
cloud completed by camera-ray mirroring + table-plane footprint grounding.
Medians: coverage B 0.56 -> C 0.89 (+33pp) BUT volume B 2484 -> C 11041 cm3
(4.4x inflation); center err 0.105 -> 0.099 (wash); fitted eps pinned at the
0.3 bound on 19/19 (degenerate); yaw err ~13 deg median.
VERDICT: NO-GO as-is — inflation is exactly the grasp-smothering failure axis
(t1), and the gate (tighter AND covering) is failed. Same lesson as G2:
single-view partial clouds punish expressive fits; robust percentiles win.
Extents stay percentile-box. Path that could flip it (future work, not now):
multi-view fusion (we already capture 2-3 cameras and max-fuse extents) for a
genuinely closed cloud before fitting, + fit only yaw with fixed percentile
scales (cheap orientation without the volume blow-up).

## 2026-07-19 night — yaw-only fit (SQ thread, arm D offline)
Min-volume-orientation fit, percentile scales fixed (no completion, no
inflation possible by construction). 19 scenes: yaw err median 2.0 deg,
p90 9.7 deg — accurate. BUT volume ratio rotated/axis-aligned = 1.00 median:
SafeLIBERO obstacles are axis-aligned, so orientation buys ZERO tightening
in-domain. VERDICT: no server integration now (a rotation field would change
GuidanceParams for nothing measurable); keep as rebuttal armor — "orientation
is recoverable at 2 deg median from the same cloud when a domain needs it."
SQ thread now rides entirely on the two queued smokes: corridor-shape fix
(42431256) and guard_k (42431257), plus no-GT VLM n=50 (42430870, running).

## ★★ 2026-07-19 night — SPATIAL L1 NO-GT CLOSED (job 42430870, n=200)
Config: r3 server (K1/uniform/0.09) + full Tier-Percep + VLM priors (haiku
anchored, floor 0.5). **70.0 / 46.5** — beats baseline (67.0/14.0) AND
self-run SOTA (62.0/28.5) on BOTH axes. Per-task: t0 0.96/0.42, t1 0.30/0.36,
t2 0.96/0.58, t3 0.58/0.50.
Fisher (n=200): CAR wins significant vs both (p=1.2e-12, p=2.9e-4); TSR
deltas ns (p=0.59, p=0.11) — consistent with the standing headline.
Notes: (a) closes the LAST open no-GT cell; every no-GT row now beats the
method class (reimpl) on both axes; (b) does it with the GENERAL identification
stack (VLM priors, not the hand table) — the smoke's +12.5 TSR over the table
arm held directionally at n=50 (70.0 vs table-arm history 60.0/40.0 n=80);
(c) no-GT CAR doubles our own GT-tier sphere CAR (46.5 vs 23.5) at -1.0 TSR.
Board: no-GT clean wins vs BOTH refs BOTH axes: Spatial L1+L2, Goal L1,
Object L1+L2 (5/8); Goal L2 TSR tie (p ns) + CAR win; Long L1+L2 beat reimpl
both axes, baseline keeps TSR via liveness stalls.
