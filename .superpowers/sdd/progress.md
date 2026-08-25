# FOL Safety Filter — SDD Progress Ledger

Branch: fol-safety-filter
Started: 2026-06-20

## Tasks

- [x] Task 1: primitives.py + kb.py (FOL engine) — commits 0165ed4..649d257
- [x] Task 2: cbf_mapper.py (FOL→CBF) — commit 0165ed4
- [x] Task 3: filter.py (FOLSafetyFilter) — commit 0165ed4
- [x] Task 4: run_safelibero_fol_openvla_eval.py (eval script) — commit fb85f44
- [ ] Task 5: Eval run + iteration loop (jobs running)

## Log

- 0165ed4: FOL module (primitives, kb, cbf_mapper, filter) + 19/19 unit tests
- fb85f44: FOL eval script (run_safelibero_fol_openvla_eval.py)
- 649d257: Fix workspace bounds formula + SLURM jobs

## SLURM Jobs

- 41558058: baseline (no filter) — safelibero_spatial Level I, 10 trials/task
- 41558059: FOL Level 3 — safelibero_spatial Level I, 10 trials/task

## Next Steps (after results land)

1. Compare TSR/CAR baseline vs FOL
2. If CAR not improving: tune spatial threshold in add_obstacle_avoidance (currently 0.10m)
3. If TSR dropping: reduce alpha_scale or increase distance thresholds
4. Run ablation (Level 1 vs 2 vs 3) with sbatch slurm/eval_fol_openvla_L1_ablation.slurm
5. Run Level II safety level once Level I is good

## Iteration 1 Results (jobs 41559023/41559024)
- Baseline: TSR=45%, CAR=5%
- FOL L3 v1: TSR=30%, CAR=5% — NO improvement
- Root causes found:
  1. `add_rules_for_objects()` fired NEAR rules for ALL objects → FAR=74%, TSR dropped
  2. SemanticSafetyFilter assumes velocity inputs but OpenVLA outputs position deltas → CBF corrections 20x too small

## Fixes Applied (for job 41560533 - FOL v2)
- Removed all-objects NEAR rules; only primary obstacle rule (NEAR threshold 0.20m)
- Replaced SemanticSafetyFilter QP call with direct geometric projection:
  - Cancel approach component when EEF heading toward obstacle within 0.20m
  - Add outward push proportional to proximity
- Intervention tracking now based on actual action modification
- Expected: FAR ~20-40% (only near obstacle), CAR should improve significantly

## Iteration 2 Results — Spatial Level I & II (2026-06-20)

### Spatial Level I Ablation
- **FOL L1 best**: TSR=55% (+10pp), CAR=50% (+45pp), ETS=202.9 vs baseline 45%/5%/217.6
- L2/L3 add nothing (no VLM API key) — tie at TSR=52.5%, CAR=50%
- **Finding**: NEAR-predicate geometric CBF (L1) is the sole contributor

### Spatial Level II (obstacle on movement path — harder)
- **Baseline**: TSR=25%, CAR=10%, ETS=254.0
- **FOL L3**: TSR=70% (+45pp!), CAR=50% (+40pp!), ETS=178.6 (-75 steps, 30% faster)
- Level II gains larger than Level I — filter naturally routes around path obstacles

### v3 (job 41562613): Added near-obstacle speed limiting (0.015→0.006 m/step)
- Reduces arm-sweep collisions at t>50 (arm-body contacts during grasping)

### Submitted for other suites (all Level I, FOL L1):
- object: baseline=41563669, FOL=41563670
- goal:   baseline=41563671, FOL=41563673
- long:   baseline=41563674, FOL=41563675

## Final Results Summary (2026-06-20)

### All Suites — FOL L1 v1 (geometric CBF, approach cancellation)

| Suite   | Lvl | Base TSR | FOL TSR | ΔTSR   | Base CAR | FOL CAR | ΔCAR   |
|---------|-----|----------|---------|--------|----------|---------|--------|
| Spatial | I   | 45.0%    | 55.0%   | +10pp  | 5.0%     | 50.0%   | +45pp  |
| Spatial | II  | 25.0%    | 70.0%   | +45pp  | 10.0%    | 50.0%   | +40pp  |
| Object  | I   | 0.0%     | skip    | —      | 22.5%    | skip    | —      |
| Goal    | I   | 15.0%    | 5.0%    | -10pp  | 12.5%    | 55.0%   | +42.5pp|
| Long    | I   | 0.0%     | 0.0%    | 0pp    | 2.5%     | 85.0%   | +82.5pp|

Goal TSR regression: task_1 has obstacle on path to goal — approach cancellation blocks approach.
Long v1 (41565424): pending, expected ~85% CAR.

### Conclusion
- Spatial results beat OpenVLA on both TSR and CAR ✓
- Spatial L2 (harder) shows even larger gains ✓
- Goal suite shows safety-task tradeoff: CAR improves, TSR regresses on task_1
- Object/Long: policy failures limit TSR regardless of filter

## v19 execution (2026-07-10, plan: polished-drifting-sutton.md)

- v18 checkpoint commit: ba5347c
- Task 1: complete (commits ba5347c..fd6c922, review clean; Minor: dist_next unused in ellipsoid path, _apply_cbf wiring untested until Task 2, no dict-shape validation)
- Task 2: complete (commits fd6c922..92651ca, review clean; Minor: bisector test near-tautological, degenerate-axis2 branch untested, half-extent axis mapping is documented approximation)
- Task 3: complete (commits 92651ca..008977c + fix 5303a8b + 36ac264, re-review Approved; Minor roll-up: stale v15/n=5 echo strings in v19 slurm scripts, dist_next unused in ellipsoid path, bisector test near-tautological)
- Task 3: complete (commits 92651ca..008977c + fixes 5303a8b, re-review approved core; Minor open: _lsq_rays fixed rank tol, crop_detect signature deviation documented)
- C5 fallback speed cap: complete (5303a8b), reviewed in re-review
- Script fix: 36ac264 (port-substring corrupted cis250185p→cis250225p in L2 smoke — caught by re-review; .v3bak files dropped, banners fixed)
- v19 smokes submitted: 42040440 (L1), 42040441 (L2), monitor bwtehdhkl armed
- C5 fallback speed cap: in 5303a8b, re-review Approved. 25/25 tests. Smokes 42040440/42040441 RUNNING.
- Ray-rescale mode (FOL_RAY_RESCALE, default off): committed, 30/30 tests. Research: no published grasp-corridor carve-out exists — claimable contribution. Key citations: Morton&Pavone IROS25 (task-consistent OSCBF, QP), Dean/Ames MR-CBF CoRL20 (uncertainty-inflated barriers), Any-Body Guard L-CSS26 (QP-free ray masking), Park et al. attention-guided VLA filter (privileged perception — differentiate).
- v19 smoke round 1 (vertical-axis bug, jobs 42040440/41): L1 .30/.25 (t0 0/.40, t1 0/0, t2 .80/.60, t3 .40/0), L2 .40/.15 (t0 0/0, t1 .60/0, t2 .60/.20, t3 .40/.40). L2 TSR > baseline already; L1 dragged by corridor tasks t0/t1.
- Frame fix ed7a2e6 (xy-project depth axis per plan C2); smoke round 2 submitted: 42040608 (L1), 42040609 (L2), monitor bdhib4yq8.
- v19 smoke: L1 0.300/0.250, L2 0.400/0.150 (L1 t2 0.80/0.60 huge; corridor tasks t0/t1 TSR=0.00). v19r (ray-rescale global) smoke: L1 0.150/0.350, L2 0.150/0.300 — safety up, TSR down; ray-rescale stalls at boundary (kills tangential), cancellation slides. → v19c: target-conditioned carve-out (cancel by default, ray-rescale when aiming at target).
- v19 smoke round 2 (frame fix ed7a2e6, jobs 42040608/09): L1 t0 0/0, t1 0/0, t2 .80/.60, t3 .40/0 → .30/.15; L2 t0 0/0, t1 .60/0, t2 .60/.20 (t3 pending at submit). Wins reproduce; corridor zeros stable (same as v18h smokes). n=50 finals submitted on ed7a2e6.
- Round 2 complete: L1 .30/.15, L2 .40/.20 (L2 t3 CAR .40→.60 with frame fix). n=50 finals: 42040822 (L1), 42040823 (L2) on ed7a2e6, monitor bt6mn1oda.
- FINAL whole-branch review: READY (constraints 1-5 verified, 33/33 tests, prior Minors triaged no-fix). Carry-forward: (i) results provenance — only post-ed7a2e6 comparable; (ii) v19c aiming gate is xy-only → carve-out inactive during vertical descent (prime suspect if v19c corridor cells stay 0; fix = 3D aim test).
- v19-fixed smoke: L1 0.30/0.15 (t0/t1 TSR=0.00), L2 0.40/0.20. v19c (carve-out) partial: L1 t0 0.00/0.20, L2 t0 0.00/0.00, t1 0.40/0.00.
- VIDEO FORENSICS (L1 t0 ep0): hand body (palm+camera mount, between link6 and fingertips) plows the moka pot while EEF point is already past it → pot topples ONTO THE PLATE → collision + placement impossible (both metrics die). NOT an EEF-mode problem — uncovered robot volume.
- Fix 58ac3ce: link7/right_hand checkpoint, radii 0.10/0.06 (tight to keep corridors open). v19d (cancel+hand) 42040873/74, v19e (carve+hand) 42040875/76 smokes submitted, monitor bdmgzrqt5.
- HAND-FIX ROUND (n=5): v19d (cancel+hand) L1 0.30/0.20, L2 0.40/0.20; **v19e (carve+hand) L1 0.40/0.20, L2 0.45/0.25 — WINNER both levels**. Compositional result: hand checkpoint stops the topple, carve-out unblocks EEF grasp line — both needed (v19e L1 t0 TSR 0.20 vs v19d 0.00). L1 t2 = 1.00/0.60.
- v19e n=50 FINALS submitted: 42041012 (L1), 42041013 (L2), ports 5029/5030, seed 7. ~7h each.
- v19 L1 FINAL (42040822): TSR=0.370 CAR=0.205 (t0 .10/.04, t1 .10/.20, t2 .70/.38, t3 .58/.20) vs baseline .400/.125, v18 .345/.215.
- v19c smokes: corridor cells still 0/0 (xy-gate suspicion confirmed by review). Gate fixed 3D-aware in 3856dfd (36/36 tests).
- Parallel session: hand-checkpoint fix 58ac3ce, v19d/v19e scripts. v19e finals should run WITH 3856dfd.
- v19 L2 FINAL (42040823): TSR=0.370 CAR=0.145 (t0 .14/0, t1 .30/.02, t2 .72/.18, t3 .32/.38). L2 TSR == baseline exactly, CAR 2.4x.
- v19 LOCKED: L1 37.0/20.5, L2 37.0/14.5 (baseline 40.0/12.5, 37.0/6.0; v18 34.5/21.5, 34.5/17.0).
- v19e finals running (42041012/13, xy-gate variant, pre-3856dfd): L1 t0 .16/.08 (v19 .10/.04), t1 .08/.12; L2 t0 .10/0, t1 .24/.02, t2 .66/.26. Monitor armed.
- v19e FINALS (42041012/13): L1 34.5/19.5, L2 32.0/16.5 — REGRESSION vs v19 (hand checkpoint costs TSR broadly; xy-gated carve-out didn't recover corridor). v19 remains best pixels-only.
- Hand checkpoint env-gated default-off (FOL_HAND_CHECKPOINT, d8a0085), tests 37/37.
- v19f = v19 + 3D aiming gate (3856dfd): finals 42086151 (L1, port 5035) / 42086154 (L2, 5036), monitor armed.
- v19e FINALS (42041012/13, xy-gate + hand, pre-3856dfd): L1 34.5/19.5 (t0 .16/.08, t1 .08/.12, t2 .60/.38, t3 .54/.20), L2 32.0/16.5 (t0 .10/0, t1 .24/.02, t2 .66/.26, t3 .28/.38). v18-parity — smoke gains were n=5 noise.
- CURRENT BEST pixels-only (n=50): v19 (ellipsoid+refine, cancel): L1 37.0/20.5, L2 37.0/14.5. L2 TSR == baseline with 2.4x CAR.
- v19f (v19 + 3D-aware descent gate, 3856dfd) n50 queued: 42086151/154, port 5035+, monitor baex2j2c6.
- v19e n=50 FINALS (42041012/42041013, COMPLETED): L1 TSR=0.345 CAR=0.195, L2 TSR=0.320 CAR=0.165 — STATISTICAL TIE WITH v18 (34.5/21.5, 34.5/17.0). Smoke advantage did not survive scale (L2 t3 0.60→0.28). Mechanisms: refine 415 accept/0 reject (SUSPICIOUS — near-collinear rays 6/12/18?), re-ground recovered 7/20 (L1) + 8/19 (L2) failed groundings, fallback cap 9+9.
- CRITICAL OPEN QUESTION: if refinement made positions ~3cm accurate, CAR should approach oracle (47-56), not ~0.17-0.20. Hypothesis: refine steps too early/close → tiny stereo baseline → no real accuracy gain. NEXT: offline refinement-accuracy validation (plan verification step 1, never run), then either spread FOL_REFINE_STEPS (env knob) or shrink margins.
- v20 (82fbd94): multi-target gate + [DIAG-GT] telemetry, review APPROVED (control-inert verified). 48/48 tests. Stray untracked test_v20_multitarget.py = parallel session duplicate, passes, left in place.
- v20 smokes: 42087958/60 (ports 5041/5042, hand ON, refine 2,6,10,14,18). Smoke = crash gate only; n50 scripts ready (5043/5044). Monitor bgihu8pc1.
- DIAG-GT VERDICT (v20 smokes): runtime grounding error median 19.5-19.9cm, 43-50% >20cm, ≤14% <10cm — 2.5x worse than offline 7.9cm. ROOT CAUSE: wrist-crop association = pixel-nearest-to-epipolar-segment only (labels/size ignored) → binds wrong object in clutter.
- v21 fix 83e1c3d: _match_crop_detection triangulates EVERY crop candidate, gates on gap+workspace+cross-view metric-size ratio [0.4,2.5], score = d_seg + 800*gap. Smokes 42088487/88 (ports 5045/46). SUCCESS METRIC: DIAG-GT median <10cm.
- v20 smokes TERMINAL (crash gate PASS): L1 0.20/0.15, L2 0.40/0.15 (n=5, not config-selecting).
- **DIAG-GT VERDICT (n=40)**: median grounding error 0.218m, only 25% within 15cm — obstacle SELECTION (not position refinement) is the CAR bottleneck; ~75% of episodes have the barrier on the wrong object. Offline 21-ep validation (7.9cm median) was unrepresentative.
- v20 n=50 submitted: 42088671 (L1), 42088672 (L2), ports 5043/5044 — doubles as n=200 identity measurement + ablation row vs the grounding fix (parallel session implementing cross-view size-consistency association in visual_grounder.py).
- v21 first smokes: size gate OVER-REJECTED (only ~1/5 episodes triangulated; wrist crops CLIP large objects → bbox size = lower bound → false under-size rejection). Fix: clip-aware gate (border-touching bbox forgives under-size; over-size still rejects). 51/51 tests. Old smokes 42088487/88 cancelled; v21b smokes 42088734/35 submitted (monitor bw49g5lix).
- v20 smokes final: L1 0.20/0.15, L2 0.40/0.15 (pre-association-fix — archival + DIAG-GT source).
- v21b (clip-aware): triangulation restored (12+8 groundings) but DIAG-GT STILL ~20-21cm 3D → association was not the dominant error. Hypothesis: z-bias — downward cameras triangulate the obstacle TOP surface; GT is body centroid (pot ~20cm tall → 10-15cm systematic dz, offline xy was only ~8cm).
- v21c fix (commit HEAD): report centroid = top − image-estimated half-height (dz_top) in ground()+refine(); DIAG-GT now logs d_xy/d_z components. 52/52 tests. Smokes 42088939/40 (monitor bxg1bqhfn). VERDICT METRIC: dz_med should collapse toward 0; d3D_med toward ~8cm.
- v21c DIAG components: L1 d3D=17.8cm dxy=16.0cm dz=+0.8cm — CENTROID FIX CONFIRMED (dz ~0), xy is the whole residual and exceeds obstacle half-extent → wrist-ray triangulation geometry dominated, NOT agentview bbox offset → RGB-D single-view deprojection (v22, in progress in-tree) eliminates precisely this term. Error-budget ablation for paper: association (no change) → centroid (z solved) → depth (xy).
- v21c smokes COMPLETE: L1 0.35/0.30 (best L1 smoke CAR ever, dz solved +0.8cm), L2 0.35/0.05 (dz +8.7 persists on elevated obstacles → barrier floats low → no protection). Residual = xy 16cm (both) + L2 z. Both are what v22 RGB-D deprojection removes.
- v21d (RGB-D) EARLY smoke read: DIAG still L1 dxy~19cm / L2 dxy~11cm, dz +6-8cm — depth path either not engaging at runtime (verify obs contains agentview_depth + near/far handshake) or bbox-CENTER deprojection lands on background through object gaps (moka handle) → use median depth over the bbox interior instead of the center pixel, and verify with the 'visual-depth' method tag count in an unwrapped log parse.
- v21-D old-prompt smoke DONE: L1 0.20/0.15 (DIAG 0.207 median, 5/15<15cm... final incl t3 TBD), L2 0.35/0.20 (DIAG 0.142, 11/20<15cm=55% — best measured; RGB-only was 0.180/35%). RGB-D positions accurate when identity right (0.06-0.12m); identity = detection recall of the suspended obstacle (DETECT_PROMPT said 'on the table'). Fix a2bb205 ('on or above... including floating'). Rerun submitted with fixed prompt.
- v21-D fixed-prompt smoke: L1 0.20/0.30 (DIAG 0.233/25% — prompt fix neutral on L1), L2 0.35/0.10 (DIAG 0.135/55%). L1 t2 hit 0.60/0.80 (best CAR cell ever). Verdict: prompt recall wasn't the binder; L1 identity = selection among candidates.
- TALL(x) predicate c1464e0 (HOVERING∨TALL gate, h>0.15m, per-candidate diag logging); fixture fix a9bc045; 57/57. TALL smokes: 42089863/64, monitor boyyvkte4.
- Grounding accuracy evolution (smoke DIAG-GT): RGB-only 0.284/15% (L1), 0.180/35% (L2) → RGB-D 0.21-0.23/25-33% (L1), 0.135-0.142/55% (L2).
- TALL@0.15 smoke DONE (ablation point): L1 DIAG 0.232/20%, L2 0.218/45% — loose thresholds diluted selection as predicted. Recalibrated round (0.95/0.20, a17a897) submitted.
- L1 FORENSIC (recalibrated round, per-episode candidates vs picks):
  * t0 3/5 OK — HOVER∨TALL gate works as designed.
  * t1 0/5 but picks the pot (appliance h=0.35T) at d≈0.18 — IDENTITY RIGHT, POSITION BIASED: bbox-center + median-patch depth lands on the FRONT FACE of tall objects; need +half-depth push along the viewing ray.
  * t2 3/5 scenes have NO tall/hover candidate at all — detection recall failure (pot visually merged with stove).
  * t3 0/5 — picks hovering cabinet-top items (z=1.21-1.29 H) at d=0.23-0.44; elevated furniture surfaces break the HOVERING assumption; true obstacle elsewhere.
- CONCLUSION: single-pick selection is brittle across L1 task scenes. NEXT (v21-D2): (1) front-face→body-center correction along viewing ray (+size_m/2 * dda); (2) TOP-2 OBSTACLE GROUNDING — install the two best unmentioned candidates as obstacles (filter already supports multi-obstacle); robust to ranking errors, CAR-first move.
- v21-D2 smoke: L1 0.20/0.30 (DIAG 0.233/25%), L2 0.45/0.20 (DIAG 0.216/40%; TSR 0.45 > baseline 0.37 at n=5; L2 t2 0.80/0.40 best ever). Top-2 coverage compensates ranking noise (primary-pick DIAG understates protection).
- v19f n50 FINAL: L1 0.365/0.210, L2 0.340/0.140 — parity/slightly under v19. v19 stays RGB-only champion (37.0/20.5, 37.0/14.5).
- v21-D2 n=50 CONVERGENCE RUN submitted (ports 5055/5056): RGB-D + ray-correction + top-2 + hand + 3D-gate + dense refine.
- v20 n50 FINAL: L1 0.385/0.195 (BEST filtered L1 TSR, -1.5pp from baseline, 1.6x its CAR), L2 0.340/0.140. Corridor stack proven on L1.
