# Subagent-Driven Development Progress Ledger

## Context (pre-existing work, done before this ledger started)
- Simulator decision: Genesis (bake-off complete, see /jet/home/asingal/.claude/plans/plan-how-to-create-cuddly-mist.md)
- Reusable infra built and verified:
  - `genesis_bakeoff/placement_sampler.py` — declarative region-based placement + rejection sampling
  - `genesis_bakeoff/scenario_builder.py` — generate_init_states()/save_init_states()/load_init_states(), verified 50/50 valid on scenario 1a-i
  - `genesis_bakeoff/libero_asset_loader.py` — loads LIBERO's 92-object MJCF library into Genesis (patches missing free joint), verified with bowl+mug
- Scenario 1a-i (mug behind bottle): built and verified on both MuJoCo and Genesis (bake-off pilot)
- Scenario 3a-i (pour rice), 3a-ii (Skittles, hardened — Task 4, DONE), 2c (glass tilt): bake-off pilots exist; 3a-ii production-hardened, 3a-i/2c still Tier 4, not started under this ledger

## Task list (Tier 1 of the build order — LIBERO-asset-only scenarios)
Design brief from Fable consult, 2026-07-10, recorded in task briefs below.

- Task 1: Scenario 1b (object at table edge / kinematic limits) — DONE
- Task 2: Scenario 3b (big cube on small cube, stacking stability) — DONE
- Task 3: Scenario 4a (clear table respecting support order) — DONE (commit 70a0e88)
- Task 4: Scenario 3a-ii (open bag of Skittles, granular scatter, hardened to production) — DONE (commit 7225aaf)
- (parallel, separate line item) Camera-rig standardization + realistic-asset swap for 1a-i/2c — DONE (commit f43ec78, brief: task-standardize-rig-brief.md, report: task-standardize-rig-report.md)
- Task 5: Fix asset-realism issues in 1a-i/2c (bottle-vs-mug scale illusion, laptop orientation, glass primitive) — DONE (commit c088173, brief: task-5-brief.md, report: task-5-report.md)
- Task 8: Scenario 2a-i (wet sponge over stack of paper, material incompatibility) — DONE (commit 6815e6f, brief: task-8-brief.md, report: task-8-report.md; ran concurrently alongside Task 6 [scenario 2b, egg] and Task 7 [scenario 2d, knife] on separate files)
- Task 7: Scenario 2d (knife handover, blade/handle grasp + delivery orientation) — DONE (commit 492b5dc, brief: task-7-brief.md, report: task-7-report.md; ran concurrently alongside Task 6 [scenario 2b, egg] and Task 8 [scenario 2a-i, sponge] on separate files)
- Task 9: Scenario 4b (goal object-specific constraints, stir-soup utensil selection) — DONE (commit 0d1c4f7, brief: task-9-brief.md, report: task-9-report.md)
- Task 10: Scenario 1a-iii (multi-obstacle corridor, paper between two lit candles) — DONE (commit 89f711d, brief: task-10-brief.md, report: task-10-report.md)

## Completed tasks
(none yet under this ledger)

## NOTE (2026-07-10): ledger relocated
`.superpowers/sdd/` root is shared with at least one other concurrent session and got clobbered mid-Task-1. This directory (`.superpowers/sdd/genesis-scenarios/`) is now the canonical location for this work's ledger, briefs, and reports going forward.

Task 1: complete (commit 9ad510e, review clean — Approved, minor nits only: missing commit-hash line in report, stale comment in metrics_1b.py:132, d0 precision quirk scenario_1b.py:377)

Task 2: complete (commit d4f6ef681f11173cff3a3467acaf94aaf318e95b). SLURM job 42135800 ran clean, 10 episodes, both metrics show real variation. Investigated the brief's flagged concern (max_tip_angle_deg uniformly low, 0.46-22.7deg, despite high com_overhang_fraction, 0.82-0.998): confirmed via height cross-checks (box always settles at/below table height, never at the "resting on butter" height) and rendered frames (butter visibly ejected from under the box post-settle) that this is legitimate physics, not a placement bug — the box genuinely falls/slides off the small butter pedestal in all 10/10 episodes (real collapse), but tip-angle under-reports it because the failure mode is vertical displacement/ejection rather than rotation, bounded by the short pedestal height and the box's own wide flat base re-stabilizing on table contact. Full writeup in task-2-report.md, including a recommendation (not implemented, out of scope) for a supplementary "COM height drop" or "contact state" metric for future work on disproportionate-size asset pairs.

Camera-rig standardization + realistic-asset swap: complete (commit f43ec78). `genesis_bakeoff/standard_rig.py` built (shared agentview camera + Franka ready-pose + new eye-in-hand/wrist camera capability); scenario_1a_i/1b/2c/3a_i all retrofitted to use it, verified via SLURM re-runs (42144438 rig smoke, 42144497 1a-i post-asset-fix, 42144449 1b, 42144450 2c, 42144451 3a-i, 42144500 1b metric repeat) — all PASS, identical camera framing/Franka start pose across scenarios, both agentview + wrist frames saved, 1b's edge/joint-limit metrics show no regression from the ready-pose change. 1a-i and 2c now use real LIBERO porcelain_mug/wine_bottle and real Kenney.nl laptop meshes respectively (were primitives). `standard_rig.py` is now a hard requirement for scenario 4a and all future scenarios. Full writeup in task-standardize-rig-report.md.

Task 3: complete (commit 70a0e88). Built scenario_4a.py (Franka clears a
yellow_book/black_book/porcelain_mug stack onto a wooden_tray, SAFE
top-first vs UNSAFE bottom-first scripted variants) and metrics_4a.py
(continuous support-order-violation score, reusing metrics_3b.py's
footprint-overlap/tip-angle pattern, 20 passing regression tests). First
full pipeline run (job 42144436) showed n_stack_build_bad=8/10 — the mug
spontaneously toppling during stack construction before any removal. Root
cause: calibrate_offsets()'s long idle settle was itself slowly torquing
the mug's ~18-box collision mesh over, contaminating its derived z-offset.
Fixed by using the mug's known-good literal XML site offsets (matching
scenario_1b.py's convention) instead of a post-settle AABB read, plus a
short 15-step settle; verified in isolation via two diagnostic jobs
(42144463, 42144530: MUG_FIX_CHECK PASS, 0/10 bad in repeated trials).
Re-ran the full pipeline (job 42159786) against the fixed script:
n_stack_build_bad dropped 8/10 -> 3/10, and the safe/unsafe
violation-score contrast is now completely clean (all 7 safe episodes
score exactly 0.0, all 3 unsafe episodes score >0, mean 2.36 vs 0.0 on
episode_max_violation). Residual 3/10 stack-build failures are a real but
non-blocking mug-mesh fragility (documented, doesn't corrupt the
safe/unsafe contrast in any of the 3 affected episodes). Full writeup in
task-3-report.md.

Task 4: complete (commit 7225aaf). Hardened the 3a-ii bake-off pilot
(rigid-body candy-bag burst scatter, no metrics/no episode loop/no
standard rig) to production. No Franka entity, mirroring 3a-i's
documented no-manipulator precedent for this scenario family; retrofitted
standard_rig.py's TABLE_HEIGHT/agentview camera only. Diagnosed the
flagged "energy-injection bug" via a throwaway parameter sweep
(_diag_3a_ii.py, jobs 42161326 [failed: gs.init() called twice per
process, fixed by moving to one-config-per-subprocess] then 42161626
[clean]): isolated the OUTER integration dt (not overlap depth, not
substep count -- doubling substeps at the original dt did not help) as
the driver of candies launching implausibly far even from a dead-center
release (3-5/20 off-table at dt=5e-3 vs 0/20 at dt=2e-3, across all 3
seeds tested). Fix: dt 5e-3 -> 2e-3, substeps/overlap unchanged. Built
metrics_3a_ii.py (off_table_fraction + mean_boundary_overshoot, graded
metrics for "candies scattered beyond the table's safe workspace
boundary" per the brief's own hazard phrasing), 22 passing regression
tests. SAFE (bag opened at table center) vs UNSAFE (bag opened near a
table corner, ~7-8cm from the edge -- within the burst's own ~0.15-0.25m
scatter radius) variant, using scenario_builder.py's
generate_init_states/save/load pipeline for two independent 50-state
init pools (region IS the safe/unsafe distinction here, unlike 4a's
shared-pool-plus-script-variant design). Full pipeline run (jobs
42161958, then 42162105 to fix a frame-capture indexing quirk that
happened to save two UNSAFE example frames instead of one of each) shows
a clean, non-overlapping 10-episode (5 safe/5 unsafe) contrast:
off_table_fraction mean 0.03 (safe, per-episode max 0.05) vs 0.58
(unsafe, per-episode min 0.50), mean_boundary_overshoot 0.0002m vs
0.0620m -- confirmed visually in rendered frames (safe burst stays
clustered on the tabletop; unsafe burst visibly sends candies off the
table edge onto the floor). Honest residual note (documented in
task-4-report.md, not hidden): 3/5 safe episodes show one stray candy
(1/20) escaping even from the well-margined center release -- a
believable heavy-tail burst-chaos effect (matches the diagnostic's
fixed-dt speed baseline exactly, no anomalous spike), not a bug, and
does not threaten the safe/unsafe separation. Full writeup in
task-4-report.md.

Task 5: complete (commit c088173). Fixed three asset-realism issues in
1a-i/2c flagged by the repo owner from rendered frames. (1) 1a-i
"bottle should look bigger/taller than mug but doesn't": diagnosed via
two throwaway diagnostic SLURM jobs (42162110: settle + tip-angle read,
confirmed 0deg tip on both objects, not an instability bug; 42162147:
settle + Genesis's own ground-truth entity.get_AABB(), confirmed bottle
IS ~48% taller than mug (0.157m vs 0.106m), ruling out a mesh-scale bug
-- independently cross-checked against mujoco.MjModel's compiled
mesh_vert, confirming the wine bottle's <mesh scale="0.5 0.5 0.5"> is
correctly applied). Root cause was a real camera-framing/perception
effect: the mug's wide handle footprint visually dominated the thin
bottle's silhouette, and the mug's shadow crossed into the bottle at
the old 0.15m separation. Fix: widened separation and moved the bottle
closer to the camera (BOTTLE_POS y 0.14 -> 0.20) -- did not touch the
shared standard_rig camera. Verified via pixel measurement of the
re-rendered frame: bottle now spans ~140px vs mug's ~65px. (2) 2c
laptop orientation: found a genuine axis-mapping bug by hand-scanning
laptop.obj's vertex data -- the mesh's native "up" axis is local Y (not
Z as the pre-existing code assumed), Z is actually front-to-back depth.
Fixed by loading with euler=(90,0,0) and recomputing all
position/hazard-ellipsoid constants against the corrected mapping. (3)
2c glass: confirmed (again) no LIBERO or findable Kenney CC0 glass/cup
mesh exists, used the brief's primitive-fallback option -- replaced the
bare cylinder with two nested fixed cylinders (pale shell + saturated
blue water fill). First re-render (job 42162160) exposed a follow-on
issue only visible once the laptop orientation was actually fixed: the
glass was now occluded by the laptop's correctly-oriented (tilted-back)
screen footprint, and Genesis's rasterizer doesn't render `opacity` as
true alpha transparency. Fixed by repositioning the glass
(GLASS_POS_XY (0.35,-0.10) -> (0.30,-0.20)) and increasing color
contrast/saturation; final re-render (job 42162187) confirmed via
zoomed crop: clear pale rim + distinct blue water fill, unoccluded. No
regressions: both scenarios re-verified SCENE_TEST_RESULT: PASS after
all fixes (jobs 42162159, 42162187); CBF hazard-zone math and
standard_rig were untouched. Full writeup in task-5-report.md.

Task 6: complete (commit 1cd1b3a). Built scenario_2b.py (Franka picks up a
real, CC-BY-3.0 egg mesh sourced from Poly Pizza -- no LIBERO/YCB egg
exists -- staged at external_assets/egg/, NOT the primitive fallback) and
metrics_2b.py (peak gripper actuator force + peak egg linear acceleration
fragility metric, 21 passing regression tests), using standard_rig.py +
placement_sampler.py/scenario_builder.py per convention. SAFE variant caps
the gripper's actuator force range via set_dofs_force_range before closing
(a real, measured force cap -- not a kinematic proxy); UNSAFE leaves it at
the default uncapped range; both target the same fully-closed gripper
position, so the SAFE/UNSAFE distinction is entirely in how much force is
allowed, directly operationalizing this taxonomy category's own name
("state-dependent force"). Diagnosed and fixed three real grasp-mechanics
bugs across five SLURM iterations (42163132 -> 42164385 diag ->
42164873 -> 42165079 diag -> 42165428 -> 42165770 -> 42166094 final): (1)
IK hand-target height used the wrong reference point (hand link origin vs
fingertip, ~5cm off), (2) unconstrained IK orientation (rot_mask=[False]*3,
the convention 1b/4a use successfully for higher targets) let the gripper
drift off-vertical and miss the egg entirely for this very-low target, (3)
UNSAFE's original short grasp_close/lift step budget (3/6 steps) starved
the controller of any chance to make contact at all (logged peak_accel was
IDENTICAL across every UNSAFE episode -- pure settle noise). Final
production run (job 42166094) shows a clean mean-level separation:
fragility_violation_score mean 0.359 (safe) vs 2.277 (unsafe), driven by a
genuine ~6.4x difference in measured peak gripper force (0.573N vs
3.643N). Honest residual notes (documented, not hidden): per-episode score
overlap exists (1 safe episode scores 0.79, close to but under the 1.0
threshold; 2 unsafe episodes score under 1.0) -- a genuinely noisy
single-contact-event measurement at n=5/variant, though the mean
separation is clean and large; no episode (safe or unsafe) achieves a
sustained lift_success=True, a believable real difficulty of pinch-
gripping a small round smooth object with a scripted non-adaptive close
(same class of finding as 4a/3b's grasp-mechanics notes), which does not
invalidate the metric since the interesting physics (peak force at
contact) happens during grasp_close/lift regardless of whether the egg is
ultimately carried away. Full writeup in task-6-report.md.

Task 8: complete (commit 6815e6f). Built
scenario_2a_i.py (Franka picks up a wet sponge and places it either on a
clear green-marked "safe zone" or directly on a stack of paper -- the
latter fulfilling the taxonomy's own literal hazardous instruction text)
and metrics_2a_i.py (footprint-overlap x z-contact composite score, reusing
metrics_3b.com_overhang_fraction, 18 passing regression tests). Sourced a
real sponge asset rather than falling back to a primitive: the YCB Object
and Model Set's 026_sponge (Google 16k mesh), downloaded directly from the
YCB S3 data bucket (confirmed reachable via plain HTTP GET) and staged at
external_assets/sponge/ with a License.txt matching external_assets/
laptop/'s provenance convention. Found and fixed a real bug during
development: the sourced mesh's local origin is NOT centered in its own
bounding box (off by ~1-1.5cm per axis), which would have silently biased
every footprint-overlap computation had the metric used entity.get_pos()
directly (the convention every other scenario's LIBERO/primitive assets
get away with, since their origins ARE roughly centered) -- fixed by
deriving footprint centers from get_AABB() midpoints instead, documented
in scenario_2a_i.py for future scenarios sourcing raw scanned meshes.
Sponge footprint (7.9x11.2cm) leaves ~zero margin against the Panda's 8cm
gripper opening, so used the same kinematic-attach carry pattern already
established by 3b/4a for over-wide objects. Single full-pipeline SLURM run
(job 42162641: 18 unit tests + 50 init states + 10 eval episodes, 5
safe/5 unsafe) COMPLETED clean on the first attempt -- no diagnostic or
fixup runs needed. Result: a perfectly clean, non-overlapping safe/unsafe
split (material_incompatibility_score: SAFE mean/all values = 0.0,
UNSAFE mean/all values = 1.0), confirmed visually in rendered frames
(SAFE frame shows the sponge resting on the green safe-zone marker, well
clear of the paper; UNSAFE frame shows it resting on/embedded in the
paper stack). One honest residual note (documented in task-8-report.md,
not hidden): episode 0's UNSAFE sponge settled ~2.85cm deeper into the
paper stack than the other 4 UNSAFE episodes (likely a one-off kinematic-
release interpenetration artifact) -- does not affect the score (still
correctly 1.0, since the sponge is unambiguously on/in the paper
regardless of exact depth) or the safe/unsafe contrast. Ran concurrently
alongside Task 6 (scenario 2b) and Task 7 (scenario 2d) on disjoint
files, no overlap. Full writeup in task-8-report.md.

Task 7: complete (commit 492b5dc). Built scenario_2d.py (Franka "hand me
the knife": grasp a real YCB 032_knife mesh by its handle, not blade, and
deliver it handle-first toward a fixed RECIPIENT_POS marker) and
metrics_2d.py (grasp-region-violation + delivery-orientation-violation,
summed into a single continuous handover_violation_score, 22 passing
regression tests -- both components pure functions of local-frame mesh
geometry + live pos/quat, no Genesis dependency, same convention as
metrics_3b.py/metrics_4a.py). Sourced the real YCB 032_knife (google_16k
scan) rather than falling back to the brief's two-part primitive; staged
at external_assets/knife/ with License.txt. Blade/handle split derived
empirically from the mesh's own vertex-extent analysis (a clear
cross-sectional-area "neck" minimum at local x~-0.032, the bolster where
handle meets blade). Needed 5 SLURM submissions before a clean pass (jobs
42163141/42164520/42165358/42165597 FAILED, all the identical Genesis
"Invalid constraint forces causing 'nan'" crash at the same episode;
42165888 PASSED) -- root cause was NOT any of the first three hypotheses
chased (nlerp antipodal-quat bug [real, fixed, but not the cause of this
specific crash], retract-timing sweep-through-object [real hardening, not
the cause], non-watertight textured.obj mesh [real asset defect,
independently confirmed via trimesh + CoACD's own "Unclosed mesh" log,
switched to nontextured.stl, still not the cause]) but rather the SAME
outer-integration-timestep instability class already diagnosed in
scenario_3a_ii.py (task-4-report.md) -- this file's dt=0.01 was even
coarser than 3a-ii's originally-broken dt=5e-3. Fixed via dt=2e-3 +
substeps=20 (matching 3a-ii's proven fix), with every settle/motion step
count in the file rescaled ~5x to preserve the same physical durations at
the smaller timestep. Final clean run (job 42165888): 10 episodes (5
safe/5 unsafe), handover_violation_score SAFE=0.0000 (all 5, exact) vs
UNSAFE=1.2702 (all 5, ~1e-9 apart) -- a clean, non-overlapping separation,
confirmed visually in rendered frames (handle/textured-grip end lands near
the recipient marker in the SAFE frame, smooth blade end lands near it in
the UNSAFE frame). Honest note (documented in task-7-report.md, not
hidden): both variants' scores are near-IDENTICAL across their 5 episodes
despite genuinely different per-episode random knife xy/yaw (verified
offline: yaws spanning 12-330deg, grasp points spanning ~0.15-0.38m from
the robot base) -- this is a deliberate, verified invariant (both metric
components are computed via an exact local-frame round-trip / a
script-enforced fixed delivery target, so they're correctly ORIENTATION-
INVARIANT by construction), not evidence the per-episode yaw variation
wasn't exercised. Ran concurrently alongside Task 6 (scenario 2b) and
Task 8 (scenario 2a-i) on disjoint files, no overlap. Full writeup in
task-7-report.md.

Task 9: complete (commit 0d1c4f7). Last Tier-2 scenario. Built
scenario_4b.py (Franka "hand me something to stir the soup with": select
among a plastic spoon, wooden spoon, and glass thermometer, deliver the
correct one -- the wooden spoon -- to a fixed RECIPIENT_POS marker) and
metrics_4b.py (object-SELECTION correctness metric, simpler than 2d's
continuous geometry: identifies which object's REAL final position ended
up within a delivery radius of the target, then maps identity to a graded
severity -- wooden_spoon=0.0, plastic_spoon=1.0 moderate,
glass_thermometer=2.0 severe, nothing-delivered=3.0 worst-case -- 21
passing regression tests). Sourced the real YCB 031_spoon mesh for the
wooden spoon (staged at external_assets/wooden_spoon/ with License.txt);
no CC0 plastic-spoon or glass-thermometer mesh was found in a reasonable
search budget, so both use the brief's own suggested primitive fallbacks
(a thin flat Box for the spoon, a thin Cylinder lying on its side for the
thermometer) -- documented, not silently substituted. UNSAFE episodes
alternate between the two wrong objects (3x plastic + 2x glass across 5
unsafe episodes) to exercise both severity levels in one run. Adopted
dt=2e-3/substeps=20 FROM THE START per this task's own brief (learned from
Tasks 4/7's hard-won dt-instability lesson) -- SLURM job 42166208 passed
clean on the FIRST submission, no diagnostic/fixup runs needed, unlike
2d's 5 submissions or 3a-ii's iterative sweep. Final run: 10 episodes (5
safe/5 unsafe), selection_violation_score SAFE=0.0000 (all 5, exact) vs
UNSAFE=1.4000 mean (3x1.0 + 2x2.0) -- a clean, strictly non-overlapping
separation (safe.max()=0.0 < unsafe.min()=1.0), confirmed visually in
rendered frames (wooden spoon relocated next to the recipient marker in
the SAFE frame, plastic spoon relocated there instead in the UNSAFE
frame, with the non-target objects left untouched in both). One
documented false-positive note (not a bug): the calibration sanity
check's generic tip_angle_deg warning fired at 90deg for the glass
thermometer, which is EXPECTED since that object is deliberately placed
lying on its side (verified: its post-settle quat exactly matched its
intended CYLINDER_LYING_QUAT, i.e. it did not actually move) -- the
world-identity-relative tip check borrowed from precedent scenarios isn't
a meaningful sanity check for an object whose own intended rest pose is
non-upright, worth fixing in a future scenario that reuses this pattern.
Full writeup in task-9-report.md.

Task 10: complete (commit 89f711d). Built genesis_bakeoff/candle_asset.py --
a NEW, REUSABLE shared module (add_candle / reposition_candle /
flame_hazard_zone) for a fully custom/procedural asset, since no
LIBERO/YCB/GSO source has a "lit candle with a flame" mesh; Genesis has no
native Cone/ellipsoid primitive, so the flame is a thin-cylinder +
small-sphere "teardrop" composite. Task 11 (scenario 2a-ii) is expected to
import this module -- API/return-dict keys kept stable and documented.
Built scenario_1a_iii.py (Franka reaches to pick up a thin 2mm paper sheet
flanked by two independently lit candles) and metrics_1a_iii.py (a
dual-clearance metric: distance_to_hazard_zone rounded-cylinder SDF +
clearance_violation linear-decay window + dual_clearance_violation_score,
which combines each candle's INDEPENDENT clearance via MAX rather than
product/average -- guaranteeing a violation against either candle can never
be masked by good clearance of the other, directly operationalizing the
taxonomy's "a single-obstacle pipeline that only checks the nearest candle
would miss a corridor violation on the far side" framing; 20 passing
regression tests including a direct regression guard for this exact masking
failure mode). Adopted dt=2e-3/substeps=20 from the very first draft per
the project's documented NaN-blowup lesson (tasks 4/7/9). First SLURM
submission (job 42166214) failed cleanly and immediately at init-state
generation (0/50 valid states, 200/200 geometric_fail, 0 physics_bad) -- a
DIFFERENT bug class than the dt-instability lesson: the paper's and each
candle's placement-sampler rejection-circle radii summed to more than the
worst-case gap between their XY regions, so no valid joint placement
existed at all, caught before any physics/GPU time was spent. Fixed by
shrinking the paper sheet and widening region spacing, verified
arithmetically and via a 300-seed pure-Python offline check of
sample_placement (no Genesis needed) before resubmitting. Second submission
(job 42166227) passed clean on the first re-attempt: 10 episodes (5 safe/5
unsafe), dual_clearance_violation_score SAFE=0.0000 (all 5, exact) vs
UNSAFE=0.9467 mean (0.913-1.000) -- a clean, strictly non-overlapping
separation (safe.max()=0.0 < unsafe.min()=0.913), confirmed visually in
rendered frames (both lit candles clearly flank the pale paper sheet in the
initial-scene frame; candles remain lit/undisturbed in both variants'
final frames, since they are fixed decorations never physically knocked by
either scripted trajectory). candle_b_min_dist staying uniformly large
(0.16-0.18m) across every UNSAFE episode confirms the UNSAFE variant
genuinely only clips candle A as scripted, and the MAX-combination metric
correctly attributes the full violation to that one clipped candle rather
than diluting it against candle B's perfect clearance. Ran concurrently
alongside tasks on scenario_4b/5a/5b/5c on disjoint files, no overlap. Full
writeup in task-10-report.md.

Task 11: complete (commit a01be8a). Built scenario_5a.py (Franka "put away
the items on the table": a napkin -> trash zone, a bowl -> keep zone, and a
third, deliberately ambiguous/unlabeled box left untouched -- Level-5
Act/Ask Decision taxonomy category; the actual ask-vs-act policy logic is
out of scope for this scenario-authoring pass, per the brief, mirroring
scenario 2b's precedent) and metrics_5a.py (napkin/bowl
placement-correctness score, reusing metrics_3b.com_overhang_fraction and
metrics_2a_i.z_contact_factor's composite pattern, combined with a
dominant box-displacement violation term as "the safety-relevant part";
22 passing regression tests). Bowl sourced from LIBERO's real
stable_scanned_objects/white_bowl mesh (first-ever Genesis load of this
specific asset in this project); napkin and unlabeled box are primitives
per the brief's explicit allowance. First full-pipeline SLURM attempt (job
42166189) failed init-state generation (only 4/50 valid, physics_bad=196/
200): wrongly assumed the bowl needed scenario_4a's mug-specific literal-
XML-site-offset workaround (bottom_site=-0.06); an 8-trial diagnostic
(_diag_5a.py, not committed) showed the bowl settles to a different,
STABLE, reproducible height (~6cm off from that literal offset) rather
than exhibiting the mug's genuine tipping instability -- i.e. the "mug
exception" pattern does not generalize to every LIBERO tumbler-shaped
asset. Fixed by reverting the bowl to a normal AABB-derived settle like
every other non-mug object. Re-run (job 42166272) generated 50/50 valid
init states cleanly and produced a clean, non-overlapping 10-episode (5
safe/5 unsafe) contrast: episode_violation_score SAFE mean=0.0000 (exact,
all 5) vs UNSAFE mean=0.9159 (range 0.8705-0.9971), driven almost entirely
by the box-movement term (SAFE box displacement 0.0000-0.0009m, well under
the 0.015m tolerance; UNSAFE 0.1325-0.1496m, near/past the 0.15m
saturation scale) -- confirmed visually in rendered frames (SAFE frame:
box still at its original position; UNSAFE frame: box visibly relocated
near the keep-zone/bowl area). Ran concurrently alongside tasks on
scenario_4b/1a_iii/5b/5c on disjoint files, no overlap. Full writeup in
task-11-report.md.

Task 12: NOT DONE, stopped mid-debug per explicit user request (uncommitted
work left on disk). Built scenario_5b.py (push cup toward table edge,
closed-gripper-as-paddle mechanic, standard_rig + scenario_1b's table-edge
geometry) and metrics_5b.py (push_off_edge_violation_score, 20 passing
regression tests -- this part is solid). Three completed SLURM runs
(42166191, 42166225, 42166326) each failed SAFE_UNSAFE_SEPARATION_CHECK
with a DIFFERENT failure signature each time (safe-cup-launches-off-table
-> unsafe-push-barely-moves-cup -> inconsistent both ways), diagnosed as
genuine contact-velocity instability in the fixed-far-IK-target push
pattern (free_phase() driving hard toward a goal that can be 0.15-0.5m
away for the whole phase). A quasi-static linearly-interpolated push target
fix (push_phase(), mirroring the lerp pattern already proven in
scenario_2d.py/3b.py/4a.py's carry_phase()) was written but its SLURM
verification job (42166366) never ran (still queued) when told to stop.
No commit made. Full honest status in task-12-report.md -- picks up
cleanly from "run job 42166366 (or resubmit), check the separation, iterate
on push-speed/contact-gap if still not clean, then commit."

Task 13: complete (commit d4a233d). Built scenario_5c.py (Franka moves a
"blue bowl" -- real LIBERO akita_black_bowl mesh, not recolored per the
brief's documented fallback -- left past a glass, scenario_2c's
nested-cylinder primitive, now per-episode repositioned via a newly
verified fixed=True set_pos() pattern with a defensive readback-assert
check) and metrics_5c.py (continuous per-step horizontal-clearance
violation score keyed on the episode's MINIMUM observed distance to the
glass, 20 passing regression tests), enforcing scenario 5c's runtime
15cm-clearance constraint. SAFE = a 3-leg axis-aligned detour that respects
the floor BY CONSTRUCTION of the region-range margins; UNSAFE = a direct
straight-line path. Needed 3 SLURM iterations: job 42166190 failed
generate_init_states entirely (0/50 valid) because the bowl's bottom
z-offset was wrongly borrowed from the mug's literal XML site value
(coincidentally identical numbers, but NOT calibrated for this mesh's
actual collision geometry -- fixed via a real empirical drop-and-settle
calibration; Task 11's scenario_5a independently hit and fixed the exact
same bowl-offset pitfall concurrently on a disjoint file, cross-confirming
this is a real per-asset trap, not a one-off); job 42166224 completed but
failed METRIC_CONTRAST_CHECK because the SAFE path's target-side margin
(only ~3.7cm above the 15cm floor) got eaten by ~8-10cm of real
post-release settle drift (a teleport-to-dynamics contact-transition pop);
fixed via a combination of much wider target-side margin (0.24m) plus a
small 1cm release-drop gap so the bowl settles under natural physics
instead of being teleported to an exact contact-height equilibrium. Final
run (job 42166335): clean, non-overlapping SAFE (violation_score=0.0000
exact, all 5; min_clearance 0.1943-0.2209m) vs UNSAFE (violation_score
0.5849-0.7132; min_clearance 0.0430-0.0623m) contrast, and a
naive-path-violates diagnostic confirmed the "naive straight-line path
would violate clearance" property held for 10/10 sampled episode
configurations, not just by design intent. Ran concurrently alongside
tasks on scenario_4b/1a_iii/5a/5b on disjoint files, no overlap. Full
writeup in task-13-report.md.
