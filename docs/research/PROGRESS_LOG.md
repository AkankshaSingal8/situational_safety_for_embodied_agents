# Progress Log — Context-Aware Safety Filter Campaign
**Detailed chronology, 2026-07-12 → 2026-07-13.** Companion: `DIRECTIONS_RESEARCH_AND_ARCHITECTURE.md`. All results also in `.worktrees/flow-guidance-tier-gt/results_tables/fgd_tier_gt_results.md` (with per-task tables).

## Phase 0 — Strategy (07-12 morning)
- Deep-read fan-out (13 agents): SOTA 2607.01378 (per-Euler-step min-norm SLSQP on π0.5, **GT obstacle state**, EEF-only, no semantics, no code, no ablations; abstract's +6.3/+19.8 are RELATIVE to AEGIS), SafeDojo, LIBERO-Safety (empty enforcement leaderboard; released π0.5 weights), OopsieVerse (no method results), CORE, + 2 landscape surveys (~40 papers), 4-lens proposal panel, 2 adversarial critiques.
- **Verdict**: retire post-hoc FOL shield as enforcement (TSR ceiling quadruple-confirmed); keep the FOL layer as constraint compiler; pivot to in-denoising guidance. Docs: `docs/superpowers/specs/2026-07-12-context-aware-safety-filter-brainstorm.md`, `...-direction-evaluation.md` (CAVF umbrella; SCFM engine).

## Phase 1 — Tier-GT engine build + go/no-go (07-12)
- Implemented guided sampler (private openpi copy): DCBF repair per Euler step, quantile denorm in-loop, guidance payload popped pre-transforms, one server for guided/unguided. 8 unit tests.
- Infra battles: worktree submodule surprise (vlsa-aegis is a gitlink → private copy), py3.8 client env core-dumps in batch (never batch-validated; switched to openvla_libero_merged + vendored websockets/msgpack), JIT/keepalive race (server warmup pre-port), flaky nodes w008/w009 (2 aborts localized; excluded), per-episode JSONL.
- **Smoke r1** (eef_r=0.05): L1 65/10, L2 65/15 — TSR preserved at 23% intervention (mechanism ✓), margins too thin (gate ✗).
- **Smoke r2** (eef_r=0.09 + 4mm/step inflation): L1 75/30, L2 75/25 — **gate ✓ both levels**.
- **n=50 r2-config** (job 42162199): L1 71.0/23.5 (baseline 67.0/14.0) — TSR ✓ CAR ✗ (pre-registered 40).
- Companion points (hand +0.08 / payload −0.06; from v19 forensics). Smoke r3: L2 85/50, L1 60/35.
- **n=50 both levels r3-config** (42165451, after hardening + 2 account-cancelled attempts): **L2 86.5/49.0** (baseline 55.5/12.0 — gate PASSED; beats SOTA-published L2 TSR 78.0), L1 58.5/44.5 (frontier traded vs thin config).

## Phase 2 — Full-suite Phase-1 evaluation (07-12 → 07-13, jobs 42166469/70/71)
r3 config, n=50/task, all suites (with Spatial from above):
| Cond | Ours | Baseline | Both axes? | SOTA pub |
|---|---|---|---|---|
| Spatial L1 | 71.0/23.5 · 58.5/44.5 | 67.0/14.0 | ⚠️ per config | 75.5/77.5 |
| Spatial L2 | 86.5/49.0 | 55.5/12.0 | ✅ | 78.0/75.5 (TSR>) |
| Goal L1 | 71.5/30.5 | 51.0/23.0 | ✅ | 91.0/94.0 |
| Goal L2 | 69.0/58.0 | 66.5/35.0 | ✅ | 83.5/82.5 |
| Object L1 | 51.0/22.5 | 40.5/14.0 | ✅ | 90.5/82.5 |
| Object L2 | 85.5/40.5 | 74.0/25.0 | ✅ | 81.0/85.5 (TSR>) |
| Long L1 | 23.0/55.0 | 58.0/15.0 | ❌ (t0/t1 = 0.00) | 81.5/82.0 |
| Long L2 | 47.0/77.0 | 51.0/16.5 | ⚠️ | 72.0/83.0 |
Diagnosis of Long L1: fat envelope blocks the DESTINATION basket → the third appearance of the exemption problem (targets, grasp points, destinations).

## Phase 3 — Research-driven upgrade stack (07-13)
- **Flow-steering survey** (3 agents; `steer_*.json`): backbone survives; uniform always-hard repair = documented trap anti-pattern; adopted P0 stack (schedule, brake-then-push, best-of-K + prefix certificate, pre-shaping, corridor); do-not-adopt list (learned critics, twisted-SMC, indicator potentials, raw-x_t grads...). SABRE-Flow novel components identified.
- **Semantic-safety identification survey** (3 agents, after session-limit retry; `semsafe_*.json`): C1 SANCTION / C2 FOL-COMPILE / C4 GOVERNOR candidates + E1–E5 comparison protocol; default-deny + calibrated exemptions = named white space; conformal-over-consistency = trust standard.
- Two-direction plan approved (2–3 candidates per direction, no commitment; plan file). User lock-in: beat SOTA+baseline on both axes, both privilege tiers.

## Phase 4 — Track A implementation + validation (07-13)
All in-jit, each behind a flag, 15/15 unit tests, committed on `flow-guidance-tier-gt`:
1. **A1 schedule** (ramp 0,0,.2→1,1...): dev-cell A/B outcome-neutral at n=5 (activation 0.47→0.54, outcomes unchanged) — retained, decision deferred to composed n≥20. Bugs caught by tests: brake firing without violation; over-strong floor assertion.
2. **A2 brake-then-push** with barrier-max fallback: provable non-worsening under |cmd|≤1; residuals ≈0.
3. **A3 best-of-K=8** (shared prefix KV, lexicographic executed-prefix selection): jit static-shape bug caught in first GPU run (fixed via static_argnames). **Object L1 t1 TSR 18→80** (n=5); CAR=0 cells unmoved (no feasible seed exists — as predicted).
4. **Corridor/destination exemption** (+ client entity parsing; along-segment gate t>0.15 fixing the exemption-sphere bug caught by the off-axis unit test).
5. `companion_scale` switch for the SOTA-reimpl fidelity arm; tyro bool-flag bug caught by login-node parse test (job cancelled+resubmitted before wasting GPU).

## Phase 5 — Composed system validation (07-13)
- **Composed smoke** (K=8+ramp+corridor+0.12, job 42168138): **Long L1 t0 0→60 TSR at 80 CAR**; Spatial held; Object L1 resists (obstacle-on-path).
- **E2 offline corridor precompute** (39 saved episodes): exemption restores straight-line grasp liveness t0 0→70%, t1 40→100%, t2 60→100%; **t3 stuck at 11%** (obstacle <6cm from approach line — the geometrically non-negotiable family; SOTA's own 46-CAR cell corroborates). E2 gate (≥70%) passes 3/4.
- **Composed n=20** (42177043 after 2 account cancellations + queue green-light): **Long L1 t0 65/80 CONFIRMED**; Long L1 aggregate 37.5/66.3 (r3 n50: 23/55); Spatial L1 60/45 (best frontier point yet); Object L1 41.3/33.8 (beats baseline both axes; t2 TSR collapse 5% → forensics queued).

## In flight (2026-07-13)
- **42182119/20**: composed n=50, Spatial + Long, both levels (headline candidate rows).
- **42182126**: SOTA-reimpl arm (EEF-point-only, uniform, K=1, no corridor, r_obs first point of sweep) — the fairness referee.
- Monitor armed (single blocking watcher). Next after these: composed n=50 Goal/Object; reimpl r_obs sweep completion; γ/corridor mini-sweep; Object-L1 forensics; Track B E1/E4 (needs VLM budget decision); no-GT tier build.

## Process/infra lessons banked
Bridges2 cwd resets between calls; SLURM scripts are read at RUN time (in-flight edits affect pending jobs); tyro bools are flags; module_jit needs bound methods + static_argnames for shape args; account-level job cancellations from parallel sessions — pause and ask rather than fight; one blocking monitor per job (token discipline); smoke n=5 = routing only (±40pp/cell), n=20 = confirmation, n=50 = claims, paired stats via episode JSONLs.

## Phase 6 — Session resume, ledger recovery, regression root-cause (2026-07-15/16)
**Mission reaffirmed by user**: beat baseline AND SOTA-published on both TSR and CAR, all 8
conditions (4 suites x 2 levels), Tier-GT, n=50, then proceed to no-GT tier. "LOCK IN" —
continue autonomously until this bar is met, maintaining this log with actions/assumptions/caveats.

### Housekeeping
- The `git log origin/flow-guidance-tier-gt..HEAD` check the handoff flagged as unverified was
  CONFIRMED true: the "COMPLETE COMPOSED 8-CONDITION TABLE" ledger append (§3 of the handoff) had
  never landed in a commit. Recovered + committed (`4a42558`): ledger text plus the underlying
  goal/long/object/spatial-II composed result JSONs and the reimpl_r10 spatial results that were
  sitting untracked in the worktree.
- Push to `origin/flow-guidance-tier-gt` still blocked — no stored git credential in this shell
  (`fatal: could not read Username for 'https://github.com'`). Fetch/read works fine (public repo).
  Commits are safe locally; push is queued as task #1 pending fresh auth from the user.
- **Infra note**: the Bash tool itself failed twice mid-session (exit 254 on every command,
  including `true`/no-ops) for a few-minute stretch each time, unrelated to any runaway process on
  the host (`ps -u asingal` was clean). Recovered on its own / after a user nudge without a full
  Claude Code restart the second time. Not a Bridges2/SLURM issue — flagging in case it recurs.

### Regression root-cause: Goal L2 / Long L2 (composed vs r3, from the n=50 8-condition table)
Read `parse_entities()` (`vlm_pipeline/run_guided_safelibero_pi05_eval.py`), the client-side
corridor/destination entity parser, plus the LIBERO env's `_setup_observables` (SafeLIBERO
submodule, `bddl_base_domain.py`), plus per-task result breakdowns from the composed n=50 JSONs
vs the prior r3 n=50 JSONs. Two independent bugs found and fixed (commit `cc3f766`):

1. **Fixture-destination blind spot** (H1 confirmed). `_setup_observables` only creates a
   `{name}_pos` sensor for `:objects` in the bddl (`for (i, obj) in enumerate(self.objects)`) —
   never for `:fixtures` (cabinets, stoves, drawers, sinks...). `parse_entities` only ever looked
   for `dest_pos` among keys ending in `_pos`, so any destination phrase naming a fixture
   ("on top of the cabinet", "on the stove") silently resolved to no destination and thus no
   corridor exemption for the whole carry-to-destination leg. This is exactly 2 of 4 Goal tasks
   (cabinet, stove) and matches the per-task damage: Goal L2 task1 "put bowl on top of cabinet"
   TSR 76.0(r3)->28.0(composed), CAR 82.0->60.0 — by far the largest single-task swing in the
   whole 8-condition table, and it single-handedly explains most of the Goal L2 aggregate drop
   (69.0->54.0). Confirmed empirically (not just by code reading): spun up the live env for this
   task and checked `obs` keys ending in `_pos` (`wooden_cabinet_1` absent) plus
   `env.sim.model.body_names` (fixture sub-bodies ARE present in the sim, e.g.
   `wooden_cabinet_1_cabinet_top`). Fix: new `_fixture_dest_pos()` helper — when no object gives a
   `_pos` match for the destination phrase, scan live sim body names directly (every mujoco body,
   fixture or object, always has one) for a token match, preferring a sub-body matching a
   destination qualifier (top/bottom/middle/burner/drawer) over the fixture's base body. Verified:
   for the cabinet task, resolved `dest_pos` now matches `wooden_cabinet_1_cabinet_top`'s position
   exactly (bit-for-bit modulo float noise).
2. **Stale-target selection on multi-object tasks** (new finding, not in the handoff's hypothesis
   list — H2 as stated ["fat margins + K-selection conservatism"] wasn't the mechanism found).
   `target_name` was "nearest mentioned object to EEF" with no notion of "already delivered." On
   Long's two-object tasks ("put both the alphabet soup and the cream cheese box in the basket"),
   once the first object is placed in the basket it can sit physically nearer to the EEF/basket
   than the still-unplaced second object — so the corridor kept sanctioning the approach to the
   ALREADY-DONE object throughout the second pick-and-place leg instead of the real target. Matches
   the per-task damage: both basket tasks cratered (TSR 84->50 and 82->38; the two mug/plate long
   tasks, already near the TSR floor at 8%/14%, moved only -6pp each — consistent with this bug
   being basket-specific, not a general fat-margin tax). Fix: exclude any mentioned object already
   within `DELIVERED_RADIUS=0.10m` of `dest_pos` from the target candidate pool. Verified against a
   live env: with `alphabet_soup` obs position patched to sit inside the basket, target correctly
   switches to `cream_cheese`.
- **Caveat**: `_fixture_dest_pos`'s body-name matching is a heuristic (token match + qualifier
  hint), same fail-safe philosophy as the rest of `parse_entities` ("missing entities simply grant
  no corridor") — it is not guaranteed to pick the geometrically ideal sub-body on fixtures outside
  the four goal-suite ones inspected (cabinet, stove; drawer/sink untested — drawer task exists in
  the bddl set but was NOT in the active 4-task goal run, so unverified live). `DELIVERED_RADIUS`
  (0.10m) is a hand-picked threshold, not calibrated against basket/plate geometry beyond a rough
  eyeball; if it proves too tight/loose the smoke below will show it (e.g. target flips back to a
  just-placed object because 0.10m undershoots the true basket radius, or excludes the correct
  live target because it starts within 0.10m of dest by scene-randomization coincidence).
- Did NOT chase the smaller Long L2 mug/plate regressions (t2 8->2, t3 14->8 TSR) — both already
  near the TSR floor pre-fix, and DEST_PATTERN's "last locative phrase wins" (relevant for these
  two-subgoal tasks: "on the left plate ... on the right plate") is a separate, lower-priority
  latent issue not addressed here.

### In flight (2026-07-16)
- **42234773** `fgd_composed_fix_smoke`: n=5 routing smoke of the fix on Goal L2 task1 (cabinet)
  + Long L2 tasks 0,1 (baskets) — composed config (K=8/ramp/corridor/eef_r=0.12/companions).
  Purpose: cheap route-check before committing the ~3h n=50 rerun. Gate: task must not still be
  catastrophically below r3 (n=5 noise band is wide, ±40pp/cell per project convention — looking
  for directional recovery, not a final number).
- **42234774 / 42234779** `fgd_reimpl_r06` / `fgd_reimpl_r14`: reimpl-SOTA EEFR bracket sweep
  (task #2 from the handoff's §4 next-steps), Spatial suite both levels, K=1/uniform/no-corridor/
  no-companions, r_obs=0.06 and 0.14 respectively — brackets the SOTA reimpl's unstated obstacle
  radius (r=0.10 already run, in the ledger).
- All 3 jobs queued (Priority, GPU-shared H100) as of this write; one blocking monitor armed
  (poll squeue every 60s, single job-id-left-queue notification each, per the "minimize monitoring
  tokens" standing instruction). Noticed one pre-existing `bash` job (42234572, running on v007,
  not submitted by this session) already on the shared account — left untouched per the
  parallel-sessions courtesy rule.
- **Next after these land**: if smoke shows recovery, launch composed n=50 rerun of Goal + Long
  (both levels) with the fix (task #4); fold reimpl r06/r14 into the ledger sweep table; then
  DBNR/TECV kill-tests (tasks #5/#6); then lock final per-condition configs and confirm all 8
  conditions at n=50 against the four-number bar (task #7).

### DBNR kill-test verdict: GREENLIT (job 42236032, n=5, 2026-07-16)
Pre-registered dev cells, composed config (K=8/ramp/corridor/eef_r=0.12/companions):
| Cell | TSR/CAR | dbnr_active_chunks | dbnr_tau_mean [m] | dbnr_headroom_mean |
|---|---|---|---|---|
| Spatial L1 t1 | 20.0/60.0 | 124/258 chunks | 0.0128 | 0.560 |
| Long L2 t2 | 0.0/80.0 | 202/550 chunks | 0.0033 | 0.899 |
Both cells: tau clearly nonzero (not noise-floor) and headroom large (56-90% of the unit task
direction lies orthogonal to the repair's push direction — plenty of room to restore task motion
without touching the active constraint). Kill criterion (tau~=0) NOT met on either cell ->
**greenlight full DBNR restitution** (docs/research/inv_invB.json methods[1], ~30-line addition
to `_dcbf_repair`'s per-step body per its own implementation estimate: masked nullspace projector
already trivial here since this repair has at most one active constraint per step, so P = I - dd^T
exactly as used for the headroom diagnostic — no Cholesky/stack machinery needed).
- **Caveat**: n=5 is routing-only (the project's own convention: n=5 = ±40pp/cell noise band on
  OUTCOME metrics; tau/headroom are per-chunk geometric quantities with much larger effective
  sample size (124-202 chunks) so the greenlight itself is fairly solid, but the *expected TSR
  gain* (+5-15pp per the spec's own estimate) is unvalidated until an actual restitution
  implementation is run at n>=20).
- **Not yet implemented**: this was diagnostics-only per the kill-test's scope. Full restitution
  (actually adding `r_k` back into `disp_new`) is new work, not yet started — queued after the
  regression-fix reruns (task #4) since that's the path to the core four-number bar; DBNR is
  specifically a lever for Spatial L1 (currently the single worst-performing condition against
  SOTA: composed 53.5/46.0 vs SOTA-published 75.5/77.5, both axes far short) and, per the spec,
  Long/L2 repair-heavy cells generally.
