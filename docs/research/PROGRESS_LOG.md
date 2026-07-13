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
