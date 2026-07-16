# COMPREHENSIVE CONTEXT & NEXT-STEPS FILE
**Written 2026-07-13 (late), MAJOR UPDATE 2026-07-16. Purpose: complete working context for any session (human or Claude) continuing this project. Read this FIRST, then the doc map in §9.**

## ⚡ 2026-07-16 STATE DELTA (supersedes §3–§6 below; full detail in the worktree ledger)
**Tier-GT is CLOSED.** Pre-registered rule applied after the fix n=20 arbiter (job 42238691):
the paper reports the per-condition best-config table — composed for Spatial L2 / Goal L1 /
Object L1+L2 / Long L1; r3 for Spatial L1(thin) / Goal L2 / Long L2. 6/8 beat baseline on
both axes; Long L1/L2 trade TSR (−15.5/−4) for CAR (+56/+60.5). Spatial L2 = full
four-number win (85.0/76.5), every component individually significant
(`vlm_pipeline/stats_tests.py`; Fisher + paired McNemar in ledger).
- **Referee closed** (#14): no obstacle radius reproduces SOTA-published (TSR,CAR) on both
  levels under our >1mm criterion (r06/r10/r14 swept); honest CAR comparison = self-run reimpl.
- **DBNR KILLED** (β-bracket 0.5/1.0/2.0: τ≈0 — repairs remove almost no task-velocity;
  stalls are route-homotopy, not velocity loss). **HDC v1 KILLED** (lateral candidate bias,
  no dev-cell lift — no route commitment across chunks). **TECV RETIRED without build**
  (no strength-shaped failure mode remains). All post-mortems in ledger = ablation narrative.
- **Epistemic signals VALIDATED** (#16): K-dispersion/margin-spread telemetry (now logged in
  every run) predicts near-obstacle stall AUC 0.88–0.91 on Spatial (0.6–0.67 on Goal/Long —
  suite-dependent, report per-suite). `vlm_pipeline/analyze_uncertainty_signals.py`.
- **ROCS v1 mechanism DONE** (#17): gripper convention verified (+1 close/−1 open,
  passthrough); release parse + forbidden-release + delay clamp in
  `openpi_guided/src/openpi/models/rocs.py` (tests in tests/test_rocs.py). Server
  integration awaits LIBERO-Safety harness (not yet cloned).
- **No-GT tier is WELL UNDERWAY** (#20): (a) runtime RGB-D localization
  (`vlm_pipeline/percep_obstacle.py`) — agentview-only 4.5–8.4 cm, z−0.03 correction,
  live-sim validated; (b) **E3 geometry-swap smoke: Spatial L2 85/70 vs Tier-GT 85.0/76.5 —
  geometry leg ≈ free**; n=20 promotion = job 42239799; (c) identity = v17 symbolic grounder
  VENDORED (`vlm_pipeline/symbolic_identity.py`, 39/39 offline on spatial; direct-VLM
  prompting is 0–10% — measured in fol worktree `fol_grounding_accuracy/`); client has
  `--obstacle_id_source symbolic` + `--obstacle_pos_source percep` (GT-anchored scoring,
  method-belief guidance); full no-GT smoke = job 42240496; all-suite identity smoke =
  job 42240503. Saved-capture data bug found: vlm_inputs arrays have inconsistent vertical
  orientation across tasks (runtime unaffected).
- **Next steps (current order)**: (1) read the 3 pending job verdicts (percep n=20, full
  no-GT smoke, all-suite identity); (2) no-GT n=20→n=50 on all conditions where identity
  holds; (3) clone LIBERO-Safety, ROCS server integration, E4 SSR offline; (4) E5
  fail-direction stress test (inject identity/position errors, measure CAR-vs-TSR error
  direction); (5) paper assembly (Table 1 = baseline | reimpl | ours-GT | ours-no-GT).

---

## 0. MISSION (user-locked)
Beat **both** our π0.5 baseline **and** the SOTA (arXiv 2607.01378) on **both TSR and CAR**, per SafeLIBERO condition, with the best method reported at **two privilege tiers**: (a) GT sim state, (b) no-GT (VLM+RGB-D perception). Directions: (1) general contextual safety identification, (2) flow-policy steering — both must be strong/general/novel. Target: ICRA 2027 (~Sept 15, 2026). Best-paper ambition. Secondary benchmarks: LIBERO-Safety (SSR = marquee for ROCS), OopsieVerse (RoboCasa track only).

## 1. WHERE EVERYTHING LIVES
- **Main repo**: `/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents`, branch `full_working_pipeline` (pushed to GitHub `AkankshaSingal8/situational_safety_for_embodied_agents`). Holds all docs (`docs/research/`, `docs/superpowers/specs/`), survey JSONs, baselines.
- **Work branch**: `flow-guidance-tier-gt` in worktree `.worktrees/flow-guidance-tier-gt` (pushed to GitHub). ALL guidance code + results live here:
  - `openpi_guided/src/openpi/models/pi0_guided.py` — the guided sampler (GuidanceParams, `_dcbf_repair`, `_corridor_scale`, `_prefix_acceptance`, `guided_sample_actions` with K-batching + shared prefix KV).
  - `openpi_guided/src/openpi/policies/guided_policy.py` — GuidedPolicy wrapper (pops `guidance` payload pre-transforms; `make_schedule`; GuidanceConfig knobs).
  - `openpi_guided/scripts/serve_policy_guided.py` — server entry (tyro CLI: `--repair_schedule ramp|uniform --num_candidates K --eef_radius R --no-use_companions --corridor_radius --corridor_relax --gamma --d_safe`). Does a warmup infer BEFORE opening the port (JIT/keepalive race fix).
  - `openpi_guided/tests/test_guided_sampler.py` — 15 tests, run: `conda activate aegis && JAX_PLATFORMS=cpu PYTHONPATH=openpi_guided/src python -m pytest openpi_guided/tests -q` (~3 min cold, ~25 s warm).
  - `vlm_pipeline/run_guided_safelibero_pi05_eval.py` — eval client (obstacle discovery via `_obstacle` names; `parse_entities()` for corridor target/dest; `--corridor`, `--disable_guidance`; per-episode JSONL; >1mm-displacement collision = SAME criterion as stored baselines).
  - `slurm/fgd_pi05_suite_n50.slurm` — parameterized runner: `sbatch -J <name> --export=ALL,SUITE=safelibero_<s>,PORT=<p>,K=8,SCHEDULE=ramp,EEFR=0.12,CORRIDOR=--corridor,TAG=composed slurm/fgd_pi05_suite_n50.slurm`. Reimpl arm: `K=1,SCHEDULE=uniform,EEFR=<r>,COMPANIONS_FLAG=--no-use_companions,TAG=reimpl_r<r>`.
  - Results: `fgd_n50_<TAG>/results/<suite>/<level>/results_*.json` (+episodes_*.jsonl); ledger `results_tables/fgd_tier_gt_results.md` (append-only; NOTE: last append attempt may not have committed — verify `git log` and re-append the COMPLETE table from §3 if missing).
- **Shared, read-only**: `vlsa-aegis/` at repo root (a git SUBMODULE — worktrees get it EMPTY; that's why `openpi_guided/` is a private copy). Checkpoint: `vlsa-aegis/checkpoints/pi05_libero`.
- **Old FOL campaign** (constraint compiler assets to reuse): `.worktrees/fol-safety-filter` — v17 symbolic grounder (`fol_safety_filter/vlm_grounder.py`, `symbolic_obstacle_id()` = ¬MENTIONED∧¬SUPPORTS∧NEAREST_TO_PATH, 21/21 offline), v16 Prompt-D runtime rule composition (built+verified), property priors, multi-prompt majority voting. Saved episode observations: `vlm_inputs/safelibero_spatial/` (metadata has GT objects/obstacle).
- **Memory** (Claude sessions): `~/.claude/projects/-ocean-...-embodied-agents/memory/` — `project-icra-lock-in-goal.md`, `project_icra_pivot_brainstorm.md` (running log), `feedback-minimize-monitoring-tokens.md`.

## 2. ENVIRONMENTS & INFRA GOTCHAS (each cost real time — do not relearn)
- Server env: `conda activate aegis` (py3.11, JAX). Client env: `conda activate openvla_libero_merged` (py3.10) — **the py3.8 `libero` env CORE-DUMPS in batch, never use it**. Client deps `websockets/msgpack` vendored at `WT/client_deps` (PYTHONPATH).
- Client PYTHONPATH order: `SafeLIBERO/safelibero : WT/client_deps : WT/openpi_guided/packages/openpi-client/src : WT/vlm_pipeline`. `MUJOCO_GL=egl`.
- **Flaky nodes**: w005, w006, w008, w009 → client aborts ("Aborted (core dumped)") — all SLURM scripts carry `--exclude=w005,w006,w008,w009`. If a job dies at ~17-20 min with abort in .err, check the node; add to exclusions.
- SLURM reads scripts from disk AT RUN TIME — in-flight edits hit pending jobs. Shell cwd resets between Claude Bash calls — always `cd` or absolute paths. tyro bools are FLAGS (`--no-use_companions`), never `--flag False`.
- jit: shape-determining kwargs (`num_candidates`, `prefix_len`, `num_steps`) are `static_argnames` in `module_jit` (guided_policy.py); actions are QUANTILE-normalized inside the loop (q01/q99 from checkpoint norm stats; `translation_scale=0.05` verified from robosuite OSC config).
- Session ops: ONE blocking background monitor per job (`while squeue -j <id> -h | grep -q .; do sleep 300; done; <print results>`), wake once. If Claude's shell dies with `fork: Resource temporarily unavailable` → the claude process hit its resource cap → **fully kill and restart claude, then --resume** (happened after ~36 h; results/commits all survive).
- Parallel sessions share the account: jobs may be CANCELLED by uid 101024 — pause submissions and ask the user rather than resubmit-warring.
- GitHub pushes: token was one-shot (user advised to revoke); ask user for auth at next push milestone.

## 3. RESULTS STATE (all n=50/task, Tier-GT, single protocol; ±7pp CI on suite aggregates; paired seeds vs baseline)
| Cond | Baseline | r3 | Composed | Reimpl-SOTA(r0.10) | SOTA-pub |
|---|---|---|---|---|---|
| Spatial L1 | 67.0/14.0 | 71.0/23.5(thin) 58.5/44.5(fat) | 53.5/46.0 | 62.0/28.5 | 75.5/77.5 |
| Spatial L2 | 55.5/12.0 | 86.5/49.0 | **85.0/76.5** | 71.5/25.5 | 78.0/75.5 |
| Goal L1 | 51.0/23.0 | 71.5/30.5 | **77.5/51.0** | — | 91.0/94.0 |
| Goal L2 | 66.5/35.0 | **69.0/58.0** | 54.0/63.5 | — | 83.5/82.5 |
| Object L1 | 40.5/14.0 | 51.0/22.5 | **48.0/43.5** | — | 90.5/82.5 |
| Object L2 | 74.0/25.0 | 85.5/40.5 | **83.0/74.5** | — | 81.0/85.5 |
| Long L1 | 58.0/15.0 | 23.0/55.0 | **42.5/71.0** | — | 81.5/82.0 |
| Long L2 | 51.0/16.5 | **47.0/77.0** | 24.5/77.0 | — | 72.0/83.0 |
Key facts: (1) **Spatial L2 = full sweep** (beats baseline, self-run reimpl, AND published SOTA on both axes). (2) Beats baseline both axes on 6/8. (3) **REFEREE FINDING: their method under our >1mm criterion = CAR 25.5–28.5 vs 75.5–77.5 published; TSR uplift (+17pp) reproduces** → their published CAR likely rests on a looser criterion; the honest comparison is vs the self-run reimpl. (4) Composed regressions: Goal L2 TSR 69→54, Long L2 47→24.5 (t2=2%, t3=8%) — suspect corridor dest-misparse on fixture destinations (stove/cabinet without `_pos` keys) and/or fat margins in clutter. (5) Spatial L1 t1 stuck at 14–28% TSR across ALL configs. Composed config = K=8 + ramp + corridor + eef_r 0.12 + companions; r3 = K=1 uniform 0.09 companions no-corridor.
Per-task numbers: in `fgd_n50_*/results/.../results_*.json` and the (to-be-verified) ledger append.

## 4. IMMEDIATE NEXT STEPS (ordered; each with method)
1. **Verify/re-commit the ledger** (§3 table) in the worktree; push both branches when user supplies auth.
2. **Reimpl sweep completion** (task #14): `EEFR∈{0.06? via r_obs proxy—NOTE: reimpl arm approximates their ellipsoid by eef_radius; arms 0.06 and 0.14}` × Spatial I+II, TAG=reimpl_r06/r14. Purpose: bracket their unstated r_obs; strengthens the referee claim. 2 jobs, ~1.5 h each.
3. **Composed-regression diagnosis** (Goal L2, Long L2): read `episodes_*.jsonl` + a few `parse_entities` traces offline. Hypothesis H1: DEST_PATTERN matches "stove/cabinet/drawer" but obs has no `<stove>_pos` → dest=None (harmless) OR matches WRONG object (harmful — check: "put the bowl on the stove", mentioned = bowl only → dest=None, corridor target-only... then why regression? H2: fat 0.12 margins in cluttered Goal/Long L2 scenes + K-selection picking conservative-but-stalling chunks; check activation rate + ETS in results JSONs, and t2/t3 Long L2 videos). Fix candidates: per-suite EEFR, corridor gating by parse confidence, dest whitelist of placeable objects. Re-run ONLY regressed conditions (2 × ~1.5 h).
4. **Per-condition best-config finals**: after (3), lock ONE primary config + report per-condition config ablation honestly (reviewers accept a config table; we have all rows).
5. **DBNR kill-test** (task #18, invention winner 8.5/10, `docs/research/inv_*.json`): add τ (task-velocity removed by repair) + nullspace-headroom diagnostics to `_dcbf_repair` (jnp, ~20 lines), run 1 dev-cell smoke (Spatial L1 t1 + Long L2 t2, n=5). If τ>0 and headroom>0 → implement DBNR (restore task velocity in constraint nullspace, budgeted by damage) — the L1-TSR lever. Kill if τ≈0.
6. **TECV kill-test** (task #19): JVP directional covariance vs K=8 endpoint dispersion correlation, offline on saved obs; if correlated → single-active-face closed-form variant.
7. **Uncertainty-signal validation** (task #16, FREE): correlate K-dispersion (mode-clustered) + denoising-path curvature vs episode outcomes using existing logged rollouts → calibrated epistemic gating thresholds (margins↑/prefix 5→3 under uncertainty). Feeds LIBERO-Safety robustness-suite claims.
8. **ROCS v1** (task #17, spec: `docs/research/OUTCOME_STEERING_SPEC.md`): event parse (gripper-dim release index — VERIFY openpi gripper open/close convention first from episode npz/JSONL), forbidden-region landing test, release-delay clamp + sanctioned-alternative attractor. Targets LIBERO-Safety SSR (safe-alternative EXECUTION vs published refusal-only F1 0.31–0.46).
9. **No-GT tier** (Phase-5; REQUIRED deliverable): swap client's GT reads for perception into the SAME payload — Qwen naming + v17 symbolic FOL selection (reuse fol worktree), RGB-D bbox-median back-projection (`camera_depths=True`, robosuite `get_real_depth_map`, conventions validated in FOL campaign: u=cx+fx·px/pz), bbox×depth/f size, +0.03–0.08 uncertainty inflation → conformal later. Also SigLIP/DINO fast loop + default-deny occupancy (C1-SANCTION) per `SYSTEM_DESIGN.md` D8/D9. Then **E3 identification-swap** (GT vs C1 vs C2, fixed enforcement) = Direction-1 arbiter.
10. **Track B offline** (task #11, BLOCKED on user VLM-budget decision: local Qwen GPU job vs API): E1 labeled ID benchmark (~150–200 scenes from `vlm_inputs/`), E4 SSR refusal-F1 offline.
11. **Breadth**: LIBERO-Safety TSA/FSHOA first method rows + quantify their Appendix-D Eq.A.3 CBF (code+π0.5 weights released, github.com/LIBERO-SAFETY/LIBERO-Safety, same robosuite stack); OopsieVerse RoboCasa behind failure-taxonomy gate (grip-crush excluded — unactuatable).
12. **Paper assembly** (~W6+): Table 1 = baseline | reimpl-SOTA(swept) | ours-GT | ours-no-GT, per condition; ablation table (every flag = a row: schedule/K/companions/corridor); mechanism figures (correction histograms, K-scaling, Long-L1 resurrection, referee criterion analysis, fail-direction E5, safety-tax telemetry); limitations = SYSTEM_DESIGN §4 honesty contract.

## 5. DECISION GATES (pre-registered)
- Composed-regression fix: regressed conditions must return to ≥ r3 levels or per-condition configs stand.
- DBNR/TECV: kill criteria in §4.5/§4.6; each earns ≥+2pp on a dev cell or is cut.
- Direction-1 headline: E2 done (corridor restores liveness 3/4 spatial tasks; t3 = obstacle-on-path family ≤11%); E3 is the arbiter; E5 (inject identification errors, measure error DIRECTION) is the unique figure. Rule: E2-fail ⇒ C2-headline; E2-pass ∧ E5-asymmetry ⇒ C1 fail-safe framing leads.
- SSR co-headline iff offline F1 > 0.5.
- Claims discipline: n=5 routes, n=20 confirms, n=50 claims; paired stats (McNemar/bootstrap) from episodes JSONLs; every table labeled Tier-GT / Tier-SemID / Tier-Percep.

## 6. OPEN PROBLEMS (honest)
Spatial L1 t1 (TSR ≤28 all configs — DBNR's target); obstacle-on-path family (Spatial t3 / Object L1 t1–t2: CAR≈0, geometrically non-negotiable per E2 — curved-approach selection or criterion discussion); Long TSR vs baseline still short at L1 (−15.5) even with corridor; composed Goal-L2/Long-L2 regressions (§4.3); their-r_obs unknown (sweep brackets); no-GT tier unbuilt; API spend limits recur (Anthropic monthly + session — check before VLM-heavy work).

## 7. DIRECTION-2 UPGRADE SHELF (from steering research; each = flag + ablation row)
Built: schedule (outcome-neutral alone — kept), brake-then-push (provable non-worsening under clamp), companions, K=8+prefix certificate, corridor. Shelf: soft hinge-gradient pre-shaping on x1-prediction (steps 2–7, analytic grads, zero NFEs, expected L1 TSR +5–8); Gauss-Seidel multi-pass sweep (sweep-vs-joint ablation); FK particle resampling w/ velocity-derived score (ONLY if feasible-seed-starvation histogram shows starvation — diagnostic is free in results JSONs `feasible_count`); DBNR/TECV (§4); SABRE components (semantic Riemannian metric, cross-chunk noise inheritance, dual-dynamics strength, forward-invariance theorem writeup — each earns its row or is cut). DO-NOT-ADOPT (survey-verified): learned critics/verifiers, twisted-SMC/tree samplers, constraint-embedded flows, Jacobian/Fisher projection, indicator potentials, raw-x_t gradients, any π0.5 fine-tuning (Guidance-as-Teacher = journal).

## 8. WORKING AGREEMENTS (user directives in force)
Terse replies; minimal monitoring tokens (one blocking monitor/job); use gathered in-context knowledge for decisions; every results table labeled with privilege tier and n; both-axes × both-references is THE bar; no commitment between C1/C2 headline until E-protocol; queue courtesy with parallel sessions.

## 9. DOC MAP (read in this order for full context)
1. This file.
2. `docs/research/SYSTEM_DESIGN.md` — target architecture + decision record D1–D14 + honesty contract.
3. `.worktrees/flow-guidance-tier-gt/results_tables/fgd_tier_gt_results.md` — full results ledger w/ per-task tables (verify last append committed).
4. `docs/research/DIRECTIONS_RESEARCH_AND_ARCHITECTURE.md` — novelty ledger + steering related-work map (§3b) + assumptions/limitations.
5. `docs/research/OUTCOME_STEERING_SPEC.md` — ROCS (runtime outcome-conditioned steering: release barriers, conditional keep-outs, sanctioned alternatives; + two-rate SigLIP/DINO context; + epistemic gating discussion in chat history).
6. `docs/research/PROGRESS_LOG.md` — chronology to 2026-07-13 morning (append §3 events).
7. `docs/research/steer_{theory,robotics,synth}.json`, `semsafe_{ident,ground,synth}.json`, `inv_{latest,theory,invA,invB,crit}.json` — raw survey/invention outputs.
8. `docs/superpowers/specs/2026-07-12-*.md` — original brainstorm + direction evaluation (CAVF umbrella, C1/C2/C4, E1–E5).
9. Plan file (Claude): `~/.claude/plans/refactored-sprouting-puffin.md` — approved two-direction comparison plan.
