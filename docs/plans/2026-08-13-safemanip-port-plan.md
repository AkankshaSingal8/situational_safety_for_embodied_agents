# Porting the FOL-elected in-denoising safety steering to SafeManip

Date: 2026-08-13. Status: PLAN (pre-registration). No code written yet.

---

## 0. Which algorithm, which branch (resolving the user's "i think")

**`fol-steering-no-gt` does not exist.** Confirmed against `git ls-remote --heads origin`:
the only steering branches are `new-steering-method`, `Consequence_Steering`,
`safe-success-steering-sota-all-suites`, `viability-steering-tournament-20260713`,
plus the local `flow-guidance-tier-gt`.

**Canonical source of the method:** worktree
`.worktrees/flow-guidance-tier-gt` @ `db06ad7` (branch `flow-guidance-tier-gt`),
work through 2026-08-13. **"no-GT" is a *configuration* of that same pipeline**
(Tier-Percep localization + `--hazard_prior_source vlm` + guard-set), not a branch.
Ledger of record: `results_tables/fgd_tier_gt_results.md`.

**What "our method" actually is** (exactly, so the port doesn't drift):

*Server side* — `openpi_guided/src/openpi/models/pi0_guided.py` (894 L) +
`openpi_guided/scripts/serve_policy_guided.py`. Additive overlay bound onto the
stock JAX Pi0 with `types.MethodType`; upstream openpi is unmodified.
After **every Euler step** of π0.5's flow denoising, the intermediate chunk's
translational dims are read as EEF deltas, rolled out single-integrator, and
repaired so the discrete ECBF chain holds along the horizon:
`B_j >= (1-γ)·B_{j-1}`, `B_j = ||p_j − o|| − r_eff`, γ = 0.9, in-order sweep
(min-norm-flavoured, solver-free, fully jit-able).
On top of the bare repair sit the pieces that the 2026-07-21 five-row board
showed are the actual separator (post-hoc ≈ degenerate in-denoising without them,
2–3× CAR difference):
denoising-time repair schedule (trust early / soft mid / exact late),
superquadric anisotropic obstacle shape (`obstacle_scales`, `sq_eps`),
companion points (hand +0.08 / payload −0.06), **semantic corridor exemption**
(margins relax inside the sanctioned eef→target / eef→dest cylinder),
guard-set M=2 second obstacle, best-of-K with executed-prefix acceptance +
progress/lookahead selection, derived tilt guidance (`tilt_lambda`).

*Client side* — the **election** stack: VLM property priors
(`vlm_pipeline/gen_vlm_priors.py`), scored obstacle identification + fail-safe
guard set (`vlm_pipeline/guard_set.py`), corridor entity election
(target/destination), and percep localization (GroundingDINO + depth) for the
no-GT tier.

**Paper position (ledger 2026-08-12 novelty sweep) — this drives the port's
emphasis:** arXiv 2607.01378 owns the in-denoising DCBF *mechanism* on frozen
π0.5. Our surviving compound claim is **semantic constraint ELECTION**
(FOL/VLM decides *which* entities become constraints, with a fail-safe guard set
and a perception-complete tier). SafeManip is therefore not "another benchmark
for the repair sweep" — it is **the generalization evidence for the election
claim**: a new simulator, a new kitchen scene distribution with many candidate
hazards, and a temporal-logic safety oracle that nobody has ever run an
enforcement method against.

**The one-sentence argument for this benchmark:** a RoboCasa kitchen presents
dozens of candidate hazards where SafeLIBERO presented one — so election, which
was nearly free there, becomes the binding problem here — and
`rc_raw_robot_contact_blocks_rte_grasp_until_sanitized` (raw-contaminated
surfaces must not touch ready-to-eat food; 64 baseline violations) is a hazard
class a purely geometric filter **cannot even express** without the semantic
layer.

---

## 1. Gate A (blocking, RESOLVED — PASS with two caveats)

*Question:* does RoboCasa's action space support the mechanism at all?

Checked `SafeManip/robocasa/robocasa/utils/env_utils.py:134`,
`robocasa/wrappers/gym_wrapper.py:107`,
`robosuite/controllers/config/robots/default_pandaomron.json`,
`openpi/examples/robocasa/eval_single_task.py`,
`openpi/src/openpi/models/pi0_config.py`.

12-dim action vector:

| dims | meaning |
|---|---|
| 0:3 | `action.end_effector_position` → OSC_POSE **delta** position |
| 3:6 | `action.end_effector_rotation` → delta axis-angle |
| 6:7 | `action.gripper_close` |
| 7:11 | `action.base_motion` → mobile base (x,y,yaw) + torso |
| 11:12 | `action.control_mode` |

Controller (`default_pandaomron.json`, `HYBRID_MOBILE_BASE` → arm `OSC_POSE`):
`input_max = 1`, `output_max = [0.05, 0.05, 0.05, ...]`, `input_type = "delta"`.
⇒ **`translation_scale = 0.05`, `cmd_clip = 1.0` — identical convention to every
SafeLIBERO row.** The repair's feasibility model transfers unchanged.

**Caveat A1 — reference frame.** `input_ref_frame = "base"`, and the policy's
state uses `state.end_effector_position_relative`. SafeLIBERO's payload was
world-frame. Fix: transform obstacle / target / dest positions into the robot
base frame at each replan (client has base pose from privileged state or from
`state.base_position`/`state.base_rotation`). This is a client-side change only;
no server math changes.

**Caveat A2 — mobile base is in-band.** Base motion (dims 7:11) displaces the EEF
in world frame and is invisible to a translational-delta rollout. Mitigations, in
order of preference: (i) run the barrier in the *base frame* (A1's fix) so base
translation largely cancels — exact when the base is static, first-order correct
when it isn't; (ii) log commanded base motion during a baseline pass and measure
the base-induced EEF drift over a 5-step executed prefix; (iii) if drift is
material, exclude `NavigateKitchen` from the headline table and declare it.
**This is a pre-registered measurement (Step 2b), not an assumption.**

**Caveat A3 — horizon.** π0.5 RoboCasa config `pi05_pretrain_human300` uses
`action_horizon = 50`, `action_dim = 32`, and the client executes a
`replan_steps = 5` prefix. SafeLIBERO used H = 10 with a longer prefix. A
50-step open-loop single-integrator rollout accumulates large tracking error, so
the repair horizon must be truncated (repair over the first J ≈ 10–15 steps,
pass through the tail) and `inflation_slope` re-tuned. Pre-registered as a small
config sweep (Step 4), not a code rewrite: J is a slice, and the schedule arrays
are already per-step.

---

## 2. Gate B (blocking, RESOLVED — STRONG PASS)

*Question:* is the geometrically-enforceable slice a large enough share of actual
baseline violations to move the headline numbers?

Mined the **already-completed baseline π0.5 runs** (`SafeManip/results/openpi/
pi0.5/pretraining/evals/target/*/rollout_data/*/*_monitor.json`, jobs 435xxxxx),
harvested **2026-08-13 ~22:00**. Caution: the run set was still being written
during the harvest (file count moved 1052 → 1055 → 1059 across passes); numbers
below are a snapshot at n = 1059 and must be re-cut once the jobs drain.

Baseline position: **`task_success` 207/1059 = 19.5 %**; 728/1059 episodes have
≥ 1 violation; **joint `safe_success` (succeeded ∧ zero violations) = 102/1059 =
9.6 %**.

Split by success stratum — this is what rules out a trivial safety win:

| viol. (succ) | viol. (fail) | property | enforceable by our machinery? |
---:|---:|---|---|
| **58** | **534** | `rc_no_forbidden_contact` | **YES** — this *is* a keep-out constraint |
| 27 | 174 | `rc_grasp_remains_safe_until_release` | PARTIAL — payload companion points; drop/settle is not |
| 41 | 112 | `rc_released_object_eventually_settles` | NO — post-release physics |
| 22 | 42 | `rc_raw_robot_contact_blocks_rte_grasp_until_sanitized` | **YES** — contamination = an *elected* keep-out on a semantic class |
| 0 | 40 | `rc_pick_preconditions_safe` | PARTIAL |
| 5 | 33 | `rc_fixture_open_obstacle_retract` | **YES** — retract-path-clear = corridor + keep-out |
| 5 | 29 | `rc_fixture_close_obstacle_retract` | **YES** |
| 0 | 14 | `rc_twist_preconditions_safe` | PARTIAL |
| 2 | 11 | `rc_place_preconditions_safe` | PARTIAL |
| 0 | 10 | `rc_reach_in_fixture_only_when_fully_open` | NO — sequencing |
| ≤1 | ≤3 | liquid/solid transfer settle, press/dump preconditions | NO |

**Verdict: PASS, and the headroom is in the right stratum.** Of the 207 episodes
that *succeeded*, **105 (50.7 %) still violated something** — so there is a large
population of "task done, unsafely done" episodes where the method can raise
safety without buying it by tanking success. The single dominant failure mode
(`rc_no_forbidden_contact`, 592 violations, 56 % of all episodes) is exactly the
property class our method enforces.

Pre-registered headline scope: `rc_no_forbidden_contact` (primary) +
`rc_fixture_{open,close}_obstacle_retract` + `rc_raw_robot_contact_blocks_rte_grasp`
(secondary). Everything else is reported honestly as out-of-scope, with the
guarantee that our method must not *regress* it.

---

## 3. Metric mapping and the comparison set

SafeLIBERO TSR/CAR do not exist here. Pre-register the mapping now:

| role | SafeManip metric | source |
|---|---|---|
| **PRIMARY** | `safe_success_per_property` — the benchmark's own joint success ∧ safe measure (baseline **9.6 %**) | `analysis/RQ1.py` `RAW_COLUMNS` |
| decomposition | `task_success` (baseline **19.5 %**) | `stats.json` |
| decomposition | `safety_satisfaction` = episodes with zero violations of *in-scope* properties | `monitor_metrics.aggregate_monitor_metrics` |
| secondary | `violation_rate`, `exposure_rate`, per-property breakdown | same |

The joint metric is primary deliberately: two separately-reported axes invite the
trivial objection that a filter raises safety by refusing to act, and SafeManip's
authors already defined the joint column. `task_success` and
`safety_satisfaction` are reported as its decomposition, never as the bar.

Acceptance bar carries over from `project-icra-lock-in-goal`: **beat the baseline
on the joint metric AND on both decomposed axes, at both privilege tiers.** SafeManip publishes *policy* evaluations
only — no enforcement method rows exist — so there is no SOTA to reimplement.
Comparison set:

1. **π0.5 baseline** — already run (1055 episodes harvested, no GPU cost).
2. **Post-hoc shield control** — our own machinery with the repair applied after
   generation (the degenerate configuration), for the architecture claim.
3. **2607.01378-mechanism control** — EEF-point-only, K=1, no corridor, no
   schedule, sphere obstacle. This is a *config* of our server
   (`companion_scale=0`, `K=1`, `corridor_relax=0`, flat schedule), not new code,
   so the "mechanism alone vs mechanism+election" ablation is nearly free.
4. **Ours, GT tier** — privileged RoboCasa entity poses (already exported).
5. **Ours, no-GT tier** — VLM priors + guard set + percep localization.

---

## 4. Work plan

### Step 1 — Isolate the workspace (0.5 d)
New worktree `safemanip-steering` off `flow-guidance-tier-gt` @ `db06ad7`.
**Do not touch `.worktrees/flow-guidance-tier-gt` modules**: jobs 43461515/16/17
(FOL ablation) and 43386826 (vanilla-tier) are queued there — the ledger records
job 42405059 killed by exactly this mistake.

### Step 2 — Instrument, measure, decide (2 d, no policy changes)
2a. Copy the guided server overlay (`pi0_guided.py`, `serve_policy_guided.py`)
    into `SafeManip/openpi`; confirm it binds and serves the RoboCasa checkpoint
    with guidance **disabled** (`enabled=0.0`) — must be bit-identical to the
    stock server on a 5-episode smoke. *Gate: bit-exact passthrough.*
2b. **Caveat-A2 measurement:** log commanded base motion + world/base EEF pose
    each step over ~50 baseline episodes; report base-induced EEF displacement
    over a 5-step prefix. *Gate: median drift < 1 cm ⇒ base-frame barrier is
    sound; otherwise scope out mobile tasks explicitly.*
2c. **Hazard ground truth from the monitor:** parse `privileged_information_*.json`
    contact streams for the geoms that trip `rc_no_forbidden_contact`, and build
    a per-scene hazard-entity label set. This is the offline election benchmark —
    the analogue of `safelibero_ident_scenes.jsonl`.

2d. Once the baseline jobs drain, **re-cut the §2 histogram at final n** and
    freeze it as the paired baseline denominator for Step 5 (per-task, per-seed,
    per-episode-index), so every arm is compared against the same episodes.

### Step 3 — Port the election layer (3–4 d) — *the paper's contribution*
3a. Scene capture for RoboCasa (analogue of `ident_scene_capture.py`): RGB from
    the three cameras, depth, entity inventory, task language.
3b. Run `gen_vlm_priors.py` + `guard_set.py` against RoboCasa entity inventories;
    add the RoboCasa semantic classes the kitchen needs — hot surfaces, raw/RTE
    hygiene classes (drives `rc_raw_robot_contact_blocks_rte_grasp`), fixture
    articulation volumes. Reuse the protected-class rule proven on LIBERO-Safety.
3c. **Offline gate (pre-registered, cheap, no GPU rollouts):** top-M guard-set
    recall against the Step-2c labels. *Gate: ≥ 90% top-2 recall on a held-out
    task split, no fail-unsafe empties.* If it fails, raise M before running
    anything — see Step 3d.
3d. **Raise the guard-set ceiling.** `GuidanceParams` hard-codes M = 2
    (`extra_obstacle_*`) as a static jit shape. A kitchen has more than two
    candidate hazards. Generalize to M = K (stacked arrays, `min` over obstacles)
    **before any rollout**, and re-verify SafeLIBERO bit-exactness at M = 2 to
    protect the frozen board.

### Step 4 — Config transfer smoke (2 d)
Repair horizon J ∈ {10, 15, 50}, `inflation_slope` re-tune, corridor radius,
`eef_r`, companion offsets for the Panda-Omron gripper. Small n (5–10 episodes ×
6 tasks), on the tasks with the highest baseline `rc_no_forbidden_contact` rate
(`PickPlaceCounterToCabinet`, `CoffeeSetupMug`, `PickPlaceSinkToCounter`,
`PickPlaceToasterToCounter`, `PickPlaceDrawerToCounter`, `PickPlaceCounterToStove`
— 48/48/48/47/46/45 violating episodes respectively).
*Screen (directional only — n = 5–10/task cannot resolve 5 pp, so this is a
configuration screen, not a gate): keep configs that reduce in-scope violations
with no visible `task_success` collapse. The real gate is Step 5's paired
comparison at full n.*

### Step 5 — Headline runs (3–4 d wall clock, GPU-bound)
Five arms (§3) × the atomic_seen + composite_seen task sets, seed-paired to the
existing baseline episodes, n sized for McNemar/Fisher per
`feedback-validation-not-noise` (n ≥ 20/task, paired). composite_unseen held for
the generalization row if budget allows.

### Step 6 — Analysis + ledger (1 d)
Reuse `SafeManip/SafeManip/analysis/RQ1.py` / `metrics.py` /
`monitor_metrics.py`. Per-property table (in-scope wins, out-of-scope
no-regression), paired significance, and a ledger block in the project's
existing format.

**Total: ~12 working days**, of which ~6 are the election layer + guard-set
generalization (the contribution) and ~4 are GPU wall clock.

---

---

## 4b. EXECUTION LOG — pointer

Execution findings now live in the project's ledger format:
**`SafeManip/results_tables/safemanip_port_ledger.md`** (dated blocks F1–F8).
This section carries only the summary; the ledger carries the evidence.

| # | finding | status |
|---|---|---|
| F1 | openpi core byte-identical; overlay is a clean drop-in (2 server adaptations) | done |
| F2 | RoboCasa checkpoint is **z-score**, not quantile — blocker; resolved by exact affine re-parameterization | gate PASS (1.4e-8) |
| F3 | hazards are **fixtures/architecture**; inscribed ellipsoid calls **47.7 %** of a wall's interior safe | drives box primitive |
| F4 | hazards/episode median 3, p90 **8** → K = 8; **19.3 %** of contacts are arm-link (outside the body model) | pre-registered |
| F5 | Step 2b unsatisfiable from existing data (8-step sampling) | needs client logging |
| F6 | passthrough gate failed; root cause = unconditional denorm round trip (1.8e-7/step → ~3e-3) | fixed, gate PASS |
| F7 | box primitive needs a **per-primitive escape direction**; centre ray is **43× weaker** at 3 m lateral | scope change |
| F8 | `sq_eps` is **dead code** — barrier is ellipsoid, not superquadric | paper-accuracy flag |

**Gates closed:** Step 2a PASS (job 43554693) — denorm equivalence + bit-exact
passthrough with a determinism control. `run_scripts/gate_2a_passthrough.sh` is
the reusable harness and the SafeLIBERO re-verification vehicle.

**Next, in order:** (i) box/slab primitive **+ escape direction** (F7) with
SafeLIBERO re-verification, (ii) M = 2 → K = 8, (iii) fixture→hazard election
against the `forbidden_contact_pairs` labels. Plus one pre-registered smoke:
old-vs-fixed kernel on the SafeLIBERO baseline arm, n=5 seed-paired, to close
F6's unmeasured-magnitude question.

## 5. Risks carried forward

- **Server changes are now two, not one** (see F3/F4): the box/slab primitive and
  M = 2 → K = 8. Both must land before any rollout, and both must leave the
  SafeLIBERO configuration bit-exact (that board is frozen). Verify with the
  same fixed-noise equivalence harness used for Step 2a.
- **H = 50 open-loop rollout** is 5× SafeLIBERO's; if the truncation sweep in
  Step 4 finds no stable J, the fallback is to re-serve with a shorter
  `action_horizon`, which changes the checkpoint's behaviour and would need its
  own baseline — flagged now, not later.
- **Out-of-scope properties** (`released_object_eventually_settles`, 153
  violations) cannot improve and could plausibly *worsen* if guidance makes
  releases jerkier. Reported as an explicit no-regression check, not hidden.
- **API budget** (`feedback-conserve-api-credits`): the election layer needs one
  VLM prior pass per RoboCasa scene class. Cache to JSON in `results_tables/` and
  reuse, exactly as the SafeLIBERO priors are cached.
