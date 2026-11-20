# Scale-up: text-conditioned residual flow on the full SafeLIBERO benchmark

Written 2026-11-02, **before any scale-up result exists**. This file freezes the protocol. Any later change is
dated and marked as a deviation, and nothing in it may be edited after a test-set number has been read.

Starting point: the Spatial study in `README.md`. Its selected recipe, **i1b**, passed all three pre-registered
bars on held-out official episodes 25–49. i1b is the image residual with text, frozen encoder BatchNorm,
oracle-driven plus DAgger data, and 25k steps at bs 128. This scale-up keeps that recipe and changes only what
the new suites and bases force.

## 1. Scope
- **Suites:** `safelibero_{spatial,object,goal,long}` × levels I and II × 4 tasks = 32 scenarios.
  - Goal L1 and L2 evaluate different 4th tasks (benchmark remap, `benchmark/__init__.py:97`).
- **Bases** (the policy whose chunk is corrected):
  1. **Flow base:** the same image flow-matching policy (`policy.py: BasePolicy`) with the same
     hyperparameters, trained per suite on that suite's raw LIBERO demos (`libero_{spatial,object,goal,10}`,
     50 demos × 10 tasks, no obstacles).
  2. **pi0.5 LIBERO** (`pi05_libero` checkpoint, served by `vlsa-aegis/openpi`): one checkpoint for all
     suites.
  3. **pi0.5 base** (`pi05_base`, not fine-tuned on LIBERO): reported as a **base row only**. On Spatial it
     scores TSR 0.00 (README). A residual trained to steer around obstacles cannot create task competence, so
     no residual is trained on it. If it scores TSR > 0.10 on any suite, this decision is revisited and the
     change is dated here.

## 2. Evaluation protocol (all arms, all bases)
- Official SafeLIBERO init states with `env.seed(0)` and **20 settle steps**: the OpenVLA-OFT/FOL driver
  convention, which the README's correction notes as the reference.
  - Max steps: 300 for spatial/object/goal, **550 for long** (paper value; `docs/experimental_setup_appendix.md`).
  - Collision: an obstacle displaced by more than 1 mm.
  - TSR ignores collisions. CAR = no collision. Safe success = success and no collision.
- pi0.5 receives its own preprocessing (180° flip, 224 px resize-with-pad, eef pos + axis-angle + gripper
  state), exactly as `vlm_pipeline/run_safelibero_pi05_eval.py` does. The env protocol is the same as above,
  so pi0.5 and flow-base rows share `env_seed`.
- **Split:**
  - **Test:** official episodes 25–49 of every scenario, i.e. 32 × 25 = **800 episodes per arm**.
  - **Validation / selection:**
    - official episodes 20–24 (closed-loop gate, 32 × 5 = 160 episodes);
    - offline fit on held-out training layouts 20–24.
  - **Training scenes:** generated from layouts of official episodes 0–19 only. Obstacles are re-sampled in
    type, count and position (`scenes.py`).
  - **Deviation from the Spatial study:** that study trained on scenes from layouts 0–24, held out 20–24
    offline and gated on official episodes 20–24, where c0 had seen those layouts. Here layouts 20–24 never
    enter training, so the gate is clean.
- Base arms are also run on all 50 episodes, for comparison with published numbers.
- **Reported at three granularities:** per task (n = 25), per suite × level (n = 100), per suite (n = 200), and
  overall (n = 800). Metrics are TSR, CAR and safe success, each with a 95% Wilson CI. Paired comparisons
  against base use exact McNemar.

## 3. Anti-overfitting rules
1. **The test set (episodes 25–49) runs exactly once per final configuration.** Every number from it is
   reported, including configurations that lose. No configuration is chosen after seeing test numbers.
2. **Hyperparameter tuning uses validation only.**
   - Grid, declared now:
     - residual steps ∈ {25k, 50k}, kept separate from data size, since data grows about 4×;
     - learning rate ∈ {3e-4, 1e-4};
     - encoder lr multiplier ∈ {0.1}, fixed;
     - everything else exactly i1b.
   - Selection criterion, declared now: closed-loop validation safe success (episodes 20–24, pooled over
     suites). Ties within 2 pp go to higher offline validation R².
   - The grid is run with training seed 0 only.
3. **Seeds:** the selected configuration is trained with **3 training seeds, {0, 1, 2}, fixed now**.
   - Every seed is evaluated on the test set and reported individually, along with mean ± sd.
   - There is no best-seed selection. Seeds 0–2 are the only ones that will ever be trained for the headline.
4. **The base policy is a single training run per suite.** It is the object being corrected, not the claim,
   and it is fixed before any residual is trained. Its training uses no obstacle data and no information from
   the test episodes.
5. **Text controls are kept, unchanged from the Spatial study:**
   - wrong-object sentence;
   - no sentence;
   - held-out wordings and nouns;
   - a **no-text residual trained on identical data with the same 3 seeds**.
6. The oracle uses privileged sim state and is reported as the teacher's upper bound, never as a method.

## 4. Pass bars (per base, fixed now)
On the test set, pooled over all 32 scenarios, with seed-mean metrics. Paired tests use each seed's rollouts
and must hold for **all three seeds**:
- **(a)** Residual with the correct sentence vs base: CAR higher, McNemar p < 0.05. TSR no lower than base − 3 pp.
- **(b)** Correct vs wrong-object sentence: CAR higher, p < 0.05.
- **(c)** Correct sentence vs the no-text residual: CAR higher, p < 0.05. Failing (c) means the safety gain is
  not attributable to the sentence.
- Per-suite results are reported for every suite. A suite where (a) fails is listed as a failure, not dropped.
  Per-suite p-values are Holm-corrected over the 4 suites.

## 5. Phases and go/no-go
| phase | what | go criterion |
|---|---|---|
| P0 | environments on pleiades; retrain the Spatial flow base | Spatial base TSR on official episodes 0–24 (settle 10, seed 0) within ±8 pp of the PSC run (`runs/gate/base`) |
| P1 | flow base for object, goal, long | none (reported as is); also obstacle-removed TSR on episodes 0–9 |
| P2 | scenes + oracle gate per suite and base (episodes 0–19 layouts; gate on episodes 0–19) | as README: oracle CAR ≥ base + 20 pp, TSR ≥ base − 5 pp. A suite failing it gets no residual (its teacher can't help), and that is reported |
| P3 | round-0 oracle data → c0-style feature residual → DAgger data → i1b image residual (text and no-text) | none |
| P4 | grid on validation (seed 0), then seeds 1 and 2 of the selected configuration | — |
| P5 | single test run of everything: flow base and pi0.5 track | bars in §4 |

The flow-base track runs first. The pi0.5 track reuses the scenes, oracle and recipe. Its residual reads the same
frozen flow-base encoder features as the flow track (the same suite's encoder, applied to the same 128 px
frames) together with its own trainable image encoders. The base chunk input is pi0.5's chunk, normalized with
the flow base's action range. Only the chunk length changes, to pi0.5's.

## 6. Log (dated; written before any residual number on any suite)

**2026-11-02. P0 passed.** Spatial base retrained on pleiades, official episodes 0–24, settle 10:
TSR 0.480 vs PSC `runs/gate/base` 0.445 (+3.5 pp, inside ±8 pp). CAR 0.090 vs 0.085.

**2026-11-02. P1 flow bases** (50k steps, seed 0; all 50 official episodes, settle 20; obstacle-removed on episodes 0–9):

| suite | TSR I / II | CAR I / II | safe I / II | no-obstacle TSR I / II |
|---|---|---|---|---|
| spatial | 0.520 / 0.420 | 0.055 / 0.105 | 0.040 / 0.105 | 0.850 / 0.775 |
| object | 0.010 / 0.510 | 0.010 / 0.545 | 0.000 / 0.360 | 0.000 / 0.675 |
| goal | 0.075 / 0.290 | 0.080 / 0.040 | 0.000 / 0.015 | 0.400 / 0.225 |
| long | 0.035 / 0.135 | 0.305 / 0.220 | 0.000 / 0.055 | 0.400 / 0.225 |

Object level I is 0/40 even with obstacles removed. Its layouts put the target object outside the LIBERO-Object
demo distribution (e.g. pudding at y = −0.44, milk at y = −0.39), so this is a coverage limit of a
demo-trained base, not an obstacle effect. **Decision: bases are not retrained or re-tuned.** P1 says
"reported as is", and retraining after seeing these numbers would be base-side tuning that the protocol
does not allow. Where base TSR ≈ 0 (object I, long I), safe success cannot rise, and CAR is the informative
metric. This is stated with the results rather than hidden. The pi0.5 LIBERO base is the stronger-base track
for these suites.

**2026-11-02. P2 gates (flow base, official episodes 0–19, n = 160 per arm)**, by the §5 rule (oracle CAR ≥ base + 20 pp
and TSR ≥ base − 5 pp):

| suite | base TSR / CAR | oracle TSR / CAR | ΔCAR | ΔTSR | gate |
|---|---|---|---|---|---|
| spatial | 0.431 / 0.069 | 0.525 / 0.706 | +63.7 | +9.4 | pass |
| object | 0.275 / 0.269 | 0.325 / 0.787 | +51.8 | +5.0 | pass |
| goal | 0.206 / 0.063 | 0.131 / 0.306 | +24.4 | **−7.5** | **fail** |
| long | 0.081 / 0.294 | 0.031 / 0.756 | +46.3 | −5.0 | pass, at the boundary (McNemar on TSR b/c 1/9, p = 0.02) |

Goal therefore gets no flow-base residual: `res_goal` (job 56298) was cancelled during its gate stage. Goal
is reported as a gate failure, with its base and oracle rows, and it is excluded from the pooled config
selection (`select_config.py --suites spatial,object,long`). Long is kept because the pre-registered rule
passes it as written. Its oracle's significant TSR loss is reported next to its results, and the rule was
not re-read in either direction.

**2026-11-02. pi0.5 base rows, first attempt void (infrastructure).** Jobs 56300–56307: the openpi client's default 20 s
websocket keepalive closed the connection while the server was blocked (first-call JAX compile with 4 workers queued).
Every episode then errored, except 100 rows of long/pi05_libero. Fix in `pi05_runner.py`: no client pings, one
reconnect on a closed socket. The outputs were moved to `eval_official_errored_20261102` (none were used) and all 8
arms were resubmitted unchanged (jobs 56340–56347).

**2026-11-02. pi0.5 base rows** (jobs 56340–56347, all 50 official episodes, 0 errors; full tables in `report_bases.md`).
TSR I / II: pi05_libero spatial 0.675/0.555, object 0.405/0.695, goal 0.490/0.710, long 0.625/0.490.
pi05_base scores TSR 0.000 on every scenario. As pre-registered (revisit only if TSR > 0.10), it stays a base row
only and gets no residual. Its CAR is high (0.20–0.78) only because it doesn't approach the objects.
The pi05_libero residual track (`BASE=pi05`, server started inside the job) was submitted for all 4 suites
(jobs 56365–56368). Its gates are run per suite and per base, so Goal gets its own pi0.5 gate.

**2026-11-02. P2 gates, pi05_libero base** (official episodes 0–19, n = 160 per arm, same §5 rule):

| suite | base TSR / CAR | oracle TSR / CAR | ΔCAR | ΔTSR | gate |
|---|---|---|---|---|---|
| spatial | 0.662 / 0.138 | 0.744 / 0.706 | +56.8 | +8.2 | pass |
| object | 0.544 / 0.194 | 0.669 / 0.769 | +57.5 | +12.5 | pass |
| goal | 0.613 / 0.181 | 0.537 / 0.400 | +21.9 | **−7.5** | **fail** |
| long | 0.531 / 0.131 | 0.375 / 0.706 | +57.5 | **−15.6** | **fail** |

`respi_goal` (56377) and `respi_long` (56378) were cancelled. pi0.5 residual selection pools spatial and object only.
The flow-base track (`res_long` 56299) finished its select stage. Its selection waits for spatial and object.
Goal fails the gate under both bases. The geometric detour oracle costs too much task success on Goal (and on Long
under pi0.5), so no residual trained to imitate it is run there.
