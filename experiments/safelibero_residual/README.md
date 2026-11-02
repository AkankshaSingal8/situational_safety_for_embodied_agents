# Text-conditioned residual flow on SafeLIBERO-Spatial (obstacle avoidance)

Scale-up of `experiments/residual_flow_poc` (dummy MuJoCo env) to SafeLIBERO-Spatial,
levels I and II. Started 2026-10-31.

1. **Base**: an image flow-matching policy trained from scratch on the LIBERO-Spatial
   demos (10 tasks, 50 demos each, no obstacles), evaluated on SafeLIBERO-Spatial.
2. **Residual**: a flow model over a correction to the base chunk's xyz deltas,
   conditioned on the base's frozen features, the base chunk and a sentence ("don't hit
   the moka pot"). Targets come from a geometric detour oracle that uses privileged sim
   state. The sentence decides which object the detour avoids.

## Decisions (user, 2026-10-31)
- Image-based base; more SU to be arranged (32 SU on `cis240028p` at the start).
- Residual targets come from a geometric detour oracle, not best-of-K or the FOL filter.
- Text picks which object to avoid: training scenes hold 1–2 of the six obstacle types,
  and the oracle target differs depending on which one is named.
- Scope: SafeLIBERO-Spatial, levels I and II.

## Protocol
- **Official episodes.** `env.seed(0)` and the official init states, with 10 settle steps
  and 300 max steps. A collision is the obstacle moving more than 1 mm, as in
  `vlm_pipeline/run_safelibero_openvla_oft_eval.py`. TSR ignores collisions;
  "safe success" requires success with no collision.
  - **Correction (2026-11-01):** the protocol only partly matches the reference drivers.
    - Settle steps: 10 here, against 20 in both the OpenVLA-OFT and pi0.5 drivers.
    - Env seed: 0 here, the same as the OpenVLA-OFT driver; the pi0.5 driver
      (`vlm_pipeline/run_safelibero_pi05_eval.py`) uses `env.seed(7)`.
    - `eval_sl.py --env_seed 7 --settle 20` reproduces the pi0.5 protocol; the base rerun under it is
      `runs/base/eval_official_pi05proto`.
    - All residual comparisons in this README are paired within this protocol, so they are unaffected.
- **Split.** Residual training scenes reuse the layouts of official episodes 0–24 (bowls,
  plate and so on), with obstacles re-sampled in type, count and position. **Official
  episodes 25–49 are the held-out test** (200 episodes: 4 tasks × 2 levels × 25). The base
  never sees obstacles.
- **Images.** Raw env orientation at 128 px, the same as the raw hdf5. The env frame
  correlates 0.92 with the demo frame, against −0.24 when flipped.
- **Action-to-motion gain** for the oracle is 0.0107 / 0.0120 / 0.0111 m per unit per step
  (x/y/z), fitted on the demos with correlation 0.95.

## Pass bars, fixed before any result
| stage | bar |
|---|---|
| Base on SafeLIBERO | Pooled TSR on all 400 official episodes at least the OpenVLA-OFT number on disk (42% / 35%, pooled 38.5%), and TSR with the obstacle removed of at least 70%. |
| Oracle gate (official episodes 0–24) | Oracle CAR at least base + 20 pp (paired McNemar p < 0.05), and TSR no more than 5 pp below base. If it fails, stop and report: the residual can't beat its teacher. |
| Residual, held-out episodes 25–49 | (a) Residual with the correct sentence: CAR above base, McNemar p < 0.05, and TSR at least base − 3 pp. (b) Text is used: CAR with the correct sentence above CAR with a wrong-object sentence, p < 0.05. (c) On two-obstacle held-out scenes, the named obstacle is hit less often than the unnamed one. |

The no-text residual is reported next to (a). If it matches the text residual on (a),
(a) is not evidence of text conditioning, and only (b) and (c) are.

## Results

### Base (2026-11-01)
- **Official SafeLIBERO-Spatial, all 400 episodes:** TSR 0.48 (level I) / 0.40 (level II), pooled 0.44.
  That's above the OpenVLA-OFT number on disk (0.385), so the bar passes. CAR is 0.065 / 0.125: the base hits
  the obstacle in most episodes.
- **Obstacle removed (episodes 0–9, n=80):** pooled TSR 0.79 (0.825 / 0.75), which passes the 0.70 bar.
  - Caveat: this run also removes the level-II platform (`box_base_1`).
  - An earlier run that left the platform in got 0.65 (0.80 / 0.50).

### Base vs pi0.5 under the same protocol (all 400 official episodes, `env.seed(7)`, 20 settle steps)

| policy | I TSR | I CAR | I safe&succ | II TSR | II CAR | II safe&succ |
|---|---|---|---|---|---|---|
| pi0.5 LIBERO checkpoint (`pi05_libero`) | 0.670 | 0.140 | 0.140 | 0.555 | 0.120 | 0.115 |
| pi0.5 base checkpoint (`pi05_base`, no LIBERO fine-tune) | 0.000 | 0.690 | 0.000 | 0.000 | 0.215 | 0.000 |
| base flow policy (ours, `runs/base/eval_official_pi05proto`) | 0.435 | 0.060 | 0.040 | 0.445 | 0.125 | 0.120 |

- **Sources for the pi0.5 rows:**
  - Driver: `vlm_pipeline/run_safelibero_pi05_eval.py`.
  - Slurm scripts: `slurm/safelibero/eval_pi05_safelibero.slurm` and `eval_pi05base_safelibero.slurm`.
  - Per-episode `success=…, collision=…` lines: `slurm_logs/pi05_spatial_L{I,II}_4148372{6,7}.err` and `slurm_logs/pi05base_eval_5010705{8,9}.err`.
- **`pi05_base`** doesn't do the task. Its CAR measures an arm that doesn't reach the bowl, not avoidance.

### Oracle gate (official episodes 0–24, n=200, c0 = auto up/sideways + arm checkpoints)
CAR 0.085 → 0.74 (131/0), TSR 0.445 → 0.51 (46/33, p=0.18), safe&success 0.08 → 0.425. Passed.
Configs c1–c3 were tuned on episodes 0–9 only (`runs/gate/tuning.json`).

### Residual c0, held-out official episodes 25–49 (n=200, one test run)

**Data:** 907 episodes on the cached training scenes (`data/residual_c0`), giving 21,807 chunks and 71k rows; 46% of the present-obstacle rows have a nonzero target.

**Model:** frozen pooled base features + base chunk + BGE text, trained 20k steps.

| arm | TSR | CAR | safe&succ | noContact | vs base: TSR b/c, p | vs base: CAR b/c, p |
|---|---|---|---|---|---|---|
| base | 0.385 | 0.080 | 0.075 | 0.110 | | |
| oracle (teacher) | 0.535 | 0.745 | 0.455 | 0.775 | 56/26, 1e-3 | 135/2, 1e-37 |
| residual, correct sentence | 0.535 | 0.225 | 0.180 | 0.275 | 44/14, 1e-4 | 31/2, 1e-7 |
| residual, held-out wording | 0.535 | 0.250 | 0.210 | 0.275 | 46/16, 2e-4 | 37/3, 2e-8 |
| residual, wrong object | 0.445 | 0.130 | 0.120 | 0.150 | 31/19, 0.12 | 11/1, 6e-3 |
| residual, no sentence | 0.455 | 0.095 | 0.090 | 0.115 | 34/20, 0.08 | 3/0, 0.25 |
| no-text residual | 0.600 | 0.130 | 0.110 | 0.160 | 59/16, 6e-7 | 11/1, 6e-3 |

- **(a) passes.** CAR is above base (p=1e-7) and TSR is above base (+15 pp).
- **(b) passes.** Correct vs wrong sentence: CAR 26/7, p=0.0013.
- **Text vs no-text residual:**
  - CAR 0.225 vs 0.13 (24/5, p=5e-4), so the safety gain depends on the text.
  - The no-text residual has the *higher* TSR (0.60 vs 0.535; 25/38, p=0.13).
  - The wrong-sentence and no-sentence arms also gain TSR.
  - So the task-success gain mostly does **not** depend on the text. A residual trained on oracle-driven states improves the base policy whatever the sentence says.
- **Distillation gap:** the residual recovers only about 22% of the teacher's CAR gain (0.08 → 0.225 against 0.745).

**Two-obstacle held-out scenes** (layouts 25–49, n=177 rather than 200 because scene generation hit its try cap; the sentence names the first obstacle):

| arm | TSR | CAR | named hit | other hit |
|---|---|---|---|---|
| base | 0.469 | 0.203 | 0.644 | 0.503 |
| oracle | 0.463 | 0.237 | 0.384 | 0.627 |
| residual, correct sentence | 0.565 | 0.209 | 0.565 | 0.554 |
| no-text residual | 0.542 | 0.226 | 0.616 | 0.531 |

- **(c) fails.** Named 0.565 vs other 0.554.
- The bar is confounded by placement: in base, the named obstacle is hit more often (0.644 vs 0.503), because the first obstacle goes on the platform when the scene is elevated.
- The oracle passes (c), so the bar is reachable.
- *Post hoc, not a test:* on flat scenes (n=126), the residual's named-hit rate is 0.468 vs 0.556 for the other obstacle, against 0.556 vs 0.484 in base. Elevated scenes (n=51) are where it fails: named 0.80 against base 0.86.

### After c0 (gate episodes only, no test-set runs)

**Training-set fit of c0** (`diag_fit.py`, present rows with a nonzero target):

| model | R² | pred/target magnitude | cosine |
|---|---|---|---|
| text residual | 0.57 | 0.76 | 0.65 |
| no-text residual | 0.17 | 0.30 | 0.40 |

- The sentence carries real information.
- Even on training rows, the detours point only partly the right way. The pooled base features look like the ceiling here.

**Text classifier-free guidance** (official episodes 0–24, n=200, correct sentence):

| guidance w | TSR | CAR | safe&succ |
|---|---|---|---|
| 1 | 0.56 | 0.245 | 0.21 |
| 2 | 0.465 | 0.33 | 0.205 |
| 3 | 0.39 | 0.42 | 0.18 |

Guidance trades task success for collision avoidance one-for-one. **It was dropped and never run on the test set.**

**i1 / i1b: image residual + DAgger states** (selection only)

**Data:** two collections with images on the cached training scenes: oracle-driven, plus the c0 residual driving (DAgger states). That's 42,873 chunks and 141k rows. Every chunk from training-scene layouts 20–24 is held out (8,554 chunks).

**Image residual:** its own two ResNet-18 encoders, initialised from the base trunks with lr ×0.1. It keeps the 4×4 layer-4 map and pools it with text-queried attention, on top of the frozen base features, the base chunk and the text.

**i1 bug:** the image residual's EMA predicted almost zero. During training its BatchNorm layers used batch statistics, but the EMA kept the base policy's running statistics.
- Re-scoring with batch statistics gives training R² 0.46, which confirms it.
- i1b freezes encoder BN (`ResidualFlowImg.freeze_bn`), so training and inference behave the same.

Offline fit (present rows with a nonzero target) and closed-loop gate on official episodes 20–24 (n=40, the held-out layouts):

| model | train R² | held-out R² | held-out magnitude ratio | held-out cos | gate TSR | gate CAR | gate safe&succ |
|---|---|---|---|---|---|---|---|
| base | | | | | 0.45 | 0.15 | 0.15 |
| c0 feature residual | (0.57) | n/a | | | 0.45 | 0.30 | 0.275 |
| i1 feature residual, text | 0.43 | 0.00 | 0.32 | 0.24 | 0.525 | 0.275 | 0.275 |
| i1 feature residual, no-text | 0.14 | 0.00 | 0.21 | 0.20 | | | |
| **i1b image residual, text** | 0.47 | **0.09** | **0.56** | **0.39** | **0.60** | **0.55** (16/0, p=3e-5) | **0.375** (10/1) |
| i1b image residual, text, wrong object | | | | | 0.475 | 0.10 | 0.10 |
| i1b image residual, no-text | 0.15 | 0.03 | 0.25 | 0.26 | 0.55 | 0.225 | 0.20 |

- The c0 row is **not** a held-out comparison on these episodes: c0 was trained on scenes from layouts 20–24, and it generated the DAgger states. It only affected selection, not the test.
- The pooled base features do not generalise to new layouts: held-out R² is about 0.
- Trainable image encoders with text-queried spatial pooling do generalise partly.
- **Selected: i1b image residual (text), as trained, without retraining on layouts 20–24.**
- Several variants were looked at on the gate (c0 at w=1/2/3, i1 feature, i1 image, i1b image), so the gate number is optimistic. The single test run below settles it.

**i1b (image residual, encoder BatchNorm frozen; `run_img2.slurm`)**
- **Held-out-layout fit:** val R² 0.087, ratio 0.56, cosine 0.39. That is the same as i1 after BN recalibration. The no-text model gets val R² 0.03.
- **Gate, official episodes 20–24 (n=40):**

  | arm | TSR | CAR | safe&succ |
  |---|---|---|---|
  | correct sentence | 0.60 | 0.55 (16/0, p=3e-5) | 0.375 (p=0.012) |
  | wrong sentence | 0.475 | 0.10 | 0.10 |
  | no-text model | 0.55 | 0.225 (p=0.45) | 0.20 |

- **Selected for the single test run:** `run_test.slurm` TAG=i1b, job 50108791, submitted by a parallel session.
  - A duplicate job, 50108805, was cancelled after it had re-run 15 held-wording and 60 wrong-sentence episodes into the same directories.
  - `compare.load` keeps the first row per episode in file order. That rule was fixed before looking at these results, and `compare.py` prints the duplicate count.
- **The unseen-object test** (`run_unseen.slurm`, job 50108806; results below):
  - The same recipe trains with milk carton and red mug fully excluded: no scene containing them and no sentence naming them.
  - It's evaluated on official episodes 25–49, split into unseen and seen obstacle types.

### Residual i1b, held-out official episodes 25–49 (n=200, one test run, 2026-11-01)

Same episodes and seeding as the c0 test. The base and oracle rows reuse those runs. McNemar statistics are b/c against base.

| arm | TSR | CAR | safe&succ | noContact | TSR b/c, p | CAR b/c, p |
|---|---|---|---|---|---|---|
| base | 0.385 | 0.080 | 0.075 | 0.110 | | |
| oracle (teacher, privileged state) | 0.535 | 0.745 | 0.455 | 0.775 | 56/26, 1e-3 | 135/2, 1e-37 |
| c0 feature residual, correct sentence | 0.535 | 0.225 | 0.180 | 0.275 | 44/14, 1e-4 | 31/2, 1e-7 |
| **i1b image residual, correct sentence** | **0.570** | **0.450** | **0.345** | **0.480** | 54/17, 1e-5 | 76/2, 2e-20 |
| i1b, held-out wording or noun | 0.580 | 0.435 | 0.360 | 0.495 | 54/15, 3e-6 | 73/2, 2e-19 |
| i1b, wrong object | 0.420 | 0.095 | 0.085 | 0.115 | 30/23, 0.41 | 4/1, 0.38 |
| i1b, no sentence | 0.385 | 0.085 | 0.075 | 0.115 | 22/22, 1 | 2/1, 1 |
| i1b no-text residual | 0.485 | 0.175 | 0.135 | 0.190 | 39/19, 0.012 | 20/1, 2e-5 |

Per level, correct sentence: level I TSR 0.57 / CAR 0.39 (base 0.47 / 0.05); level II TSR 0.57 / CAR 0.51 (base 0.30 / 0.11).

- **(a) passes.** CAR 0.08 → 0.45 (p=2e-20), TSR 0.385 → 0.57 (+18.5 pp, p=1e-5), safe&success 0.075 → 0.345.
- **(b) passes.** Correct vs wrong object: CAR 73/2 (p=1.5e-19), TSR 49/19 (p=4e-4).
- **Text vs no-text residual** (same data and architecture): CAR 61/6 (p=1.5e-12), safe&succ 48/6, TSR 44/27 (p=0.057).
  - Unlike c0, the wrong-object and no-sentence arms now sit at base level on both TSR and CAR. So both the safety gain and most of the task gain come from the sentence.
- **Held-out wordings and nouns** (for example "stovetop espresso maker" or "give the X plenty of room") match the training wordings.

**Two-obstacle held-out scenes** (n=177, the sentence names the first obstacle):

| arm | TSR | CAR | named hit | other hit | within scene, other-only / named-only hit |
|---|---|---|---|---|---|
| base | 0.469 | 0.203 | 0.644 | 0.503 | 27/52 (p=0.007, named hit more) |
| oracle | 0.463 | 0.237 | 0.384 | 0.627 | 67/24 (p=7e-6) |
| c0 residual | 0.565 | 0.209 | 0.565 | 0.554 | 40/42 (p=0.91) |
| **i1b residual, correct sentence** | 0.486 | 0.254 | **0.441** | **0.599** | **54/26 (p=0.002)** |
| i1b no-text residual | 0.503 | 0.209 | 0.610 | 0.576 | 32/38 (p=0.55) |

- **(c) passes.** Named 0.441 < other 0.599. The within-scene paired test reverses base's bias (p=0.002); the no-text model shows no preference.
- Like the oracle, the residual partly pushes collisions onto the unnamed obstacle (other hit 0.503 → 0.599). Overall CAR in two-obstacle scenes changes little (0.20 → 0.25, p=0.12).

**Write collision in the test directories.**
- Another session's job (`slres_test`, 50108805, 19:39–19:45, cancelled) ran the same i1b arms while this test was running. It appended to `runs/test/res_i1b_held` (15 duplicate episode keys) and `res_i1b_wrong` (60 duplicate keys).
- The duplicates have the same sentences, and their outcomes differ only by sampling noise. I couldn't recover which checkpoint that job loaded.
- `compare.py` keeps the first row for each key in file order (rule fixed before results were read; the table above uses it). First-row vs last-row metrics differ by at most 1.5 pp (held TSR 0.58 / 0.57; wrong CAR 0.095 either way).
- The correct-sentence, no-sentence, no-text and base arms have no duplicates, so bar (a) is unaffected.
- Excluding every duplicated key (125 clean episodes):
  - Correct sentence vs base: CAR 0.42 vs 0.10 (42/1, p=1e-11).
  - Correct vs wrong object: CAR 0.42 vs 0.12 (40/2, p=4e-10), TSR 0.54 vs 0.42 (p=0.03).
- The raw files are left untouched.

**All bars pass on the held-out test for i1b.** The c0 test run (above) passed (a) and (b) and failed (c). Both test runs are reported.

**Remaining gaps:**
- The residual gets about 55% of the teacher's CAR gain (0.45 vs 0.745).
- Held-out-layout fit is still low (R² 0.09). Layout diversity (only 25 training layouts) is the obvious next lever.
- Single training seed.

**Test-set hygiene from here on:** every new variant is selected on official episodes 0–24 or the training scenes. Episodes 25–49 run once per chosen variant, and every test run is reported, c0 included.

### Unseen obstacle types (i1b recipe, milk carton + red mug excluded from training; official episodes 25–49, one run)

Models `runs/res_text_img_uo`, `runs/res_notext_img_uo`: no training scene contains either object and no training sentence names them. Fit on held-out layouts + unseen objects: text model R² −0.03 (train 0.66), so poor transfer was expected before the rollouts.

| arm | unseen types (n=75) TSR | CAR | safe&succ | seen types (n=125) TSR | CAR | safe&succ |
|---|---|---|---|---|---|---|
| base | 0.347 | 0.067 | 0.067 | 0.408 | 0.088 | 0.080 |
| text model, correct sentence | 0.347 | 0.107 (4/1, p=0.38) | 0.093 | 0.488 | **0.320** (31/2, p=1e-7) | 0.248 |
| text model, no sentence | 0.280 | 0.053 | 0.053 | 0.464 | 0.104 | 0.080 |
| no-text model | 0.480 | 0.107 | 0.093 | 0.608 | 0.200 (16/2) | 0.176 |

- **Text-conditioned avoidance does not transfer to unseen object types.** On milk carton / red mug the correct sentence gives no significant CAR gain over base (0.067 → 0.107) and no TSR gain.
- On seen types the same model reproduces the main result at lower strength (CAR 0.09 → 0.32), so the failure is specific to the new objects, not a broken model.
- Unseen *wordings and nouns for seen objects* do transfer (main test, held-wording arm). The limit is visual/semantic grounding of new object categories from only 4 training types (moka pot, storage box, wine bottle, yellow book).
- **Second training of the recipe (replication check against the main test):** the CAR edge of text over the no-text model replicates (seen types 0.32 vs 0.20). The TSR edge does not: the no-text model has higher TSR (0.608 vs 0.488 on seen types). Together with main-test TSR 44/27 (p=0.057), the claim "text raises TSR" holds against base and the no-sentence arm, but not robustly against a model trained without text.
- **Input deviation from the requested spec:** the residual takes images and the safety sentence as text (BGE). The task instruction enters only through the base policy's task-ID embedding (10 LIBERO-Spatial tasks, `policy.py:115`), which reaches the residual through the frozen base features. It is not read as text.
- Next lever: more obstacle types in training (BDDL edits, scene regeneration, recollection), then a scaling curve of unseen-type CAR against the number of training types. Not run: about 13 SU left.

## Files
| file | role |
|---|---|
| `prep_data.py` | pack the hdf5 demos (no-op filter, joints + gripper proprio) |
| `policy.py` | base image flow policy (2× ResNet-18, 1-D U-Net flow head) and residual flow |
| `train_base.py`, `train_residual.py` | training |
| `sl_env.py` | env construction, obstacles, AABBs, teleporting |
| `scenes.py` | generated obstacle scenes (train layouts 0–24, test layouts 25–49) |
| `oracle.py` | geometric detour for the named object |
| `collect_residual.py` | base + oracle closed loop; records targets for several sentences per state |
| `eval_sl.py`, `controllers.py`, `residual_ctrl.py` | parallel closed-loop evaluation, arms base / oracle / residual |
| `texts.py` | training and held-out wordings and nouns; BGE-base embeddings |
| `checks.py` | fast correctness checks |
