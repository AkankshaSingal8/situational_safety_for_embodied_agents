# Text-conditioned residual flow on SafeLIBERO: research notes

Last updated 2026-11-02. This file explains why the experiment exists, what it tests, how it is built, and which choices were made along the way and why. `README.md` in this directory holds the full result tables, and this file links to them rather than copying every row.

## Research direction

The project's earlier method (`fol_safety_filter/`) filters a policy's actions at run time with control barrier functions built from VLM predicates. It works, but it is a hand-built pipeline on top of the policy, and every new kind of hazard needs a new rule or geometry. The direction here moves the safety behaviour into a learned model: a flow policy that takes the camera images, the task instruction and a plain-language safety request ("don't hit the moka pot") and produces actions that respect the request.

The work went in two steps. The proof of concept in `experiments/residual_flow_poc/` used a toy MuJoCo scene with primitive shapes and a scripted base path. It showed that a residual flow conditioned on a sentence embedding can pick which object to avoid, and that it fails on composed sentences ("avoid A and B" scored 0/1200). This directory scales the single-constraint case up to a real benchmark: SafeLIBERO-Spatial obstacle avoidance at levels I and II, with a learned image policy as the base.

## Hypotheses

H1. A flow policy trained only on obstacle-free LIBERO-Spatial demos completes SafeLIBERO-Spatial tasks at a rate comparable to published VLA baselines, but hits the obstacle in most episodes. This sets up the problem: task skill without safety.

H2. A residual flow trained on top of that base, with targets from a privileged detour teacher and a safety sentence as input, raises the collision avoidance rate (CAR) on held-out episodes without lowering task success (TSR). Because the base often fails by knocking the obstacle into the target, TSR may rise too.

H3. The residual reads the sentence. With the wrong object named, or no sentence, it should behave like the base. With two obstacles present, the named one should be hit less often than the other.

H4. The text-conditioned residual transfers to new wordings and nouns for objects it has seen.

H5 (not yet tested). Transfer to object types never seen in training grows with the number of object types in training. A first test with 4 training types says it does not transfer at that scale (see Results). The object-scaling study is built to test H5 directly.

## Approach

### Base policy

Two ResNet-18 encoders (agent view and wrist, 128 px) feed a 1-D U-Net flow-matching head that outputs a chunk of 16 actions (7-D: xyz, rotation, gripper). The task enters as a learned embedding over the 10 LIBERO-Spatial task IDs, together with the joint and gripper proprioception. It was trained for 50k steps on the 500 LIBERO-Spatial demos (10 tasks × 50), about 70 minutes on one V100. It never sees an obstacle during training. Code: `policy.py`, `train_base.py`, `prep_data.py`.

### Detour teacher (oracle)

`oracle.py` takes the base chunk and the privileged sim state and computes where the end effector will go, using a per-axis action-to-motion gain fitted on the demos (0.0107 / 0.0120 / 0.0111 m per unit per step, correlation 0.95). The gripper, a ring around it for a held bowl, and four arm links are treated as spheres. Where one of them would enter the named object's bounding box (inflated by a margin), the teacher bends the path up or sideways, whichever is the smaller move, and smooths the bend. Only xyz changes; rotation and gripper are left alone. If the sentence names an object that is absent, or names nothing, the target is zero.

### Residual flow

The residual is a second flow model whose output is an xyz correction to the base chunk. Two versions were tested on the held-out episodes:

- c0 takes the frozen pooled base features, the base chunk and a BGE-base sentence embedding (768-D).
- i1b adds its own two ResNet-18 encoders, initialised from the base encoders and trained at 0.1× the learning rate, with BatchNorm frozen. It keeps the 4×4 layer-4 feature map and pools it with attention queried by the sentence (`TextSpatialPool`), so the sentence can pick out the region of the image that holds the named object.

At run time the executed chunk is the base chunk plus the sampled correction. Code: `policy.py` (`ResidualFlow`, `ResidualFlowImg`), `train_residual.py`, `residual_ctrl.py`.

### Training data

`scenes.py` generates training scenes by taking the layouts (bowls, plate, ramekin, fixtures) of official episodes 0–24 and re-sampling the obstacles: which of the six types, one or two of them, and where. `collect_residual.py` runs the base with the teacher in closed loop. At each chunk it records the teacher's target for several sentences at the same state: each present obstacle (its detour), one absent obstacle (zero) and no sentence (zero). Because the same image gets different targets depending on the sentence, the model can only fit the data by reading the sentence.

The i1b data has two parts. One is collected with the teacher driving. The other is a DAgger round where the c0 residual drives and the teacher still labels, so the model also sees the states its own corrections lead to. Together that is 42,873 chunks.

## Methodology

### Metrics

These follow the SafeLIBERO protocol in `vlm_pipeline/run_safelibero_openvla_oft_eval.py`. A collision is any obstacle moving more than 1 mm. TSR counts task completion and ignores collisions. CAR is the fraction of episodes with no collision. Safe success requires both. Episodes run 300 steps at most.

### Split

The base is trained on LIBERO demos only, so every SafeLIBERO episode is new to it. For the residual:

| part | official SafeLIBERO episodes | use |
|---|---|---|
| training layouts | 0–19 | generated training scenes |
| validation layouts | 20–24 | held out from residual training for the i1/i1b fit check and gate |
| gate | 0–24 | choosing variants and settings |
| test | 25–49 (200 episodes: 4 tasks × 2 levels × 25) | one run per chosen variant, every run reported |

The c0 model trained on layouts 0–24 and had no validation split. That was added for i1, after c0's held-out-layout fit turned out to be near zero.

### Statistics

Every comparison is paired: two arms run on the same episodes with the same seeds, and differences are tested with McNemar's exact test on the discordant pairs (b/c in the tables). Pass bars were written into `README.md` before any residual result existed, and they were not changed afterwards.

| stage | bar |
|---|---|
| Base | Pooled TSR on all 400 official episodes at least OpenVLA-OFT's 0.385, and at least 0.70 with the obstacle removed. |
| Teacher gate (episodes 0–24) | Teacher CAR at least base + 20 points (p < 0.05), with TSR at most 5 points below base. If the teacher fails, stop, because the residual cannot beat it. |
| Residual test (episodes 25–49) | (a) correct sentence beats base on CAR (p < 0.05) with TSR at least base − 3 points. (b) correct sentence beats a wrong-object sentence on CAR (p < 0.05). (c) in two-obstacle scenes, the named obstacle is hit less often than the other one. |

A no-text residual (same data and architecture, sentence removed) is reported next to every test. If it matches the text model on bar (a), then (a) says nothing about text, and only (b) and (c) count.

## Design decisions and why

Decisions the user made on 2026-10-31: an image-based base policy, a geometric detour teacher (rather than best-of-K sampling or the FOL filter), text that picks which object to avoid, and SafeLIBERO-Spatial levels I and II as scope.

Decisions made while building and running it:

1. The base trains on obstacle-free LIBERO demos. SafeLIBERO ships no demos with obstacles, and keeping the base obstacle-free means every collision it makes is a real zero-shot failure that the residual has to fix.
2. The base was trained from scratch instead of fine-tuning pi0.5. A small model trains in about an hour on our allocation, exposes features the residual can read, and gives full control of the action space. The cost is a weaker base: under the pi0.5 protocol it gets TSR 0.435 / 0.445 against pi0.5-LIBERO's 0.670 / 0.555.
3. The residual corrects the base rather than replacing it, so the base's task skill is kept and the residual only has to learn the detour. It changes xyz only, because the teacher only defines xyz detours.
4. At level II the obstacles stand on a 15 cm platform (`box_base_1`) right in front of the gripper's start position, and collisions happen at steps 4–7. The teacher treats the platform as part of the obstacle.
5. Each state is labelled for several sentences, including absent objects and the null sentence. Without this, the sentence would be redundant with the image and the model could ignore it.
6. Text classifier-free guidance (w = 2, 3) raised CAR on the gate but lowered TSR by about the same amount. It was dropped and never run on the test set.
7. c0's pooled base features fit training rows (R² 0.57) but reached R² 0.00 on held-out layouts. That motivated the image residual with its own encoders and text-queried spatial pooling, plus the DAgger states.
8. The first image residual (i1) predicted almost nothing at run time. Its EMA copy kept the base policy's BatchNorm running statistics while training used batch statistics. i1b freezes encoder BatchNorm, so training and inference match. Held-out R² went to 0.09.
9. Variant selection used episodes 0–24 only. The test episodes 25–49 were run once for c0 and once for i1b, and both runs are reported, including the c0 failure on bar (c).
10. Our eval used `env.seed(0)` and 10 settle steps, while the pi0.5 driver uses `env.seed(7)` and 20. Flags `--env_seed` and `--settle` were added, and the base was rerun under the pi0.5 protocol for the comparison table. Residual comparisons stay paired within our protocol, so the mismatch doesn't affect them.
11. A parallel job wrote duplicate rows into two test directories. `compare.py` keeps the first row per episode, a rule fixed before the results were read. Dropping every duplicated episode instead leaves the conclusions unchanged (README, "Write collision").
12. For the object-scaling study, SafeLIBERO-Spatial uses only 6 of the 19 SafeLIBERO obstacle classes. `extend_bddl.py` adds the other 13 to the task files, parked off the table so the constructor's random placement can't fail. `sl_env.apply_layout` then copies an official layout into the extended env by name (joints and fixture poses). Fixture poses live in the model rather than the state vector, so they are reapplied after every reset. A smoke test at both levels gave 0.00 mm offsets.
13. In the scaling study each pool size trains on the same number of chunks (`--max_train_chunks 12000`), so the curve measures object diversity and not data volume. The pools are nested (K = 6 ⊂ 10 ⊂ 14) and five types are never trained on.
14. The residual so far sees the task only through the base's task-ID embedding. The user asked for image, instruction and safety text as inputs, so `train_residual.py --instr` adds a BGE embedding of the instruction text to the residual's conditioning.

## Results so far

Full tables are in `README.md`. The headline numbers:

The base reaches TSR 0.48 (level I) and 0.40 (level II) on all 400 official episodes, pooled 0.44, which passes the 0.385 bar. CAR is 0.065 / 0.125. With the obstacle and platform removed, TSR is 0.79. H1 holds.

Base against pi0.5, all 400 episodes, `env.seed(7)` and 20 settle steps:

| policy | I TSR | I CAR | I safe success | II TSR | II CAR | II safe success |
|---|---|---|---|---|---|---|
| pi0.5 LIBERO checkpoint | 0.670 | 0.140 | 0.140 | 0.555 | 0.120 | 0.115 |
| pi0.5 base checkpoint (no LIBERO fine-tune) | 0.000 | 0.690 | 0.000 | 0.000 | 0.215 | 0.000 |
| our base flow policy | 0.435 | 0.060 | 0.040 | 0.445 | 0.125 | 0.120 |

The teacher passed its gate: CAR 0.085 → 0.74 (131/0), TSR 0.445 → 0.51.

Held-out test, official episodes 25–49, n = 200:

| arm | TSR | CAR | safe success |
|---|---|---|---|
| base | 0.385 | 0.080 | 0.075 |
| teacher (privileged state) | 0.535 | 0.745 | 0.455 |
| c0 residual, correct sentence | 0.535 | 0.225 | 0.180 |
| i1b residual, correct sentence | 0.570 | 0.450 | 0.345 |
| i1b, held-out wording or noun | 0.580 | 0.435 | 0.360 |
| i1b, wrong object named | 0.420 | 0.095 | 0.085 |
| i1b, no sentence | 0.385 | 0.085 | 0.075 |
| i1b architecture trained without text | 0.485 | 0.175 | 0.135 |

i1b passes all three bars. On CAR the correct sentence beats base 76/2 (p = 2e-20) and beats the wrong-object sentence 73/2. With no sentence the model is at base level on both TSR and CAR. In two-obstacle scenes (n = 177) the named obstacle is hit 0.441 of the time against 0.599 for the other, and the within-scene paired test reverses the base's bias (54/26, p = 0.002). So H2, H3 and H4 hold for the six original obstacle types. c0 passed (a) and (b) and failed (c).

The TSR gain from text is weaker than the CAR gain. Against the no-text model, TSR is 44/27 (p = 0.057) on the main test, and in the second training of the recipe (the unseen-type run) the no-text model had the higher TSR. The claim "text raises TSR" holds against base and against the no-sentence arm, but not robustly against a model trained without text.

Unseen object types fail. Trained with milk carton and red mug excluded, the model gets CAR 0.067 → 0.107 on those types (4/1, p = 0.38) and no TSR gain, while the same model on seen types gets 0.09 → 0.32 (31/2). With only 4 training types, the model does not ground new object categories.

### Known gaps

- i1b recovers about 55% of the teacher's CAR gain (0.45 against 0.745).
- Fit on held-out layouts is low (R² 0.09), with only 20–25 training layouts.
- Each result comes from a single training seed.
- In two-obstacle scenes the residual pushes some collisions onto the unnamed obstacle (0.503 → 0.599), as the teacher does.
- The model is a frozen base plus a text-conditioned residual. It is not one end-to-end policy trained with text.

## Status and next steps

The object-scaling study is implemented and smoke-tested but has not run. `cis240028p` is at −11 of 480 GPU SU. `cis250185p` has GPU SU, but this user is not associated with it.

To do, in order:

1. Get compute: a top-up on `cis240028p` or association with `cis250185p`. The scaling study needs about 10–12 SU.
2. Decide the model form. Either keep frozen base + residual (ready to run) or fine-tune the whole flow policy end to end on demos (with the null sentence) plus teacher-corrected chunks, with instruction and safety text as inputs. The end-to-end version needs roughly 2–3 more SU and some code.
3. Run `run_scale.slurm` with `STAGE=scenes`, then `collect`, `train` and `eval`, one job per stage. Training runs K = 6, 10 and 14, each with a text model and a no-text model.
4. Report, for each K, three test sets (official episodes 25–49, extended seen-type scenes, extended unseen-type scenes) and five arms (base, teacher, residual with the correct sentence, the same residual with no sentence, and the no-text residual). The study supports H5 if unseen-type CAR with the correct sentence rises with K, while the no-sentence arm stays at base level.
5. Train at least two more seeds of the chosen recipe before any paper claim.
6. Narrow the gap to the teacher: more layouts, a second DAgger round with i1b driving, larger image resolution.
7. Return to composed sentences ("avoid A and B"), which failed in the toy proof of concept, once single-object transfer works.

## Where things are

| what | where |
|---|---|
| full results, pass bars, protocol notes | `README.md` |
| base policy and residuals | `policy.py`, `train_base.py`, `train_residual.py` |
| teacher | `oracle.py` |
| scenes and extended scenes | `scenes.py`, `extend_bddl.py`, `bddl_x/`, `sl_env.py` |
| data collection | `collect_residual.py` |
| evaluation and paired tests | `eval_sl.py`, `residual_ctrl.py`, `controllers.py`, `compare.py` |
| sentences and embeddings | `texts.py`, `data/text_emb.npz` |
| correctness checks | `checks.py` |
| test run scripts | `run_test.slurm`, `run_unseen.slurm`, `run_scale.slurm` |
| toy proof of concept | `../residual_flow_poc/` |
