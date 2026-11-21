# ex2: residual action expert for SafeLIBERO

ex2 corrects pi0.5's actions near obstacles. pi0.5 (`pi05_libero`) proposes a 10-step action chunk; ex2 predicts an xyz residual for all 10 steps, which is added before execution (5 steps run per chunk).

## Model

- Backbone: the **pi05_base** PaliGemma, frozen in bf16.
- Head: the **pi05_base** action expert (Gemma-300M), fine-tuned (430M trainable parameters). The pi05_libero head is not used.
- New layer: `cond_proj` (zero-initialized, lr 1e-3) feeds pi0.5's planned chunk and the robot state to the expert.
- Output: an xyz residual over 10 steps, trained with flow matching (`res_expert.py`).
- Prompt: `"<instruction>. <safety sentence>"`, where the sentence names the obstacle to avoid.

## Training data

Labels come from a MuJoCo lookahead shield (`shield.py`). When pi0.5's chunk would hit an obstacle, the shield replaces it with the first candidate that does not collide. An episode is kept only if it ends in **safe success** (task done, no hit).

- 19,046 training episodes from 4 suites: 11,264 SafeLIBERO and 7,782 with GSO obstacles (525 objects).
- 6,268 episodes contain at least one corrected chunk (38,452 corrected chunks).
- SafeLIBERO layouts 0–39 only. Layouts 40–49 and 46 GSO objects (no name-family overlap with training) are held out.
- Validation: 719 episodes on 48 held-out GSO objects. The checkpoint was picked by validation only, never by the test tables.

## Training settings

These match pi05_libero fine-tuning where possible.

| setting | value |
|---|---|
| steps | 30,000 |
| batch | 64 (one RTX 6000 Ada, 23 GB peak) |
| optimizer | AdamW b1 0.9, b2 0.95, eps 1e-8, wd 1e-10, grad clip 1.0 |
| lr | 5e-5 peak, 10k warmup, then constant (new layer 1e-3) |
| EMA | 0.999 |
| augmentation | random crop 95% + rotate ±5° (agent view), color jitter (both views) |
| row mix | corrected 35%, safe 30%, wrong sentence 15%, no sentence 20%, balanced per suite |

The full config is in `runs_v2/expert/ex2_1gpu_bs64/config.json` and is also stored inside the checkpoint.

Command:

```bash
export ARGS="--gso_root data_gso,data_gso_r2,data_gso_retry,data_gso_r2_retry --rounds r0,r0_official,r1_*,r0_retry \
  --mix corr=0.35,safe=0.30,wrong=0.15,none=0.20 --steps 30000 --warmup 10000 --lr 5e-5 --b2 0.95 --wd 1e-10 \
  --lr_end_frac 1.0 --ema 0.999 --aug --balance_suites --keep_steps 20000"
sbatch -J ex2_1gpu -t 2-00:00:00 --gres=gpu:1 --export=ALL,OUT=runs_v2/expert/ex2_1gpu,BS=64,BS_FALLBACK=32 run_train_fallback.slurm
```

Pass ARGS through the environment. `sbatch --export=ARGS=...` splits the value at commas.

Held-out validation (`cos_nz` = cosine between predicted and true correction on corrected chunks):

| step | 6k | 10k | **14k (best.pkl)** | 20k | 30k |
|---|---|---|---|---|---|
| val_loss | 0.216 | 0.194 | **0.187** | 0.194 | 0.212 |
| cos_nz | 0.433 | 0.466 | **0.498** | 0.477 | 0.437 |

## Checkpoint

`best.pkl` (step 14k, EMA weights, 1.7 GB) is attached to the GitHub release `ex2-v1` of this repository. Download it to `runs_v2/expert/ex2_1gpu_bs64/best.pkl`.

The file is a pickle with:
- `train_state`: the expert and `cond_proj` weights;
- `stats`: state and action normalization stats.

The frozen PaliGemma is loaded from the `pi05_base` openpi checkpoint (`gs://openpi-assets/checkpoints/pi05_base`, cached under `OPENPI_DATA_HOME`).

## Evaluating with the checkpoint

Each eval starts two servers:
- the pi0.5 server (`pi05_libero`);
- the ex2 server: `res_server.py --ckpt <best.pkl>`, which answers `{agent, wrist, prompt, state, base}` with `{delta}`.

The SafeLIBERO controller is `eval_sl.py --arm residual_ex`. GPU memory needed is about 16–20 GB in total, so one 48 GB GPU holds both servers.

```bash
EX=runs_v2/expert/ex2_1gpu_bs64/best.pkl
# Table 1: official episodes
for E in 0-19 20-39 40-49; do sbatch -J t1_spatial_$E --export=ALL,S=spatial,ARMS=true,EX=$EX,EPS=$E run_eval_pg.slurm; done
# Table 2: 25 unseen objects
sbatch --export=ALL,S=spatial,ARMS=true,EX=$EX,SCENES=../../data_gso/spatial/scenes_newobj25 run_eval_pg.slurm
# Table 3: held-out layouts
sbatch --export=ALL,S=spatial,ARMS=true,EX=$EX,SCENES=scenes_newlayout run_eval_pg.slurm
# pi0.5 alone: same commands with ARMS=base and no EX
python make_tables.py --ex ex2_1gpu_bs64     # prints the 4 tables below
```

- Settings shared by both arms: env_seed 0, settle 20, 5 executed steps per chunk, 256 px render, 6 workers.
- `ARMS=true` gives the correct safety sentence. `none` and `wrong` are the control arms.
- Outputs go to `runs_v2/<suite>/<test|scenes_newobj25|scenes_newlayout>/ex2_1gpu_bs64/true/`.

## Running on another machine

`replicate_ex2_eval.sh` does everything on a plain GPU node with no SLURM. It clones the repo, builds both envs, downloads the checkpoints and meshes, runs both arms on all 4 tables, and prints the tables:

```bash
export GH_TOKEN=...   # only if the repo is private
bash replicate_ex2_eval.sh          # or STEPS="eval tables" to rerun only the evals
```

The details below are what the script does.

What the repo has and what you need to get:

| item | where |
|---|---|
| code, configs, test object lists | this branch |
| Table 2 scenes (25 new objects) | `eval_scenes/<suite>/scenes_newobj25.npz`; copy to `data_gso/<suite>/` |
| Table 3 scenes (held-out layouts) | `eval_scenes/<suite>/scenes_newlayout.npz`; copy to `data_v2/<suite>/` |
| Table 1 episodes | the official SafeLIBERO init states (ship with the benchmark) |
| ex2 checkpoint | release `ex2-v1`, `best.pkl` |
| base-policy files (pi0.5 runner reads action range and features) | release `ex2-v1`, `<suite>_ema_last.pt` -> `runs_scale/<suite>/base/ema_last.pt` |
| pi0.5 checkpoints | openpi `pi05_libero` (base policy) and `pi05_base` (frozen ex2 backbone), downloaded by openpi on first use |
| GSO meshes for Table 2 | `git clone https://github.com/kevinzakka/mujoco_scanned_objects` then `tar cf gso_models.tar models` (Google Scanned Objects, CC-BY 4.0); set `TAR` in `gso_objects.py` |

Paths to edit: 26 scripts hard-code `/home/asingal2/...` or `/data/asingal2/...` (envs, openpi checkout, cache, logs). Search for `asingal2` and replace those paths.

## Results

Each cell is **TSR / CAR / SR** in %:
- TSR: task success.
- CAR: no obstacle collision.
- SR: success with no collision.

Table 1b uses the training layouts (0–39), but with the official obstacle positions, which were never in training data. Table 2 uses the 25 held-out objects that got 4 placements in every suite and level, picked in md5 order (`runs_v2/gso_test25.json`). Table 3 re-places the benchmark obstacles on layouts 40–49. Long II has 195 runs because one placement failed.

pi0.5 avoids obstacles more often in Tables 2–3 (CAR 32–86%) than in Table 1 (3–29%). The generated scenes put obstacles near the target, but not always in the robot's path.

ex2 numbers are pending: the eval jobs are queued. Rerun `python make_tables.py` to fill them.

### Table 1a: official episodes 0-49

| Suite | Level | Runs | pi0.5 | ex2 |
|---|---|---|---|---|
| Spatial | I | 200 | 68.5 / 13.5 / 12.5 | pending |
| Spatial | II | 200 | 59.0 / 12.5 / 11.5 | pending |
| Object | I | 200 | 41.0 / 14.5 / 13.5 | pending |
| Object | II | 200 | 72.0 / 25.5 / 23.0 | pending |
| Goal | I | 200 | 58.0 / 3.0 / 1.5 | pending |
| Goal | II | 200 | 64.5 / 29.0 / 26.0 | pending |
| Long | I | 200 | 57.5 / 13.0 / 11.0 | pending |
| Long | II | 200 | 52.0 / 15.5 / 12.5 | pending |

### Table 1b: official episodes 0-39 (training layouts)

| Suite | Level | Runs | pi0.5 | ex2 |
|---|---|---|---|---|
| Spatial | I | 160 | 68.1 / 12.5 / 11.2 | pending |
| Spatial | II | 160 | 58.1 / 11.9 / 11.9 | pending |
| Object | I | 160 | 40.0 / 13.8 / 12.5 | pending |
| Object | II | 160 | 73.8 / 25.0 / 22.5 | pending |
| Goal | I | 160 | 57.5 / 3.1 / 1.9 | pending |
| Goal | II | 160 | 65.0 / 29.4 / 26.2 | pending |
| Long | I | 160 | 58.1 / 13.1 / 11.2 | pending |
| Long | II | 160 | 54.4 / 15.0 / 12.5 | pending |

### Table 2: 25 unseen objects, training layouts 0-39

| Suite | Level | Objects x runs | Runs | pi0.5 | ex2 |
|---|---|---|---|---|---|
| Spatial | I | 25 x 4 | 100 | 79.0 / 63.0 / 62.0 | pending |
| Spatial | II | 25 x 4 | 100 | 81.0 / 47.0 / 45.0 | pending |
| Object | I | 25 x 4 | 100 | 52.0 / 81.0 / 48.0 | pending |
| Object | II | 25 x 4 | 100 | 93.0 / 86.0 / 85.0 | pending |
| Goal | I | 25 x 4 | 100 | 79.0 / 55.0 / 51.0 | pending |
| Goal | II | 25 x 4 | 100 | 77.0 / 50.0 / 43.0 | pending |
| Long | I | 25 x 4 | 100 | 61.0 / 64.0 / 55.0 | pending |
| Long | II | 25 x 4 | 100 | 53.0 / 52.0 / 43.0 | pending |

### Table 3: benchmark obstacles, held-out layouts 40-49

| Suite | Level | Runs | pi0.5 | ex2 |
|---|---|---|---|---|
| Spatial | I | 200 | 75.0 / 41.0 / 39.5 | pending |
| Spatial | II | 200 | 66.5 / 32.0 / 31.5 | pending |
| Object | I | 200 | 45.5 / 74.0 / 36.0 | pending |
| Object | II | 200 | 88.0 / 74.0 / 73.0 | pending |
| Goal | I | 200 | 84.0 / 43.5 / 42.5 | pending |
| Goal | II | 200 | 69.0 / 38.5 / 34.0 | pending |
| Long | I | 200 | 67.0 / 57.5 / 50.0 | pending |
| Long | II | 195 | 54.9 / 47.7 / 41.5 | pending |

