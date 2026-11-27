# SafeLIBERO residual expert: full report

This report covers the whole pipeline behind the current results: data collection, the training method, the evaluation protocol, how to run pi0.5 alone and our method, which model is best, and what you need to reproduce every number.

All paths are relative to `experiments/safelibero_residual/` unless they start with `/`. All GPU work runs through SLURM on the `pleiades` partition.

```
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
```

---

## 1. Task and method in one paragraph

SafeLIBERO puts obstacles on the table in the four LIBERO suites (Spatial, Object, Goal, Long), each at two levels (I, and II, where obstacles can sit on a platform). The base policy is pi0.5 fine-tuned on LIBERO (`pi05_libero`, from openpi). It completes most tasks but hits obstacles often. Our method keeps pi0.5 frozen and adds a learned residual.
- **Each chunk:** pi0.5 proposes a 10-step action chunk. The residual model sees both camera images, the robot state, pi0.5's chunk and a sentence naming the obstacle ("don't hit the yellow book"). It outputs an xyz correction for all 10 steps.
- **What runs:** the corrected chunk is clipped to [-1, 1], and the first 5 steps are executed.
- **The residual model:** pi05_base's action expert (Gemma-300M), trained with flow matching. It reads a frozen pi05_base PaliGemma (SigLIP + Gemma-2B) through a rank-16 LoRA.
- **The labels:** a simulator lookahead shield produces them. Training keeps only episodes that the shield-corrected policy finished without a collision.

---

## 2. Dataset collection

### 2.1 Scene splits (no test leakage)

- **SafeLIBERO official episodes:** each (suite, level, task) has 50 initial states ("layouts").
  - Layouts 0-39 are the training layouts.
  - Layouts 40-49 are held out and used only in Table 1b.
- **Training scenes:**
  - `data_v2/<suite>/scenes_train*.npz` hold obstacle placements on layouts 0-39.
  - `data_v2/<suite>/scenes_val_sl.npz` holds fresh placements used only for validation.
- **Google Scanned Objects (GSO):** used as extra and novel obstacles.
  - Source: `/data/asingal2/assets/gso_models.tar`, 1030 models, indexed by `gso_objects.py`.
  - Size filter: height 0.025-0.35 m, width 0.025-0.28 m, thickness at least 0.02 m.
  - Train split: 583 usable objects. A fixed 8% of them (48 objects, chosen by name hash, `is_val_object` in `train_expert.py`) are never trained on and form the novel-object validation set.
  - Test split: 113 usable objects. The 25 Table 2 objects come only from this split.
- **Table 3 positions:** `data_v2/<suite>/scenes_newpos.npz` (built by `newpos_scenes.py`) moves the obstacle on the 40 training layouts.
  - Each position is at least 3 cm from every train and validation placement of the same obstacle in that layout, task and level.
  - Every later training placement (DAgger round 2) is kept at least 3 cm from these too (`--extra_refs`).

### 2.2 The label generator: lookahead shield (`shield.py`)

For every chunk, the shield simulates the candidate chunk in a copy of the MuJoCo state and checks for contact between the robot (or the held object) and the named obstacle or its platform. It tries candidates in this order:
1. pi0.5's own chunk.
2. Up and sideways detours.
3. Lift or retreat.
4. Hold.

It executes the first candidate with no contact. The label for that chunk is `drive_tgt = executed/planned chunk - pi0.5 chunk` on xyz: zero when pi0.5's chunk was already safe, non-zero when the shield changed it. A geometric CBF filter (`safety_filter.py`) was compared first and rejected, because it stalled and failed tasks on Goal and Long.

### 2.3 Round 0: shield-driven episodes

**SafeLIBERO obstacles** (`collect_v2.py`, outputs `data_v2/<suite>/r0`, `r0_official`, `r0_retry`):
- Each scene gets one driving sentence. With probability 0.8 it names a present obstacle; otherwise it names nothing.
- **Named:** pi0.5 plus the shield runs the episode. It's kept only if it ends in safe success: task done, and the named obstacle displaced no more than 1 mm (the benchmark's collision test). Failures are retried with the next shield config, up to 3 tries (shield, shield, shield_wide).
- **Nothing named:** pi0.5 runs alone and the episode is always kept. These rows teach "no sentence, no correction".

**GSO obstacles** (`collect_gso.py --split train --layouts 0-39`, outputs `data_gso/<suite>`, `data_gso_r2`, `data_gso_retry`, `data_gso_r2_retry`, `data_gso_r3`):
- One GSO object goes in the first obstacle slot, upright with a random yaw. `scenes.make_scene` places it near the target object, on its path or between gripper and object; on level II, a platform is used with the usual probability.
- Up to 8 placement attempts per item.
- The episode is then run as in `collect_v2`, with the object as the named obstacle.

**Each kept episode is one npz file, with one row per executed chunk:**

| Field | Shape | Meaning |
|---|---|---|
| `agent`, `wrist` | (C, 224, 224, 3) uint8 | the two camera frames, raw env orientation |
| `state` | (C, 8) | end-effector position, axis-angle, gripper (pi0.5's state) |
| `base` | (C, 10, 7) | pi0.5's chunk |
| `plan` | (C, 10, 7) | the shield's planned chunk |
| `exec` | (C, 5, 7) | actions actually executed |
| `drive_tgt` | (C, 10, 3) | the label |
| `meta` | json | scene, layout, level, task, instruction, driving obstacle, GSO name/noun, outcome |

### 2.4 DAgger rounds: the expert drives, the shield labels

`run_dagger_ex.slurm` (MODE=gso calls `collect_gso.py`; MODE=v2 calls `collect_v2.py` on a scene file):
- pi0.5 plus the current residual drives the robot.
- The shield checks each chunk and runs its own plan instead whenever the residual's chunk would collide.
- Every visited state is labeled with the shield's plan minus pi0.5's chunk.
- An episode with no collision is kept whatever the task outcome. Its meta records `dagger`, `success` and `ex_frac`.

| Round | Driver | Data | Notes |
|---|---|---|---|
| dag1 | ex2 | `data_gso_dag1/<suite>`, `data_v2/<suite>/dag1_*` | first DAgger round |
| dag2 | ex8 (step 6k), seed 22 | `data_gso_dag2`: spatial 846, object 552, goal 750, long 709 | GSO training objects |
| dag2 (v2) | ex8, seed 22 | `data_v2/<suite>/dag2_*`: spatial 2753, object 2710, goal 1501, long 2700 files | new training placements `scenes_train_d.npz` (seed 61, 3 per layout) built by `run_dag2.slurm`, at least 3 cm from all train, val and Table 3 placements |
| dag2b | ex8, seed 23, 2 per layout | `data_gso_dag2b`: spatial 563, object 378, goal 507, long 472 | used from ex12 on |
| rs1 | ex11, seed 32, 3 per layout | `data_gso_rs1`: spatial 795, object 714, goal 644, long 866 | 270 rescaled GSO training objects (below), used from ex12 on |

**Rescaled objects (rs1).** 286 train-split GSO objects failed the size filter. `python gso_objects.py rescale` gives each one with a valid scale range a log-uniform scale factor drawn from its name hash, then writes `gso_index_rs.json` (270 variants, listed in `runs_v2/gso_rs_objects.json`). Training on this data needs `GSO_RESCALED=1`.

**Totals.** About 45,500 episode files through dag2, plus 1,920 (dag2b) and 3,019 (rs1). Never train on `data_gso_rs_smoke` (smoke test only).

---

## 3. Training method

### 3.1 Model (`res_expert.py`)

- **Prefix:** agent image (256 tokens), wrist image (256 tokens), and prompt tokens go through the pi05_base PaliGemma, giving a KV cache.
  - Prompt format: "<task instruction>. <safety sentence>", with the 8-value state discretized into it as pi0.5 does.
  - The PaliGemma weights are frozen; a rank-16 LoRA is trained from ex8 on.
- **Suffix:** 10 action tokens. Token h is `action_in_proj(noisy residual_h) + cond_proj([pi0.5 action_h (7), state (8)])`.
- **Expert:** pi05_base's Gemma-300M action expert with adaRMS flow time. It attends to the prefix KV cache and outputs the flow velocity of the xyz residual (10 × 3).
- **`cond_proj` is new and zero-initialized,** so at step 0 the expert sees exactly what it was pre-trained on.
- **`--motion` (ex13 only):** `cond_proj` takes 8 more inputs, the state change since the previous chunk. On a warm start these weights are zero-padded, so training starts from the same function as the parent model.
- **Flow matching:**
  - Interpolation: `x_t = t·noise + (1-t)·x`; target velocity `noise - x`.
  - t is drawn from a Beta(1.5, 1) distribution, with 2 flow times per sample.
  - Sampling: 10 Euler steps from t = 1 to t = 0.
- **Normalization:** targets are scaled by 1 / std(non-zero labels). The base chunk and the state are standardized. A warm start keeps its parent's statistics.

### 3.2 Training rows (`train_expert.py --mix`)

Each batch element is one chunk of one episode, paired with one prompt:

| Kind | Share | Sentence | Chunk | Target |
|---|---|---|---|---|
| corr | 0.30 | true object name | one the shield corrected | the correction |
| safe | 0.40 | true object name | one the shield left alone | 0 |
| wrong | 0.10 | names an object not in the scene | any chunk | 0 |
| none | 0.20 | instruction only | any chunk | 0 |

- **Sampling and augmentation:**
  - `--balance_suites` draws corr rows suite-first.
  - `--aug` applies pi0.5's image augmentation (crop, rotation, color jitter).
  - `--dup dag=4[,rs1=4]` oversamples DAgger and rescaled episodes 4×.
- **Validation inside training:** batches from the 48 held-out GSO objects, logging:
  - `cos_nz`: direction match on corrected chunks.
  - `mag_true`: push size on corrected chunks.
  - `mag_safe`: push size on safe chunks (should be 0).
  - Gate accuracy.

### 3.3 Model lineage (each warm-starts from the previous)

Shared recipe for ex7 onward:

```
--mix corr=0.30,safe=0.40,wrong=0.10,none=0.20 --bs 64 --per_ep 2 --lr 2e-5 --lr_new 1e-3 --warmup 500
--aug --balance_suites --b2 0.95 --wd 1e-10 --lr_end_frac 1.0 --ema 0.999 --val_every 1000
```

| Model | Init | Steps | Data (`--gso_root` / `--rounds`) | Extra |
|---|---|---|---|---|
| ex7 (`ex7_bs64`) | ex3b_bs64 best | 8000 | data_gso, r2, retry, r2_retry, dag1, r3 / r0, r0_official, r1_*, r0_retry, dag1_* | `--dup dag=4`, no LoRA |
| ex8 (`ex8_lora`) | ex7 last | 6000 | same as ex7 | `--lora --lr_lora 1e-4` |
| ex11 (`ex11_dag2`) | ex8 last | 6000 | + data_gso_dag2 / + dag2_* | LoRA |
| ex12 (`ex12_rs`) | ex11 last | 6000 | + data_gso_dag2b, data_gso_rs1 | `--dup dag=4,rs1=4`, `GSO_RESCALED=1` |
| ex13 (`ex13_motion`) | ex12 last | 6000 | same as ex12 | `--motion` |

Training runs on 4 GPUs: `sbatch -p pleiades --gres=gpu:4 -c 16 --mem=160G -t 14:00:00`. The global batch of 64 is split across them, and each run takes about 4.5 h at 2.67 s/step. Checkpoints are `step_2000.pkl`, `step_4000.pkl` and `last.pkl` (EMA weights).

**Checkpoints used in the tables** (each is a folder holding a `best.pkl` link):
- `runs_v2/expert/ex7last` → ex7_bs64/last.pkl
- `runs_v2/expert/ex8s6k` → ex8_lora/last.pkl
- `runs_v2/expert/ex11s6k` → ex11_dag2/last.pkl
- `runs_v2/expert/ex12s6k` → ex12_rs/last.pkl
- `runs_v2/expert/ex13s4k` → ex13_motion/step_4000.pkl

### 3.4 Exact training command (ex13 as the example)

```
export GSO_RESCALED=1
export ARGS="--gso_root data_gso,data_gso_r2,data_gso_retry,data_gso_r2_retry,data_gso_dag1,data_gso_r3,data_gso_dag2,data_gso_dag2b,data_gso_rs1 \
 --rounds r0,r0_official,r1_*,r0_retry,dag1_*,dag2_* --mix corr=0.30,safe=0.40,wrong=0.10,none=0.20 --steps 6000 --bs 64 \
 --per_ep 2 --lr 2e-5 --warmup 500 --val_every 1000 --aug --balance_suites --keep_steps 2000,4000 --b2 0.95 --wd 1e-10 \
 --lr_end_frac 1.0 --ema 0.999 --dup dag=4,rs1=4 --init runs_v2/expert/ex12_rs/last.pkl --lora --lr_lora 1e-4 --motion"
sbatch -J ex13 -p pleiades --gres=gpu:4 -c 16 --mem=160G -t 14:00:00 --export=ALL,OUT=runs_v2/expert/ex13_motion run_train_expert.slurm
```

For the other models, change `--gso_root`, `--rounds`, `--dup` and `--init`, and drop `--motion`, following the table in 3.3.

---

## 4. Evaluation

### 4.1 Metrics

- **TSR:** task success rate.
- **CAR:** collision-avoidance rate, the share of episodes in which no active obstacle moved more than 1 mm.
- **SR:** safe success, meaning the task succeeded with no collision. This is the headline number.
- **Gain:** SR(method) - SR(pi0.5) on the same scenes. The goal was a gain above +5 on every row.

### 4.2 Fixed eval settings (same for every arm)

`run_eval_pg.slurm` calls `eval_sl.py` with:

```
--env_seed 0 --settle 20 --exec_steps 5 --res 256 --workers 6
```

- **Step limits:** 300 steps, or 550 for Long.
- **What the method arm sees:** `ARMS=true`, the true sentence "don't hit the <obstacle>" with the obstacle's audited name.
- **Error handling:** an episode that crashes (env error, GPU out of memory) is scored as a failure. Check for errors before reading any table, and rerun the affected folder.

### 4.3 Test tables

| Table | Scenes | n per row (suite × level) |
|---|---|---|
| 1a | official episodes, layouts 0-39 | 160 |
| 1b | official episodes, layouts 40-49 (unseen layouts) | 40 |
| 1 | 1a + 1b pooled, layouts 0-49 | 200 |
| 2 | 25 novel GSO test-split objects, 4 episodes per object per level (`data_gso/<suite>/scenes_newobj25.npz`) | 100 |
| 3 | 200 new obstacle positions per level on layouts 0-39, each run once (`data_v2/<suite>/scenes_newpos.npz`) | 200 |

**How the Table 3 positions were made.** `newpos_scenes.py gen` (seed 51) made 3 candidates per layout, task and level (3,837 in total). `pick` kept those that pi0.5 plus the shield finished without a collision, then sampled 200 per level balanced across tasks and layouts. Model results played no part.

### 4.4 Validation (checkpoints are chosen on this only, never on test)

- **`scenes_val`:** the 48 held-out GSO training objects.
- **`scenes_val_sl`:** fresh SafeLIBERO placements.
- **Size:** 16 rows (4 suites × 2 levels × 2 sets), about 48 episodes each, so roughly ±7 noise per row.
- **Command:** `val_tables.py` builds the table, and the SR gain per arm is summarized from it.

---

## 5. How to run pi0.5 alone (the baseline)

Run this once per base policy; every residual arm shares it.

```
for S in spatial object goal long; do
  for E in 0-19 20-39 40-49; do
    sbatch -J base_${S}_$E -p pleiades -t 12:00:00 --export=ALL,S=$S,ARMS=base,EPS=$E run_eval_pg.slurm
  done
  sbatch -J base_${S}_obj -p pleiades -t 12:00:00 --export=ALL,S=$S,ARMS=base,SCENES=../../data_gso/$S/scenes_newobj25 run_eval_pg.slurm
  sbatch -J base_${S}_pos -p pleiades -t 12:00:00 --export=ALL,S=$S,ARMS=base,SCENES=scenes_newpos run_eval_pg.slurm
done
```

- Results go to `runs_v2/<suite>/{test,scenes_newobj25,scenes_newpos}/pi05/...`.
- The script starts the pi0.5 policy server (`run_pi05_server.sh pi05_libero`) on the same GPU.
- Checkpoints are read from `/data/asingal2/openpi_cache/openpi-assets/checkpoints/`.

## 6. How to run our method

```
NAME=ex13s4k                                   # any folder under runs_v2/expert/ holding best.pkl
EX=runs_v2/expert/$NAME/best.pkl
for S in spatial object goal long; do
  for E in 0-19 20-39 40-49; do
    sbatch -J t_${S}_$E -p pleiades -t 12:00:00 --export=ALL,S=$S,ARMS=true,EX=$EX,EPS=$E run_eval_pg.slurm
  done
  sbatch -J t_${S}_obj -p pleiades -t 12:00:00 --export=ALL,S=$S,ARMS=true,EX=$EX,SCENES=../../data_gso/$S/scenes_newobj25 run_eval_pg.slurm
  sbatch -J t_${S}_pos -p pleiades -t 12:00:00 --export=ALL,S=$S,ARMS=true,EX=$EX,SCENES=scenes_newpos run_eval_pg.slurm
done
```

- **What each job runs:** the pi0.5 server plus `res_server.py` serving the checkpoint, so each job needs one 48 GB GPU. Nodes with smaller GPUs are excluded.
- **Results:** `runs_v2/<suite>/<table folder>/$NAME/true/<eps folder or all>/part_*.jsonl`.
- **Building the tables:** `tables_cmp.py` makes Tables 1a, 1b, 2 and 3; the pooled Table 1 needs a short snippet. Both commands, and the error check, are in `claude_prompts/eval_tables_residual.md`.

---

## 7. Results

### 7.1 Validation SR gain over pi0.5 (all 16 rows)

| Model | TSR | CAR | SR | Rows > +5 | Novel-object SR | Novel rows > +5 |
|---|---|---|---|---|---|---|
| ex7 | +0.3 | +11.5 | +7.6 | 11/16 | | |
| ex8 | +2.6 | +12.3 | +8.6 | 12/16 | +6.4 | 4/8 |
| ex11 | -0.8 | +12.8 | +7.8 | 13/16 | +7.2 | 6/8 |
| ex12 | +1.4 | +14.6 | +8.9 | 10/16 | +9.9 | 5/8 |
| ex13 step 4k | -0.5 | +15.2 | **+10.0** | 13/16 | +8.8 | 6/8 |

### 7.2 Test SR gain over pi0.5 (✗ = at or below +5)

| Table | Row | pi0.5 SR | ex7 | ex8 | ex11 | ex12 | ex13 |
|---|---|---|---|---|---|---|---|
| 1 | Spatial I | 12.5 | +16.5 | +17.5 | +17.5 | +21.5 | +20.0 |
| 1 | Spatial II | 11.5 | +42.0 | +36.0 | +35.0 | +37.0 | +37.5 |
| 1 | Object I | 13.5 | +13.5 | +11.0 | +16.5 | +17.5 | +18.0 |
| 1 | Object II | 23.0 | +5.0 ✗ | +5.0 ✗ | +6.5 | +17.0 | +12.0 |
| 1 | Goal I | 1.5 | +35.0 | +38.5 | +42.5 | +40.5 | +35.0 |
| 1 | Goal II | 26.0 | +38.5 | +32.5 | +30.5 | +30.0 | +33.5 |
| 1 | Long I | 11.0 | +25.5 | +27.0 | +25.5 | +28.5 | +23.5 |
| 1 | Long II | 12.5 | +12.5 | +10.5 | +13.0 | +12.0 | +7.0 |
| 2 | Spatial I | 62 | +1 ✗ | -4 ✗ | +1 ✗ | -1 ✗ | +2 ✗ |
| 2 | Spatial II | 45 | +9 | +13 | +14 | +8 | +11 |
| 2 | Object I | 48 | +4 ✗ | +8 | +7 | -3 ✗ | -1 ✗ |
| 2 | Object II | 85 | -2 ✗ | -1 ✗ | 0 ✗ | +1 ✗ | 0 ✗ |
| 2 | Goal I | 51 | +10 | +8 | +6 | +9 | +10 |
| 2 | Goal II | 43 | +15 | +6 | +11 | +12 | +13 |
| 2 | Long I | 55 | +3 ✗ | -1 ✗ | +1 ✗ | +2 ✗ | +3 ✗ |
| 2 | Long II | 43 | +2 ✗ | +9 | +3 ✗ | +2 ✗ | +3 ✗ |
| 3 | Spatial I | 51.5 | +10.5 | +14.0 | +14.5 | +15.5 | +22.0 |
| 3 | Spatial II | 45.5 | +9.5 | +7.5 | +11.0 | +14.0 | +18.5 |
| 3 | Object I | 58.0 | +4.0 ✗ | +2.0 ✗ | +3.0 ✗ | +2.5 ✗ | +2.5 ✗ |
| 3 | Object II | 82.0 | 0 ✗ | +2.0 ✗ | -0.5 ✗ | -1.5 ✗ | +1.0 ✗ |
| 3 | Goal I | 55.0 | +2.5 ✗ | -2.5 ✗ | +6.0 | +2.5 ✗ | +4.0 ✗ |
| 3 | Goal II | 42.0 | +14.5 | +11.5 | +17.5 | +19.0 | +18.5 |
| 3 | Long I | 61.5 | +5.0 ✗ | +7.0 | +3.0 ✗ | +11.5 | +7.0 |
| 3 | Long II | 49.5 | +12.0 | +12.5 | +12.5 | +13.0 | +14.5 |

| Model | Table 1 rows > +5 / mean | Table 2 | Table 3 | Total rows > +5 |
|---|---|---|---|---|
| ex7 | 7/8, +23.6 | 3/8, +5.3 | 4/8, +7.7 | 14/24 |
| ex8 | 7/8, +22.3 | 5/8, +4.8 | 5/8, +6.8 | 16/24 |
| ex11 | 8/8, +23.4 | 4/8, +5.4 | 5/8, +8.4 | **17/24** |
| ex12 | 8/8, **+25.5** | 3/8, +3.8 | 5/8, +9.6 | 16/24 |
| ex13 | 8/8, +23.3 | 3/8, +5.1 | 5/8, **+11.0** | 16/24 |

Full cells with TSR / CAR / SR are in the scratchpad table outputs (`t13.txt`, `t13pool.txt`). Rebuild them with the commands in `eval_tables_residual.md`.

### 7.3 Which method is best

- **ex11 passes the most rows (17 of 24).** It's the safest single choice when the criterion is the number of rows above +5.
- **ex13 (step 4k) is best on Table 3** (mean +11.0) **and on validation** (mean +10.0). It's tied with ex11 for validation rows above +5.
- **ex12 has the best Table 1 mean (+25.5)** but the weakest Table 2.
- **No model clears +5 on all 24 rows.** These rows fail for every model:
  - Table 2: Spatial I, Object II and Long I.
  - Table 3: Object I and Object II.
- **Why:** on these rows pi0.5 already avoids the obstacle most of the time (CAR 63-86), and the residual costs about as much task success as it saves in collisions.

### 7.4 What was tried and did not help

| Change | Result |
|---|---|
| Scaling the residual ×0.75 / ×0.5 (ex11) | validation SR +6.1 / +4.8 vs +7.8 at ×1.0 |
| Averaging 4 flow samples per chunk (ex3b) | validation SR +5.6 vs +8.4 |
| Separately trained correction gate, threshold 0.3 (ex8 + ex9 gate) | validation SR +7.9 vs +8.6 |
| More DAgger data (ex11) | +1 row vs ex8 |
| Rescaled GSO objects (ex12) | Tables 1 and 3 up, Table 2 down |
| Arm-motion input (ex13) | Table 3 Spatial up by 7-8, Table 2 unchanged |
| 2× step limit on collision-free timeouts (ex7, ex8, ex11) | the extra time adds 0 to 2 SR on most rows. Most "recovered" episodes succeed within the original limit on a second rollout. Only Table 2 Object I gains from the extra time (ex11 +8). pi0.5 was not rerun, so this is diagnostic only. |

---

## 8. Reproducing the results

### 8.1 Environments and checkpoints

- **Python environments:**
  - `/data/asingal2/envs/openpi` (JAX, openpi): policy servers and training.
  - `/data/asingal2/envs/slres` (robosuite / LIBERO / SafeLIBERO): simulation and tables.
  - Build scripts and frozen package lists are in `setup/`, from commits `9f2121c1` and `bb3eaf92` on branch `safelibero-residual-scaleup`.
  - A one-command eval replication script for a non-SLURM node is in commit `b0bedaf8`.
- **openpi source:** `vlsa-aegis/openpi`.
- **pi0.5 checkpoints:** `pi05_libero` (the base policy) and `pi05_base` (the residual's backbone and expert), under `/data/asingal2/openpi_cache/openpi-assets/checkpoints/`.
- **GSO models:** `/data/asingal2/assets/gso_models.tar`, copied to node-local `/tmp/gso_$USER` by the jobs.
- **Rendering:** `MUJOCO_GL=egl` is set by `pleiades_run.sh`.

### 8.2 Order of operations from scratch

1. **Collect round 0.**
   - Run `collect_v2.py` for each suite on `scenes_train*.npz`, writing `data_v2/<suite>/r0` and `r0_official`.
   - Run `collect_gso.py --split train --layouts 0-39` for each suite, writing `data_gso/<suite>`.
   - Run the retry and r2/r3 top-ups for scenes that failed.
2. **Train the early models,** ex1 → ex3b, from these data. Then collect dag1 with `run_dagger_ex.slurm`.
3. **Train ex7, then ex8,** using the commands in 3.3 and 3.4.
4. **DAgger round 2.**
   - `sbatch --export=ALL,S=<suite> run_dag2.slurm` builds `scenes_train_d.npz`.
   - Then run `run_dagger_ex.slurm` with `MODE=v2 ROUND=dag2 SEED=22 SCENES=data_v2/<S>/scenes_train_d.npz EX=runs_v2/expert/ex8s6k/best.pkl`, in chunks of 240 scene ids.
   - Also run `MODE=gso ROUND=dag2 SEED=22` with the same driver.
5. **Train ex11.**
6. **Collect the extra round-2 and rescaled data.**
   - dag2b: `MODE=gso ROUND=dag2b SEED=23 PER=2`, driven by ex8.
   - Rescaled objects:
     - `python gso_objects.py rescale` writes the index.
     - Then `GSO_RESCALED=1 MODE=gso ROUND=rs1 SEED=32 PER=3 OBJS=runs_v2/gso_rs_objects.json`, driven by ex11.
7. **Train ex12** (with `GSO_RESCALED=1`), **then ex13** (`--motion`).
8. **Validate** each checkpoint at steps 2k, 4k and 6k on `scenes_val` and `scenes_val_sl`, and pick by validation SR.
9. **Run the test tables** (sections 5 and 6), check for eval errors, and build the tables.

### 8.3 Things that change numbers if done wrong

- **Use the same eval settings for every arm.** That means `env_seed 0`, `settle 20` and `exec_steps 5`. The simulator seed moves object positions even with a fixed initial state.
- **Rerun any folder with eval errors.** GPU out-of-memory crashes happen when a node is crowded, and those episodes count as failures until rerun.
- **Don't edit `shield.py` or `safety_filter.py` while collection jobs are running,** since the jobs import them.
- **`GSO_RESCALED=1` is needed to train on `data_gso_rs1`.** Without it, the trainer can't find those objects' names. It isn't needed at eval.
- **Single rows are noisy.** About ±7 at n=200, ±10 at n=100 and ±16 at n=20. Don't pick checkpoints or settings from individual test rows.
- **The `MAXM`, `SCENE_LIST` and `EPISODES_FILE` options in `run_eval_pg.slurm` are for the timeout study only.** Leave them unset for the main tables.
- **The env step cap:** robosuite's episode cap is raised above its default of 1000 only when a longer step limit is requested, so normal evals are unchanged.
