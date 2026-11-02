# Text-conditioned residual flow — proof of concept

Implements `docs/Text_Conditioned_Residual_Flow_POC.pdf`:
ΔA ~ p_θ(ΔA | s, A_base, E_text(T)), A_corrected = A_base + ΔA.

## Pipeline

| step | command | env |
|---|---|---|
| embed instructions (frozen BGE-base, CLS, L2) | `BGE_PATH=<snapshot> LD_LIBRARY_PATH=$ENV/lib python embed_texts.py` | `libero` (transformers 4.21) |
| scenes, base trajectories, oracle residuals | `python gen_data.py --n_train 4000 --n_test 100` | `flow_policy` |
| pre-training control check | `python oracle_check.py` | `flow_policy` |
| train main / notext / shuffled + evaluate | `sbatch run_all.slurm` | `flow_policy` (GPU) |
| tables + paired stats | `python analyze.py --eval runs/eval` | `flow_policy` |
| Exp 3 follow-ups (stacking, residual sum) | `python compose_eval.py` | `flow_policy` (GPU) |
| Exp 2 follow-up (`main_aug` + fresh paraphrase set) | `python train.py --variant main --aug --out runs/main_aug && python eval_para.py` | `flow_policy` (GPU) |
| example figure | `python make_figure.py --scene 3` | `flow_policy` |

Results and interpretation: `RESULTS.md`.

## Round 2 (v2, 2026-10-20): wider scenes, train vs test scenes, combined training

| step | command | env |
|---|---|---|
| re-embed (adds TRAIN_AB, COMBINED_FAR) | `BGE_PATH=<snapshot> LD_LIBRARY_PATH=$ENV/lib python embed_texts.py --out data_v2/text_emb.npz` | `libero` |
| full pipeline: v2 scenes with the simulated control filter (smoke mode), training of main / notext / shuffled / main_ab, evaluation on 100 train + 100 test scenes, analysis, video rendering | `sbatch --export=ALL,MODE=smoke run_v2.slurm`, then `MODE=full` | `flow_policy` (GPU) |
| tables + paired stats | `python analyze_v2.py --evals runs_v2/eval_train runs_v2/eval_test --out runs_v2` | `flow_policy` |
| report figures | `python report/make_report_figs_v2.py` | `flow_policy` |
| rollout-viewer page | `python make_artifact_v2.py --art runs_v2/artifact` | `flow_policy` |

- `src/traj.py: sample_scene_v2` widens mug, target and laptop ranges and the base tilt, and adds
  base-path habits (bend, lateral and height wobble). `gen_data.py --profile v2 --sim_filter` keeps a
  scene only if the base breaks both constraints and each scripted reference fixes exactly its own
  in MuJoCo. With the new code, v1 data regenerates bit-identically, v1 rollouts replay exactly, and v1 training step-0 losses match.
- `main_ab` trains on 30/30/20/20 A/B/null/A+B with the five `TRAIN_AB` sentences. `COMBINED` (set 1)
  and `COMBINED_FAR` (set 2, worded away from `TRAIN_AB`) are held out from every model.
- Report: `report/report_v2.pdf` (updated 2026-10-31 with three seeds and the failure review;
  the 2026-10-20 version is kept as `report/report_v2_2026-10-20.pdf`). Full tables:
  `runs_v2/results_v2.md`, three seeds in `runs_v2/seeds/seeds.md`, review in `CRITIQUE.md`.
- Page (report view + videos): https://claude.ai/artifact/S8SGRe55C1Rnw3YGbNf7GW. The report view is
  `report_section.html`, a hand-kept copy of `report_v2.tex`: edit both together, then run
  `report/make_report_figs_v3.py`, export the figures to `runs_v2/artifact/figs/` and run
  `make_artifact_v2.py`.

## Design (frozen before training)

- **Scene** (`src/sim.py`): MuJoCo 3.2.3. Franka Panda kinematic chain copied from the
  robosuite Panda (link offsets, joint axes/limits, masses) with primitive geoms;
  joint position servos with gravity compensation track damped-least-squares IK
  set-points on a grip site. Grasp = weld equality hand↔mug (on at waypoint 7, off
  at 35). Mug uses a box collision proxy (a MuJoCo cylinder rocks and rolls on
  release). Laptop = base slab + open screen. Arm geoms do not collide.
- **Randomization** (`src/traj.py`): mug start, target, laptop pose/size, carry height,
  and the base policy's tilt (35–60°, random horizontal axis, random timing) and
  progress profile. Detour side is decided by geometry (~50/50).
- **Actions**: 40 grip-site waypoints × (xyz + world tilt rotation vector).
- **Base trajectory** violates both constraints on every scene (straight carry over
  the laptop + mid-carry tilt). Oracles: A = minimal lateral detour on the cheaper
  side (3 cm clearance); B = zero tilt; A+B = both (test split only, never trained).
- **Metrics** (from simulated mug state):
  viol A = mug centre inside the mug-inflated laptop footprint while grasped;
  viol B = mug tilt > 15° while grasped;
  success = after release + 1.2 s settle, mug within 5 cm of target, tilt < 10°, on table;
  deviation = mean executed grip-site distance from the executed base path;
  correction magnitude = mean commanded ‖Δpos‖ (m) and ‖Δrot‖ (deg).
- **Model** (`src/model.py`): rectified-flow MLP (6 FiLM residual blocks, width 1024)
  over the 240-d residual, conditioned on scene features, the base trajectory and a
  projected BGE embedding; the empty instruction is a learned null token.
- **Training mix**: 40% A target + random TRAIN_A phrasing, 40% B target + TRAIN_B
  phrasing, 20% zero target + empty instruction. No A+B text or target, ever.
- **Baselines**: identity (base), no-text model, shuffled-text model (text drawn
  independently of target class), per-instruction mean residual, test-time swap.
- **Exp. 3 variants**: (i) the unseen combined sentence fed directly (headline);
  (ii) inference-time composition v = v_∅ + Σ_i (v_{clause i} − v_∅) over a clause split.
