# Situational Safety for Embodied Agents

A VLM-grounded safety filter for generalist vision-language-action (VLA) policies in
tabletop manipulation. A vision-language model proposes open-vocabulary safety
predicates from robot observations; those predicates are grounded into parametric
spatial constraints, assembled into a control barrier function (CBF), and used to
filter the VLA's nominal action through a CBF-QP at every timestep.

The repository covers experiments on two safety benchmarks:

- **[SafeLIBERO](https://huggingface.co/datasets/THURCSCT/SafeLIBERO)** — four LIBERO
  suites (`spatial`, `object`, `goal`, `long`) at two hazard levels (I, II).
- **[LIBERO-Safety](https://github.com/LIBERO-SAFETY/LIBERO-Safety)** — four safety
  suites: `human_safety`, `obstacle_avoidance`, `obstacle_avoidance_human`,
  `affordance`.

Base policies evaluated: OpenVLA, OpenVLA-OFT, π0 / π0.5 (openpi), NVIDIA Cosmos
Policy, and Fast-WAM.

See [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for the research brief and the
six-level safety taxonomy, and
[`Hierarchical_symbolic_safety.md`](Hierarchical_symbolic_safety.md) for the symbolic
formulation.

## Repository layout

| Path | Contents |
|---|---|
| `fol_safety_filter/` | First-order-logic safety filter: predicate primitives, knowledge base, rule composer, VLM grounding, CBF mapper, runtime filter (`filter.py`) |
| `vlm_prompt_runner/` | Multi-model VLM prompt harness (Anthropic / OpenAI / Gemini / Qwen) with majority voting and accuracy scoring |
| `semantic_cbf/` | Standalone semantic-CBF prototypes: VLM→CBF pipeline, multi-prompt strategy, VLA integration, latent-space CBF |
| `vlm_pipeline/` | SafeLIBERO eval drivers, perception/grounding accuracy studies |
| `run_libsafety_*.py` | LIBERO-Safety eval drivers (OpenVLA/OFT, openpi, Cosmos, Fast-WAM, flow-CBF, AEGIS) |
| `libsafety_*.sh`, `libsafety_env_utils.py` | LIBERO-Safety environment setup, asset/checkpoint download, policy server |
| `epistemic_uncertainty/` | Uncertainty quantification (MC dropout, deep ensembles, entropy, density OOD) |
| `prompts/` | Prompt templates for obstacle ID, safety predicates, STL specifications |
| `slurm/`, `experiments/` | Cluster job definitions (Bridges2-specific paths — see below) |
| `patches/` | Patches applied to the pinned submodules |
| `results_tables/`, `*_benchmark*/`, `fol_*_n50/`, `baseline_*/` | Aggregated tables and per-condition result JSONs |

Rollout videos, raw VLM observation dumps, CBF visualisation HTML, and cluster logs are
not tracked — they are regenerable and are excluded by `.gitignore`.

## Setup

```bash
git clone --recurse-submodules https://github.com/AkankshaSingal8/situational_safety_for_embodied_agents.git
cd situational_safety_for_embodied_agents
```

Submodules: `SafeLIBERO`, `LIBERO-Safety`, `openvla-oft`, `cosmos-policy`, `vlsa-aegis`.

Headless rendering (any cluster or CI machine) requires:

```bash
export MUJOCO_GL=egl
```

API-backed VLMs read keys from the environment — copy `.env.example` to `.env` and fill
in the providers you intend to use:

```bash
cp .env.example .env
```

### Conda environments

The two benchmarks need different stacks; each has a setup script.

```bash
# SafeLIBERO
bash libero_env_setup.sh            # LIBERO + robosuite + MuJoCo (Python 3.8)
bash openvla_safelibero_setup.sh    # OpenVLA-OFT policy stack (Python 3.10)
bash qwen_vlm_env_setup.sh          # local Qwen-VL server
bash openpi_setup.sh                # π0.5 policy stack

# LIBERO-Safety
bash libsafety_client_setup.sh      # LIBERO-Safety rollout client (Python 3.10+;
                                    #   3.8 does NOT work — the benchmark uses PEP 604 syntax)
bash libsafety_openvla_setup.sh     # OpenVLA / OpenVLA-OFT eval env
bash libsafety_openpi_setup.sh      # π0 / π0.5 server env (Python 3.11)
```

---

## Running SafeLIBERO experiments

Evaluate OpenVLA-OFT with the FOL safety filter. This is the exact form used in the
reported n=50 campaigns:

```bash
export MUJOCO_GL=egl
export PYTHONPATH="$PWD:$PWD/vlm_pipeline:$PWD/openvla-oft:$PWD/SafeLIBERO/safelibero:$PYTHONPATH"

python vlm_pipeline/run_safelibero_fol_openvla_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name safelibero_spatial --safety_level I \
    --num_trials_per_task 50 \
    --use_fol_filter True --fol_level 1 \
    --results_output_dir fol_spatial_L1_n50 \
    --center_crop True --seed 7
```

- Drop `--use_fol_filter` for the unfiltered baseline of the same condition.
- `--task_suite_name`: `safelibero_{spatial,object,goal,long}`.
- `--safety_level`: `I` or `II`.
- `--fol_level`: taxonomy level the filter enforces (1 = geometric, 3 = semantic).

Aggregate the result JSONs:

```bash
python scripts/aggregate_results.py --help
python final_results.py          # prints the cross-version comparison table
python compare_results.py --help
```

### VLM prompt experiments

```bash
python vlm_prompt_runner/run_experiment.py --help          # single or --models multi
python vlm_prompt_runner/run_majority_vote_experiment.py --help
```

---

## Running LIBERO-Safety experiments

### 1. Fetch assets and checkpoints

```bash
bash libsafety_download_assets.sh        # needs huggingface_hub in the active env
bash libsafety_download_checkpoints.sh   # pi0 / pi0.5 openpi weights
```

### 2. Point LIBERO at its config directory

`LIBERO_CONFIG_PATH` must name a **directory** containing `config.yaml` (created by the
client setup script), not the YAML file itself:

```bash
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/LIBERO-Safety/.libero_config"
```

### 3. Run an evaluation

**OpenVLA / OpenVLA-OFT** — in the `libsafety_openvla` env:

```bash
python run_libsafety_eval_openvla.py \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name human_safety \
    --all_tasks True \
    --num_trials_per_task 3 \
    --results_out_path libsafety_results
```

Use `--task_index N` instead of `--all_tasks True` to run a single task.

**π0 / π0.5** — start the policy server, then run the client:

```bash
bash libsafety_serve_openpi.sh pi05_libero 8000    # in the libsafety_openpi env
python run_libsafety_eval_openpi.py \
    --task_suite_name obstacle_avoidance --all_tasks True \
    --num_trials_per_task 3 --port 8000 \
    --checkpoint_name pi05_libero
```

`--checkpoint_name` is required — the driver asserts on it, since results are keyed by
checkpoint and would otherwise clobber another checkpoint's output.

**Cosmos Policy:**

```bash
python run_libsafety_cosmos_policy_eval.py \
    --task_suite_name affordance --all_tasks True \
    --num_trials_per_task 3 \
    --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --results_output_dir cosmos_benchmark_libsafety
```

**Fast-WAM:**

```bash
python run_libsafety_fastwam_eval.py \
    --task_suite_name human_safety \
    --task_indices 0 1 2 3 \
    --num_trials_per_task 3 \
    --ckpt_path  <fastwam_ckpt_dir>/libero_uncond_2cam224.pt \
    --stats_path <fastwam_ckpt_dir>/libero_uncond_2cam224_dataset_stats.json \
    --results_output_dir fastwam_benchmark_libsafety \
    --replan_steps 10 --action_horizon 32 --seed 195
```

Note this driver uses `--task_indices` (a list) rather than the `--all_tasks` flag the
other drivers take, and needs both the checkpoint and its dataset-stats JSON.

Suites: `human_safety`, `obstacle_avoidance`, `obstacle_avoidance_human`, `affordance`.

### 4. Aggregate

```bash
python aggregate_libsafety_results.py --results_dir cosmos_benchmark_libsafety
python aggregate_table3_replication.py
```

### Flow-CBF variant

```bash
python run_libsafety_eval_flowcbf.py --help
python validate_flow_cbf.py            # offline barrier validation
python validate_flow_cbf_trajectory.py
```

---

## Cluster scripts

`slurm/` holds the author's Bridges2 job scripts for every reported condition. They are
kept as an exact record of how each number was produced, **not** as runnable scripts:
they hardcode a specific cluster allocation path, and 149 of them resolve the source
tree through a git worktree (`$REPO/.worktrees/...`) that is not part of this
repository. Adapt the `REPO` / `WT` variables at the top of a script before submitting.

## Tests

```bash
python -m pytest fol_safety_filter/tests vlm_prompt_runner/tests
```

The `epistemic_uncertainty/` tests additionally require `torch`, so run them from the
policy environment:

```bash
python -m pytest epistemic_uncertainty/tests
```

## Metrics

- **TSR** — Task Success Rate: episodes completing the task goal.
- **CAR** — Collision Avoidance Rate: episodes with no obstacle contact.
- **ETS** — Execution Time Steps: mean episode length.

Headline results: [`results_tables/summary_table.md`](results_tables/summary_table.md)
and [`fol_safety_filter/README.md`](fol_safety_filter/README.md).

## License

MIT — see [`LICENSE`](LICENSE). Submodules and benchmark datasets carry their own
licenses.
