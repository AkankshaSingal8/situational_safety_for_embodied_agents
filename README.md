# Situational Safety for Embodied Agents

A VLM-grounded safety filter for generalist vision-language-action (VLA) policies in
tabletop manipulation. A vision-language model proposes open-vocabulary safety
predicates from robot observations; those predicates are grounded into parametric
spatial constraints, assembled into a control barrier function (CBF), and used to
filter the VLA's nominal action through a CBF-QP at every timestep.

Evaluated on [SafeLIBERO](https://huggingface.co/datasets/THURCSCT/SafeLIBERO) with
[OpenVLA-OFT](https://github.com/moojink/openvla-oft) and π0.5 as base policies.

See [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for the full research brief, including
the six-level safety taxonomy the pipeline is designed against, and
[`Hierarchical_symbolic_safety.md`](Hierarchical_symbolic_safety.md) for the symbolic
formulation.

## Repository layout

| Path | Contents |
|---|---|
| `fol_safety_filter/` | First-order-logic safety filter: predicate primitives, knowledge base, rule composer, VLM grounding, CBF mapper, and the runtime filter (`filter.py`) |
| `vlm_prompt_runner/` | Multi-model VLM prompt harness (Anthropic / OpenAI / Gemini / Qwen backends) with majority voting and accuracy scoring |
| `semantic_cbf/` | Standalone semantic-CBF prototypes: VLM→CBF pipeline, multi-prompt strategy, VLA integration, latent-space CBF |
| `vlm_pipeline/` | Perception and grounding evaluation: detection/grounding accuracy studies, integrated LIBERO eval |
| `epistemic_uncertainty/` | Uncertainty quantification (MC dropout, deep ensembles, entropy monitoring, density OOD) |
| `prompts/` | Prompt templates for obstacle identification, safety predicates, and STL specifications |
| `slurm/`, `experiments/` | Cluster job definitions for the evaluation campaigns (Bridges2-specific paths; see Running) |
| `patches/` | Patches applied to the pinned submodules |
| `results_tables/` | Aggregated benchmark tables |
| `fol_*_n50/`, `baseline_*/`, `*_benchmark/` | Per-condition evaluation result JSONs (n=50 per task) |

Rollout videos, raw VLM observation dumps, CBF visualisation HTML, and cluster logs are
not tracked — they are regenerable from the scripts above and are excluded by
`.gitignore`.

## Setup

```bash
git clone --recurse-submodules https://github.com/AkankshaSingal8/situational_safety_for_embodied_agents.git
cd situational_safety_for_embodied_agents
```

The project uses several conda environments; each has a setup script:

```bash
bash libero_env_setup.sh          # LIBERO + robosuite + MuJoCo (Python 3.8)
bash openvla_safelibero_setup.sh  # OpenVLA-OFT policy stack (Python 3.10)
bash qwen_vlm_env_setup.sh        # local Qwen-VL server
bash openpi_setup.sh              # π0.5 policy stack
```

API-backed VLMs read their keys from the environment. Copy `.env.example` to `.env` and
fill in whichever providers you intend to use:

```bash
cp .env.example .env
```

Headless rendering (any cluster or CI machine) requires:

```bash
export MUJOCO_GL=egl
```

## Running

Evaluate OpenVLA-OFT on a SafeLIBERO suite with the FOL safety filter enabled
(this is the exact form used in the reported n=50 campaigns; see `slurm/` for the
full job scripts):

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

Drop `--use_fol_filter` to produce the unfiltered baseline for the same condition.
`--safety_level` selects the SafeLIBERO hazard level (`I` or `II`); `--task_suite_name`
accepts `safelibero_{spatial,object,goal,long}`.

Run the VLM prompt experiments (backend selected by `FOL_VLM_BACKEND` or the runner's
own flags):

```bash
python vlm_prompt_runner/run_prompt_experiment.py --help
python vlm_prompt_runner/run_majority_vote_experiment.py --help
python vlm_prompt_runner/run_multi_model_experiment.py --help
```

Aggregate result JSONs into the summary tables:

```bash
python scripts/aggregate_results.py --help
python final_results.py
```

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

Headline results are in [`results_tables/summary_table.md`](results_tables/summary_table.md)
and [`fol_safety_filter/README.md`](fol_safety_filter/README.md).

## License

MIT — see [`LICENSE`](LICENSE). Submodules and benchmark datasets carry their own licenses.
