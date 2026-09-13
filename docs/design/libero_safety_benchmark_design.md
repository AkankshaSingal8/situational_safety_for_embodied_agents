# LIBERO-Safety Benchmark Replication — Design Spec

Date: 2026-07-10
Status: Approved for planning

## Goal

Replicate the environment/benchmark from the paper *LIBERO-Safety: A Comprehensive
Benchmark for Physical and Semantic Safety in Vision-Language-Action Models*
(arXiv:2606.23686, ECCV 2026) well enough to run eval experiments for four
policies — **pi0**, **pi0.5**, **OpenVLA**, **OpenVLA-OFT** — on their benchmark
tasks, using the same eval-setup conventions already used in this repo for
SafeLIBERO (per-policy setup script + draccus-config eval script, or
server/client split for openpi policies).

Non-goals: no training/fine-tuning, no reproducing the paper's own fine-tuned
checkpoints (only `pi05_libero_safety` is public; pi0/OpenVLA/OpenVLA-OFT
checkpoints and the data-generation code are not released), no modifying any
existing environment, repo, or branch.

## Source material

- Paper: https://arxiv.org/abs/2606.23686
- Project page: https://libero-safety.github.io/
- Benchmark repo: https://github.com/LIBERO-SAFETY/LIBERO-Safety
- Assets: `LIBERO-Safety/libero_safety_assets` (HF dataset)
- Dataset (demos, not needed for eval-only): `LIBERO-Safety/libero_safety`
- Their released checkpoint (not used here): `LIBERO-Safety/pi05_libero_safety`

## Isolation strategy

- New branch `libero-safety-benchmark`, created off `full_working_pipeline`
  (current HEAD, already pushed to origin).
- New top-level sibling directory `LIBERO-Safety/` in the project root,
  cloned from the benchmark repo — fully separate from `SafeLIBERO/`. Its own
  `~/.libero/config.yaml`-equivalent (via `LIBERO_CONFIG_PATH` env var pointed
  at a path under `LIBERO-Safety/`) so it never collides with the existing
  LIBERO config used by `SafeLIBERO/` evals.
- Two brand-new conda envs at
  `/ocean/projects/cis250185p/asingal/envs/`, mirroring the existing
  per-policy-family split already used in this repo, but fully isolated from
  the existing `aegis` and `openvla_libero_merged` envs:
  - `libsafety_openpi` — JAX/openpi stack for pi0 + pi0.5 (mirrors
    `openpi_setup.sh` / the `aegis` env versions)
  - `libsafety_openvla` — PyTorch/transformers stack for OpenVLA +
    OpenVLA-OFT (mirrors `openvla_safelibero_setup.sh` / `openvla_libero_merged`)

No existing env, repo directory, branch, or config file is modified.

## Checkpoint strategy

Off-the-shelf, standard LIBERO-pretrained checkpoints for all four policies
(not the paper's own fine-tuned weights):

| Policy | Checkpoint source |
|---|---|
| pi0 | `physical-intelligence` openpi `pi0_libero` base checkpoint |
| pi0.5 | `physical-intelligence` openpi `pi05_libero` base checkpoint (same one already used by `aegis`, downloaded fresh into the new env's checkpoint dir) |
| OpenVLA | `openvla/openvla-7b-finetuned-libero-*` (base, non-OFT) |
| OpenVLA-OFT | `moojink/openvla-7b-oft-finetuned-libero-*` (already used in this repo's SafeLIBERO eval) |

These are zero-shot evals of standard checkpoints against the new
LIBERO-Safety task suites/safety tiers — not a claim of matching the paper's
exact reported numbers (which depend on their unreleased fine-tuned models).

## Eval setup — mirrors existing SafeLIBERO conventions

**OpenVLA / OpenVLA-OFT (single-process, draccus config):**
Follow the `run_libero_eval.py` pattern — a `GenerateConfig` dataclass
(`model_family`, `pretrained_checkpoint`, task-suite enum, etc.), loop over
`LIBERO-Safety` task suites, log Success Rate / Refusal Rate / Collision Rate.
New script: `LIBERO-Safety/run_libsafety_eval_openvla.py`, run inside
`libsafety_openvla`.

**pi0 / pi0.5 (server/client, like `running_pi05_safelibero.sh`):**
1. Start policy server: `openpi/scripts/serve_policy.py policy:checkpoint
   --policy.config=<pi0_libero|pi05_libero> --policy.dir=<checkpoint>` inside
   `libsafety_openpi`.
2. Run client eval script `LIBERO-Safety/run_libsafety_eval_openpi.py`
   (adapted from `vlsa-aegis/main/pi05_evaluation.py`) inside a LIBERO-capable
   env, pointed at `LIBERO-Safety`'s task suites via `PYTHONPATH`.

Both eval scripts write videos/metrics into a new `LIBERO-Safety/rollouts/`
and `LIBERO-Safety/results/` directory (mirroring this repo's `rollouts/` /
`results/` convention), separate from the existing SafeLIBERO output dirs.

## Validation

- Each conda env: import smoke test (`import openpi` / `import openvla_oft`,
  `torch.cuda.is_available()` / `jax.devices()`).
- `LIBERO-Safety` repo installs and its bddl/task-suite loader runs
  (`benchmark.get_benchmark_dict()` equivalent) without touching the existing
  `SafeLIBERO` config.
- One smoke episode per policy (1 task, 1 episode, `--num_trials_per_task 1`)
  completes end-to-end and produces a rollout video + metrics JSON, before
  scaling to full task-suite runs.

## Open items deferred to implementation

- Exact system-dependency install path on Bridges2 (no `sudo` — likely
  `module load` or user-local builds for `libexpat1`/`libfontconfig1-dev`/etc.)
- Exact HF checkpoint IDs for base `openvla-7b-finetuned-libero-*` per task
  suite (spatial/object/goal/long), confirmed at implementation time.
