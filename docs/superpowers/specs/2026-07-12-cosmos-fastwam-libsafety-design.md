# Cosmos Policy + Fast-WAM on LIBERO-Safety — Design

## Context

The LIBERO-Safety worktree already has off-the-shelf eval drivers for OpenVLA (base + OFT)
and pi0.5 (`run_libsafety_eval_openvla.py`, `run_libsafety_eval_openpi.py`), producing the
results table reported earlier this session (3/180, 1/180, 3/180 successes across the four
task suites: `human_safety`, `obstacle_avoidance`, `obstacle_avoidance_human`, `affordance`).

Separately, the SafeLIBERO benchmark (main repo, not this worktree) already has two additional
world-model-based baseline policies wired up and validated:

- **Cosmos Policy** (`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`) — video diffusion transformer,
  run inside an Apptainer container (`cosmos-policy/cosmos_policy.sif` + `.venv_rhel8`) because
  `transformer_engine` needs glibc ≥ 2.29 (Bridges2 nodes run RHEL8/glibc 2.28).
- **Fast-WAM** (`yuanty/fastwam`, built on Wan2.2-TI2V-5B) — video diffusion policy, run in a
  dedicated conda env (`fastwam_env`, torch 2.7.1+cu128) because it conflicts with the
  OpenVLA/OFT torch 2.2.0 env.

Both are documented in `docs/cosmos_policy_safelibero.md` and `fast-wam/{README,NOTES}.md`
(this worktree) and already fully built/downloaded in the **main repo**
(`/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/{cosmos-policy,fast-wam}`).

Goal: bring these same two baselines to LIBERO-Safety, reusing the existing built artifacts
read-only (per explicit user constraint: must not risk breaking the currently-working SafeLIBERO
pipeline), following the fork-and-adapt pattern already established by
`run_libsafety_eval_openvla.py` relative to `run_libero_eval.py`.

## Non-goals

- No changes to any file in the main repo (`cosmos-policy/`, `fast-wam/`, `vlm_pipeline/`,
  SafeLIBERO eval scripts). Everything referenced there is read-only.
- No refactor of shared env-setup logic into a common module across SafeLIBERO/LIBERO-Safety.
- No `--safety_level` flag — LIBERO-Safety encodes safety level per-task
  (`task.level`/`task.level_id`), unlike SafeLIBERO's global L1/L2 argument.
- Full-scale eval (4 suites × 45 episodes) is a later step, gated on smoke tests passing first.

## Design

### New files (all in the LIBERO-Safety worktree)

1. `run_libsafety_cosmos_policy_eval.py`
2. `run_libsafety_fastwam_eval.py`
3. `slurm/smoke_cosmos_libsafety.slurm`
4. `slurm/smoke_fastwam_libsafety.slurm`
5. (later, once smoke tests pass) `slurm/eval_cosmos_libsafety_<suite>.slurm` ×4,
   `slurm/eval_fastwam_libsafety_<suite>.slurm` ×4

### What's reused verbatim vs. adapted

| Concern | Source | Treatment |
|---|---|---|
| Model loading (`get_model`, checkpoint/backbone loading) | `run_safelibero_cosmos_policy_eval.py`, `run_safelibero_fastwam_eval.py` | Copied verbatim |
| Action inference loop, image preprocessing, proprio construction | same | Copied verbatim |
| Collision tracking (obstacle displacement > 1mm) | same | Copied verbatim — both scripts already compute TSR *and* CAR independently, unlike the openvla driver |
| Environment creation | `run_libsafety_eval_openvla.py` (`get_libsafety_env`) | Adapted: `task_suite.get_task_bddl_file_path_by_level_id(task.level, task.level_id)` instead of SafeLIBERO's simpler bddl path lookup |
| Task-suite / init-state loading | `run_libsafety_eval_openvla.py` | Adapted: `task_suite.get_task_init_states(task.level, task.level_id)` |
| Path setup for external repos | `run_libsafety_eval_openvla.py` (`OPENVLA_OFT_PATH` hardcoded absolute path) | Same convention: hardcode `COSMOS_POLICY_PATH` / `FASTWAM_PATH` to the **main repo** absolute paths |

### Artifact reuse (read-only, absolute paths into the main repo)

```
COSMOS_SIF   = /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy/cosmos_policy.sif
COSMOS_VENV  = /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy/.venv_rhel8
COSMOS_SRC   = /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy
FASTWAM_SRC  = /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/fast-wam/FastWAM
FASTWAM_CKPT = /ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/fast-wam/checkpoints
FASTWAM_ENV  = /ocean/projects/cis250185p/asingal/envs/fastwam_env   (conda env, shared across worktrees)
```

Nothing under these paths is written to. New eval scripts only *read* checkpoints/venvs and
*import* code from `COSMOS_SRC`/`FASTWAM_SRC` via `sys.path.insert`, exactly like
`OPENVLA_OFT_PATH` is used today.

### Output locations (new, worktree-local — never collide with SafeLIBERO's)

```
cosmos_benchmark_libsafety/   # JSON results
cosmos_video_libsafety/       # rollout mp4s
fastwam_benchmark_libsafety/  # JSON results
fastwam_video_libsafety/      # rollout mp4s
```

Result JSON schema mirrors the existing SafeLIBERO cosmos/fastwam schema (per-task TSR/CAR/ETS
+ an `overall` block), but with `task_suite`/`level` fields matching the LIBERO-Safety
convention used by `run_libsafety_eval_openvla.py`'s output so all four models' results can be
aggregated into one table later.

### Testing / rollout plan

1. **Smoke test** each script: `--task_suite_name human_safety --task_index 0 --num_trials_per_task 1`.
   Confirms env creation, container/venv or conda-env activation, checkpoint loading, and the
   action inference loop all work end-to-end against LIBERO-Safety's env/task-suite API.
2. Only after both smoke tests exit 0 and produce a plausible-looking result JSON: submit full
   evals (4 suites × 45 episodes) as separate SLURM jobs, mirroring existing resource specs
   (H100, apptainer for cosmos / `fastwam_env` conda for fast-wam).
3. No unit tests — this is an eval-harness port; correctness is judged by the smoke run
   producing sane step counts (<300) and a non-crashing full episode loop, same bar used when
   the openvla/pi0.5 drivers were originally validated.

### Error handling

Same pattern as `run_libsafety_eval_openvla.py`: per-task `try/except` so one task's exception
doesn't kill the whole suite; `env.close()` in a `finally` block for cleanup between tasks.
