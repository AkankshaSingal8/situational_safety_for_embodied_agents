# Safety Memory — Proof of Life

Tests one question: **does persisting a fact across episodes change safety outcomes?**
Predicates and rules are fixed and human-written. Nothing is induced.

Results: [`RESULTS.md`](RESULTS.md). Plan:
[`docs/design/2026-09-19-safety-memory-proof-of-life-protocol.md`](../../docs/design/2026-09-19-safety-memory-proof-of-life-protocol.md).

## The idea in one line

With fixed rules, memory can only affect safety when the rule's antecedent is not
recoverable from the current observation. So the endpoint is an interaction: memory
should be decisive when the hazard is hidden and a no-op when it is visible.

## Layout

| path | role |
|---|---|
| `src/ledger.py` | independent ground truth; never reads the runtime's beliefs |
| `src/memory.py` | durable JSON fact store (facts, never rules) |
| `src/rules.py` | the single fixed rule and its grounding |
| `src/cells.py` | the four cells and the four belief-source arms |
| `src/filter.py` | discrete control-barrier projection, shared by every arm |
| `src/checkpoints.py` | robot body checkpoints, so the barrier describes the arm |
| `src/policy.py` | scripted waypoint controller |
| `src/runner.py` | exposure episode, settle, scored episode |
| `src/kinematic_sim.py` | simulator-free stand-in with the same interface |
| `run_campaign.py` | runs the cell x arm panel |
| `analyze.py` | per-cell rates, paired tests, the interaction, deletion panel |
| `leak_check.py` | proves the hidden property is not in the scored observation |

## Reproduce

Tests (CPU, no simulator, seconds):

```bash
python -m pytest tests/ -q          # 44 tests
```

Kinematic board and leak check (CPU, about a minute):

```bash
python run_campaign.py --sim kinematic --pairs 40 \
    --arms memoryless,memory,memory_deleted,oracle --out runs/kin
python analyze.py --runs runs/kin
python leak_check.py --sim kinematic --pairs 60
```

Physics board (batch scheduler required, see below):

```bash
sbatch --export=ALL,PAIRS=20,OUT=runs/libero,SEED0=2000 final.slurm
```

## Environment notes that cost time

- **The login node hangs** on environment construction under both `egl` and `osmesa`,
  blocked at near-zero CPU. Even physics-only MuJoCo must go through `sbatch`.
- The installed `libero` **is** the SafeLIBERO fork; only `safelibero_*` suites exist.
- Use `ControlEnv(..., use_camera_obs=False, has_offscreen_renderer=False,
  has_renderer=False, camera_names=[])` for physics without a renderer.
- **The scene is not at rest after `reset()`.** Objects drift on their own, so every
  measurement starts after a settle phase and unsettled scenes are excluded. See
  `RESULTS.md`.
- Measured delta-EEF action scale is **0.0113 m/step**, not the 0.05 implied by config.
