# Learned Safe-Success Critic + Tournament Completion — Design

Date: 2026-07-13
Branch/worktree: `safe-success-steering-sota-all-suites` (`.worktrees/viability-steering-tournament-20260713`)

## Context

The viability steering tournament (`results_tables/viability_steering_tournament.md`)
compared candidate-steering variants for in-denoising DCBF guidance on pi0.5 /
SafeLIBERO Spatial. Its conclusion: a single distance-based phase switch can't
distinguish grasp-critical target contact from avoidable obstacle contact, and
`semantic_best_of_k`'s hand-tuned clearance-minus-correction selector picks
trajectories that are geometrically safe but task-inviable. The recommended next
step: combine calibrated in-denoising candidate generation with a **learned,
semantic safe-success critic**, keeping FOL/hard geometry as the final verifier
rather than the candidate selector.

This spec covers building that critic and using it to complete the tournament
with a directly comparable result row.

## Existing infrastructure being reused

- `vlm_pipeline/run_guided_safelibero_pi05_eval.py` — the tournament's single
  eval client. Implements `--steering_mode {fixed,phase_adaptive,best_of_k,
  semantic_best_of_k,oracle_counterfactual}`.
- `_infer_diverse_candidates()` (line 205) and `_inject_topology_experts()`
  (line 230) — generate K qualitatively diverse candidates (varied DCBF
  guidance mode + hand-crafted geometric detour/retreat experts). Reused
  unchanged for the new mode.
- `_oracle_counterfactual_score()` (line 313) — forks the live sim
  (`env.get_sim_state()` / `env.regenerate_obs_from_state()`), steps each
  candidate's actions for real, and computes a lexicographic composite score
  from collision, task completion, minimum separation, progress, and DCBF
  correction norm. This is the **label source** for the critic — no new
  scoring logic needed, only logging of what it already computes.
- Server-side DCBF/geometry guidance (`openpi_guided/scripts/serve_policy_guided.py`)
  is applied during denoising for whichever candidate is requested, independent
  of how that candidate was selected. This means "geometry remains the final
  verifier" is already true of the architecture and requires no new code — the
  critic only changes which candidate gets requested from the guided policy
  server, not whether guidance runs.
- `viability_tournament_smoke.slurm` — existing SLURM launcher pattern
  (server + client, env-var-driven mode/K/scale/trials). Reused for both the
  data-collection run and the new mode's smoke/promoted runs.

## Architecture

```
Data collection run (oracle_counterfactual + --log_training_data)
  → per-candidate rows: {clip_embed, candidate_features, oracle_label}
  → semantic_cbf/train_safe_success_critic.py (offline)
  → semantic_cbf/safe_success_critic.py checkpoint

Live eval, steering_mode=learned_critic:
  current frame → CLIP embed (once per replan step)
  _infer_diverse_candidates + _inject_topology_experts (existing, no sim fork)
  → per-candidate features (existing, already computed for other modes)
  → critic.forward(clip_embed, candidate_features) → score
  → argmax select → send to guided policy server (unchanged DCBF guidance)
```

### 1. Data logging

Add `--log_training_data <path>` to `run_guided_safelibero_pi05_eval.py`,
active only when `steering_mode == oracle_counterfactual`. For every candidate
already scored by `_oracle_counterfactual_score`, append one row to a
per-episode `.npz` shard under `<path>/<task>/<level>/<episode_id>.npz`:

- `clip_embed`: float32[512] — CLIP ViT-B/32 embedding of `agentview_image`
  for the current replan step, computed once and shared across all candidates
  of that step (candidates share the same starting frame; only the oracle's
  simulated rollout differs, and that's exactly the outcome being labeled).
- `candidate_features`: float32[~12] — `min_clearance`, `correction_norm`,
  `expert` (one-hot over the known expert labels), endpoint delta from EEF,
  obstacle distance, phase (approach/transport one-hot), progress-to-target
  distance before the step.
- `label`: the `_oracle_counterfactual_score` composite float, plus raw
  `collided`/`done` booleans for diagnostics.

No changes to `_oracle_counterfactual_score` itself — only a logging hook
around its existing call site.

### 2. Data collection run

Rerun `oracle_counterfactual` mode (existing SLURM script, new env var for
`--log_training_data`) on `safelibero_spatial`, levels I and II, `TRIALS=3`
(bounded — oracle forking is the slow path; this is a data-volume choice, not
a final-eval one). Expected: ~24 episodes × ~dozens of replan steps ×
K candidates ≈ several thousand labeled rows.

### 3. Critic

New `semantic_cbf/safe_success_critic.py`:

```python
class SafeSuccessCritic(nn.Module):
    # [clip_embed(512); candidate_features(~12)] -> 256 -> 256 -> 1
```

Same MLP spirit as `SafetyMarginNetwork` in `latent_cbf.py`, but a new class —
`SafetyMarginNetwork`'s `(scene_embed, object_embed, state)` signature doesn't
fit a single-image + candidate-features input and forcing it would be a worse
fit than a small new class.

`semantic_cbf/train_safe_success_critic.py`: loads the `.npz` shards, splits
by episode (not by row, to avoid leakage across a step's candidates), trains
with MSE against the oracle composite score, reports held-out correlation/MAE.
Saves a checkpoint under `semantic_cbf/checkpoints/safe_success_critic.pt`.

**Gate before touching the eval loop:** held-out score must show the critic
tracks the oracle ranking (e.g. Spearman correlation clearly positive, and the
critic's argmax-selected candidate matches the oracle's argmax on a majority
of held-out steps). If it doesn't, stop and diagnose before running any live
eval — a live smoke run isn't a substitute for checking the offline fit.

### 4. New steering mode `learned_critic`

Added to the `--steering_mode` choices. Per replan step:
1. One CLIP forward pass on the current frame (embedding cached for that
   step, matching the training-time convention).
2. `_infer_diverse_candidates` + `_inject_topology_experts` — identical to
   `oracle_counterfactual`, minus the sim fork.
3. Extract the same `candidate_features` used in training (already computed
   inline for the other modes' diagnostics).
4. `critic.forward(...)` per candidate, `argmax` select.
5. Selected candidate's actions proceed exactly as today — server-side DCBF
   guidance already ran during that candidate's denoising, unchanged.

No server-side changes. No changes to `_candidate_score` or
`_oracle_counterfactual_score` (both remain available for the older modes).

### 5. Tournament completion

1. Micro-smoke: `TRIALS=2`, both levels, `learned_critic` mode, same SLURM
   pattern as existing smokes.
2. Gate: TSR/CAR should not be worse than `best_of_k` on CAR, and should move
   toward `oracle_counterfactual`-quality selection without its per-step sim-fork
   cost. If the smoke result is clearly broken (e.g. critic always picks the
   same expert, CAR collapses), stop and diagnose rather than promoting.
3. If the gate passes: promote to `TRIALS=5`, both levels (matches the
   existing "promoted" convention).
4. Append a new row (or rows, if both smoke and promoted are worth keeping
   for the record) to `results_tables/viability_steering_tournament.md`,
   alongside the existing variants and the pi0.5 baseline / r3 full-geometry /
   arXiv:2607.01378 reference points already in that table.
5. State a final recommended method in the doc's Interpretation section,
   updating (not replacing) the existing analysis paragraph.

## Testing

- Offline: critic held-out correlation/MAE check (step 3's gate) before any
  live run.
- Smoke run (step 5.1-5.2) before the promoted run — same iterate-fast
  convention the existing tournament used.
- No unit tests needed for `_infer_diverse_candidates` /
  `_inject_topology_experts` / server guidance — unchanged, already exercised
  by the existing `oracle_counterfactual` and `semantic_best_of_k` runs.
- New code (logging hook, critic module, training script, `learned_critic`
  selection path) is client-side Python — verify with a short offline
  smoke test of the CLIP embedding call and critic forward pass shapes before
  the first live SLURM run, to avoid burning a GPU job on a shape bug.

## Out of scope

- No changes to the DCBF/geometry guidance itself (server-side).
- No new task suites beyond `safelibero_spatial` (matches the existing
  tournament's scope).
- No hyperparameter sweep over CLIP model choice, MLP size, or K — use the
  documented defaults (CLIP ViT-B/32, 256/256 MLP, existing tournament's K)
  unless the offline gate or smoke run indicates a specific problem.
