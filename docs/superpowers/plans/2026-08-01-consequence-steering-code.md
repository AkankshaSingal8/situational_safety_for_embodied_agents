# Plan: consequence-steering code phases (B/C/D of the GO/NO-GO prototype)

Parent design (requirements authority): `docs/superpowers/specs/2026-08-01-consequence-steering-design.md`.

## Global Constraints

- All new flags default off; defaults-off behavior and server payloads
  byte-identical to current HEAD. Full existing CPU test suite
  (`vlm_pipeline/tests/`) must keep passing; new tests CPU-only.
- Never use the `_obstacle_` name substring in decision logic.
- No API/VLM calls. No new heavyweight dependencies (torch is available in the
  `aegis` env for training; the CLIENT env must not require torch — logging is
  pure JSON).
- Commit per task on branch `new-steering-method`, worktree
  `.worktrees/flow-guidance-tier-gt`; plain commits (parallel session shares
  the branch).
- Test env for client tests: `source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh
  && conda activate openvla_libero_merged && python -m pytest vlm_pipeline/tests/ -q`.

## Task 1: `--log_transitions` in both eval clients

Add flag `--log_transitions` (store_true, default off) to
`vlm_pipeline/run_guided_libero_safety_eval.py` AND
`vlm_pipeline/run_guided_safelibero_pi05_eval.py`.

When on: for every `env.step()`, append one JSON line to
`<results_output_dir>/transitions_<suite-or-task>_<level>.jsonl` with:
`{"task": int, "ep": int, "t": int, "eef": [x,y,z], "grip": [q0,q1],
"action": [7 floats as sent to env], "entities": {name: [x,y,z], ...}}`
where `entities` uses the client's existing GT position source for all tracked
hazards/objects that the runner already resolves positions for (reuse the
existing pos_of/entity machinery — do NOT build new perception). Flush per
episode. File handle opened lazily on first write; recorded in results JSON
metadata (`"log_transitions": bool`).

Requirements: zero overhead when off (no dict building on the off path);
pure helper `_transition_record(task, ep, t, eef, grip, action, entities)`
returning the dict (unit-testable); writing stays in the runner loop.
Tests (new file `vlm_pipeline/tests/test_transition_logging.py`): record
schema/values, JSON round-trip, defaults-off no-op (no file, no payload
change), and a source-level test that both clients expose the flag.

## Task 2: consequence model + training + offline G1 evaluation

New file `vlm_pipeline/consequence_model.py` (runs in the `aegis` env, torch):

- Dataset: load `transitions_*.jsonl` files from a directory tree; build
  samples: input = (eef, grip, per-entity relative positions padded to E=8
  slots with presence mask, next H=10 action translations, domain tag one-hot),
  targets = (realized eef path over next H steps; CONTACT_i = min over window
  dist(eef, entity_i) < 0.05 for each entity slot; HOLDING at t+H = grip width
  in (0.002, 0.045) band; progress = dist-to-target(t) − dist-to-target(t+H),
  target entity name passed per episode). Episode-level train/held-out split
  (80/20 by episode id, seeded, split saved to JSON).
- Model: GRU over the H action steps + MLP trunk; heads: path (H×3),
  contact logits (E), holding logit, progress scalar. Ensemble = 3 seeds.
- Training loop: Adam, early stop on held-out loss; save
  `consequence_model_<domain>.pt` (3 members) + norm stats.
- `evaluate_g1()`: on held-out episodes, compute (a) median terminal-point
  error vs the calibrated single-integrator baseline (translation_scale 2.0 ×
  cumsum, cmd_clip 0.025) restricted to samples with any entity within 0.25 m;
  (b) CONTACT AUC (pooled over slots with positives); (c) Spearman(progress
  pred, realized). Print a G1 verdict line
  `G1: PASS/FAIL err_ratio=.. auc=.. spearman=..` and write
  `g1_report.json`. Thresholds from the design doc: ratio ≤ 0.5, AUC ≥ 0.90,
  Spearman ≥ 0.6.
- CLI: `python consequence_model.py train --data_dir D --out_dir O`,
  `python consequence_model.py g1 --data_dir D --model_dir O`.

Tests (`vlm_pipeline/tests/test_consequence_model.py`, CPU, tiny synthetic
JSONL fixtures written by the test): dataset parsing + padding/mask, label
computation (contact/holding/progress) against hand-computed values,
single-integrator baseline math, split determinism, one 2-epoch smoke train on
synthetic data (tiny dims) verifying loss decreases and G1 report is emitted.
Guard imports so the client test suite still passes in envs without torch
(skip markers).

## Task 3: `--consequence_select` selection layer in the guided server

In `openpi_guided/scripts/serve_policy_guided.py` + the guided model
(`openpi_guided/src/openpi/models/pi0_guided.py`): flag
`--consequence_select MODEL_DIR` (default None = off; off path byte-identical).

When on: after the existing best-of-K candidates are fully denoised, run the
ensemble on all K chunks (batch): feasible = P(contact any hazard) ≤ 0.10
with pessimism penalty λ=1.0 × ensemble std added to the probability;
among feasible pick max predicted progress; if feasible empty → existing
selection/repair path unchanged (fallback = backstop). Log per-query
`consequence_select: {"feasible": k, "chosen": i, "fallback": bool}` into the
server's existing per-query debug channel if one exists, else stdout at INFO.
Model loads once at server start (fail fast with a clear error if MODEL_DIR
invalid). Latency: single batched forward for K=8; assert < 10 ms on CPU in a
test with tiny dims (structural, not a benchmark).

Tests (`vlm_pipeline/tests/test_consequence_select.py`, CPU with tiny synthetic
checkpoint written by the test): feasibility rule incl. pessimism, argmax
progress among feasible, empty-feasible fallback flagged, defaults-off (None)
path leaves candidate selection untouched (source-level + unit), checkpoint
load error handling. Server-side code must be importable without a GPU.
