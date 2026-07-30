# CDC implementation plan — Certificate-Distillation Critic

Approved design: `docs/superpowers/specs/2026-07-30-cdc-design.md` (written
alongside this plan). Branch: `flow-guidance-tier-gt` worktree. Goal: distill
the geometric safety certificate into a language-conditioned chunk critic;
actuator = best-of-K selection; phases D0 (dataset) → D1 (train/offline eval)
→ D2 (runtime swap dev cells) → D3 (paper campaign), auto-gated.

## Global Constraints

- Python only; client code runs in conda env `openvla_libero_merged`
  (torch available); server code is jax (openpi_guided) — no torch on server.
- NO behavior change to production paths unless the new flag is passed:
  `--log_critic_dataset` (client) and `return_all_candidates` (payload)
  must default OFF and leave existing runs bit-identical.
- Never call paid APIs. All models local/frozen (DINOv2-S via torch.hub
  cache, MiniLM or similar via transformers local cache); pre-download in a
  setup step, never at job runtime from scratch.
- Unit tests must run on CPU login node (no GPU assumptions in tests).
- Results JSONs store rates as fractions; scripts that compare must ×100.
- SLURM: `source /jet/home/asingal/miniconda3/etc/profile.d/conda.sh`,
  `export MUJOCO_GL=egl`, exclude nodes `w002,w004,w005,w006,w007,w008,w009,w010`,
  logs to /ocean/projects/cis250185p/asingal/slurm_logs/, always pass
  explicit `--export=ALL,...` values.
- Commit messages end with the Claude Co-Authored-By trailer.

## Task 1: Dataset logger in the eval client

Extend `vlm_pipeline/run_guided_safelibero_pi05_eval.py` with
`--log_critic_dataset DIR` (default None = off).

The client already has per-replan K-candidate diagnostics logging (added for
RMCS Phase 0, commit 157028e — grep `k_diag` in the file). When the new flag
is set, at every replan ALSO write:

- a JPEG of the CURRENT agentview observation, 224×224 (use the same
  rotation/resize convention as the policy element construction in this
  file: image is `obs["agentview_image"][::-1, ::-1]` then resize_with_pad
  to 224), saved under `DIR/<suite>/<level>/t<task>/ep<idx>/replan<k>.jpg`,
  JPEG quality 85;
- one JSONL record per replan appended to
  `DIR/<suite>/<level>/t<task>/critic_ds.jsonl` with fields:
  `{img: <relpath from DIR>, suite, level, task_id, ep, replan_idx,
    eef_pos: [3], hazard_text: str, obstacle_ids: [str],
    K: int, chunks: [K][H][3] (translational dims of each candidate chunk),
    feasible: [K], min_margin: [K], tail_margin: [K], best: int}`
  where the per-candidate arrays come from the server's per-candidate
  diagnostics in `result["guidance"]` (grep for the keys the k_diag path
  already consumes; if candidate chunks are not currently returned, take
  them from `result` when the payload asks for them — see the existing
  k_diag machinery first and reuse whatever per-candidate data it already
  gets; if chunks are unavailable today, record the SELECTED chunk under
  `chunks_selected: [H][3]` and set `chunks: null` — Task 4 adds full-K).
- `hazard_text`: the hazard phrase used for this episode's guidance — the
  obstacle identity string(s) the client resolves (e.g. from the VLM-prior /
  symbolic identity source already in scope at payload-assembly time);
  join multiple with "; ".
- At episode end, append one record `{ep_summary: true, ep, task_id,
  success: bool, collision: bool, timeout: bool, steps: int}` to the same
  JSONL (timeout = not success and not collision). Consumers backfill
  outcomes onto that episode's replan records by (task_id, ep).

Tests (new `vlm_pipeline/tests/test_critic_logger.py`, CPU): factor the
record-building + backfill-key logic into small pure functions
(`vlm_pipeline/critic/logging_utils.py` is fine) and unit-test: record
schema round-trips through json, image relpath layout, ep_summary matching.
Do NOT try to run the simulator in tests.

Also: a `--log_critic_dataset` dry-run must not change any behavior when
the flag is off — verify by grepping that all new code is inside
`if args.log_critic_dataset:` guards (state this in the report).

## Task 2: Critic package — dataset, features, model, training

New package `vlm_pipeline/critic/` (torch, runs in openvla_libero_merged):

- `dataset.py`: loads one or more `critic_ds.jsonl` trees (Task 1 format),
  backfills outcomes from ep_summary records, expands to per-CANDIDATE
  examples: x = (img_path, hazard_text, chunk[H][3]), y =
  {infeasible: 1-feasible[i], min_margin: float}. Skips records with
  `chunks: null`. Split helpers: `split_random(seed, frac)` and
  `split_leave_one_suite_out(suite)` — split by EPISODE (never leak replans
  of one episode across splits), and for LOSO by suite.
- `features.py`: precompute-and-cache embeddings. Image: DINOv2-S
  (`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`, frozen,
  384-d CLS), cache to `<ds_dir>/feat_cache/img/<sha1(relpath)>.npy`.
  Text: frozen `sentence-transformers/all-MiniLM-L6-v2` if importable via
  transformers, else DistilBERT CLS; cache per unique hazard string to
  `<ds_dir>/feat_cache/text.json` (few dozen strings). CLI:
  `python -m vlm_pipeline.critic.features --ds_dir D [--device cuda]`.
- `model.py`: `ChunkCritic(nn.Module)`: inputs img_emb (384), text_emb
  (D_t), chunk (H*3 flattened, normalized by a stored per-dim scale).
  Chunk+img trunk MLP [512, 256]; text conditions via FiLM (linear from
  text_emb to per-layer scale+shift). Heads: `logit_infeasible` (1) and
  `margin_pred` (1). ~2-5M params.
- `train.py`: BCE(infeasible) + 0.5*SmoothL1(margin, only where margin is
  finite), AdamW lr 3e-4, cosine, 30 epochs, batch 1024 (features are
  cached vectors — this is minutes). CLI takes `--ds_dir --split
  random|loso:<suite> --out ckpt.pt`; writes `metrics.json` with held-out
  ROC-AUC (feasibility) and margin MAE. Deterministic seeding.
- Tests `vlm_pipeline/tests/test_critic.py` (CPU, no downloads): model
  forward shapes; FiLM actually consumes text (zeroing text_emb changes
  output); dataset expansion + episode-level split disjointness on a tiny
  synthetic JSONL written by the test; training loop runs 2 epochs on
  synthetic random features and loss decreases.

## Task 3: Offline ranking eval + gate scripts

- `vlm_pipeline/critic/eval_ranking.py`: given a trained ckpt and a ds_dir,
  for each replan record with full-K chunks, rank candidates by critic score
  (feasible-first via p(infeasible), tie-break by margin_pred) and report:
  top-1 agreement with the certificate's own selection rule (feasibility
  desc, then min_margin desc — implement that reference rule here), mean
  regret in certificate min_margin, grouped by suite. Writes JSON.
- `vlm_pipeline/critic/gate_g0.py DS_DIR`: counts replan records,
  certificate-infeasible candidate examples, collision episodes; GO iff
  records >= 40000 AND infeasible_examples >= 8000 AND collision_eps >= 60;
  writes `DS_DIR/gate_g0.json {go: bool, counts...}`; exit 0 on GO, 3 on
  NO-GO.
- `vlm_pipeline/critic/gate_g1.py`: reads the three metrics.json files
  (random split, loso splits) + eval_ranking JSON; GO iff held-out AUC
  >= 0.85 (random) AND top1 >= 0.70 AND (random_auc - min_loso_auc) <= 0.10;
  same output/exit convention.
- `vlm_pipeline/critic/gate_g2.py`: takes production and swap-arm results
  JSONs (fraction rates ×100); GO iff swap within 5.0 pp of production on
  both TSR and CAR; separately reports headline-arm vs baseline
  (CAR >= baseline+15 AND TSR >= baseline-5). Same output/exit convention.
- Perfect-teacher sanity: `vlm_pipeline/tests/test_gates.py` (CPU) builds a
  synthetic dataset where a linear rule generates labels, "trains" a
  stand-in scorer = the true rule, and asserts eval_ranking reports top1 =
  1.0 and gate_g1 logic fires GO/NO-GO correctly on crafted metrics dicts;
  also unit-test g0/g2 threshold edges.

## Task 4: Server `return_all_candidates` + client selection path

Server (`openpi_guided/src/openpi/models/pi0_guided.py`,
`openpi_guided/src/openpi/policies/guided_policy.py`):

- New GuidanceParams field `return_all_candidates` (payload-readable like
  `hdc_scale`; default 0/off). When on and K>1 in `guided_sample_actions`:
  still compute `best` and the selected chunk as today (response `actions`
  UNCHANGED — the executed chunk must stay the server-selected one unless
  the client overrides), but ALSO return in diagnostics:
  `cand_actions` (K,H,3 translational dims of x_0), plus per-candidate
  `cand_feasible`, `cand_min_margin`, `cand_tail_margin` (K,), and `best`
  (scalar). K is static so jit shapes are fine.
- Policy wrapper: the diag coercion currently forces every entry to
  `float(np.asarray(v)[0])` — route entries whose name starts with `cand_`
  (and `best`) through `np.asarray(v)` instead (msgpack needs numpy arrays,
  never jax arrays).
- Client (`run_guided_safelibero_pi05_eval.py`): when
  `--critic_select CKPT` is passed, send `return_all_candidates: 1.0` in the
  guidance payload, load the Task-2 critic once at startup (torch, CPU or
  cuda if free), score the K returned candidates from the CURRENT agentview
  frame + hazard text + candidate chunks, and execute the critic-chosen
  candidate's chunk instead of `result["actions"]`. New flag
  `--critic_no_repair` additionally sets the server to schedule-none…
  NOTE: repair config is server-side; for the headline arm the SERVER is
  launched with `--repair_schedule none`; the client flag only needs
  `--critic_select`. Record per-replan chosen-vs-server-best agreement into
  the episode JSONL (`critic_agree_frac`).
- Task-1 logger upgrade: when full-K `cand_actions` are present in the
  response, fill `chunks` properly (remove the null fallback path if now
  dead, or leave both paths tested).
- Tests: pure-function tests for the client-side scoring/selection glue
  (mock response dict → chosen index), and for payload construction with
  the new flag. No server-side jax tests required beyond an import check;
  the D2 job smoke is the integration test.

## Task 5: SLURM chain D0→D3 with auto-gates

New scripts in `slurm/` (model on existing `fgd_vlmfin_r10_09.slurm` /
`sq_full_n50.slurm` two-process pattern: guided server + client):

- `cdc_d0.slurm`: array over 4 suites × {I,II}; server = production no-GT
  config (ramp K8 r0.09, num_candidates 8); client = board no-GT config for
  that suite (percep/symbolic/vlm-priors/corridor; Object adds
  `--entity_cameras agentview,birdview`) + `--log_critic_dataset
  $WT/cdc_ds` + n=10/task. Epilogue on the LAST array task (use a
  completion-count file or afterok dependency job): run gate_g0; on GO
  sbatch `cdc_d1.slurm`.
- `cdc_d1.slurm` (1 GPU): features precompute (cuda), then train random
  split + 4 LOSO splits, then eval_ranking on held-out, then gate_g1; on GO
  sbatch `cdc_d2.slurm`. Pre-download models in this job's prologue only if
  the torch.hub/HF caches are missing (they should be pre-warmed on the
  login node before D0 — document the exact warm-up commands in the script
  header as comments).
- `cdc_d2.slurm`: dev cells Spatial L1 + Object L1, paired n=20, three
  server+client arms: production (control), swap (`--critic_select`, server
  repair unchanged), headline (`--critic_select`, server
  `--repair_schedule none`); then gate_g2; on GO sbatch `cdc_d3.slurm`.
- `cdc_d3.slurm`: n=200 no-GT board cells Spatial L1+L2, Object L1+L2:
  headline arm vs production; plus LS transfer cell obstacle_avoidance L1
  (LS client script, critic trained on SafeLIBERO only), n=50.
- Every gate writes its JSON verdict into `$WT/cdc_ds/` or the arm output
  dir and appends one line to
  `results_tables/fgd_tier_gt_results.md`? NO — slurm jobs must not edit
  the ledger (merge conflicts with live sessions). Gates write JSON only;
  the controller session ledgers verdicts.
- Do NOT submit any job in this task — creating the scripts + a
  `bash -n` syntax check + a dry-run flag audit is the deliverable.
  Submission happens from the controller after final review.
