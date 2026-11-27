# Astra arms: GPT-6 Astra as base policy and as residual, against pi0.5 and ex2

Two closed-loop SafeLIBERO arms that put OpenAI `gpt-6-astra` into the ex2 evaluation, with everything else held
fixed, so the results drop into the same tables as pi0.5 and ex2 (see `EX2.md`).

| arm | what Astra replaces | Astra sees | Astra returns | executed |
|---|---|---|---|---|
| `astra_base` (exp 1) | pi0.5 (the whole policy) | agent + wrist frame, state8, prompt | 10 x 7 chunk, env units | first 5 steps |
| `astra_res` (exp 2) | ex2's expert (`res_server.py`) | agent + wrist frame, state8, prompt, pi0.5's 10 x 7 chunk | 10 x 3 xyz delta | `clip(A[:, :3] + delta, -1, 1)`, first 5 steps |

## Why

- Exp 1 asks whether a frontier VLM, zero-shot, can produce the low-level control that pi0.5 produces, under the
  same interface. It is the "no robot training at all" baseline for ex2.
- Exp 2 asks whether the same VLM can do ex2's job: keep pi0.5's chunk and only correct it near the obstacle.
  It is the "no residual training" baseline: same inputs and output contract as ex2's expert.

## What is held fixed (fair comparison)

Everything in the ex2 eval except the model that produces the numbers:

- `eval_sl.py` loop, metrics, episode lists and scene files; env_seed 0, settle 20, 256 px render, `MAX_STEPS`
  300 (550 Long).
- Chunk length H = 10, 5 executed steps per chunk, then re-query (pi0.5 and ex2 use exactly this).
- Action format: 7-D in env units, the same numbers pi0.5 returns and the env consumes: dx dy dz (OSC position
  delta, [-1, 1], 1.0 = 5 cm per step), droll dpitch dyaw (axis-angle delta, [-1, 1], 1.0 = 0.5 rad per step),
  gripper (-1 open, +1 close). Astra's numbers are clipped to [-1, 1] like ex2's sum; no rescaling, no smoothing.
- Observation: the frames pi0.5 gets (both cameras flipped 180 deg, resize-with-pad to 224) and state8
  (eef pos 3 + axis-angle 3 + gripper qpos 2). No history, no privileged sim state (pi0.5 and ex2 have none).
- Prompt: ex2's prompt `"<instruction>. <safety sentence>"` with `text_mode true` (the sentence names the
  obstacle). Both Astra arms get it, because the arms are baselines for ex2, which gets it.
- Exp 2 runs pi0.5 from the same `pi05_libero` server, so the base chunk is identical to ex2's.

## Astra call

- Chat Completions, `model=gpt-6-astra`, one request per chunk (the only per-chunk choice is the model).
- Fixed system prompt (`astra_ctrl.py`, constant across all calls): the robot, camera views, action space and
  units as above, frame axes (robot base: +x forward away from the robot, +y left, +z up), 20 Hz, H = 10 and
  that only the first 5 run before the next query. Exp 2's prompt adds: the 10 x 7 chunk is pi0.5's plan,
  return only the xyz correction (zeros when the plan is already safe), and what the correction is added to.
- User message: the two images (PNG, 224 px) and a short text block with the prompt, state8 and (exp 2) the
  chunk, numbers rounded to 4 decimals.
- Output: structured output (JSON schema, strict) `{"actions": [[7 numbers] x 10]}` or
  `{"delta": [[3 numbers] x 10]}`. Wrong shape or non-finite values count as a failed call.
- Reasoning effort and image detail: one setting for every call, picked once from the smoke test on latency and
  cost (default unless latency makes the full run impossible), and written into each run's `config.json`.
- Failure handling (after 4 retries with backoff): exp 1 holds still (zero motion, gripper keeps its last
  command); exp 2 returns a zero delta (= pi0.5's chunk). Every failure is counted per episode, so the tables can
  report the failure rate.
- Logging: every call writes request id, latency, prompt/completion/reasoning tokens and the raw output to
  `astra_calls/part_<worker>.jsonl` inside the run folder; `astra_cost.py` sums tokens per table.

## Code

| file | role |
|---|---|
| `astra_client.py` | OpenAI call (requests, no SDK), image encoding, schema, retries, call log |
| `astra_ctrl.py` | `AstraBaseController` (exp 1) and `AstraResController` (exp 2), the `eval_sl` controller interface |
| `controllers.py` | arms `astra_base`, `astra_res` |
| `run_eval_astra.slurm` | exp 1: CPU-only job, no GPU or servers; exp 2: pi0.5 server on one GPU + Astra |
| `make_tables.py` | `--astra`: adds Astra columns and the tables named below |

Outputs follow ex2's layout: `runs_v2/<suite>/<test|scenes_newobj25|scenes_newlayout>/<astra_base|astra_res>/true/<eps|all>`.

## Tables

Cell = TSR / CAR / SR in %; columns pi0.5 | ex2 | Astra base | Astra residual.

- Table 1: official episodes 0-49 (all)
- Table 1a: official episodes 0-39 (training layouts of ex2)
- Table 1b: official episodes 40-49
- Table 2: 25 unseen GSO objects x 4 runs, layouts 0-39 (`scenes_newobj25`)
- Table 3: benchmark obstacles on held-out layouts 40-49 (`scenes_newlayout`)

Per arm: 1,600 + 800 + 1,595 = 3,995 episodes; about 60 calls per Spatial/Object/Goal episode and up to 110 per
Long episode that runs to time-out.

## Order

1. Smoke: 1 episode per suite per arm on a login-free CPU job; check shapes, actions reach the env, call log,
   latency and tokens per call. Estimate total tokens and wall-clock for all tables.
2. Table 1b (episodes 40-49, 400 episodes per arm), then the rest of Table 1, then Tables 2 and 3.
3. `python make_tables.py --astra` after each stage.

## Measured per call (client test, 2026-11-22)

Default reasoning effort: ~606 prompt tokens (two 224 px images + text), ~170 completion tokens (~20 reasoning),
3-9 s latency. `reasoning_effort=low`: same tokens, ~3 s. Rate limit on this key: 15,000 requests/min, 40M
tokens/min, far above what 256 parallel episodes need. Plan: default effort (latency is acceptable).

Estimate for all tables, both arms: ~290k calls (Astra base, most episodes run to time-out) + ~220k calls (Astra
residual; base-chunk text adds ~300 prompt tokens) = ~510k calls, ~370M prompt + ~80M completion tokens.
Wall-clock with 16 jobs x 16 workers: about 2-4 h per arm, Long the slowest.

## Launch (not yet submitted)

```bash
# smoke: 2 episodes per arm (Spatial, level I, task 0, eps 40-41)
for A in astra_base astra_res; do sbatch -J smoke_$A -t 03:00:00 \
  --export=ALL,S=spatial,ARM=$A,EPS=40-41,WORKERS=2,EXTRA="--levels I --tasks 0",OUT=runs_v2/astra_smoke/$A run_eval_astra.slurm; done
# full: Table 1 (official eps), Table 2 (25 unseen objects), Table 3 (held-out layouts), per suite and arm
for A in astra_base astra_res; do for S in spatial object goal long; do
  for E in 0-39 40-49; do sbatch -J ${A}_${S}_$E --export=ALL,S=$S,ARM=$A,EPS=$E run_eval_astra.slurm; done
  sbatch -J ${A}_${S}_obj25 --export=ALL,S=$S,ARM=$A,SCENES=../../data_gso/$S/scenes_newobj25 run_eval_astra.slurm
  for I in 0-199 200-399; do sbatch -J ${A}_${S}_lay$I --export=ALL,S=$S,ARM=$A,SCENES=scenes_newlayout,SCENE_IDS=$I,OUT=runs_v2/$S/scenes_newlayout/$A/true/all_$I run_eval_astra.slurm; done
done; done
python make_tables.py --astra
```

## Expectations and caveats

- Exp 1 will probably fail most tasks: Astra has never seen LIBERO's action scale or 20 Hz control. That is the
  result the baseline exists to measure; CAR alone will look good for a policy that does not move, so read SR.
- Astra is a hosted model: its version can change during the run and the calls are not seeded. Record the
  `model` field returned by every call.
- The API key belongs to a collaborator; the call log is the record of spend.
