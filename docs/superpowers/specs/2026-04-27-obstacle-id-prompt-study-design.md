# Obstacle Identification Prompt Study — Design Spec

**Date:** 2026-04-27  
**Scope:** task_0 / level_I / safelibero_spatial (10 episodes)  
**Goal:** Find the prompt strategy that maximises single-obstacle identification accuracy across 3 Qwen models.

---

## Problem Statement

The existing `stl_single_object.md` prompt achieves 0/10 accuracy on task_0 because:
1. It asks the VLM to generate full STL constraints (complex, distracting task).
2. The VLM ignores the object list and invents names or picks the wrong object.
3. It does not explicitly exclude the manipulated object or destination object from consideration.

The new prompts focus purely on *obstacle identification* (one object name + reasoning), with varying amounts of metadata supplied.

---

## Ground Truth & Scoring

- **Ground truth:** `metadata.json["obstacle"]["name"]` per episode.
- **Matching:** `names_match()` in `vlm_prompt_runner/accuracy.py` — strips `_obstacle`, `_N` suffixes, checks word intersection. Cleaned VLM output names score correctly.
- **Metric:** correct/total per prompt per model, reported as accuracy %.

---

## Prompt Strategies

All 5 prompts live in `prompts/obstacle_id/` and share the same output schema:

```json
{
  "reasoning": "<chain-of-thought>",
  "object": "<canonical_object_name>"
}
```

The `reasoning` field is required in all prompts so chain-of-thought is always saved to disk.

| File | Template vars used | Metadata level |
|---|---|---|
| `p1_vision_only.md` | `{task_description}` | Minimal — task description only |
| `p2_task_roles.md` | `{task_description}` | Explicit: "don't pick the object being manipulated or its destination" |
| `p3_candidate_list.md` | `{task_description}`, `{object_list}` | + cleaned candidate list (no `_obstacle_N` suffix) |
| `p4_positions.md` | `{task_description}`, `{object_list_with_positions}` | + 3D xyz positions per object |
| `p5_cot_structured.md` | `{task_description}`, `{object_list}` | + explicit step-by-step reasoning scaffold |

### P1 — Vision Only
No object list. VLM must identify the obstacle purely from images + task description. Hardest for the model; establishes a baseline for how much metadata helps.

### P2 — Task Roles
Explicitly tells the VLM: "the object being picked up and its destination are NOT obstacles — ignore them." Does not provide a candidate list; VLM names the obstacle from vision.

### P3 — Candidate List
Provides a cleaned list of all scene objects (task objects excluded, `_obstacle_N` stripped from names). VLM picks exactly one. No positions — relies on visual disambiguation.

### P4 — Positions
Same as P3 plus each candidate's 3D position. Objects with `|x| > 5` or `|y| > 5` are far outside the workspace and invisible; VLM can use positions to reason spatially.

### P5 — Structured CoT
Same candidate list as P3, but the prompt enforces explicit step-by-step reasoning:
1. Identify what is being manipulated.
2. Identify where it goes.
3. List which candidates are visible in the images.
4. Reason about which one obstructs the robot's path.
5. Output one name.

---

## Metadata Injection (Approach B)

### Template substitution in `runner.py`

`build_prompt(system_prompt, task_description, template_vars=None)` — before concatenation, `system_prompt` is processed with `system_prompt.format(**template_vars)` if `template_vars` is provided. Prompt files use standard Python `{placeholder}` syntax.

### New fields in `load_episode` (episode.py)

`load_episode(ep_dir)` already returns `task_description`, `agentview`, `eye_in_hand`, `backview`. Add:

- `object_list: str` — newline-separated cleaned names of scene objects, excluding structural objects (robot, table, walls, cabinet, stove, box_base) and objects whose names appear as task-object references in the task description.
- `object_list_with_positions: str` — same list with `(x=..., y=..., z=...)` appended per line.

Cleaning rule: strip trailing `_\d+` and `_obstacle` from each name for display; keep original canonical name as the identifier the VLM should output (so `names_match` works).

**Important:** `object_list` provides *cleaned display names* alongside *canonical names* so the VLM can output the canonical form:
```
moka_pot  [moka_pot_obstacle_1]
white_storage_box  [white_storage_box_obstacle_1]
...
```

The VLM is instructed to output the canonical name in brackets.

### Structural object filter

Objects whose names start with any of: `robot`, `mount`, `gripper`, `wall`, `floor`, `table`, `box_base`, `wooden_cabinet`, `flat_stove` are excluded from `object_list`.

### Task object exclusion

Objects whose cleaned name appears as a substring in `task_description` are flagged as task-related. For the prompt, we note these are task objects; they are still included in `object_list` so the VLM can explicitly reason about excluding them.

---

## Experiment Workflow

### Phase 1 — Prompt selection (qwen3-vl-8b)

Run all 5 prompts against all 10 episodes of task_0/level_I with `qwen3-vl-8b`.
Output dir per prompt: `vlm_prompt_runner/<prompt_stem>/safelibero_spatial/level_I/task_0/`
Compute accuracy for each prompt. Best prompt = highest `correct/total`.

### Phase 2 — Model comparison

Run the best-performing prompt with `qwen2.5-vl-7b` and `qwen2.5-vl-3b` on the same episodes.

### Output artefacts

- Per-episode `output.json` with `reasoning`, `object`, `_meta` fields.
- Per-run accuracy JSON written to `vlm_prompt_runner/results/obstacle_id_study.json`.
- Human-readable table printed to stdout and appended to `prompts/obstacle_id/README.md`.

---

## New / Modified Files

| Path | Action | Notes |
|---|---|---|
| `prompts/obstacle_id/p1_vision_only.md` | Create | No object list |
| `prompts/obstacle_id/p2_task_roles.md` | Create | Explicit role exclusion |
| `prompts/obstacle_id/p3_candidate_list.md` | Create | Candidate list, no positions |
| `prompts/obstacle_id/p4_positions.md` | Create | Candidate list + xyz |
| `prompts/obstacle_id/p5_cot_structured.md` | Create | Structured CoT scaffold |
| `prompts/obstacle_id/README.md` | Create | Rationale + results table |
| `vlm_prompt_runner/episode.py` | Modify | Add `object_list`, `object_list_with_positions` |
| `vlm_prompt_runner/runner.py` | Modify | Template substitution in `build_prompt` |
| `vlm_prompt_runner/run_prompt_experiment.py` | Create | Experiment orchestrator CLI |
| `run_obstacle_id_study.sh` | Create | Two-phase bash launcher |
| `vlm_prompt_runner/tests/test_runner.py` | Modify | Update for new `build_prompt` signature |
| `vlm_prompt_runner/tests/test_episode.py` | Modify | Test new `object_list` fields |

---

## Constraints & Non-Goals

- The active obstacle name is **never** placed in the prompt. `metadata["obstacle"]["name"]` is only used for ground-truth scoring.
- No STL generation in these prompts.
- Evaluation is on task_0/level_I only (10 episodes). Other tasks/levels are out of scope for this study.
- No majority voting across prompt repetitions — single-shot per episode per prompt.
