# Obstacle Identification Prompts

Five prompt strategies for identifying exactly one obstacle object in a robot manipulation scene.
Results are scored by `vlm_prompt_runner/accuracy.py` against `metadata.json["obstacle"]["name"]`.

## Why these prompts?

The original `stl_single_object.md` prompt asked the VLM to generate full STL safety constraints —
a complex multi-step task that caused 0/10 accuracy. The VLM picked the wrong object or invented
names not in the list. These new prompts focus solely on *obstacle identification*, with varying
amounts of metadata to measure how much context the VLM actually needs.

## Prompts

### p1_vision_only — Pure vision baseline
**Metadata given:** task description only.
**Rationale:** Establishes a lower bound. The VLM must identify the obstacle purely from images.
No object list, so the model cannot cheat by pattern-matching names.

### p2_task_roles — Explicit task-object exclusion
**Metadata given:** task description (with instruction to identify manipulated and destination objects).
**Rationale:** The biggest failure mode of the original prompt was treating the manipulated object
or destination as an obstacle. This prompt explicitly tells the VLM which roles to exclude.

### p3_candidate_list — Candidate list, no positions
**Metadata given:** task description + full list of non-structural object names (`{object_list}`).
**Rationale:** Gives the VLM a concrete set of names to choose from. Eliminates invented names.
Still requires vision to determine which candidate is actually present on the table.

### p4_positions — Candidate list + 3D positions
**Metadata given:** task description + object names + xyz coordinates (`{object_list_with_positions}`).
**Rationale:** In each episode only one obstacle is placed near the workspace (|x|<2, |y|<2).
Other obstacle candidates are far away (|x|>5 or |y|>5). Providing positions gives the VLM
a near-deterministic signal: pick the candidate near (0, 0.2, 0.9).

### p5_cot_structured — Structured chain-of-thought
**Metadata given:** task description + candidate list (`{object_list}`).
**Rationale:** Forces the VLM to answer five explicit sub-questions in order before naming
the obstacle. Tests whether structured reasoning helps compared to free-form prompting (p3).

## Template Variables

All prompts use standard Python `{placeholder}` syntax. The runner substitutes:

| Variable | Source | Used by |
|---|---|---|
| `{task_description}` | `metadata.json["task_description"]` | Injected automatically by runner |
| `{object_list}` | Non-structural object names, one per line | p3, p5 |
| `{object_list_with_positions}` | Same + `(x=..., y=..., z=...)` | p4 |

## Running the Study

```bash
# Full two-phase study (qwen3 for prompt selection, all 3 models for comparison)
bash run_obstacle_id_study.sh

# Single prompt, single model (quick test)
python -m vlm_prompt_runner.run_prompt_experiment \
    --prompts-dir prompts/obstacle_id \
    --prompts p3_candidate_list \
    --model qwen3-vl-8b \
    --suite safelibero_spatial --level I --task 0 \
    --results-out vlm_prompt_runner/results/quick_test.json
```

Results are saved to `vlm_prompt_runner/results/`.

## Output Format

Each episode produces `output.json` with:
```json
{
  "reasoning": "<VLM chain-of-thought>",
  "object": "<predicted obstacle canonical name>",
  "_meta": {
    "task_description": "...",
    "ep_dir": "..."
  }
}
```

## Results

<!-- Updated after running the study -->

| Model | Prompt | Correct | Total | Accuracy |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |
