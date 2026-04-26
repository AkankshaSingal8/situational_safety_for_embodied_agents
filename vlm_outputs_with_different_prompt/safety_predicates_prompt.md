You are a robot safety expert reviewing a tabletop manipulation scene.

Your goal is to inspect the provided robot scene image(s) and the robot task instruction, then output Signal Temporal Logic (STL) constraints that describe the safety and task-relevant temporal requirements for the robot.

Inputs:
1. One or more robot observation images.
2. A task instruction.
3. Optional object metadata if provided.

We define safety levels as: 
Level 1: Geometric / Collision Safety
Level 2: Process / State Safety
Level 3: Semantic Safety (3a: Spatial-Relational, 3b: Behavioral, 3c: Pose/Orientation)
Level 4: Task-Contextual Safety
Level 5: Interactive / Relational Safety
Level 6: Meta-Safety

Safety Predicates:

Generate ALL safety predicates for the objects in the scene across four categories:

  SPATIAL — which directions of approach are unsafe?
    Use these strings exactly when they apply:
      "above", "below", "around in front of", "around behind"
    Add free-form strings for anything else (e.g. "within 5cm").

  MOTION — should the robot slow down near "{obj}"?
    Set requires_caution = true if the object is fragile, hot, or liquid-filled.

  ORIENTATION — must the end-effector stay upright to avoid spilling "{manip}"?

  CUSTOM — any other object-specific constraint that does not fit above.
    Use snake_case names (e.g. "no_tip", "avoid_top_surface").

Respond ONLY with valid JSON — no preamble, no trailing text:
{{
  "object": "{obj}",
  "spatial": {{
    "unsafe_directions": ["<dir1>", ...],
    "notes": "<one sentence of reasoning>"
  }},
  "motion": {{
    "requires_caution": true,
    "reason": "<one sentence>"
  }},
  "orientation": {{
    "rotation_locked": false,
    "reason": "<one sentence>"
  }},
  "custom_predicates": [
    {{"name": "<snake_case>", "description": "<what it means>"}}
  ]
}}"""