You are a robot safety assistant. Your job is to rank all objects from the candidate list below by their likelihood of being physically struck by the robot arm during task execution — from highest collision risk to lowest.

**Candidate objects and their 3D positions in the robot workspace:**
{object_list_with_positions}

The robot workspace is centred near (x=0, y=0.2, z=0.9). Objects with |x| > 1.5 or |y| > 1.5 are outside the workspace.

Work through these steps exactly:

STEP 1 — Filter by workspace position.
Keep only candidates where |x| ≤ 1.5 AND |y| ≤ 1.5. Objects outside this range cannot be collided with — assign them collision risk: NONE and exclude from ranking.

STEP 2 — Remove task objects.
From the robot task instruction identify:
- The object being picked up (pick target) — note its (x, y) position
- The destination where it will be placed — note its (x, y) position
These are not obstacles. Exclude them from the ranking.

STEP 3 — Remove spatial reference objects.
Remove any remaining candidate that is named only as a spatial landmark in the task instruction (e.g. "between the plate and the ramekin" — the ramekin is a landmark). Assign these collision risk: NONE.

STEP 4 — Rank remaining candidates by collision risk.
For each remaining in-workspace, non-task candidate:
  (a) Compute distance to the pick target: d_pick = sqrt((x - x_pick)² + (y - y_pick)²)
  (b) Compute distance to the destination: d_dest = sqrt((x - x_dest)² + (y - y_dest)²)
  (c) Collision score = 1 / (min(d_pick, d_dest) + 0.01)   [higher score = higher risk]

Rank all candidates from highest score to lowest. Assign a risk label:
  - HIGH: score > 5  (very close to pick or place target)
  - MEDIUM: score 2–5
  - LOW: score < 2

STEP 5 — Output your answer.
The primary obstacle is the top-ranked candidate (highest collision risk).

---

Output ONLY valid JSON with no additional text:

```json
{{
  "reasoning": "<answer Steps 1–4 in order, including distance calculations>",
  "ranked_obstacles": [
    {{"object": "<name>", "risk": "<HIGH|MEDIUM|LOW>", "score": <float>, "reason": "<one line>"}},
    ...
  ],
  "object": "<top-ranked candidate — exact name from the candidate list>"
}}
```
