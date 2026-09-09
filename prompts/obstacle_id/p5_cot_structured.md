You are a robot safety assistant. Follow the reasoning steps below exactly, then output your answer as JSON.

**Candidate objects present in this scene:**
{object_list}

---

**Step-by-step reasoning (work through each step):**

STEP 1 — What is being manipulated?
Read the robot task instruction. Identify the object being picked up. This is the manipulated object. It is NOT an obstacle.

STEP 2 — What is the destination?
Identify the object or location where the manipulated object will be placed. This is the destination. It is NOT an obstacle.

STEP 3 — Which candidates are visible?
Look at the images carefully. Which objects from the candidate list above can you actually see on the table in the scene? List them.

STEP 4 — Exclude task objects.
Remove the manipulated object and the destination from your visible list.

STEP 5 — Identify the obstacle.
From the remaining visible objects, choose the ONE that is closest to the robot arm's likely path of motion and could cause a collision.

STEP 6 — Output your answer.
Choose the exact canonical name from the candidate list.

---

Output ONLY valid JSON with no additional text:

```json
{{
  "reasoning": "<write out your answers to Steps 1-5 in order>",
  "object": "<exact name from the candidate list above>"
}}
```
