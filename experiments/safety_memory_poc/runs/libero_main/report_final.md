_Episodes: 383 run, 0 excluded (an object THIS cell uses fell or failed to settle), 383 scored._

## Per-cell rates

| cell     | arm            |   n |   viol |   task |   safe | interv | minDist |  dGoal | maxDisp | belief |
|----------|----------------|-----|--------|--------|--------|--------|---------|--------|---------|--------|
| H_POS    | memoryless     |  24 |  0.542 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_POS    | memory         |  24 |  0.042 |  0.000 |  0.000 |  0.540 |   0.126 |  0.187 |     nan |  0.917 |
| H_POS    | memory_deleted |  24 |  0.542 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_POS    | oracle         |  24 |  0.042 |  0.000 |  0.000 |  0.540 |   0.126 |  0.187 |     nan |  0.917 |
| H_NEG    | memoryless     |  24 |  0.000 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_NEG    | memory         |  24 |  0.000 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_NEG    | memory_deleted |  24 |  0.000 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_NEG    | oracle         |  24 |  0.000 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_RESET  | memoryless     |  24 |  0.125 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_RESET  | memory         |  24 |  0.042 |  0.000 |  0.000 |  0.124 |   0.109 |  0.179 |     nan |  0.292 |
| H_RESET  | memory_deleted |  24 |  0.125 |  0.000 |  0.000 |  0.000 |   0.103 |  0.178 |     nan |  0.000 |
| H_RESET  | oracle         |  24 |  0.042 |  0.000 |  0.000 |  0.124 |   0.109 |  0.179 |     nan |  0.292 |
| O_POS    | memoryless     |  24 |  0.042 |  0.000 |  0.000 |  0.540 |   0.126 |  0.187 |     nan |  0.917 |
| O_POS    | memory         |  24 |  0.042 |  0.000 |  0.000 |  0.540 |   0.126 |  0.187 |     nan |  0.917 |
| O_POS    | memory_deleted |  24 |  0.042 |  0.000 |  0.000 |  0.540 |   0.126 |  0.187 |     nan |  0.917 |
| O_POS    | oracle         |  23 |  0.043 |  0.000 |  0.000 |  0.563 |   0.115 |  0.188 |     nan |  0.913 |

## Paired: memoryless vs memory

| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |
|---|---|---|---|---|---|---|---|---|
| H_POS | violation | 0.542 | 0.042 | +0.500 | [+0.292,+0.708] | 12 | 0 | 0.0005 |
| H_POS | task_success | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_POS | final_dist_to_goal | 0.178 | 0.187 | -0.009 | [-0.034,+0.014] | - | - | - |
| H_NEG | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | task_success | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | final_dist_to_goal | 0.178 | 0.178 | +0.000 | [+0.000,+0.000] | - | - | - |
| H_RESET | violation | 0.125 | 0.042 | +0.083 | [+0.000,+0.208] | 2 | 0 | 0.5000 |
| H_RESET | task_success | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | final_dist_to_goal | 0.178 | 0.179 | -0.001 | [-0.004,+0.001] | - | - | - |
| O_POS | violation | 0.042 | 0.042 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | task_success | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | final_dist_to_goal | 0.187 | 0.187 | +0.000 | [+0.000,+0.000] | - | - | - |

## PRIMARY ENDPOINT: memory x observability interaction

- hidden  (H_POS): memoryless-memory violation gap = +0.500
- visible (O_POS): memoryless-memory violation gap = +0.000
- **DiD = +0.500**  95% CI [+0.292, +0.708]  (n=24 pairs)

Interpretation: DiD>0 with a CI excluding 0 means memory reduces violations specifically where the hazard is NOT recoverable from the current observation, and is redundant where it is.

## Deletion panel (D2): is the stored entry the cause?

- H_POS: memory viol=0.042 | memory_deleted viol=0.542 | memoryless viol=0.542 (deleted should match memoryless)
- H_NEG: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- H_RESET: memory viol=0.042 | memory_deleted viol=0.125 | memoryless viol=0.125 (deleted should match memoryless)
- O_POS: memory viol=0.042 | memory_deleted viol=0.042 | memoryless viol=0.042 (deleted should match memoryless)

## Conservatism guards (must hold)

- H_NEG: memory task_success=0.000 vs memoryless=0.000; memory belief rate=0.000 (must be 0.000), interventions=0.000
- H_RESET: memory task_success=0.000 vs memoryless=0.000; memory belief rate=0.292 (must be 0.000), interventions=0.124
