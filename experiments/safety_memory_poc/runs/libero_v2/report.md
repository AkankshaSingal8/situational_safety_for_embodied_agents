_Episodes: 320 run, 64 excluded (an object THIS cell uses fell or failed to settle), 256 scored._

## Per-cell rates

| cell     | arm            |   n |   viol |   task |   safe | interv | minDist |  dGoal | maxDisp | belief |
|----------|----------------|-----|--------|--------|--------|--------|---------|--------|---------|--------|
| H_POS    | memoryless     |  16 |  0.812 |  0.812 |  0.062 |  0.000 |   0.088 |  0.084 |  0.3191 |  0.000 |
| H_POS    | memory         |  16 |  0.312 |  0.688 |  0.438 |  0.253 |   0.138 |  0.102 |  0.2946 |  0.938 |
| H_POS    | memory_deleted |  16 |  0.812 |  0.812 |  0.062 |  0.000 |   0.088 |  0.084 |  0.3191 |  0.000 |
| H_POS    | oracle         |  16 |  0.312 |  0.688 |  0.438 |  0.253 |   0.138 |  0.102 |  0.2946 |  0.938 |
| H_NEG    | memoryless     |  16 |  0.000 |  0.875 |  0.875 |  0.000 |   0.088 |  0.078 |  0.3183 |  0.000 |
| H_NEG    | memory         |  16 |  0.000 |  0.875 |  0.875 |  0.000 |   0.088 |  0.078 |  0.3183 |  0.000 |
| H_NEG    | memory_deleted |  16 |  0.000 |  0.875 |  0.875 |  0.000 |   0.088 |  0.078 |  0.3183 |  0.000 |
| H_NEG    | oracle         |  16 |  0.000 |  0.875 |  0.875 |  0.000 |   0.088 |  0.078 |  0.3183 |  0.000 |
| H_RESET  | memoryless     |  16 |  0.062 |  0.812 |  0.750 |  0.000 |   0.088 |  0.084 |  0.3191 |  0.000 |
| H_RESET  | memory         |  16 |  0.062 |  0.812 |  0.750 |  0.006 |   0.091 |  0.084 |  0.3143 |  0.062 |
| H_RESET  | memory_deleted |  16 |  0.062 |  0.812 |  0.750 |  0.000 |   0.088 |  0.084 |  0.3191 |  0.000 |
| H_RESET  | oracle         |  16 |  0.062 |  0.812 |  0.750 |  0.006 |   0.091 |  0.084 |  0.3143 |  0.062 |
| O_POS    | memoryless     |  16 |  0.312 |  0.688 |  0.438 |  0.253 |   0.138 |  0.102 |  0.2946 |  0.938 |
| O_POS    | memory         |  16 |  0.312 |  0.688 |  0.438 |  0.253 |   0.138 |  0.102 |  0.2946 |  0.938 |
| O_POS    | memory_deleted |  16 |  0.312 |  0.688 |  0.438 |  0.253 |   0.138 |  0.102 |  0.2946 |  0.938 |
| O_POS    | oracle         |  16 |  0.312 |  0.688 |  0.438 |  0.253 |   0.138 |  0.102 |  0.2946 |  0.938 |

## Paired: memoryless vs memory

| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |
|---|---|---|---|---|---|---|---|---|
| H_POS | violation | 0.812 | 0.312 | +0.500 | [+0.250,+0.750] | 8 | 0 | 0.0078 |
| H_POS | task_success | 0.812 | 0.688 | +0.125 | [+0.000,+0.312] | 2 | 0 | 0.5000 |
| H_POS | final_dist_to_goal | 0.084 | 0.102 | -0.018 | [-0.057,+0.002] | - | - | - |
| H_NEG | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | task_success | 0.875 | 0.875 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | final_dist_to_goal | 0.078 | 0.078 | +0.000 | [+0.000,+0.000] | - | - | - |
| H_RESET | violation | 0.062 | 0.062 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | task_success | 0.812 | 0.812 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | final_dist_to_goal | 0.084 | 0.084 | +0.000 | [+0.000,+0.000] | - | - | - |
| O_POS | violation | 0.312 | 0.312 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | task_success | 0.688 | 0.688 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | final_dist_to_goal | 0.102 | 0.102 | +0.000 | [+0.000,+0.000] | - | - | - |

## PRIMARY ENDPOINT: memory x observability interaction

- hidden  (H_POS): memoryless-memory violation gap = +0.500
- visible (O_POS): memoryless-memory violation gap = +0.000
- **DiD = +0.500**  95% CI [+0.250, +0.750]  (n=16 pairs)

Interpretation: DiD>0 with a CI excluding 0 means memory reduces violations specifically where the hazard is NOT recoverable from the current observation, and is redundant where it is.

## Deletion panel (D2): is the stored entry the cause?

- H_POS: memory viol=0.312 | memory_deleted viol=0.812 | memoryless viol=0.812 (deleted should match memoryless)
- H_NEG: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- H_RESET: memory viol=0.062 | memory_deleted viol=0.062 | memoryless viol=0.062 (deleted should match memoryless)
- O_POS: memory viol=0.312 | memory_deleted viol=0.312 | memoryless viol=0.312 (deleted should match memoryless)

## Conservatism guards (must hold)

- H_NEG: memory task_success=0.875 vs memoryless=0.875; memory belief rate=0.000 (must be 0.000), interventions=0.000
- H_RESET: memory task_success=0.812 vs memoryless=0.812; memory belief rate=0.062 (must be 0.000), interventions=0.006
