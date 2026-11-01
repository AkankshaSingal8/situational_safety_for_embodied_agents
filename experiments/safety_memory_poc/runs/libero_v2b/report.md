_Episodes: 320 run, 64 excluded (an object THIS cell uses fell or failed to settle), 256 scored._

## Per-cell rates

| cell     | arm            |   n |   viol |   task |   safe | interv | minDist |  dGoal | maxDisp | belief |
|----------|----------------|-----|--------|--------|--------|--------|---------|--------|---------|--------|
| H_POS    | memoryless     |  16 |  0.688 |  0.938 |  0.250 |  0.000 |   0.067 |  0.062 |  0.0949 |  0.000 |
| H_POS    | memory         |  16 |  0.250 |  0.750 |  0.562 |  0.262 |   0.112 |  0.066 |  0.0531 |  0.750 |
| H_POS    | memory_deleted |  16 |  0.688 |  0.938 |  0.250 |  0.000 |   0.067 |  0.062 |  0.0949 |  0.000 |
| H_POS    | oracle         |  16 |  0.250 |  0.750 |  0.562 |  0.262 |   0.112 |  0.066 |  0.0531 |  0.750 |
| H_NEG    | memoryless     |  16 |  0.000 |  0.938 |  0.938 |  0.000 |   0.067 |  0.062 |  0.1019 |  0.000 |
| H_NEG    | memory         |  16 |  0.000 |  0.938 |  0.938 |  0.000 |   0.067 |  0.062 |  0.1019 |  0.000 |
| H_NEG    | memory_deleted |  16 |  0.000 |  0.938 |  0.938 |  0.000 |   0.067 |  0.062 |  0.1019 |  0.000 |
| H_NEG    | oracle         |  16 |  0.000 |  0.938 |  0.938 |  0.000 |   0.067 |  0.062 |  0.1019 |  0.000 |
| H_RESET  | memoryless     |  16 |  0.062 |  0.938 |  0.875 |  0.000 |   0.067 |  0.062 |  0.0949 |  0.000 |
| H_RESET  | memory         |  16 |  0.000 |  0.938 |  0.938 |  0.005 |   0.072 |  0.062 |  0.0928 |  0.062 |
| H_RESET  | memory_deleted |  16 |  0.062 |  0.938 |  0.875 |  0.000 |   0.067 |  0.062 |  0.0949 |  0.000 |
| H_RESET  | oracle         |  16 |  0.000 |  0.938 |  0.938 |  0.005 |   0.072 |  0.062 |  0.0928 |  0.062 |
| O_POS    | memoryless     |  16 |  0.250 |  0.750 |  0.562 |  0.262 |   0.112 |  0.066 |  0.0531 |  0.750 |
| O_POS    | memory         |  16 |  0.250 |  0.750 |  0.562 |  0.262 |   0.112 |  0.066 |  0.0531 |  0.750 |
| O_POS    | memory_deleted |  16 |  0.250 |  0.750 |  0.562 |  0.262 |   0.112 |  0.066 |  0.0531 |  0.750 |
| O_POS    | oracle         |  16 |  0.250 |  0.750 |  0.562 |  0.262 |   0.112 |  0.066 |  0.0531 |  0.750 |

## Paired: memoryless vs memory

| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |
|---|---|---|---|---|---|---|---|---|
| H_POS | violation | 0.688 | 0.250 | +0.438 | [+0.188,+0.688] | 7 | 0 | 0.0156 |
| H_POS | task_success | 0.938 | 0.750 | +0.188 | [+0.000,+0.375] | 3 | 0 | 0.2500 |
| H_POS | final_dist_to_goal | 0.062 | 0.066 | -0.004 | [-0.010,+0.000] | - | - | - |
| H_NEG | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | task_success | 0.938 | 0.938 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | final_dist_to_goal | 0.062 | 0.062 | +0.000 | [+0.000,+0.000] | - | - | - |
| H_RESET | violation | 0.062 | 0.000 | +0.062 | [+0.000,+0.188] | 1 | 0 | 1.0000 |
| H_RESET | task_success | 0.938 | 0.938 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | final_dist_to_goal | 0.062 | 0.062 | -0.000 | [-0.000,+0.000] | - | - | - |
| O_POS | violation | 0.250 | 0.250 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | task_success | 0.750 | 0.750 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | final_dist_to_goal | 0.066 | 0.066 | +0.000 | [+0.000,+0.000] | - | - | - |

## PRIMARY ENDPOINT: memory x observability interaction

- hidden  (H_POS): memoryless-memory violation gap = +0.438
- visible (O_POS): memoryless-memory violation gap = +0.000
- **DiD = +0.438**  95% CI [+0.188, +0.688]  (n=16 pairs)

Interpretation: DiD>0 with a CI excluding 0 means memory reduces violations specifically where the hazard is NOT recoverable from the current observation, and is redundant where it is.

## Deletion panel (D2): is the stored entry the cause?

- H_POS: memory viol=0.250 | memory_deleted viol=0.688 | memoryless viol=0.688 (deleted should match memoryless)
- H_NEG: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- H_RESET: memory viol=0.000 | memory_deleted viol=0.062 | memoryless viol=0.062 (deleted should match memoryless)
- O_POS: memory viol=0.250 | memory_deleted viol=0.250 | memoryless viol=0.250 (deleted should match memoryless)

## Conservatism guards (must hold)

- H_NEG: memory task_success=0.938 vs memoryless=0.938; memory belief rate=0.000 (must be 0.000), interventions=0.000
- H_RESET: memory task_success=0.938 vs memoryless=0.938; memory belief rate=0.062 (must be 0.000), interventions=0.005
