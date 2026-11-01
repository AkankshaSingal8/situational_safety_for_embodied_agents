_Episodes: 640 run, 0 excluded as unsettled/invalid scenes, 640 scored._

## Per-cell rates

| cell     | arm            |   n |   viol |   task |   safe | interv | minDist |  dGoal | maxDisp | belief |
|----------|----------------|-----|--------|--------|--------|--------|---------|--------|---------|--------|
| H_POS    | memoryless     |  40 |  1.000 |  1.000 |  0.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_POS    | memory         |  40 |  0.000 |  1.000 |  1.000 |  0.204 |   0.090 |  0.023 |  0.0000 |  1.000 |
| H_POS    | memory_deleted |  40 |  1.000 |  1.000 |  0.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_POS    | oracle         |  40 |  0.000 |  1.000 |  1.000 |  0.204 |   0.090 |  0.023 |  0.0000 |  1.000 |
| H_NEG    | memoryless     |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_NEG    | memory         |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_NEG    | memory_deleted |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_NEG    | oracle         |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_RESET  | memoryless     |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_RESET  | memory         |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_RESET  | memory_deleted |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| H_RESET  | oracle         |  40 |  0.000 |  1.000 |  1.000 |  0.000 |   0.045 |  0.029 |  0.0000 |  0.000 |
| O_POS    | memoryless     |  40 |  0.000 |  1.000 |  1.000 |  0.204 |   0.090 |  0.023 |  0.0000 |  1.000 |
| O_POS    | memory         |  40 |  0.000 |  1.000 |  1.000 |  0.204 |   0.090 |  0.023 |  0.0000 |  1.000 |
| O_POS    | memory_deleted |  40 |  0.000 |  1.000 |  1.000 |  0.204 |   0.090 |  0.023 |  0.0000 |  1.000 |
| O_POS    | oracle         |  40 |  0.000 |  1.000 |  1.000 |  0.204 |   0.090 |  0.023 |  0.0000 |  1.000 |

## Paired: memoryless vs memory

| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |
|---|---|---|---|---|---|---|---|---|
| H_POS | violation | 1.000 | 0.000 | +1.000 | [+1.000,+1.000] | 40 | 0 | 0.0000 |
| H_POS | task_success | 1.000 | 1.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_POS | final_dist_to_goal | 0.029 | 0.023 | +0.006 | [+0.001,+0.010] | - | - | - |
| H_NEG | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | task_success | 1.000 | 1.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | final_dist_to_goal | 0.029 | 0.029 | +0.000 | [+0.000,+0.000] | - | - | - |
| H_RESET | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | task_success | 1.000 | 1.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | final_dist_to_goal | 0.029 | 0.029 | +0.000 | [+0.000,+0.000] | - | - | - |
| O_POS | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | task_success | 1.000 | 1.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | final_dist_to_goal | 0.023 | 0.023 | +0.000 | [+0.000,+0.000] | - | - | - |

## PRIMARY ENDPOINT: memory x observability interaction

- hidden  (H_POS): memoryless-memory violation gap = +1.000
- visible (O_POS): memoryless-memory violation gap = +0.000
- **DiD = +1.000**  95% CI [+1.000, +1.000]  (n=40 pairs)

Interpretation: DiD>0 with a CI excluding 0 means memory reduces violations specifically where the hazard is NOT recoverable from the current observation, and is redundant where it is.

## Deletion panel (D2): is the stored entry the cause?

- H_POS: memory viol=0.000 | memory_deleted viol=1.000 | memoryless viol=1.000 (deleted should match memoryless)
- H_NEG: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- H_RESET: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- O_POS: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)

## Conservatism guards (must hold)

- H_NEG: memory task_success=1.000 vs memoryless=1.000; memory belief rate=0.000 (must be 0.000), interventions=0.000
- H_RESET: memory task_success=1.000 vs memoryless=1.000; memory belief rate=0.000 (must be 0.000), interventions=0.000
