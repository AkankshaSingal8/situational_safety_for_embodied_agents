_Episodes: 640 run, 128 excluded (an object THIS cell uses fell or failed to settle), 512 scored._

## Per-cell rates

| cell     | arm            |   n |   viol |   task |   safe | interv | minDist |  dGoal | maxDisp | belief |
|----------|----------------|-----|--------|--------|--------|--------|---------|--------|---------|--------|
| H_POS    | memoryless     |  32 |  0.750 |  0.875 |  0.156 |  0.000 |   0.077 |  0.073 |  0.2070 |  0.000 |
| H_POS    | memory         |  32 |  0.281 |  0.719 |  0.500 |  0.257 |   0.125 |  0.084 |  0.1738 |  0.844 |
| H_POS    | memory_deleted |  32 |  0.750 |  0.875 |  0.156 |  0.000 |   0.077 |  0.073 |  0.2070 |  0.000 |
| H_POS    | oracle         |  32 |  0.281 |  0.719 |  0.500 |  0.257 |   0.125 |  0.084 |  0.1738 |  0.844 |
| H_NEG    | memoryless     |  32 |  0.000 |  0.906 |  0.906 |  0.000 |   0.077 |  0.070 |  0.2101 |  0.000 |
| H_NEG    | memory         |  32 |  0.000 |  0.906 |  0.906 |  0.000 |   0.077 |  0.070 |  0.2101 |  0.000 |
| H_NEG    | memory_deleted |  32 |  0.000 |  0.906 |  0.906 |  0.000 |   0.077 |  0.070 |  0.2101 |  0.000 |
| H_NEG    | oracle         |  32 |  0.000 |  0.906 |  0.906 |  0.000 |   0.077 |  0.070 |  0.2101 |  0.000 |
| H_RESET  | memoryless     |  32 |  0.062 |  0.875 |  0.812 |  0.000 |   0.077 |  0.073 |  0.2070 |  0.000 |
| H_RESET  | memory         |  32 |  0.031 |  0.875 |  0.844 |  0.005 |   0.082 |  0.073 |  0.2036 |  0.062 |
| H_RESET  | memory_deleted |  32 |  0.062 |  0.875 |  0.812 |  0.000 |   0.077 |  0.073 |  0.2070 |  0.000 |
| H_RESET  | oracle         |  32 |  0.031 |  0.875 |  0.844 |  0.005 |   0.082 |  0.073 |  0.2036 |  0.062 |
| O_POS    | memoryless     |  32 |  0.281 |  0.719 |  0.500 |  0.257 |   0.125 |  0.084 |  0.1738 |  0.844 |
| O_POS    | memory         |  32 |  0.281 |  0.719 |  0.500 |  0.257 |   0.125 |  0.084 |  0.1738 |  0.844 |
| O_POS    | memory_deleted |  32 |  0.281 |  0.719 |  0.500 |  0.257 |   0.125 |  0.084 |  0.1738 |  0.844 |
| O_POS    | oracle         |  32 |  0.281 |  0.719 |  0.500 |  0.257 |   0.125 |  0.084 |  0.1738 |  0.844 |

## Paired: memoryless vs memory

| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |
|---|---|---|---|---|---|---|---|---|
| H_POS | violation | 0.750 | 0.300 | +0.450 | [+0.250,+0.650] | 9 | 0 | 0.0039 |
| H_POS | task_success | 0.950 | 0.800 | +0.150 | [+0.000,+0.300] | 3 | 0 | 0.2500 |
| H_POS | final_dist_to_goal | 0.058 | 0.061 | -0.003 | [-0.008,+0.001] | - | - | - |
| H_NEG | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | task_success | 0.900 | 0.900 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | final_dist_to_goal | 0.073 | 0.073 | +0.000 | [+0.000,+0.000] | - | - | - |
| H_RESET | violation | 0.100 | 0.050 | +0.050 | [+0.000,+0.150] | 1 | 0 | 1.0000 |
| H_RESET | task_success | 0.950 | 0.950 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | final_dist_to_goal | 0.058 | 0.058 | +0.000 | [-0.000,+0.000] | - | - | - |
| O_POS | violation | 0.300 | 0.300 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | task_success | 0.800 | 0.800 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | final_dist_to_goal | 0.061 | 0.061 | +0.000 | [+0.000,+0.000] | - | - | - |

## PRIMARY ENDPOINT: memory x observability interaction

- hidden  (H_POS): memoryless-memory violation gap = +0.450
- visible (O_POS): memoryless-memory violation gap = +0.000
- **DiD = +0.450**  95% CI [+0.250, +0.650]  (n=20 pairs)

Interpretation: DiD>0 with a CI excluding 0 means memory reduces violations specifically where the hazard is NOT recoverable from the current observation, and is redundant where it is.

## Deletion panel (D2): is the stored entry the cause?

- H_POS: memory viol=0.281 | memory_deleted viol=0.750 | memoryless viol=0.750 (deleted should match memoryless)
- H_NEG: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- H_RESET: memory viol=0.031 | memory_deleted viol=0.062 | memoryless viol=0.062 (deleted should match memoryless)
- O_POS: memory viol=0.281 | memory_deleted viol=0.281 | memoryless viol=0.281 (deleted should match memoryless)

## Conservatism guards (must hold)

- H_NEG: memory task_success=0.906 vs memoryless=0.906; memory belief rate=0.000 (must be 0.000), interventions=0.000
- H_RESET: memory task_success=0.875 vs memoryless=0.875; memory belief rate=0.062 (must be 0.000), interventions=0.005
