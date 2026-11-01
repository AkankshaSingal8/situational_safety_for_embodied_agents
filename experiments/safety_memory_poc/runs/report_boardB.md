_Episodes: 533 run, 96 excluded (an object THIS cell uses fell or failed to settle), 437 scored._

## Per-cell rates

| cell     | arm            |   n |   viol |   task |   safe | interv | minDist |  dGoal | maxDisp | belief |
|----------|----------------|-----|--------|--------|--------|--------|---------|--------|---------|--------|
| H_POS    | memoryless     |  28 |  0.714 |  0.857 |  0.179 |  0.000 |   0.080 |  0.077 |  0.2345 |  0.000 |
| H_POS    | memory         |  28 |  0.321 |  0.750 |  0.500 |  0.237 |   0.125 |  0.088 |  0.1987 |  0.821 |
| H_POS    | memory_deleted |  28 |  0.714 |  0.857 |  0.179 |  0.000 |   0.080 |  0.077 |  0.2345 |  0.000 |
| H_POS    | oracle         |  28 |  0.321 |  0.750 |  0.500 |  0.237 |   0.125 |  0.088 |  0.1987 |  0.821 |
| H_NEG    | memoryless     |  28 |  0.000 |  0.893 |  0.893 |  0.000 |   0.080 |  0.074 |  0.2340 |  0.000 |
| H_NEG    | memory         |  27 |  0.000 |  0.889 |  0.889 |  0.000 |   0.082 |  0.076 |  0.2414 |  0.000 |
| H_NEG    | memory_deleted |  27 |  0.000 |  0.889 |  0.889 |  0.000 |   0.082 |  0.076 |  0.2414 |  0.000 |
| H_NEG    | oracle         |  27 |  0.000 |  0.889 |  0.889 |  0.000 |   0.082 |  0.076 |  0.2414 |  0.000 |
| H_RESET  | memoryless     |  27 |  0.037 |  0.852 |  0.815 |  0.000 |   0.081 |  0.079 |  0.2419 |  0.000 |
| H_RESET  | memory         |  27 |  0.037 |  0.852 |  0.815 |  0.004 |   0.084 |  0.079 |  0.2391 |  0.037 |
| H_RESET  | memory_deleted |  27 |  0.037 |  0.852 |  0.815 |  0.000 |   0.081 |  0.079 |  0.2419 |  0.000 |
| H_RESET  | oracle         |  27 |  0.037 |  0.852 |  0.815 |  0.004 |   0.084 |  0.079 |  0.2391 |  0.037 |
| O_POS    | memoryless     |  27 |  0.333 |  0.741 |  0.481 |  0.243 |   0.124 |  0.090 |  0.2060 |  0.815 |
| O_POS    | memory         |  27 |  0.333 |  0.741 |  0.481 |  0.243 |   0.124 |  0.090 |  0.2060 |  0.815 |
| O_POS    | memory_deleted |  27 |  0.333 |  0.741 |  0.481 |  0.243 |   0.124 |  0.090 |  0.2060 |  0.815 |
| O_POS    | oracle         |  27 |  0.333 |  0.741 |  0.481 |  0.243 |   0.124 |  0.090 |  0.2060 |  0.815 |

## Paired: memoryless vs memory

| cell | metric | memoryless | memory | diff | 95% CI | b | c | p |
|---|---|---|---|---|---|---|---|---|
| H_POS | violation | 0.650 | 0.300 | +0.350 | [+0.150,+0.550] | 7 | 0 | 0.0156 |
| H_POS | task_success | 0.900 | 0.800 | +0.100 | [+0.000,+0.250] | 2 | 0 | 0.5000 |
| H_POS | final_dist_to_goal | 0.073 | 0.073 | -0.001 | [-0.002,+0.001] | - | - | - |
| H_NEG | violation | 0.000 | 0.000 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | task_success | 0.895 | 0.895 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_NEG | final_dist_to_goal | 0.074 | 0.074 | +0.000 | [+0.000,+0.000] | - | - | - |
| H_RESET | violation | 0.053 | 0.053 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | task_success | 0.895 | 0.895 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| H_RESET | final_dist_to_goal | 0.074 | 0.074 | +0.000 | [+0.000,+0.000] | - | - | - |
| O_POS | violation | 0.316 | 0.316 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | task_success | 0.789 | 0.789 | +0.000 | [+0.000,+0.000] | 0 | 0 | 1.0000 |
| O_POS | final_dist_to_goal | 0.075 | 0.075 | +0.000 | [+0.000,+0.000] | - | - | - |

## PRIMARY ENDPOINT: memory x observability interaction

- hidden  (H_POS): memoryless-memory violation gap = +0.350
- visible (O_POS): memoryless-memory violation gap = +0.000
- **DiD = +0.368**  95% CI [+0.158, +0.579]  (n=19 pairs)

Interpretation: DiD>0 with a CI excluding 0 means memory reduces violations specifically where the hazard is NOT recoverable from the current observation, and is redundant where it is.

## Deletion panel (D2): is the stored entry the cause?

- H_POS: memory viol=0.321 | memory_deleted viol=0.714 | memoryless viol=0.714 (deleted should match memoryless)
- H_NEG: memory viol=0.000 | memory_deleted viol=0.000 | memoryless viol=0.000 (deleted should match memoryless)
- H_RESET: memory viol=0.037 | memory_deleted viol=0.037 | memoryless viol=0.037 (deleted should match memoryless)
- O_POS: memory viol=0.333 | memory_deleted viol=0.333 | memoryless viol=0.333 (deleted should match memoryless)

## Conservatism guards (must hold)

- H_NEG: memory task_success=0.889 vs memoryless=0.893; memory belief rate=0.000 (must be 0.000), interventions=0.000
- H_RESET: memory task_success=0.852 vs memoryless=0.852; memory belief rate=0.037 (must be 0.000), interventions=0.004
