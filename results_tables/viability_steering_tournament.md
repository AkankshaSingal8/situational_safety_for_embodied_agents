# Viability Steering Tournament — SafeLIBERO Spatial (2026-07-13)

All new variants use pi0.5, matched SafeLIBERO initial-state ordering, in-denoising
DCBF guidance, and the benchmark collision criterion (>1 mm obstacle displacement).
Micro-smokes use two episodes per task (8 per level). The promoted result uses five
episodes per task (20 per level).

| Variant | L1 TSR/CAR | L2 TSR/CAR | Decision |
|---|---:|---:|---|
| Phase-adaptive EEF/full-envelope hybrid, legacy 0.05 scale | 62.5/37.5 | 75.0/25.0 | No-go: dominated on L2 |
| Best-of-4, clearance minus correction selector | 50.0/50.0 | 87.5/37.5 | No-go: safe-looking timeouts; heuristic lacks task viability |
| Calibrated phase-adaptive hybrid, 0.00523 scale (micro) | 62.5/62.5 | 62.5/75.0 | Promote |
| **Calibrated phase-adaptive hybrid, n=5/task** | **55.0/50.0** | **85.0/60.0** | **L1 no-go; L2 partial go** |

References:

- Stored pi0.5 baseline: L1 67.0/14.0; L2 55.5/12.0.
- Existing r3 full-geometry guidance: L1 58.5/44.5; L2 86.5/49.0.
- arXiv:2607.01378 reference: L1 75.5/77.5; L2 78.0/75.5.

## Promoted run per-task results

L1: task 0 80/60, task 1 20/20, task 2 80/60, task 3 40/60.

L2: task 0 80/40, task 1 60/40, task 2 100/80, task 3 100/80.

## Interpretation

Controller calibration is essential and materially improves the safety/task frontier.
However, a single distance-based phase switch cannot distinguish grasp-critical target
contact from avoidable obstacle contact. Best-of-K candidate generation is useful, but
clearance-minus-correction selection chooses trajectories that are geometrically safe
and task-inviable. The next primary approach should therefore combine calibrated
in-denoising candidate generation with a learned, semantic safe-success critic. FOL or
hard geometry remains the final verifier, not the candidate selector.
