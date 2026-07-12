# Contextual Predictive Filter — Pilot Decision

Date: 2026-07-11  
Branch: `contextual-predictive-safety-filter`  
Implementation commit: `df2d617` (subsequent integration fixes through `b39dd5b`)

## Outcome

**NO-GO for the current tangential action-chunk projector.** Do not run the
planned 50-episode-per-task evaluation or extend this implementation to
LIBERO-Safety/OopsieVerse. The technical gate showed a regression large enough
that a full run would not plausibly satisfy the pre-registered continuation
criteria.

## Valid gate result

OpenVLA-OFT, SafeLIBERO-Spatial Level I, fixed episode 0 for each of four tasks:

| Method | Episodes | TSR | CAR | Interventions |
|---|---:|---:|---:|---:|
| Stored unfiltered baseline (all 200 episodes) | 200 | 42.0% | 12.0% | — |
| Contextual predictive filter technical gate | 4 | 25.0% | 0.0% | 34 / 131 chunks (26.0%) |

The gate completed one task but collided in all four episodes. Several collisions
occurred much earlier than in preliminary runs, demonstrating that the tangent
selection can steer the end effector or swept robot/object volume into the
obstacle even when the predicted EEF point is outside the spherical envelope.
Mean filter latency was below 1 ms, so runtime was not the limiting factor.

Result artifact:
`contextual_smoke_v3/openvla/results/safelibero_spatial/results_EVAL-safelibero_spatial-levelI-openvla-2026_07_11-22_30_38--contextual-predictive.json`

## Invalid diagnostic runs

Two earlier four-episode runs are retained only as debugging evidence. They
incorrectly treated controller commands as metric deltas and/or imposed an
invalid EEF height floor, causing 100% intervention rates. They must not be used
as scientific results.

## Pi0.5 integration status

The shared action-chunk API, Pi0.5 adapter, server launcher, checkpoint loading,
and WebSocket startup were implemented and tested. The LIBERO client initially
hit the repository's documented `GLIBCXX_3.4.29` mismatch; the launcher now
preloads the conda environment C++ runtime. A corrected performance run was not
continued after the OpenVLA architecture gate failed, because adapter success
cannot rescue the unsafe geometric projector.

## Why the approach failed

1. EEF-only sphere checks omit the arm, gripper, and manipulated-object swept
   volumes used by SafeLIBERO's collision signal.
2. Choosing a tangent locally has no global route or task-manifold information;
   the chosen side can be dynamically worse than the nominal path.
3. Open-loop cumulative Cartesian prediction is not an accurate rollout of the
   closed-loop OSC controller and changing scene.
4. A filter cannot be expected to improve TSR merely by projecting actions; it
   needs recovery-aware candidate generation or replanning after intervention.

## Required redesign before another pilot

- Replace analytic cumulative-delta rollout with simulator-free learned dynamics
  or policy/world-model future prediction calibrated against executed EEF and
  object trajectories.
- Score swept volumes for robot links, gripper, and held objects, not just EEF
  points.
- Generate multiple policy-conditioned candidate chunks and use risk-constrained
  selection/guidance; do not invent a tangent independently of the policy.
- Add intervention-conditioned replanning so corrected actions remain on the task
  manifold.
- Validate prediction calibration and collision recall offline on stored baseline
  rollouts before spending policy inference compute.

