## 1. Situational safety for robot manipulation

Class presentation. The two completed studies ask what a robot can avoid from current visual observations and what changes when a safety-relevant fact persists across episodes. Results are simulation-only. Recommended speaking time: 15–20 minutes.

Akanksha Singal · October 2026

## 2. What counts as a safe rollout?

SafeLIBERO: spatial, object, goal and long suites, each at hazard levels I and II. The main baseline comparisons here use 50 episodes per task, four tasks, 200 episodes per condition. TSR and CAR are separate marginal rates; neither is the joint safe-success rate. Level I places hazards near the target; Level II places hazards along the movement path.

Source: results/safelibero/policy_baselines/; results/README.md

## 3. Task success leaves a safety gap

Completed spatial Level-I runs, 200 episodes per policy. Cosmos 7.5/10.5, Fast-WAM 45/5, OpenVLA-OFT 42/12, pi0.5 67/14 (TSR/CAR). These describe the evaluated checkpoints and settings, not universal model rankings. Fast-WAM uses the completed fastwam_50ep run. Full baseline board is in the appendix.

SafeLIBERO spatial · Level I · n = 200 per policy

## 4. Use the images to locate the obstacle

Completed visual filter v19: vision-based obstacle selection and multi-view triangulation produce a 3D obstacle estimate; uncertainty determines ellipsoid geometry, refined using later views. No ground-truth obstacle positions are used in this control path. The diagram summarizes the evaluated obstacle-avoidance mechanism, not the aspirational six-level taxonomy.

Source: fol_safety_filter/filter.py; final campaign report

## 5. Adjust motion near the estimated hazard

The implemented geometric filter removes the approaching component near the warning boundary and pushes outward inside the hard boundary. It monitors end-effector and arm checkpoints. v19 uses uncertainty-shaped ellipsoids and multi-frame refinement. This is a diagram of the implemented action correction, not a claim of a verified global safety guarantee or an exact QP solve.

Source: FOLSafetyFilter._apply_cbf and _apply_point_cbf

## 6. Collision avoidance improves; failures remain

Matched final campaign, OpenVLA-OFT og10 checkpoint, seed 7. Each condition has 200 episodes. v19 Level I: TSR 40 to 37, CAR 12.5 to 20.5. Level II: TSR stays 37, CAR 6 to 14.5, a gain of 8.5 percentage points. No statistical significance claim. The run working tree was not fully committed; these are recorded experimental results, not a tagged reproducibility claim.

SafeLIBERO spatial · n = 200 per method and level · seed 7

## 7. More avoidance can cost task success

All points are completed 200-episode campaign conditions. v20 adds a target corridor mechanism; v21-D2 uses RGB-D and two obstacle candidates. GT oracle v17.2 uses simulator positions and differs in filter version, so the gap is diagnostic rather than a clean isolated perception ablation. Do not claim the full gap is caused solely by grounding. Unfinished variants are excluded.

Source: results/safelibero/fol/{baseline_og10,v19,v20,v21d2,v172}/

## 8. Watch the remaining failure

Same spatial Level-II task 3, episode 0, baseline og10 versus v19. Both recordings show the moka pot obstacle being toppled. These illustrate a failure case, not a successful intervention. The campaign rates on preceding slides summarize all 200 episodes. Playback is the stored 30 fps video, not a timing measurement.

Task: move the bowl from the cabinet to the plate · illustrative episode

## 9. Check what the benchmark actually measures

Completed code audit of LIBERO-Safety, a different benchmark from SafeLIBERO. Robot-contact checks compare geometry integer IDs against string names, so that channel cannot fire. Affordance has no constraints. Duplicate predicate keys and success-step accounting also affect scoring. These findings prevent interpreting older 100% CAR values as safety evidence. No claim that the fixes or reruns are complete.

Source: docs/code_review_findings.md · LIBERO-Safety audit

## 10. Some hazards outlast the image

Separate proof of concept with a scripted waypoint controller, fixed human-written rules and a JSON fact store. A proximity event marks the tool in a marker-transfer analogue. The next episode resets and randomizes object poses while preserving identity. A hidden mark cannot be recovered from the current observation. No VLM, learned rule induction or physical contamination model is evaluated.

Source: experiments/safety_memory_poc/README.md and RESULTS.md

## 11. Change the information; keep the controller fixed

Four cells: hidden marked source, clean lookalike, marked then washed, visible mark. Four arms share the same scripted policy and barrier projection: memoryless, memory, deleted memory and oracle ledger. The memoryless arm has a perfect current-frame grounder. Observed state features of hidden marked/clean twins are identical in the documented leak check; this is not a claim about all raw camera pixels.

Fixed rules · same policy and filter · four belief sources

## 12. Memory reduces hidden-hazard violations

Full scored hidden-positive cell: 32 episodes per arm. Memoryless: violation 24/32, task 28/32, safe 5/32. Memory: violation 9/32, task 23/32, safe 16/32. Deleted memory matches memoryless. Entire board: 640 episodes run, 128 excluded by cell-aware settle criteria, 512 scored. The paired analysis uses a smaller matched subset of 20; do not mix its rates with these full-cell rates. Success is lower with memory; lack of significance does not establish no cost.

Hidden hazard · n = 32 per arm · 512 scored / 640 run across the full board

## 13. The effect depends on what can be observed

Chart uses full scored cells, n=32 per arm. On the paired subset, hidden violation is 75% versus 30%; visible violation is 30% in both arms. The interaction is 45 percentage points, 95% CI [25,65], n=20 paired comparisons. Full-cell reset belief remains 6.2%, so do not claim perfect wash release. Memory and oracle match, but both derive their marking signal from the same proximity event.

Full scored cells: n = 32 per arm · paired interaction: 45 pp, 95% CI [25, 65]

## 14. Replay: the same task with and without memory

Animation replays the stored x/y end-effector trajectory for hidden-positive pair 0 in kin_final_v3. This is a projection of a 3D kinematic testbed trace, not MuJoCo video and not the physics board used in the preceding plots. Frame progression interpolates the display time over stored samples, not real-time simulation speed. Square: food location. Shaded disk: 6 cm proximity threshold in the top-down view. Star: goal. Exact 3D violation/success labels come from the recorded episode.

Kinematic testbed · recorded pair 0 · top view · no new rollout

## 15. What we have demonstrated

Completed work: policy baseline evaluation, visual geometric safety filtering and its campaign, an audit of benchmark scoring, and persistent-fact enforcement in a controlled scripted-policy experiment. These are separate pieces, not an integrated learned memory/VLA system. Visual control still fails in many episodes. Memory improves safe success here but with reduced task success; external perception, learned policy integration and real robots are not established by these results.

Simulation results · visual filter and memory experiments evaluated separately

## 16. Completed baseline board

Each entry is task success / collision avoidance (%), n=200. Values are read from completed JSONs. The pi0.5 Long-I stale markdown entry is replaced with its complete result (58 / 15). Fast-WAM Spatial-I uses the completed 50-episode-per-task run. Cross-policy comparisons are descriptive of these checkpoint configurations. All exact source paths are stored in sources.json.

Each cell: task success / collision avoidance (%) · n = 200

## Added comparison slides (current slides 8 and 9)

compare_safelibero

SafeLIBERO spatial Level I, task 0, episode 0. Both videos use OpenVLA-OFT og10. Left: baseline_og10_L1_video. Right: fol_v19_spatial_L1_n50_video. Full 10-second stored recordings, unchanged speed. Task: pick up the black bowl between the plate and ramekin and place it on the plate. The baseline topples the moka pot; it remains upright in this filtered recording. This illustrates obstacle avoidance, not proof of task completion or collision-free behavior for every robot link. v19 uses visual obstacle grounding and the geometric filter described earlier. Exact sources are in artifacts/class_progress_2026_10_08/comparison_demos/sources.json.

compare_liberosafety

LIBERO-Safety obstacle_avoidance_human, level 2, task 14, episode 1. Task: put the white-yellow mug in the microwave and close it. Both arms use the LIBERO-Safety fine-tuned pi0.5 checkpoint (models--LIBERO-Safety--pi05_libero_safety), confirmed in .worktrees/flow-guidance-tier-gt/slurm/ls_teaser_render.slurm. Left disables guidance; right uses FOL obstacle selection and perceived entity positions with guided action generation (ramp schedule, 8 candidates, 0.09 m EEF radius, 0.30 m engagement radius, 60-degree cone). This is the guided pi0.5 variant, distinct from the OpenVLA-OFT v19 filter on the preceding slide. Result filenames and footage show baseline timeout (600 steps) and guided completion (510 steps). The full source recordings are shown at 2x stored playback speed: 10.0 s and 8.5 s. Original videos sample every five environment steps at 6 fps; playback is not wall-clock robot speed. Same task and episode index; do not claim pixel-identical starts or infer aggregate improvement from this selected example. Avoid interpreting the benchmark's defective robot-contact metric as proof of safety.

