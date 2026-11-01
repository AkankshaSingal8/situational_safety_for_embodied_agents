# Proof-of-concept benchmark choice for learned safety memory

Survey date: 11 October 2026. This is a primary-source and source-code survey, not an installation or performance comparison. No new simulator, checkpoint, or evaluation job was launched.

**Decision: use VLA-Arena as the host, starting with one `safety_state_preservation` mug-transport task and a small, explicitly named memory extension.** Keep an unmodified task as the control. Use MIKASA-Robo-VLA later as a separate memory diagnostic if needed. Do not migrate to a larger household simulator for the first result.

The choice is based on current usability and the scientific question, not the fact that this project already uses MuJoCo. VLA-Arena supplies the difficult infrastructure—robot, task, simulator, policy adapter, and released task-adapted policy—while the extension supplies the missing experiment: a safety condition visible earlier but unavailable at the decision point, followed by reuse of a learned rule on new objects.

## What the benchmark must establish

Three claims require different tests:

1. **Fact memory:** an earlier observation affects a later safety decision.
2. **Rule learning:** the agent learns which combination of context and action predicts harm, rather than being given the prohibition.
3. **Rule reuse:** the learned relation survives process/episode boundaries and improves decisions on unseen instances.

The native suites reviewed do not, by their default protocols, cleanly isolate all three. Long episodes do not automatically require memory. A color-recall task does not automatically measure safety. Fine-tuning on a fixed set of tasks does not by itself demonstrate deployment-time rule acquisition.

## Survey and selection

| Candidate | What is available / relevant | Gap for this proof of concept | Decision |
|---|---|---|---|
| [VLA-Arena](https://github.com/PKU-Alignment/VLA-Arena) | MuJoCo/robosuite manipulation; safety suites; task/config files; model adapters and released checkpoints | Requires controlled partial observability and an explicit acquisition/transfer protocol | **Best overall host** |
| [MIKASA-Robo-VLA](https://github.com/CognitiveAISystems/MIKASA-Robo) | 90 language-conditioned tasks, hidden-cue tasks and reactive controls; ManiSkill GPU simulation; datasets | Native objectives mostly assess recall/task success; no native persistent safety-rule acquisition protocol. The inspected pi05 baseline README is a placeholder | Best dedicated memory diagnostic; less certain route to a ready VLA safety experiment |
| [LIBERO-Mem](https://github.com/libero-mem/libero-mem) | Object-level memory in partially observable LIBERO manipulation | No ready safety objective or rule-learning protocol; additional fork/dependency integration | Reasonable alternative for memory-only follow-up |
| [RMBench](https://github.com/RoboTwin-Platform/RMBench) / [RoboTwin-MeM](https://github.com/InternRobotics/EventVLA) | Memory-dependent manipulation, data, released memory-policy weights | RoboTwin/SAPIEN stack, policy/environment setup, and no native safety-rule acquisition objective | Strong later validation; more moving parts for the first experiment |
| [SafeLIBERO / AEGIS](https://github.com/THU-RCSCT/vlsa-aegis) | Collision-oriented continuous VLA tasks; already integrated here | Most obstacles are currently observable; persistence and rule learning need new structure | Keep current integration as backup and geometry control |
| [LIBERO-Safety](https://arxiv.org/abs/2606.23686) | Physical and semantic safety scenarios | Current local evaluation has documented scoring problems; native protocol does not isolate memory | Do not make its current local metric the primary learning target |
| [SafeVLA-Bench](https://safevla.org/) | Independent task-aware safety measurement over existing rollouts | A measurement layer, not a memory-learning environment | Useful metrics reference; not the host benchmark |
| [SafeAgentBench](https://github.com/shengyin1224/SafeAgentBench) | 750 safe/unsafe planning tasks; AI2-THOR controller with high-level actions | Moves the study toward LLM planning/refusal; additional temporal/transfer design still needed | Choose only if the research goal changes to high-level agent safety |
| [Safety-CHORES / SafeVLA](https://github.com/PKU-Alignment/SafeVLA) | Explicit safety costs, navigation/manipulation and safe-learning stack | Larger embodied navigation/training setup than needed to isolate a local learned memory rule | Poor first-step cost/complexity tradeoff |
| [BEHAVIOR-1K / OmniGibson](https://behavior.stanford.edu/getting_started/installation.html) | Rich household states and interactions | Isaac/Omniverse dependencies and RTX-oriented graphics requirements; substantial environment setup | Attractive eventual semantic validation, excessive initial scope |
| [ForesightSafety-VLA](https://arxiv.org/abs/2606.27079) | Broad RoboTwin-based diagnostic safety taxonomy | No benchmark-specific runnable release verified in this survey; no isolated rule-memory protocol | Follow, but do not depend on it for the first result |
| [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) / [POPGym](https://github.com/proroklab/popgym) | Cheap controlled safety or memory algorithm tests | Considerable distance from object-centric VLA manipulation and semantic predicate learning | Optional unit tests only |

These rankings are engineering judgments from the released interfaces and task definitions. They are not measured installation times or speed comparisons.

## Source-code findings that matter

**A working policy path is more valuable than a paper score.** VLA-Arena's [OpenVLA-OFT evaluation config](https://github.com/PKU-Alignment/VLA-Arena/blob/main/vla_arena/configs/evaluation/openvla_oft.yaml) points to `VLA-Arena/openvla-7b-oft-finetuned-vla-arena`. Its [Hugging Face repository](https://huggingface.co/VLA-Arena/openvla-7b-oft-finetuned-vla-arena) is accessible and contains model/action-head files. This verifies release availability, not successful execution on PSC. Use the repository's matching loader/config; the model card contains generic OpenVLA descriptions that are insufficient to reconstruct the OFT setup by themselves.

**Use the actual filled task, not an arbitrary task with “safety” in its suite name.** The inspected [level-1 green-mug task](https://github.com/PKU-Alignment/VLA-Arena/blob/main/vla_arena/vla_arena/bddl_files/safety_state_preservation/level_1/pick_up_the_green_mug_on_the_table_and_place_it_on_the_wooden_cabinet.bddl) specifies seven `water_balls` inside the mug and a `Spilled` cost. The inspected level-0 blue-mug counterpart has neither. Native levels also differ in other ways; do not use L0 versus L1 as the causal empty-versus-filled comparison. Construct paired variants of the same scene.

**Spill ground truth needs an episode ledger.** In the inspected [environment implementation](https://github.com/PKU-Alignment/VLA-Arena/blob/main/vla_arena/vla_arena/envs/bddl_base_domain.py), `check_spilled` counts current contacts between water-ball geometry and table/target geometry. `_check_cost` evaluates predicates according to temporal registration or episode termination. This is not automatically an “ever spilled during the episode” signal. Build a separate per-step monitor that retains particle escape/contact events, including terminal steps. Test a particle that escapes and later stops touching the target. This is a targeted inspection, not a complete benchmark audit.

**MIKASA's new and old releases differ.** The [2026 VLA paper](https://arxiv.org/abs/2610.00604) describes 90 tasks; the original MIKASA-Robo suite had 32. The [installation instructions](https://mikasarobo.github.io/installation.html) distinguish the new stack from the legacy package. The [pi05 baseline README](https://github.com/CognitiveAISystems/MIKASA-Robo/blob/main/baselines/pi05/README.md) currently says “Coming soon.” This does not mean the environments or datasets are unavailable, but it prevents assuming a one-command reproduction of the reported VLA baseline.

**MIKASA negative memory is not harm avoidance.** Its task table defines `FindImposterColor3-VLA-v0` as choosing a color not previously shown. Relabeling that choice as “safety” would give a clean toy study, but would not test physical harm or discovery of a safety mechanism. Its [canonical protocol](https://mikasarobo.github.io/benchmarking.html) is valuable for memory evaluation; the [required wrappers](https://mikasarobo.github.io/quickstart.html) and observation restrictions must be preserved.

## The first experiment: remembered fill state and spill prevention

Working label: **VLA-Arena remembered-fill diagnostic**. This is a proposed derivative task, not an official named benchmark.

Use one Panda, one mug, and one placement target. The task instruction is simply to move the mug to the target. Keep the task objective independent of whether the mug is filled. Filling is represented by the existing particle approximation; make no claims about accurate fluid mechanics.

1. **Acquire evidence in simulation.** Collect short controlled trials varying fill, tilt, speed, container appearance, and pose. Include safe upright transport of filled containers and safe diagnostic tilts of empty containers. Do not let color or object ID determine the label. Observe actual particle outcomes, rather than defining harm using the same tilt threshold the learner will fit.
2. **Learn a condition and restriction.** From images/events and outcomes, fit a fill-relevant detector or latent feature, then a conditional action-risk relation. A possible learned interpretation is `contains_spillable_contents(x) AND risky_transport(u,x) -> restrict(u)`. The predicate name and relation are examples of an output, not inputs supplied to the learner. Begin with slow/quasi-static probes to isolate tilt; broaden to acceleration only after that works.
3. **Make memory necessary.** Show the contents during an initial observation phase, then hide the interior in both policy camera streams during transport. Use a controlled observation wrapper with the same occluded region in filled and empty conditions; do not erase particles from physics. Audit all policy inputs, including language, state dictionaries, overlays, and camera views. A render-only occlusion is a deliberate diagnostic intervention and must be reported as such.
4. **Transfer the rule.** Freeze learned detector/rule parameters, restart the process, load the library, and test new container instances/appearances and layouts with the same controlled physical range. Reinitialize object facts when physical identity changes; retain the schema. Establish generalization across geometry only after appearance/instance transfer, since spill thresholds depend on geometry.
5. **Check usefulness.** The monitor must preserve successful transport. A universal upright restriction may solve the filled tasks; test false interventions and safe empty-container tilt probes to show whether learning the condition adds value. If all methods achieve the same task and safety outcomes, acknowledge that the benchmark has not demonstrated a practical advantage for conditional memory.

Avoid beginning with contamination propagation, uncertain object reassociation, multi-step cleaning, moving humans, or cross-embodiment transfer. Those introduce mechanisms unrelated to establishing the first memory effect. A single fill condition plus an action-dependent consequence is enough for a first result.

## Controls, data split, and acceptance gates

Use four matched evaluation conditions: visible filled, visible empty, hidden filled, hidden empty. Keep current observations as similar as practical within hidden pairs; quantify residual detectability with a held-out frame-only classifier. Do not claim perfect observational aliasing unless actually measured. Particle mass can affect later dynamics, so specify exactly at what decision time information is withheld.

Five initial methods:

- learned current-observation monitor, without memory;
- learned monitor with persistent rule library and object-state memory;
- the same method with object-state memory deleted after the cue;
- a small recurrent action-conditioned risk head on the same observations;
- oracle fill state/rule with the same controller, explicitly privileged.

Also delete or replace the learned schema in a focused transfer test; deleting only a current object fact does not demonstrate persistent rule reuse. Keep base-policy weights fixed. Report training/acquisition performance separately from transfer tests. No held-out outcomes enter library updates before their reported evaluation.

First pilot: 20 paired seeds, native task, visible variants, and an oracle monitor. Continue only if the base policy can manipulate the chosen scene and the controller has a meaningful safe-completion improvement to recover. If the nominal policy never spills, report a lack of safety headroom; do not silently introduce forced perturbations and present them as natural-policy failures. Controlled probes remain useful training/diagnostic data, explicitly separated from nominal rollout results.

One small board: one held-out container family × four conditions × 20 paired seeds × five methods = **400 scored rollouts**, plus separate acquisition trials. Expand families after the first causal result. Primary metrics are any spill, number/fraction of escaped particles, task completion, and safe completion. Secondary metrics are false restriction on verified safe probes, action deviation, duration, detector calibration, and failure after memory deletion. Preserve paired seeds and use episode/task-level uncertainty estimates.

A successful result requires measured harm reduction on hidden-filled cases, retention of useful completion, and evidence that the rule transfers. A memory module that only memorizes “green mug means unsafe” fails. A named `filled` detector plus a manually written prohibition establishes grounding/retention, not rule induction.

## PSC and implementation effort

Start with a **single L40S-48 job** for the official frozen policy and simulator; this is a resource proposal, not a verified model-fit result. Use H100-80 if the matched model stack or memory footprint requires it. Avoid assuming that V100 supports all BF16/attention kernels in a newly released policy adapter. Reuse the existing OpenVLA-OFT familiarity, but keep VLA-Arena in an isolated environment to avoid collisions with installed LIBERO forks.

A 7–8B policy has roughly 14–16 GB of two-byte weights before vision/runtime overhead. Keep the VLM proposal stage offline and train small detector heads from cached features. A pilot must determine peak VRAM and actual total rollout time. At 120 seconds per rollout, 400 rollouts cost about 13.3 GPU-hours; at 300 seconds they cost 33.3 GPU-hours, before acquisition, setup, and fitting. Neither estimate includes queue delay or promises current available allocation.

MIKASA requires SAPIEN/Vulkan rendering in addition to CUDA; its [installation guide](https://mikasarobo.github.io/installation.html) makes that explicit. This is an additional commissioning step, not evidence that PSC cannot run it. BEHAVIOR's [RTX-oriented installation requirements](https://behavior.stanford.edu/getting_started/installation.html) make L40S a more natural starting point than H100 for that simulator, but its additional scope is unnecessary here.

Minimum new benchmark code is an observation-cue wrapper, an independent spill ledger, and an acquisition/transfer runner. Learning and monitoring are the research system, not benchmark scaffolding. Keep the existing safety-memory proof of life as a regression test; its fixed predicates and scripted controller already answer the narrower question of whether retaining a fact changes behavior.

**Final selection: VLA-Arena, one spill-preservation task, frozen OpenVLA-OFT, and the smallest controlled memory extension.** MIKASA is the alternate choice if the objective becomes memory evaluation without a physical-safety learning claim.
