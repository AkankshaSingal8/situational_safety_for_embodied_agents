## Feature: SafeLIBERO + OpenVLA-OFT + Qwen Safety Filter Integrated Evaluation Plan

**What it does:**
Create a detailed implementation plan by updating `@integrated_eval_readme.md` for a full evaluation pipeline based on `@run_safelibero_openvla_oft_eval.py`, where the user can choose the SafeLIBERO suite and level, OpenVLA-OFT provides nominal actions, and a Qwen-based safety filter runs at the same control frequency to generate semantic constraints, construct ellipsoidal safety regions around relevant objects, solve a CBF-QP, and output safe actions. The plan must also include how to calculate and report evaluation metrics: Collision Avoidance Rate, Task Success Rate, and Episode Steps.

**Scope:**
- IN:
  - Read and analyze `@run_safelibero_openvla_oft_eval.py`
  - Update `@integrated_eval_readme.md` with a concrete implementation plan
  - Support user-specified SafeLIBERO suite and level
  - Define pipeline stages for:
    - OpenVLA-OFT policy inference
    - Qwen-based semantic safety filter
    - constraint extraction from Qwen
    - ellipsoid construction around objects
    - CBF-QP safety action optimization
    - environment stepping and evaluation logging
  - Include plan for computing:
    - Collision Avoidance Rate
    - Task Success Rate
    - Episode Steps
  - Include system design decision on how Qwen should run using the `qwen` environment
  - Compare:
    - `conda run -n qwen ...`
    - running Qwen and OpenVLA/Libero in two separate terminals/processes
  - Recommend the better option with technical reasoning
  - Identify assumptions, dependencies, interfaces, and possible bottlenecks
  - Ensure the plan is staged and reviewable before implementation
- OUT:
  - Do not implement code yet
  - Do not modify evaluation behavior without explicitly documenting assumptions
  - Do not introduce vague high-level suggestions without file-level or module-level planning
  - Do not assume Qwen and OpenVLA can safely share one environment unless verified
  - Do not omit synchronization/frequency considerations between policy and safety filter

**Files to modify/create:**
- Modify: `@integrated_eval_readme.md`
- Inspect: `@run_safelibero_openvla_oft_eval.py`
- Suggest after planning:
  - any new integration module(s)
  - any metrics utility file(s)
  - any Qwen runner/service wrapper file(s)
  - any config updates needed for suite/level/user selection

**Follow these patterns:**
- Mirror the structure and conventions already used in `@run_safelibero_openvla_oft_eval.py`
- Reuse existing evaluation flow, config handling, logging style, and rollout structure where possible
- Prefer minimal-intrusion integration over large refactors
- Separate nominal policy inference from safety filtering cleanly
- Keep metric computation explicit and reproducible
- If the repo already has utilities for logging, environment setup, or action post-processing, reuse them
- Clearly identify where the safety filter should be inserted in the control loop
- Explicitly document action frequency assumptions and how Qwen is synchronized with OpenVLA

**Constraints:**
- Qwen must run using the `qwen` environment
- OpenVLA-OFT and SafeLIBERO/Libero simulation may remain in their existing environment if needed
- The plan must explicitly evaluate whether `conda run -n qwen ...` is practical for per-step or per-action-frequency inference
- The plan must consider latency, IPC/process communication, fault isolation, debugging ease, and reproducibility
- The Qwen safety filter must run at the same frequency as the OpenVLA policy
- The safety filter pipeline is:
  - get observation/state/context
  - obtain constraints from Qwen
  - create ellipsoids around relevant objects
  - solve CBF-QP
  - return safe action
- The evaluation must support user-specified SafeLIBERO suite and level
- The plan must clearly define how each metric is computed:
  - Collision Avoidance Rate
  - Task Success Rate
  - Episode Steps
- Do not hallucinate existing helper functions, files, or APIs; verify from the code first
- If any part of the current script already computes one of these metrics, document that precisely instead of duplicating logic

**Testing required:**
- Planning validation that the proposed integration points are compatible with current rollout/eval flow
- Verification plan for user-selected SafeLIBERO suite and level
- Verification plan for metric correctness:
  - Task success detection
  - collision event detection / avoidance accounting
  - episode step counting
- Verification plan for synchronization between OpenVLA action generation and Qwen safety filtering
- Environment validation for running Qwen in a separate env
- Sanity-check plan for latency/performance impact of the safety filter
- Smoke-test plan for one small SafeLIBERO suite/level before full evaluation


**Next step:**
Create a detailed plan by updating `@integrated_eval_readme.md` only. The plan should include:
1. current-script analysis
2. proposed architecture
3. integration points in control loop
4. metrics computation plan
5. environment/process design recommendation for Qwen
6. file-level change plan
7. staged implementation plan
8. risks, assumptions, and validation plan

---

## Emphasis for this task

While planning, specifically answer:
- Whether Qwen should be run via `conda run -n qwen ...` per call, as a persistent background service in the `qwen` environment, or in a separate terminal/process
- Which option is best for same-frequency safety filtering and why
- Is it possible to run QWEN and OpenVLA on the same frequency

- How to compute and log:
  - Collision Avoidance Rate
  - Task Success Rate
  - Episode Steps
- What new modules/utilities are needed, if any
- What assumptions might break the integration