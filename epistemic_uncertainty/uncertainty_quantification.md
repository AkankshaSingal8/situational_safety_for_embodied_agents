I want to design an implementation plan for calculating epistemic uncertainty during OpenVLA evaluation on the SafeLIBERO benchmark.

Context:
- I am evaluating OpenVLA-OFT on SafeLIBERO.
- The evaluation should be configurable by the user.
- User should be able to specify:
  1. LIBERO / SafeLIBERO suite name
  2. Task ID or task name
  3. Number of episodes
  4. OpenVLA checkpoint path
  5. Whether to enable each uncertainty method

Goal:
Create a clean implementation plan that extends the existing OpenVLA evaluation pipeline so that, during each episode, we compute uncertainty scores, log them, and later evaluate whether uncertainty correlates with failure, unsafe behavior, or out-of-distribution scenes.

Uncertainty methods to support:

1. Entropy-based uncertainty
   - Purpose: VLM / failure-monitor uncertainty.
   - If the VLM outputs probabilities or structured confidence scores, compute entropy over candidate safety/failure labels.
   - If only text is available, propose a practical way to convert outputs into a discrete probability-like confidence estimate.
   - Output per step:
     - entropy_score
     - predicted_failure_label
     - confidence

2. MC Dropout
   - Purpose: OpenVLA action uncertainty.
   - Enable dropout at inference time if possible.
   - Run N stochastic forward passes for the same observation and language instruction.
   - Compute uncertainty over predicted actions:
     - mean action
     - action variance
     - action standard deviation
     - norm of translational action variance
     - norm of rotational action variance if applicable
   - Explain where this should be inserted in the OpenVLA action-selection loop.

3. Deep Ensemble
   - Purpose: estimate model uncertainty using multiple OpenVLA-OFT / LoRA checkpoints.
   - Load multiple checkpoints provided by the user.
   - For each observation, query all models.
   - Compute:
     - ensemble mean action
     - ensemble variance
     - pairwise action disagreement
     - max disagreement
   - Plan for memory-efficient execution, since multiple OpenVLA models may not fit on one GPU.
   - Include options:
     - sequential loading
     - separate inference servers
     - separate processes
     - precomputed action logs

4. Density Estimation / OOD Detection
   - Purpose: detect out-of-distribution scenes.
   - Extract scene features from either:
     - DINOv2 image encoder features
     - OpenVLA visual backbone features
   - Fit a density model or distance-based OOD detector on in-distribution LIBERO / SafeLIBERO training scenes.
   - Candidate methods:
     - Mahalanobis distance
     - kNN distance
     - Gaussian density
     - one-class SVM if simple enough
   - During evaluation, compute:
     - feature vector
     - OOD score
     - nearest-neighbor distance
     - is_ood flag based on threshold

5. Conformal Prediction
   - Purpose: calibrated threshold for deciding when to trust OpenVLA.
   - Use a calibration set of episodes.
   - Define nonconformity scores from uncertainty values, for example:
     - action variance
     - ensemble disagreement
     - OOD score
     - VLM entropy
   - Calibrate thresholds for a desired risk level alpha.
   - During test-time evaluation, output:
     - trust / abstain decision
     - conformal threshold
     - uncertainty score
     - whether the decision was correct after episode outcome is known

Implementation requirements:
- Do not directly implement the full code yet.
- First inspect the current evaluation script structure.
- Identify the exact files/functions/classes that need to be modified.
- Propose a modular design with minimal disruption to the existing OpenVLA evaluation code.
- Prefer adding new files/modules instead of heavily modifying core evaluation logic.
- Keep OpenVLA inference, uncertainty estimation, logging, and metric computation separated.

Expected plan structure:

1. Current code understanding
   - Explain the current OpenVLA evaluation flow.
   - Identify where observations are collected.
   - Identify where language instructions are passed.
   - Identify where OpenVLA predicts actions.
   - Identify where actions are executed in the environment.
   - Identify where success/failure is recorded.
   - Identify what metrics are necessary for recording
   - Identify granularity of the ac (or chunk of action)

2. Proposed architecture
   Design the following modules:

   - `base.py`
     - abstract base class for uncertainty estimators

   - `entropy_monitor.py`
     - entropy from VLM/failure monitor

   - `mc_dropout.py`
     - MC dropout action uncertainty

   - `deep_ensemble.py`
     - ensemble disagreement across checkpoints

   - `density_ood.py`
     - DINOv2/OpenVLA feature-based OOD scoring

   - `conformal.py`
     - conformal calibration and trust decision

   - `logger.py`
     - per-step and per-episode uncertainty logging

   - `run_safelibero_uncertainty_eval.py`
     - main evaluation entrypoint

3. CLI design
   Propose command-line arguments such as:

   ```bash
   python scripts/run_safelibero_uncertainty_eval.py \
     --suite safelibero_spatial \
     --task-id 0 \
     --num-episodes 10 \
     --openvla-checkpoint /path/to/checkpoint \
     --uncertainty-methods entropy mc_dropout density conformal \
     --mc-samples 10 \
     --ensemble-checkpoints ckpt1 ckpt2 ckpt3 \
     --ood-feature-backbone dinov2 \
     --conformal-alpha 0.1 \
     --output-dir results/uncertainty_eval