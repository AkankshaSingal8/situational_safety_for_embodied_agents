"""Shared minimal config for calling cosmos_policy.experiments.robot.cosmos_utils
(get_model/get_action) directly, without importing run_robocasa_eval.py's
PolicyEvalConfig.

That module does `from robocasa.utils.dataset_registry import ...` at
import time (the real RoboCasa pip package, from moojink's fork), which
isn't installed in the container's .venv_rhel8 (built with `--group
libero`, not `--group robocasa` -- confirmed by cosmos_risk_gate_spike.py's
first failed attempt). get_model()/get_action() only touch a handful of
plain attributes, so this local dataclass avoids that import entirely
without needing to install/rebuild anything in the shared venv. Field
defaults are copied from PolicyEvalConfig's own defaults (ROBOCASA.md's
example CLI invocation).

Used by both cosmos_risk_gate_spike.py and servers/cosmos_server.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CosmosEvalConfig:
    suite: str = "robocasa"
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    chunk_size: int = 32
    env_img_res: int = 224

    use_third_person_image: bool = True
    num_third_person_images: int = 2
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    trained_with_image_aug: bool = True
    seed: int = 195
    randomize_seed: bool = False
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 3
    future_state_ensemble_aggregation_scheme: str = "average"
    use_ensemble_value_predictions: bool = False
    num_value_predictions_in_ensemble: int = 5
    value_ensemble_aggregation_scheme: str = "average"
    search_depth: int = 1
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False
    num_queries_best_of_n: int = 1
    parallel_timeout: int = 15


DEFAULT_COSMOS_CONFIG_KWARGS = dict(
    suite="robocasa",
    config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
    ckpt_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json",
    t5_text_embeddings_path="",  # force live T5 compute for OopsieVerse's novel instructions
    num_denoising_steps_action=5,
    num_denoising_steps_future_state=1,
    num_denoising_steps_value=1,
    chunk_size=32,
    env_img_res=224,
)
