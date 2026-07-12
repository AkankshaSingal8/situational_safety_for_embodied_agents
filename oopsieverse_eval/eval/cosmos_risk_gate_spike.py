"""Risk-gate spike for Cosmos-Policy on OopsieVerse (plan Task 9).

Goal: confirm `cosmos_policy.experiments.robot.cosmos_utils` (get_model +
get_action + live T5 embedding) runs cleanly inside the existing
cosmos_policy.sif / .venv_rhel8 container -- WITHOUT ever constructing a
robosuite/RoboCasa env in the same process -- using synthetic observations
shaped like OopsieVerse's RoboCasa contract (2 third-person cameras + wrist
+ 9-dim proprio + an arbitrary instruction string not in any precomputed
T5 cache).

This does NOT touch the sim at all. If this segfaults or errors, STOP --
do not write cosmos_server.py until this passes cleanly (see plan's
Extension: Cosmos-Policy + Fast-WAM > Phased implementation > step 2).

Run via apptainer (see ../slurm/spike_cosmos_risk_gate.slurm), reusing the
existing container/venv read-only -- built by
/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy/slurm/build_cosmos_sif.slurm.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import numpy as np

COSMOS_REPO = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy"
if COSMOS_REPO not in sys.path:
    sys.path.insert(0, COSMOS_REPO)


@dataclass
class _SpikeConfig:
    """Minimal duck-typed stand-in for run_robocasa_eval.py's PolicyEvalConfig.

    NOT importing PolicyEvalConfig itself: that module does
    `from robocasa.utils.dataset_registry import ...` at import time (the
    real RoboCasa pip package, from moojink's fork), which isn't installed
    in this container's .venv_rhel8 (built with `--group libero`, not
    `--group robocasa` -- confirmed by this spike's first failed attempt).
    get_model()/get_action() only touch a handful of plain attributes, so a
    local dataclass with just those fields avoids that import entirely
    without needing to install/rebuild anything.
    """

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

    # Every other field get_action()/prepare_images_for_model() reads,
    # copied from PolicyEvalConfig's own defaults (ROBOCASA.md's example CLI
    # invocation) -- not importing that class itself, see docstring above.
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


def main():
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    print("=== Cosmos-Policy risk-gate spike (no sim construction) ===")

    cfg = _SpikeConfig(
        suite="robocasa",
        config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ckpt_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json",
        t5_text_embeddings_path="",  # deliberately empty: force a live T5 compute, not a cache hit
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        chunk_size=32,
        env_img_res=224,
    )

    print(f"Loading model from {cfg.ckpt_path} (config={cfg.config})...")
    model, model_config = get_model(cfg)
    print("Model loaded OK.")

    print(f"Loading dataset stats from {cfg.dataset_stats_path}...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    print("Dataset stats loaded OK.")

    # Deliberately empty cache -> get_action's get_t5_embedding_from_cache()
    # must compute this instruction's T5 embedding live.
    init_t5_text_embeddings_cache(None)

    novel_instruction = "carefully place the bowl onto the plate without spilling"
    print(f"Using a novel (uncached) instruction: {novel_instruction!r}")

    # Synthetic observations shaped like OopsieVerse's RoboCasa contract:
    # agentview_left (primary), agentview_right (secondary), eye_in_hand (wrist).
    obs = {
        "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "secondary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "proprio": np.random.uniform(-1, 1, size=(9,)).astype(np.float32),
    }

    print("Calling get_action() with synthetic obs (this is the segfault-risk seam)...")
    result = get_action(
        cfg,
        model,
        dataset_stats,
        obs,
        novel_instruction,
        seed=195,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    print(f"get_action() returned OK. Keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    if isinstance(result, dict) and "action" in result:
        action = np.asarray(result["action"])
        print(f"action shape: {action.shape}, dtype: {action.dtype}")

    print("\n=== RISK-GATE SPIKE: PASS ===")
    print("Safe to proceed to Task 10 (cosmos_server.py + RoboCasa client).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== RISK-GATE SPIKE: FAIL ===")
        traceback.print_exc()
        print("\nSTOP: do not build cosmos_server.py until this is resolved (see plan).")
        sys.exit(1)
