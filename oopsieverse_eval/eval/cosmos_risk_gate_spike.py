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

import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from cosmos_config import CosmosEvalConfig, DEFAULT_COSMOS_CONFIG_KWARGS  # noqa: E402

COSMOS_REPO = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/cosmos-policy"
if COSMOS_REPO not in sys.path:
    sys.path.insert(0, COSMOS_REPO)


def main():
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    print("=== Cosmos-Policy risk-gate spike (no sim construction) ===")

    cfg = CosmosEvalConfig(**DEFAULT_COSMOS_CONFIG_KWARGS)

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
