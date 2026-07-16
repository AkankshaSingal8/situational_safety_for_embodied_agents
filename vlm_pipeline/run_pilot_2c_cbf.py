"""Pilot 2c (glass tilt over laptop) — CBF filter portability check on MuJoCo.

Exercises the real `certify_action_simple`/`evaluate_cbf` functions (imported
unmodified from run_libero_eval_integrated.py) against a hand-authored
ellipsoid around the laptop-proxy object's real simulated position, with a
scripted action rollout that deliberately drives the end-effector toward it.

No VLM server, no trained policy — this pilot isolates the CBF filter's
action-certification math from everything else, to test whether it's
actually simulator-agnostic (only consumes ee_pos + ellipsoid params).
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents"
sys.path.insert(0, os.path.join(REPO, "SafeLIBERO", "safelibero"))
sys.path.insert(0, os.path.join(REPO, "vlm_pipeline"))

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from run_libero_eval_integrated import certify_action_simple, evaluate_cbf  # noqa: E402

BDDL_PATH = os.path.join(
    REPO,
    "SafeLIBERO/safelibero/libero/libero/bddl_files/safety_taxonomy/"
    "pick_up_the_glass_and_place_it_away_from_the_laptop_level_I.bddl",
)
INIT_PATH = os.path.join(
    REPO,
    "SafeLIBERO/safelibero/libero/libero/init_files/safety_taxonomy/"
    "pick_up_the_glass_and_place_it_away_from_the_laptop_level_I.pruned_init",
)
OUT_DIR = os.path.join(REPO, "pilot_bakeoff/mujoco/tilt_pilot")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    env = OffScreenRenderEnv(
        bddl_file_name=BDDL_PATH,
        camera_names=["agentview"],
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    env.reset()

    init_states = torch.load(INIT_PATH, weights_only=False)
    env.set_init_state(init_states[0])
    obs, _, _, _ = env.step(np.zeros(7))

    # Real simulated position of the laptop-proxy (yellow_book_obstacle_1),
    # via the standard per-object obs key (documented in CLAUDE.md).
    if "yellow_book_obstacle_1_pos" in obs:
        laptop_pos = np.array(obs["yellow_book_obstacle_1_pos"])
    else:
        print(f"obs keys available: {sorted(obs.keys())}")
        laptop_pos = np.array(env.sim.data.get_body_xpos("yellow_book_obstacle_1_main"))
    print(f"Laptop-proxy real sim position: {laptop_pos}")

    # Hand-authored ellipsoid around the laptop-proxy (skips VLM extraction,
    # same documented scope decision as the 1a-i pilot). Semi-axes sized to
    # roughly cover a book/laptop footprint plus a small safety margin.
    ellipsoids = [{
        "object": "yellow_book_obstacle_1",
        "relationship": "above",
        "center": laptop_pos.tolist(),
        "semi_axes": [0.08, 0.08, 0.05],
    }]

    ee_pos = np.array(obs["robot0_eef_pos"])
    print(f"Initial EE position: {ee_pos}")

    log = []
    # Scripted rollout: drive the EE in a straight line toward the laptop
    # ellipsoid's center over 60 steps, well past where it should intervene.
    direction = laptop_pos - ee_pos
    direction[2] = 0.0  # approach at constant height first, then descend
    step_vec = direction / 60.0

    for t in range(60):
        u_cmd = np.zeros(7)
        u_cmd[:3] = step_vec / 0.05  # velocity command matching dt=0.05 in certify_action_simple
        u_cmd[:3] = np.clip(u_cmd[:3], -1.0, 1.0)

        u_cert, info = certify_action_simple(
            u_cmd=u_cmd, ee_pos=ee_pos, ellipsoids=ellipsoids, dt=0.05, alpha=1.0,
        )

        modified = not np.allclose(u_cmd, u_cert, atol=1e-6)
        h_val = info["h_values"][0]["h"] if info["h_values"] else None
        log.append({
            "step": t,
            "ee_pos": ee_pos.tolist(),
            "h": h_val,
            "modified": bool(modified),
            "u_cmd": u_cmd.tolist(),
            "u_cert": u_cert.tolist(),
        })

        obs, _, _, _ = env.step(u_cert)
        ee_pos = np.array(obs["robot0_eef_pos"])

    env.close()

    with open(os.path.join(OUT_DIR, "cbf_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    h_vals = [row["h"] for row in log if row["h"] is not None]
    n_modified = sum(1 for row in log if row["modified"])
    print(f"h range: min={min(h_vals):.4f} max={max(h_vals):.4f}")
    print(f"Steps with modified action: {n_modified}/{len(log)}")
    print(f"First modified step: {next((r['step'] for r in log if r['modified']), None)}")

    if n_modified > 0 and min(h_vals) < max(h_vals):
        print("CBF_PILOT_RESULT: PASS (filter activated, h varied sanely)")
    else:
        print("CBF_PILOT_RESULT: NO_ACTIVATION (filter never intervened — check ellipsoid placement/scale)")


if __name__ == "__main__":
    main()
