"""LIBERO-Safety end-to-end smoke: their env + their pi0.5 weights via our guided server.

Phase A (recon): create envs for each safety suite/level, print obs keys, joint
names containing 'obstacle', and object `*_pos` keys — determines how the
guided client's discovery/identity/collision code ports.
Phase B (rollout): obstacle_avoidance L1 task 0, n=2 episodes through the
guided policy server (guidance disabled = pure baseline sanity).

Env: PYTHONPATH must put LIBERO-Safety BEFORE SafeLIBERO; LIBERO_CONFIG_PATH
must point to libero_safety_config.yaml.
"""

import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "/ocean/projects/cis250185p/asingal/situational_safety_for_embodied_agents/.worktrees/flow-guidance-tier-gt/vlm_pipeline")


def _quat2axisangle(quat):
    import math
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--recon_only", action="store_true")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bdict = benchmark.get_benchmark_dict()
    print("suites available:", [k for k in bdict if "safety" in k or "obstacle" in k or "affordance" in k or "reasoning" in k])

    # ---- Phase A: recon each safety suite (first task, level dirs are in bddl path)
    for suite in ["obstacle_avoidance", "human_safety", "affordance", "reasoning_safety"]:
        try:
            bm = bdict[suite]()
            n = bm.get_num_tasks()
            task = bm.get_task(0)
            env = OffScreenRenderEnv(
                bddl_file_name=bm.get_task_bddl_file_path_by_level_id(task.level, task.level_id),
                camera_heights=128, camera_widths=128,
            )
            obs = env.reset()
            try:
                init_states = bm.get_task_init_states_by_level_id(task.level, task.level_id)
                obs = env.set_init_state(init_states[0])
            except FileNotFoundError:
                init_states = None  # reasoning_safety ships no .pruned_init — reset-state recon only
            for _ in range(5):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            pos_keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot0")]
            obst_joints = [j for j in env.sim.model.joint_names if "obstacle" in j.lower()]
            print(json.dumps({
                "suite": suite, "n_tasks": n, "task0": task.name,
                "language": str(task.language)[:80],
                "obj_pos_keys": pos_keys[:12], "obstacle_joints": obst_joints[:6],
                "img_keys": [k for k in obs if "image" in k],
            }))
            env.close()
        except Exception as e:
            print(f"RECON FAIL {suite}: {type(e).__name__}: {e}")

    if args.recon_only:
        print("LS_SMOKE_RECON_COMPLETE")
        return

    # ---- Phase B: 2 rollout episodes through the guided server (guidance off)
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    bm = bdict["obstacle_avoidance"]()
    task = bm.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bm.get_task_bddl_file_path_by_level_id(task.level, task.level_id),
        camera_heights=256, camera_widths=256,
    )
    init_states = bm.get_task_init_states_by_level_id(task.level, task.level_id)
    for ep in range(2):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        plan = collections.deque()
        done = False
        t = 0
        while t < 280 and not done:
            if not plan:
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )),
                    "prompt": str(task.language),
                }
                plan.extend(client.infer(element)["actions"][:5])
            obs, reward, done, info = env.step(plan.popleft().tolist())
            t += 1
        print(json.dumps({"ep": ep, "success": bool(done), "steps": t}))
    env.close()
    print("LS_SMOKE_COMPLETE")


if __name__ == "__main__":
    main()
