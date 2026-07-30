"""S1 kill-test: can the frozen policy's own task head aim at a named obstacle?

Semantic self-guidance (contrastive velocity composition) requires that the
chunk generated under a synthesized hazard-APPROACH instruction points toward
the hazard — the policy's approach competence becomes the hazard direction
model. This probe measures exactly that, offline, no rollouts:

For each Spatial L1 scene (4 tasks x N_EP init states, settled):
  chunk(task prompt)                     -> control displacement
  chunk("pick up the <obstacle>")        -> approach displacement A1
  chunk("move the gripper to the <obstacle>") -> approach displacement A2
cos_xy(displacement, obstacle - eef) recorded per prompt.

Pre-registered GO: median cos(approach) >= 0.5 AND cos(approach) > cos(task)
on >= 70% of scenes. NO-GO otherwise (instruction swap does not redirect the
chunk => S1 dies like the other conditioning arms).

Server: plain guided server, but no `guidance` key is sent -> unguided
denoising. One inference per (scene, prompt): ~60 calls total.
"""

import argparse
import json
import logging
import math
import pathlib
import re

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as wcp

logging.basicConfig(level=logging.INFO, format="%(message)s")
RESIZE = 224
SETTLE = 10
# Task #36 GO bar is ">=15/19 scenes"; anything under this is a broken harness.
MIN_SCENES = 15


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if den < 1e-9:
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def element(obs, prompt):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, RESIZE, RESIZE)),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"])),
        "prompt": prompt,
    }


def chunk_displacement(client, obs, prompt):
    acts = np.asarray(client.infer(element(obs, prompt))["actions"])
    return acts[:, :3].sum(axis=0)


def cos_xy(d, u):
    d2, u2 = np.asarray(d)[:2], np.asarray(u)[:2]
    n = np.linalg.norm(d2) * np.linalg.norm(u2)
    return float(d2 @ u2 / n) if n > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--episodes_per_task", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = wcp.WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    rows = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                 camera_widths=256, camera_depths=False)
        env.seed(7)
        inits = suite.get_task_init_states(task_id)
        for ep in range(args.episodes_per_task):
            env.reset()
            obs = env.set_init_state(inits[ep % len(inits)])
            for _ in range(SETTLE):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
            # Obstacle selection must mirror the eval client exactly: SafeLIBERO
            # names obstacles `<object>_obstacle_<idx>`, so the observation key is
            # `<object>_obstacle_<idx>_pos` — an `endswith("_obstacle_pos")` filter
            # matches nothing and silently drops every scene.
            ob_name = None
            for n in [j.replace("_joint0", "") for j in env.sim.model.joint_names
                      if "obstacle" in j]:
                p = obs.get(f"{n}_pos")
                if p is not None and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                    ob_name = n
                    break
            if ob_name is None:
                logging.warning("t%d ep%d: no obstacle observable, skipping", task_id, ep)
                continue
            clean = re.sub(r"_obstacle(_\d+)?$", "", ob_name).replace("_", " ")
            u = np.asarray(obs[f"{ob_name}_pos"]) - np.asarray(obs["robot0_eef_pos"])
            prompts = {
                "task": task.language,
                "approach1": f"pick up the {clean}",
                "approach2": f"move the gripper to the {clean}",
            }
            row = {"task_id": task_id, "ep": ep, "obstacle": clean,
                   "dist": float(np.linalg.norm(u))}
            for key, p in prompts.items():
                d = chunk_displacement(client, obs, p)
                row[f"cos_{key}"] = cos_xy(d, u)
                row[f"norm_{key}"] = float(np.linalg.norm(d[:2]))
            rows.append(row)
            logging.info("t%d ep%d %-14s cos task % .2f  appr1 % .2f  appr2 % .2f",
                         task_id, ep, clean, row["cos_task"],
                         row["cos_approach1"], row["cos_approach2"])
        env.close()

    appr = [max(r["cos_approach1"], r["cos_approach2"]) for r in rows]
    beats = [a > r["cos_task"] for a, r in zip(appr, rows)]
    med = float(np.median(appr)) if appr else float("nan")
    frac = float(np.mean(beats)) if beats else float("nan")
    # A probe that collected nothing is a harness failure, not evidence. Writing
    # "NO-GO" on n=0 is how a zero-scene bug got recorded as a result on
    # 2026-07-29; refuse to emit a verdict unless the scenes are actually there.
    if len(rows) < MIN_SCENES:
        verdict = "INVALID"
        logging.error("S1 probe collected %d/%d scenes — NO VERDICT. Fix the "
                      "harness before interpreting this run.", len(rows), MIN_SCENES)
    else:
        verdict = "GO" if (med >= 0.5 and frac >= 0.7) else "NO-GO"
    summary = {"n_scenes": len(rows), "median_cos_approach": med,
               "frac_approach_beats_task": frac, "verdict": verdict}
    logging.info("S1 PROBE SUMMARY: %s", json.dumps(summary))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
