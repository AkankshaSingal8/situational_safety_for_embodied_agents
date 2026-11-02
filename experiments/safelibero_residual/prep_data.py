"""Pack the LIBERO-Spatial demos into flat arrays for the base flow policy.

    python prep_data.py --src /ocean/projects/cis250185p/asingal/data_libero/libero_spatial \
                        --out /ocean/projects/cis250185p/asingal/data_libero/packed_spatial

Images are kept exactly as stored in the hdf5 (the env's native orientation, 128 px); the
eval driver feeds raw env frames the same way, so no flip happens anywhere.
No-op steps (all six arm deltas ~0 and the gripper command unchanged) are dropped, as in
OpenVLA's LIBERO preprocessing; behaviour cloning otherwise learns to stall.
Proprio = 7 joint angles + 2 gripper finger positions (hdf5 joint_states/gripper_states,
env robot0_joint_pos/robot0_gripper_qpos), which avoids any quaternion/axis-angle mismatch.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def lang_from_file(path):
    base = os.path.basename(path).replace("_demo.hdf5", "")
    return base.replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_demos", type=int, default=0, help="per task, 0 = all (smoke tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.src, "*.hdf5")))
    assert files, f"no hdf5 in {a.src}"
    langs = [lang_from_file(f) for f in files]
    agent, wrist, prop, act, ep_id, task_id = [], [], [], [], [], []
    n_ep, kept, dropped = 0, 0, 0
    for ti, f in enumerate(files):
        with h5py.File(f, "r") as h:
            demos = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1]))
            if a.max_demos:
                demos = demos[: a.max_demos]
            for d in demos:
                g = h["data"][d]
                A = g["actions"][()].astype(np.float32)
                ob = g["obs"]
                keep = np.ones(len(A), bool)
                prev_grip = None
                for t in range(len(A)):
                    noop = np.linalg.norm(A[t, :6]) < 1e-4 and (prev_grip is not None and A[t, 6] == prev_grip)
                    if noop:
                        keep[t] = False
                    prev_grip = A[t, 6]
                agent.append(ob["agentview_rgb"][()][keep])
                wrist.append(ob["eye_in_hand_rgb"][()][keep])
                prop.append(np.concatenate([ob["joint_states"][()], ob["gripper_states"][()]], 1)[keep].astype(np.float32))
                act.append(A[keep])
                ep_id.append(np.full(keep.sum(), n_ep, np.int32))
                task_id.append(np.full(keep.sum(), ti, np.int16))
                kept += keep.sum(); dropped += (~keep).sum(); n_ep += 1
        print(f"task {ti}: {langs[ti]}  episodes so far {n_ep}", flush=True)
    cat = lambda xs: np.concatenate(xs, 0)  # noqa: E731
    arrs = dict(agent=cat(agent), wrist=cat(wrist), prop=cat(prop), act=cat(act), ep=cat(ep_id), task=cat(task_id))
    for k, v in arrs.items():
        np.save(os.path.join(a.out, k + ".npy"), v)
    A = arrs["act"]
    meta = dict(langs=langs, n_episodes=n_ep, n_frames=int(kept), dropped_noops=int(dropped),
                act_min=A.min(0).tolist(), act_max=A.max(0).tolist(), act_mean=A.mean(0).tolist(), act_std=A.std(0).tolist(),
                prop_mean=arrs["prop"].mean(0).tolist(), prop_std=(arrs["prop"].std(0) + 1e-6).tolist(),
                grip_values=sorted(set(np.round(A[:, 6], 3).tolist()))[:10])
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("n_episodes", "n_frames", "dropped_noops", "grip_values")}))


if __name__ == "__main__":
    main()
