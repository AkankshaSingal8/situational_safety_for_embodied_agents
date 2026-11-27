"""Policy server: GR00T-N1.6 base + residual expert (drop-in for run_gr00t_server.py --use-sim-policy-wrapper).

The base chunk comes from the unmodified GR00T-N1.6-3B (Gr00tSimPolicyWrapper, identical to the base arm); the
residual expert sees the same observation, the prompt "<instruction>. <safety sentence>" and GR00T's base xyz
chunk, and its xyz residual is added to action.end_effector_position. Run in the groot_srv_u24 env.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groot_residual import CAMS, EMB, MODEL, SENTENCES, STATE_KEYS, GrootResidual, prompt  # noqa: E402

XYZ = "action.end_effector_position"


def build(ckpt, task, text_mode, n=1, gate_thr=0.0):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.policy import BasePolicy

    base = Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EmbodimentTag[EMB], model_path=MODEL, device="cuda",
                                             strict=False), strict=False)
    res = GrootResidual()
    res.load(ckpt)
    res.eval()
    sentence = {"true": SENTENCES[task], "none": ""}.get(text_mode)
    if sentence is None:  # "wrong": another task's sentence
        sentence = SENTENCES[[t for t in SENTENCES if t != task][0]]

    class ResidualSimPolicy(BasePolicy):
        def __init__(self):
            super().__init__(strict=False)

        def check_observation(self, observation):
            base.check_observation(observation)

        def check_action(self, action):
            base.check_action(action)

        def get_modality_config(self):
            return base.get_modality_config()

        def reset(self, options=None):
            return base.reset(options)

        def _get_action(self, observation, options=None):
            act, info = base.get_action(observation)
            B = np.asarray(act[XYZ]).shape[0]
            lang = observation["annotation.human.action.task_description"]
            ims = [{c: np.asarray(observation[f"video.res256_image_{c}"])[i, -1] for c in CAMS} for i in range(B)]
            sts = [np.concatenate([np.asarray(observation[f"state.{k}"])[i, -1].reshape(-1) for k in STATE_KEYS])
                   for i in range(B)]
            txs = [prompt(lang[i], sentence) for i in range(B)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r, g = res.sample(ims, sts, txs, np.asarray(act[XYZ], np.float32), n=n, with_gate=True)
            if gate_thr > 0:  # gate says no correction needed: GR00T's chunk unchanged (res_server --gate_thr)
                r = r * (g >= gate_thr)[:, None, None]
            print(f"gate {np.round(g, 3).tolist()} |r| {float(np.abs(r).max()):.4f}", flush=True)
            act = dict(act)
            act[XYZ] = (np.asarray(act[XYZ], np.float32) + r.astype(np.float32))
            return act, info

    return ResidualSimPolicy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=list(SENTENCES))
    ap.add_argument("--text_mode", default="true", choices=["true", "wrong", "none"])
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--n", type=int, default=1, help="average n residual samples (res_server --n)")
    ap.add_argument("--gate_thr", type=float, default=0.0, help="zero the residual when gate prob < this (0 = off)")
    a = ap.parse_args()
    from gr00t.policy.server_client import PolicyServer
    pol = build(a.ckpt, a.task, a.text_mode, a.n, a.gate_thr)
    PolicyServer(policy=pol, host="0.0.0.0", port=a.port).run()


if __name__ == "__main__":
    main()
