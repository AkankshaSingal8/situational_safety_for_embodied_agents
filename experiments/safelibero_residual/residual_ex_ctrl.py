"""pi0.5 LIBERO + residual action expert (res_server.py), xyz dims (eval_sl --arm residual_ex --ex_port P).

Per chunk: pi0.5 LIBERO proposes A (H,7); the server's expert (pi05_base action expert over the frozen pi05_base
PaliGemma) returns an xyz residual for "<instruction>. <sentence>"; A[:, :3] += residual, clipped to [-1, 1];
the first exec_steps actions are executed. Frames follow the collection path (env frame -> 224 px INTER_AREA).
"""
import cv2
import numpy as np

from collect_v2 import state8
from controllers import make_runner
from residual_pg_ctrl import FeatureClient, prompt


class ResidualExController:
    def __init__(self, cfg):
        self.r = make_runner(cfg)
        self.k = cfg["exec_steps"]
        self.cl = FeatureClient(int(cfg["ex_port"]))

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.p = prompt(lang, text)
        self.n_chunks, self.mag = 0, []

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        A = self.r.unnorm(x.cpu().numpy())
        ag = cv2.resize(obs["agentview_image"], (224, 224), interpolation=cv2.INTER_AREA)
        wr = cv2.resize(obs["robot0_eye_in_hand_image"], (224, 224), interpolation=cv2.INTER_AREA)
        d = np.asarray(self.cl.ws_call(dict(agent=ag, wrist=wr, prompt=self.p, state=state8(obs),
                                            base=A.astype(np.float32)))["delta"])
        A[:, :3] = np.clip(A[:, :3] + d, -1, 1)
        self.n_chunks += 1
        self.mag.append(float(np.abs(d[: self.k]).sum(0).max()))
        return A[: self.k]

    def episode_stats(self):
        return dict(n_chunks=self.n_chunks, mean_corr=round(float(np.mean(self.mag)), 4) if self.mag else 0.0,
                    max_corr=round(float(np.max(self.mag)), 4) if self.mag else 0.0)
