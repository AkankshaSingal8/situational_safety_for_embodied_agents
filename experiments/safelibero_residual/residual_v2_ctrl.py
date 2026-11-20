"""pi0.5 + v2 residual (frozen SigLIP features, train_v2.py) on the xyz dims (eval_sl --arm residual_v2).
Frames go through exactly the collection path: 224 px INTER_AREA resize of the env frame, rotated 180 deg,
SigLIP patch tokens 7x7-pooled (cache_feats_v2.encode_images)."""
import cv2
import numpy as np
import torch

import texts as TX
from cache_feats_v2 import encode_images, encode_texts, load_siglip
from collect_v2 import state8
from controllers import make_runner
from policy_v2 import ResidualFlowV2


class ResidualV2Controller:
    def __init__(self, cfg):
        self.r = make_runner(cfg)
        ck = torch.load(cfg["residual"], map_location="cpu")
        self.m = ResidualFlowV2(horizon=ck["cfg"]["horizon"], layers=ck["cfg"]["layers"])
        self.m.load_state_dict(ck["model"]); self.m.eval().to(cfg["device"])
        self.st = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
        z = np.load(ck["text_bank"], allow_pickle=True)
        self.bank = dict(zip(z["texts"].tolist(), z["emb"].astype(np.float32)))
        self.sig, self.tok = load_siglip(cfg["device"])
        self.k, self.fs, self.dev = cfg["exec_steps"], cfg["flow_steps"], cfg["device"]
        self.n_samp = int(cfg.get("res_samples") or 1)

    def emb(self, text):
        if text not in self.bank:
            self.bank[text] = encode_texts(self.sig, self.tok, [text], self.dev)[0]
        return torch.from_numpy(self.bank[text]).unsqueeze(0).to(self.dev)

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.has = torch.tensor([float(text != TX.NULL)], device=self.dev)
        self.sent = self.emb(text) if text != TX.NULL else torch.zeros(1, 768, device=self.dev)
        self.instr = self.emb(lang)
        self.n_chunks, self.mag = 0, []

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        A = self.r.unnorm(x.cpu().numpy())
        im = np.stack([cv2.resize(obs[k], (224, 224), interpolation=cv2.INTER_AREA)
                       for k in ("agentview_image", "robot0_eye_in_hand_image")])
        f = torch.from_numpy(encode_images(self.sig, im, self.dev)).to(self.dev)
        T = lambda v: torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(self.dev)  # noqa: E731
        st = T((state8(obs) - self.st["st_m"]) / self.st["st_s"])
        bn = T(((A.reshape(-1) - self.st["b_m"]) / self.st["b_s"]).reshape(A.shape))
        d = self.m.sample(steps=self.fs, n=self.n_samp, agent=f[:1], wrist=f[1:], sent=self.sent, has_sent=self.has,
                          instr=self.instr, state_n=st, base_n=bn)[0].cpu().numpy() / float(self.st["scale"])
        A[:, :3] = np.clip(A[:, :3] + d, -1, 1)
        self.n_chunks += 1
        self.mag.append(float(np.abs(d[: self.k]).sum(0).max()))
        return A[: self.k]

    def episode_stats(self):
        return dict(n_chunks=self.n_chunks, mean_corr=round(float(np.mean(self.mag)), 4) if self.mag else 0.0,
                    max_corr=round(float(np.max(self.mag)), 4) if self.mag else 0.0)
