"""Base flow policy + sampled text-conditioned residual on the xyz dims (eval_sl --arm residual)."""
import numpy as np
import torch

import texts as TX
from controllers import BaseRunner
from policy import ResidualFlow, ResidualFlowImg


class ResidualController:
    def __init__(self, cfg):
        self.r = BaseRunner(cfg["ckpt"], cfg["device"], cfg["flow_steps"])
        ck = torch.load(cfg["residual"], map_location="cpu")
        self.variant, self.scale = ck["variant"], ck["scale"]
        self.img = self.variant.endswith("_img")
        cls = ResidualFlowImg if self.img else ResidualFlow
        self.m = cls(feat_dim=ck["feat_dim"], text_dim=768 if self.variant.startswith("text") else 0)
        self.m.load_state_dict(ck["model"]); self.m.eval().to(cfg["device"])
        self.bank = TX.TextBank()
        self.k, self.fs, self.dev = cfg["exec_steps"], cfg["flow_steps"], cfg["device"]
        self.half = (self.r.amax[:3] - self.r.amin[:3]) / 2
        self.guide = float(cfg.get("guide") or 1.0)

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.text = torch.from_numpy(self.bank(text)).unsqueeze(0).to(self.dev)
        self.n_chunks, self.mag = 0, []

    def act_chunk(self, env, obs):
        x, f = self.r.chunk(obs, self.tid)
        if self.img:
            ia, iw = self.r.obs_tensors(obs, self.tid)[:2]
            d = self.m.sample_img(ia, iw, f, x.unsqueeze(0), self.text, steps=self.fs)[0].cpu().numpy() / self.scale
        else:
            d = self.m.sample(f, x.unsqueeze(0), self.text, steps=self.fs, guide=self.guide)[0].cpu().numpy() / self.scale
        A = self.r.unnorm(x.cpu().numpy())
        dA = d * self.half
        A[:, :3] = np.clip(A[:, :3] + dA, -1, 1)
        self.n_chunks += 1
        self.mag.append(float(np.abs(dA[: self.k]).sum(0).max()))
        return A[: self.k]

    def episode_stats(self):
        return dict(n_chunks=self.n_chunks, mean_corr=round(float(np.mean(self.mag)), 4) if self.mag else 0.0,
                    max_corr=round(float(np.max(self.mag)), 4) if self.mag else 0.0)
