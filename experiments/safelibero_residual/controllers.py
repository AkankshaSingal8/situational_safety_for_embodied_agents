"""Chunk controllers used by eval_sl.py.

base      : base flow policy chunk, first exec_steps actions executed.
oracle    : base chunk + geometric detour around the named obstacle (privileged sim state).
residual  : base chunk + sampled text-conditioned residual.
"""
import json

import numpy as np
import torch

from policy import BasePolicy


class BaseRunner:
    def __init__(self, ckpt, device="cuda", flow_steps=10):
        ck = torch.load(ckpt, map_location="cpu")
        self.meta = ck["meta"]
        self.model = BasePolicy(len(self.meta["langs"]), pretrained=False)
        self.model.load_state_dict(ck["model"]); self.model.eval().to(device)
        self.dev, self.fs = device, flow_steps
        self.amin = np.array(self.meta["act_min"], np.float32); self.amax = np.array(self.meta["act_max"], np.float32)
        self.pm = np.array(self.meta["prop_mean"], np.float32); self.ps = np.array(self.meta["prop_std"], np.float32)

    def task_id(self, lang):
        return self.meta["langs"].index(lang)

    def obs_tensors(self, obs, tid):
        import sl_env
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(self.dev)  # noqa: E731
        return (T(obs["agentview_image"]), T(obs["robot0_eye_in_hand_image"]),
                T((sl_env.proprio(obs) - self.pm) / self.ps), torch.tensor([tid], device=self.dev))

    def chunk(self, obs, tid):
        """Returns (normalized chunk (H,7) tensor, features tensor (1,F))."""
        x, f = self.model.sample(*self.obs_tensors(obs, tid), steps=self.fs, return_feats=True)
        return x[0], f

    def unnorm(self, x):
        a = (np.asarray(x, np.float32) + 1) / 2 * (self.amax - self.amin) + self.amin
        a[:, 6] = np.where(a[:, 6] > 0, 1.0, -1.0)
        return a


def make_runner(cfg):
    """Flow base (default) or pi0.5 served over websocket (cfg['base'] == 'pi05')."""
    if cfg.get("base", "flow") == "pi05":
        from pi05_runner import Pi05Runner
        return Pi05Runner(cfg["ckpt"], cfg["device"], cfg["flow_steps"], port=int(cfg.get("pi_port", 8000)))
    return BaseRunner(cfg["ckpt"], cfg["device"], cfg["flow_steps"])


class BaseController:
    def __init__(self, cfg):
        self.r = make_runner(cfg)
        self.k = cfg["exec_steps"]

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.n_chunks = 0

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        self.n_chunks += 1
        return self.r.unnorm(x.cpu().numpy())[: self.k]

    def episode_stats(self):
        return dict(n_chunks=self.n_chunks)


def make_controller(cfg):
    if cfg["arm"] == "base":
        return BaseController(cfg)
    if cfg["arm"] == "oracle":
        from oracle import OracleController
        return OracleController(cfg)
    if cfg["arm"] == "shield":
        from shield import ShieldController
        return ShieldController(cfg)
    if cfg["arm"] == "filter":
        from safety_filter import FilterController
        return FilterController(cfg)
    if cfg["arm"] == "residual_pg":
        from residual_pg_ctrl import ResidualPGController
        return ResidualPGController(cfg)
    if cfg["arm"] == "residual_ex":
        from residual_ex_ctrl import ResidualExController
        return ResidualExController(cfg)
    if cfg["arm"] in ("astra_base", "fable_base"):
        from astra_ctrl import AstraBaseController
        return AstraBaseController(cfg)
    if cfg["arm"] in ("astra_res", "fable_res"):
        from astra_ctrl import AstraResController
        return AstraResController(cfg)
    if cfg["arm"] == "residual_v2":
        from residual_v2_ctrl import ResidualV2Controller
        return ResidualV2Controller(cfg)
    if cfg["arm"].startswith("residual"):
        from residual_ctrl import ResidualController
        return ResidualController(cfg)
    raise ValueError(cfg["arm"])
