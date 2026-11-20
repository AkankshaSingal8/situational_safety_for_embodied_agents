"""pi0.5 + residual over frozen pi05_base backbone tokens (train_pg.py), xyz dims (eval_sl --arm residual_pg).

Backbones (from the checkpoint):
  pg      fused PaliGemma tokens for "<instruction>. <sentence>" from pg_server.py
  sig     SigLIP-So400m tokens from pg_server.py + SigLIP-base text embeddings
  siglip  SigLIP-base image tokens (cache_feats_v2.encode_images) + SigLIP-base text embeddings
Frames follow the collection path: env frame -> 224 px INTER_AREA, raw orientation.
cfg: residual (ckpt path), pg_port (feature server), plus the usual pi0.5 runner keys.
"""
import cv2
import numpy as np
import torch

import texts as TX
from collect_v2 import state8
from controllers import make_runner
from policy_pg import ResidualFlowTok


def prompt(lang, sentence=""):
    lang = lang.strip().rstrip(".")
    return f"{lang}. {sentence.strip()}" if sentence else lang


class FeatureClient:
    def __init__(self, port, host="localhost"):
        import websockets.sync.client
        from openpi_client import msgpack_numpy
        self.uri, self.mp = f"ws://{host}:{port}", msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None, ping_interval=None,
                                                 open_timeout=600)
        self.ws.recv()  # metadata

    def __call__(self, agent, wrist, text):
        return self.ws_call({"agent": agent, "wrist": wrist, "prompt": text})

    def ws_call(self, d):
        self.ws.send(self.packer.pack(d))
        r = self.ws.recv()
        if isinstance(r, str):
            raise RuntimeError(r)
        return self.mp.unpackb(r)


class ResidualPGController:
    def __init__(self, cfg):
        self.r = make_runner(cfg)
        ck = torch.load(cfg["residual"], map_location="cpu", weights_only=False)
        self.bb = ck["backbone"]
        self.m = ResidualFlowTok(**ck["cfg"])
        self.m.load_state_dict(ck["model"]); self.m.eval().to(cfg["device"])
        self.st = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
        self.dev, self.k, self.fs = cfg["device"], cfg["exec_steps"], cfg["flow_steps"]
        self.n_samp = int(cfg.get("res_samples") or 1)
        if self.bb in ("pg", "sig"):
            self.fc = FeatureClient(int(cfg["pg_port"]))
        if self.bb != "pg":
            from cache_feats_v2 import encode_texts, load_siglip
            self.sigm, self.tokz = load_siglip(self.dev)
            self.enc_text = lambda t: encode_texts(self.sigm, self.tokz, [t], self.dev)[0]

    def reset(self, env, obs, lang, named, text):
        self.tid = self.r.task_id(lang)
        self.lang, self.text = lang, text
        if self.bb != "pg":
            S = self.m.text_dim
            self.has = torch.tensor([float(text != TX.NULL)], device=self.dev)
            self.sent = (torch.from_numpy(self.enc_text(text)).float().unsqueeze(0).to(self.dev) if text != TX.NULL
                         else torch.zeros(1, S, device=self.dev))
            self.instr = torch.from_numpy(self.enc_text(lang)).float().unsqueeze(0).to(self.dev)
        self.n_chunks, self.mag = 0, []

    def tokens(self, obs):
        ag = cv2.resize(obs["agentview_image"], (224, 224), interpolation=cv2.INTER_AREA)
        wr = cv2.resize(obs["robot0_eye_in_hand_image"], (224, 224), interpolation=cv2.INTER_AREA)
        if self.bb == "siglip":
            from cache_feats_v2 import encode_images
            f = encode_images(self.sigm, np.stack([ag, wr]), self.dev)
            tok = np.concatenate([f[:1], f[1:]], 1)
            return torch.from_numpy(tok).to(self.dev), torch.ones(tok.shape[:2], dtype=torch.bool, device=self.dev)
        o = self.fc(ag, wr, prompt(self.lang, self.text) if self.bb == "pg" else prompt(self.lang))
        if self.bb == "pg":
            tok = torch.from_numpy(np.asarray(o["pg"])).unsqueeze(0).to(self.dev)
            tm = torch.from_numpy(np.asarray(o["pg_mask"])).unsqueeze(0).to(self.dev)
            tm = torch.cat([torch.ones(1, tok.shape[1] - tm.shape[1], dtype=torch.bool, device=self.dev), tm], 1)
            return tok, tm
        tok = torch.from_numpy(np.asarray(o["sig"])).unsqueeze(0).to(self.dev)
        return tok, torch.ones(tok.shape[:2], dtype=torch.bool, device=self.dev)

    def act_chunk(self, env, obs):
        x, _ = self.r.chunk(obs, self.tid)
        A = self.r.unnorm(x.cpu().numpy())
        tok, tm = self.tokens(obs)
        T = lambda v: torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(self.dev)  # noqa: E731
        kw = dict(tokens=tok, tok_mask=tm, state_n=T((state8(obs) - self.st["st_m"]) / self.st["st_s"]),
                  base_n=T(((A.reshape(-1) - self.st["b_m"]) / self.st["b_s"]).reshape(A.shape)))
        if self.bb != "pg":
            kw.update(sent=self.sent, has_sent=self.has, instr=self.instr)
        d = self.m.sample(steps=self.fs, n=self.n_samp, **kw)[0].cpu().numpy() / float(self.st["scale"])
        A[:, :3] = np.clip(A[:, :3] + d, -1, 1)
        self.n_chunks += 1
        self.mag.append(float(np.abs(d[: self.k]).sum(0).max()))
        return A[: self.k]

    def episode_stats(self):
        return dict(n_chunks=self.n_chunks, mean_corr=round(float(np.mean(self.mag)), 4) if self.mag else 0.0,
                    max_corr=round(float(np.max(self.mag)), 4) if self.mag else 0.0)
