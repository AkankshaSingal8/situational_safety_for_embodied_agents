"""Frozen PaliGemma backbone of pi0.5 (pi05_base checkpoint, JAX/openpi) as a feature extractor.

Runs in the openpi env (/data/asingal2/envs/openpi), not in slres. Used by
  pg_feats.py   cache features for the residual training data
  pg_server.py  serve the same features to the eval controller over a websocket

Input per sample: agent-view and wrist frames (224x224 uint8, raw env orientation; flipped 180 deg here, as pi0.5
sees them) and one prompt string "<task instruction>. <safety sentence>" (or just the instruction).
The prefix [agent 256 tokens, wrist 256 tokens, text tokens] goes through Gemma-2B with full attention, exactly as
pi0.5 builds its prefix. Outputs, both projected with a fixed PCA (fit once, pca.npz):
  pg   (128, K)   Gemma last-layer outputs: agent tokens pooled 16x16 -> 8x8 (64), wrist 16x16 -> 4x4 (16),
                  text tokens (48, padded; mask in pg_mask)            -> image and sentence fused (backbone C)
  sig  (80, K)    SigLIP-So400m tokens before Gemma, same pooling        -> image only (backbone B)
"""
import os
import pathlib

import numpy as np

CKPT = os.environ.get("PG_CKPT", "/data/asingal2/openpi_cache/openpi-assets/checkpoints/pi05_base")
MAX_TXT = 48
N_IMG = 64 + 16


def prompt(lang, sentence=""):
    lang = lang.strip().rstrip(".")
    return f"{lang}. {sentence.strip()}" if sentence else lang


class Backbone:
    def __init__(self, ckpt=CKPT, pca=None):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as _model
        from openpi.models import pi0 as _pi0
        from openpi.models import tokenizer as _tok
        from openpi.training import config as _config

        cfg = _config.get_config("pi05_libero")
        params = _model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16)
        model = cfg.model.load(params)
        self.tok = _tok.PaligemmaTokenizer(max_len=MAX_TXT)
        graphdef, state = nnx.split(model)
        self.pca = None
        if pca is not None:
            self.set_pca(pca)

        def pool(x, g):  # (B,256,D) -> (B,g*g,D)
            B, _, D = x.shape
            k = 16 // g
            return x.reshape(B, g, k, g, k, D).mean((2, 4)).reshape(B, g * g, D)

        def fwd(state, img_a, img_w, toks, tmask):
            m = nnx.merge(graphdef, state)
            ia = img_a.astype(jnp.float32) / 127.5 - 1.0
            iw = img_w.astype(jnp.float32) / 127.5 - 1.0
            ta, _ = m.PaliGemma.img(ia, train=False)
            tw, _ = m.PaliGemma.img(iw, train=False)
            te = m.PaliGemma.llm(toks, method="embed")
            B = ta.shape[0]
            x = jnp.concatenate([ta, tw, te], 1)
            mask = jnp.concatenate([jnp.ones((B, 512), bool), tmask], 1)
            attn = _pi0.make_attn_mask(mask, jnp.zeros(mask.shape[1], bool))
            pos = jnp.cumsum(mask, axis=1) - 1
            (out, _), _ = m.PaliGemma.llm([x, None], mask=attn, positions=pos)
            pg = jnp.concatenate([pool(out[:, :256], 8), pool(out[:, 256:512], 4), out[:, 512:]], 1)
            sig = jnp.concatenate([pool(ta, 8), pool(tw, 4)], 1)
            return pg.astype(jnp.float32), sig.astype(jnp.float32)

        self._fwd = jax.jit(fwd)
        self._state = state
        self.jnp = jnp

    def set_pca(self, path):
        z = np.load(path)
        self.pca = {k: z[k].astype(np.float32) for k in ("pg_mean", "pg_proj", "sig_mean", "sig_proj")}

    def tokenize(self, prompts):
        T, M = zip(*(self.tok.tokenize(p) for p in prompts))
        return np.stack(T).astype(np.int32), np.stack(M).astype(bool)

    def raw(self, agent, wrist, prompts):
        """agent, wrist (B,224,224,3) uint8 raw env orientation. Returns pg (B,128,2048), sig (B,80,1152), mask (B,48)."""
        a = np.ascontiguousarray(agent[:, ::-1, ::-1])
        w = np.ascontiguousarray(wrist[:, ::-1, ::-1])
        toks, tmask = self.tokenize(prompts)
        pg, sig = self._fwd(self._state, a, w, toks, tmask)
        return np.asarray(pg), np.asarray(sig), tmask

    def __call__(self, agent, wrist, prompts):
        """PCA-projected fp16 features: pg (B,128,K), sig (B,80,K), mask (B,48)."""
        pg, sig, tmask = self.raw(agent, wrist, prompts)
        p = self.pca
        pg = (pg - p["pg_mean"]) @ p["pg_proj"]
        sig = (sig - p["sig_mean"]) @ p["sig_proj"]
        return pg.astype(np.float16), sig.astype(np.float16), tmask
