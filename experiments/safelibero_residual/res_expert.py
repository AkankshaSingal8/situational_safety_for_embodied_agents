"""Residual action expert: pi05_base's action expert fine-tuned to predict the safety residual (openpi env, JAX).

    prefix  [agent 256, wrist 256, prompt tokens]  -> frozen pi05_base PaliGemma (SigLIP + Gemma-2B), KV cache
    suffix  H action tokens, token h = action_in_proj(noisy residual_h padded to 32 dims) + cond_proj([base_h, state])
            -> pi05_base action expert (Gemma-300M, adaRMS flow time), attends to the prefix KV cache
    output  action_out_proj(...)[..., :3] = flow velocity of the xyz residual (H, 3)

Everything comes from the pi05_base checkpoint (not pi05_libero). cond_proj is new and zero-initialised, so at
step 0 the expert sees exactly the inputs it was pre-trained on. PaliGemma is frozen; the expert, its in/out
projections, the time MLP and cond_proj are trained. Flow convention as openpi: x_t = t * noise + (1 - t) * x,
u = noise - x, sampled from t = 1 to t = 0.

Images: (B,224,224,3) uint8 in raw env orientation (as collect_v2 stores them); flipped 180 deg here, as pi0.5
sees them (same path as pg_backbone.py). Prompt "<task instruction>. <safety sentence>" or the instruction alone.
"""
import os
import pathlib

import numpy as np

CKPT = os.environ.get("PG_CKPT", "/data/asingal2/openpi_cache/openpi-assets/checkpoints/pi05_base")
MAX_TXT = 96    # pi0.5 prompt format: "Task: <prompt>, State: <8 discretised values>;\nAction: "
H = 10          # pi05_libero chunk length; the residual corrects the whole chunk, the first 5 steps are executed
COND = 7 + 8    # base action (7) + state (8) per action token


def build(ckpt=CKPT, seed=0, load_bf16=False):
    """Returns (graphdef, frozen_state, train_state) with the pi05_base weights loaded."""
    import dataclasses
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as _model
    from openpi.training import config as _config

    mcfg = dataclasses.replace(_config.get_config("pi05_libero").model, action_horizon=H, max_token_len=MAX_TXT)
    # load_bf16: inference server; the trained weights come from the residual checkpoint, so the whole base can load
    # in bf16 (6.6 GB instead of 13 GB in fp32, which ran out of memory next to the pi0.5 server).
    params = _model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16 if load_bf16 else jnp.float32)
    base = mcfg.load(params)

    class ResExpert(nnx.Module):
        def __init__(self, base, rngs):
            self.base = base
            w = base.action_in_proj.out_features
            self.cond_proj = nnx.Linear(COND, w, kernel_init=nnx.initializers.zeros, rngs=rngs)

    model = ResExpert(base, nnx.Rngs(seed))
    frozen = nnx.All(nnx.Param, nnx_path_regex(r".*PaliGemma.*"), nnx.Not(nnx_path_regex(r".*llm.*_1.*")))
    graphdef, frozen_state, train_state, rest = nnx.split(model, frozen, nnx.Param, ...)
    frozen_state = jax.tree.map(lambda x: x.astype(jnp.bfloat16), frozen_state)
    return graphdef, frozen_state, train_state, rest


def nnx_path_regex(p):
    from openpi.shared import nnx_utils
    return nnx_utils.PathRegex(p)


def prefix_kv(m, agent, wrist, toks, tmask):
    """Frozen prefix pass -> (kv_cache, prefix_mask)."""
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0 as _pi0
    pg = m.base.PaliGemma
    ia = agent[:, ::-1, ::-1].astype(jnp.float32) / 127.5 - 1.0
    iw = wrist[:, ::-1, ::-1].astype(jnp.float32) / 127.5 - 1.0
    ta, _ = pg.img(ia, train=False)
    tw, _ = pg.img(iw, train=False)
    te = pg.llm(toks, method="embed")
    B = ta.shape[0]
    x = jnp.concatenate([ta, tw, te], 1)
    mask = jnp.concatenate([jnp.ones((B, ta.shape[1] + tw.shape[1]), bool), tmask], 1)
    attn = _pi0.make_attn_mask(mask, jnp.zeros(mask.shape[1], bool))
    pos = jnp.cumsum(mask, axis=1) - 1
    _, kv = pg.llm([x, None], mask=attn, positions=pos)
    return jax.lax.stop_gradient(kv), mask


def velocity(m, kv, pmask, x_t, t, cond):
    """x_t (B,H,3) noisy residual, t (B,), cond (B,H,COND) -> v (B,H,3)."""
    import einops
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import pi0 as _pi0
    b = m.base
    B, Hh, _ = x_t.shape
    xa = jnp.pad(x_t, ((0, 0), (0, 0), (0, b.action_in_proj.in_features - x_t.shape[-1])))
    tok = b.action_in_proj(xa) + m.cond_proj(cond)
    te = _pi0.posemb_sincos(t, b.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
    te = nnx.swish(b.time_mlp_out(nnx.swish(b.time_mlp_in(te))))
    smask = jnp.ones((B, Hh), bool)
    sar = jnp.array([True] + [False] * (Hh - 1))
    full = jnp.concatenate([einops.repeat(pmask, "b p -> b s p", s=Hh), _pi0.make_attn_mask(smask, sar)], -1)
    pos = jnp.sum(pmask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
    (_, out), _ = b.PaliGemma.llm([None, tok], mask=full, positions=pos, kv_cache=kv, adarms_cond=[None, te])
    return b.action_out_proj(out[:, -Hh:])[..., : x_t.shape[-1]]


def flow_loss(m, rng, agent, wrist, toks, tmask, cond, target, n_t=2):
    """Mean squared velocity error; n_t flow times per prefix (the prefix pass is shared)."""
    import jax
    import jax.numpy as jnp
    kv, pmask = prefix_kv(m, agent, wrist, toks, tmask)
    losses = []
    for k in jax.random.split(rng, n_t):
        k1, k2 = jax.random.split(k)
        noise = jax.random.normal(k1, target.shape)
        t = jax.random.beta(k2, 1.5, 1, target.shape[:1]) * 0.999 + 0.001
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * target
        v = velocity(m, kv, pmask, x_t, t, cond)
        losses.append(jnp.mean(jnp.square(v - (noise - target))))
    return sum(losses) / n_t


def sample(m, rng, agent, wrist, toks, tmask, cond, steps=10, n=1):
    """Euler sampling from t = 1 (noise) to 0; mean of n samples. Returns (B,H,3) normalised residual."""
    import jax
    import jax.numpy as jnp
    kv, pmask = prefix_kv(m, agent, wrist, toks, tmask)
    B = agent.shape[0]
    outs = []
    for k in jax.random.split(rng, n):
        x = jax.random.normal(k, (B, H, 3))
        for i in range(steps):
            t = jnp.full((B,), 1.0 - i / steps)
            x = x - velocity(m, kv, pmask, x, t, cond) / steps
        outs.append(x)
    return sum(outs) / n


class Tokenizer:
    def __init__(self):
        from openpi.models import tokenizer as _tok
        self.tok = _tok.PaligemmaTokenizer(max_len=MAX_TXT)

    def __call__(self, prompts, states):
        """states: (B,8) standardised state, mapped to [-1, 1] by /3 and clipped (pi0.5 discretises it into the prompt)."""
        st = np.clip(np.asarray(states, np.float32) / 3.0, -1, 1)
        T, M = zip(*(self.tok.tokenize(p, s) for p, s in zip(prompts, st)))
        return np.stack(T).astype(np.int32), np.stack(M).astype(bool)


def prompt(lang, sentence=""):
    lang = lang.strip().rstrip(".")
    return f"{lang}. {sentence.strip()}" if sentence else lang
