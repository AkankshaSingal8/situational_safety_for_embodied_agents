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
MOTION = 8      # --motion: + state change since the previous chunk (8), appended to cond


def build(ckpt=CKPT, seed=0, load_bf16=False, lora=False, cond_dim=COND):
    """Returns (graphdef, frozen_state, train_state) with the pi05_base weights loaded.
    lora: PaliGemma gets rank-16 LoRA adapters (openpi gemma_2b_lora) on attention and MLP; the adapters are
    trained, the PaliGemma weights stay frozen. lora_b starts at zero, so step 0 matches the frozen backbone."""
    import dataclasses
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as _model
    from openpi.training import config as _config

    mcfg = dataclasses.replace(_config.get_config("pi05_libero").model, action_horizon=H, max_token_len=MAX_TXT,
                               **({"paligemma_variant": "gemma_2b_lora"} if lora else {}))
    # load_bf16: inference server; the trained weights come from the residual checkpoint, so the whole base can load
    # in bf16 (6.6 GB instead of 13 GB in fp32, which ran out of memory next to the pi0.5 server).
    params = _model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16 if load_bf16 else jnp.float32)
    if lora:
        base = _load_with_lora(mcfg, params, seed)
    else:
        base = mcfg.load(params)

    class ResExpert(nnx.Module):
        def __init__(self, base, rngs):
            self.base = base
            w = base.action_in_proj.out_features
            self.cond_proj = nnx.Linear(cond_dim, w, kernel_init=nnx.initializers.zeros, rngs=rngs)

    model = ResExpert(base, nnx.Rngs(seed))
    frozen = nnx.All(nnx.Param, nnx_path_regex(r".*PaliGemma.*"), nnx.Not(nnx_path_regex(r".*llm.*_1.*")),
                     nnx.Not(nnx_path_regex(r".*lora.*")))
    graphdef, frozen_state, train_state, rest = nnx.split(model, frozen, nnx.Param, ...)
    frozen_state = jax.tree.map(lambda x: x.astype(jnp.bfloat16), frozen_state)
    return graphdef, frozen_state, train_state, rest


def _load_with_lora(mcfg, params, seed):
    """mcfg.load for a LoRA variant: pretrained weights where they exist, new adapters lora_a ~ N(0, 0.01),
    lora_b = 0 (the pretrained checkpoint has no adapter weights, so openpi's load would reject the tree)."""
    import jax
    import jax.numpy as jnp
    from flax import nnx, traverse_util
    model = nnx.eval_shape(mcfg.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    want = traverse_util.flatten_dict(state.to_pure_dict())
    have = traverse_util.flatten_dict(params)
    key = jax.random.key(seed + 17)
    out, n_new = {}, 0
    for k, v in want.items():
        if k in have:
            out[k] = have[k]
        elif "lora_a" in "/".join(map(str, k)):
            key, sk = jax.random.split(key)
            out[k] = 0.01 * jax.random.normal(sk, v.shape, jnp.float32); n_new += 1
        elif "lora_b" in "/".join(map(str, k)):
            out[k] = jnp.zeros(v.shape, jnp.float32); n_new += 1
        else:
            raise KeyError(f"missing pretrained weight {k}")
    print(f"lora: {n_new} new adapter tensors", flush=True)
    state.replace_by_pure_dict(traverse_util.unflatten_dict(out))
    return nnx.merge(graphdef, state)


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
    ta, tw = jax.lax.stop_gradient(ta), jax.lax.stop_gradient(tw)
    te = pg.llm(toks, method="embed")
    B = ta.shape[0]
    x = jnp.concatenate([ta, tw, te], 1)
    mask = jnp.concatenate([jnp.ones((B, ta.shape[1] + tw.shape[1]), bool), tmask], 1)
    attn = _pi0.make_attn_mask(mask, jnp.zeros(mask.shape[1], bool))
    pos = jnp.cumsum(mask, axis=1) - 1
    _, kv = pg.llm([x, None], mask=attn, positions=pos)
    # no stop_gradient: without LoRA the prefix depends only on frozen weights, so no gradient reaches it anyway;
    # with LoRA the adapters are trained through it. SigLIP has no adapters, so its tokens are cut off here.
    return kv, mask


def velocity(m, kv, pmask, x_t, t, cond):
    """x_t (B,H,3) noisy residual, t (B,), cond (B,H,COND) -> v (B,H,3)."""
    return suffix_out(m, kv, pmask, x_t, t, cond)[..., : x_t.shape[-1]]


GATE_DIM = 3    # action_out_proj output dim used as the correction-needed logit (dims 0-2 are the xyz velocity)


def gate_logit(m, kv, pmask, cond):
    """Correction-needed logit (B,): one suffix pass at t = 1 with a zero residual, output dim GATE_DIM averaged over
    the H tokens. Uses no new parameters, so older checkpoints load unchanged (their gate is untrained)."""
    import jax.numpy as jnp
    B, Hh = cond.shape[:2]
    out = suffix_out(m, kv, pmask, jnp.zeros((B, Hh, 3)), jnp.ones((B,)), cond)
    return out[..., GATE_DIM].mean(-1)


def suffix_out(m, kv, pmask, x_t, t, cond):
    """Action expert pass over the H residual tokens -> action_out_proj output (B,H,32)."""
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
    return b.action_out_proj(out[:, -Hh:])


def flow_loss(m, rng, agent, wrist, toks, tmask, cond, target, n_t=2, gate_w=0.0, flow_w=1.0, gate_balance=False):
    """flow_w * mean squared velocity error (n_t flow times per prefix; the prefix pass is shared)
    + gate_w * BCE(gate_logit, target is non-zero). flow_w = 0 trains the gate alone (a separate gate checkpoint
    for res_server --gate_ckpt). gate_balance weights positives and negatives equally within the batch."""
    import jax
    import jax.numpy as jnp
    kv, pmask = prefix_kv(m, agent, wrist, toks, tmask)
    loss = 0.0
    if flow_w:
        losses = []
        for k in jax.random.split(rng, n_t):
            k1, k2 = jax.random.split(k)
            noise = jax.random.normal(k1, target.shape)
            t = jax.random.beta(k2, 1.5, 1, target.shape[:1]) * 0.999 + 0.001
            x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * target
            v = velocity(m, kv, pmask, x_t, t, cond)
            losses.append(jnp.mean(jnp.square(v - (noise - target))))
        loss = flow_w * sum(losses) / n_t
    if gate_w:
        y = (jnp.abs(target).sum((1, 2)) > 1e-3).astype(jnp.float32)
        z = gate_logit(m, kv, pmask, cond)
        bce = jax.nn.softplus(z) - y * z
        if gate_balance:
            pos = jnp.maximum(y.sum(), 1.0); neg = jnp.maximum((1 - y).sum(), 1.0)
            loss = loss + gate_w * 0.5 * ((bce * y).sum() / pos + (bce * (1 - y)).sum() / neg)
        else:
            loss = loss + gate_w * jnp.mean(bce)
    return loss


def sample(m, rng, agent, wrist, toks, tmask, cond, steps=10, n=1, with_gate=False):
    """Euler sampling from t = 1 (noise) to 0; mean of n samples. Returns (B,H,3) normalised residual
    (and the gate probability (B,) when with_gate)."""
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
    if with_gate:
        return sum(outs) / n, jax.nn.sigmoid(gate_logit(m, kv, pmask, cond))
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
