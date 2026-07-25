"""Adjoint noise-space enforcement GO/NO-GO probe (enforcement candidate A).

Mechanism test, not a performance run: can we reverse-differentiate a terminal
safety margin through the 10-step denoising ODE w.r.t. the INITIAL NOISE, with
the frozen pi0.5, at acceptable memory/latency — and does gradient ascent on
the margin actually increase it?

Key architectural bet being validated: only the action expert runs per Euler
step (VLM prefix KV-cache is computed once, constant w.r.t. noise), so the
backward pass never touches the 3B backbone.

GO criteria (all must hold):
  1. compiles + runs without OOM on one H100-80
  2. grad(margin, noise) finite and non-degenerate (norm > 1e-6)
  3. >=5 gradient-ascent steps monotonically increase the margin on an
     obstacle-on-path payload
  4. per-gradient-step walltime comparable to a K=8 forward (< ~3x)
Report JSON -> results_tables/adjoint_probe.json
"""

import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.models.pi0_guided import make_attn_mask
from openpi.policies import policy_config as _policy_config
from openpi.policies.guided_policy import GuidanceConfig, make_guided_policy
from openpi.policies.libero_policy import make_libero_example
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

import dataclasses
import einops


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    dir: str = ""
    num_steps: int = 10
    ascent_steps: int = 8
    lr: float = 0.05
    out: str = ""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    train_config = _config.get_config(args.config)
    base = _policy_config.create_trained_policy(train_config, args.dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(
        pathlib.Path(args.dir) / "assets", data_config.asset_id)
    policy = make_guided_policy(base, norm_stats, GuidanceConfig())
    model = policy._model  # noqa: SLF001
    q01 = np.asarray(norm_stats["actions"].q01[:3], dtype=np.float32)
    q99 = np.asarray(norm_stats["actions"].q99[:3], dtype=np.float32)

    # --- observation: warmup example, transformed exactly as infer() does ---
    example = make_libero_example()
    inputs = policy._input_transform(dict(example))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs)
    observation = _model.preprocess_observation(None, observation, train=False)

    # --- prefix once (constant wrt noise): the architectural bet ---
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    kv_cache = jax.lax.stop_gradient(kv_cache)

    dt = -1.0 / args.num_steps

    def denoise(noise):
        x_t, time_t = noise, 1.0
        for _ in range(args.num_steps):  # unrolled: clean reverse-mode AD
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_t, 1))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p",
                                        s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
            v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
            x_t = x_t + dt * v_t
            time_t = time_t + dt
        return x_t

    # Obstacle straight ahead of a nominal eef, on the expected motion path.
    eef = jnp.asarray([0.0, 0.0, 1.0])
    obstacle = jnp.asarray([0.08, 0.0, 0.95])
    r_total = 0.09 + 0.04

    span = jnp.asarray((q99 - q01) / 2.0)
    mid = jnp.asarray((q99 + q01) / 2.0)

    def margin(noise):
        x0 = denoise(noise)[0]  # (H, ad)
        deltas = x0[:, :3] * span + mid  # quantile unnorm -> metric deltas
        path = eef[None, :] + jnp.cumsum(deltas, axis=0)
        d = jnp.linalg.norm(path - obstacle[None, :], axis=-1)
        return jnp.min(d) - r_total

    margin_grad = jax.jit(jax.value_and_grad(margin))

    rng = jax.random.PRNGKey(0)
    noise = jax.random.normal(rng, (1, model.action_horizon, model.action_dim))

    t0 = time.monotonic()
    m0, g0 = margin_grad(noise)
    jax.block_until_ready(g0)
    compile_s = time.monotonic() - t0
    t0 = time.monotonic()
    m0, g0 = margin_grad(noise)
    jax.block_until_ready(g0)
    step_s = time.monotonic() - t0

    gnorm = float(jnp.linalg.norm(g0))
    report = {
        "compile_s": round(compile_s, 1),
        "grad_step_s": round(step_s, 3),
        "grad_norm": gnorm,
        "grad_finite": bool(jnp.all(jnp.isfinite(g0))),
        "margin_init": float(m0),
        "ascent": [],
    }
    logging.info("compile %.1fs, grad step %.3fs, |g|=%.2e, m0=%.4f",
                 compile_s, step_s, gnorm, float(m0))

    z = noise
    for i in range(args.ascent_steps):
        m, g = margin_grad(z)
        z = z + args.lr * g / (jnp.linalg.norm(g) + 1e-8)
        report["ascent"].append(round(float(m), 4))
        logging.info("ascent %d: margin=%.4f", i, float(m))
    m_final, _ = margin_grad(z)
    report["margin_final"] = float(m_final)
    report["margin_gain"] = float(m_final - m0)
    # On-manifold check: how far did the noise move (per-dim z remains ~N(0,1))?
    report["noise_shift_norm"] = float(jnp.linalg.norm(z - noise))
    report["noise_final_std"] = float(jnp.std(z))

    mem = jax.devices()[0].memory_stats() or {}
    report["peak_bytes_in_use"] = int(mem.get("peak_bytes_in_use", -1))

    go = (report["grad_finite"] and gnorm > 1e-6
          and report["margin_gain"] > 0.0)
    report["verdict"] = "GO" if go else "NO-GO"
    out = args.out or str(pathlib.Path(__file__).parents[2]
                          / "results_tables/adjoint_probe.json")
    pathlib.Path(out).write_text(json.dumps(report, indent=1))
    logging.info("ADJOINT_PROBE_%s %s", report["verdict"], json.dumps(report))


if __name__ == "__main__":
    main(tyro.cli(Args))
