"""Policy wrapper that routes a per-request `guidance` payload into the
in-denoising DCBF sampler (`openpi.models.pi0_guided`).

The payload is popped from the raw client dict BEFORE input transforms run, so
the LiberoInputs key whitelist never sees it and the normal observation path
is untouched. Metric quantities (EEF pos, obstacle pos/radius) come straight
from the simulator on the client side; only the action quantile stats — taken
from the checkpoint's own norm stats — are needed server-side to convert
between normalized actions and metric displacements.
"""

import dataclasses
import logging
import time
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_guided import GuidanceParams, guided_sample_actions
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils


@dataclasses.dataclass(frozen=True)
class GuidanceConfig:
    gamma: float = 0.9  # DCBF decay (SOTA setting)
    d_safe: float = 0.01  # extra clearance margin [m]
    eef_radius: float = 0.09  # EEF+fingers+carried-object sphere approximation [m]
    inflation_slope: float = 0.004  # extra margin per horizon step [m]
    translation_scale: float = 0.05  # metres per unit command per control step
    default_obstacle_radius: float = 0.065
    num_denoise_steps: int = 10
    num_candidates: int = 1  # best-of-K noise seeds with executed-prefix selection
    corridor_radius: float = 0.07  # sanctioned-approach cylinder radius [m]
    corridor_relax: float = 0.6  # margin relaxation inside corridors (0 = off)
    use_companions: bool = True  # False = EEF-point-only (SOTA-reimpl fidelity arm)
    repair_schedule: str = "ramp"  # "ramp" (trust early, exact late) | "uniform" (r3 behavior)
    dbnr_beta0: float = 0.0  # DBNR restitution budget (0 = off; kill-test greenlit, see PROGRESS_LOG Phase 6)


def make_schedule(kind: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """repair_weight[k], margin_scale[k] over denoising steps. Last entry must be 1."""
    margin = np.ones(n, dtype=np.float32)
    if kind == "uniform":
        return np.ones(n, dtype=np.float32), margin
    if kind == "ramp":
        w = np.ones(n, dtype=np.float32)
        w[:2] = 0.0
        ramp = np.linspace(0.2, 1.0, num=5, dtype=np.float32)  # steps 2..6
        w[2 : 2 + len(ramp)] = ramp
        return w, margin
    raise ValueError(f"unknown repair_schedule {kind!r}")


FAR = np.array([100.0, 100.0, 100.0])  # disables a corridor segment


class GuidedPolicy(_policy.Policy):
    """Wraps a JAX openpi Policy; adds DCBF guidance inside denoising."""

    def __init__(self, base: _policy.Policy, action_stats, config: GuidanceConfig):
        if base._is_pytorch_model:  # noqa: SLF001
            raise ValueError("GuidedPolicy supports the JAX model path only.")
        # Steal the wrapped policy's members instead of re-loading the model.
        self._model = base._model  # noqa: SLF001
        self._input_transform = base._input_transform  # noqa: SLF001
        self._output_transform = base._output_transform  # noqa: SLF001
        self._sample_kwargs = base._sample_kwargs  # noqa: SLF001
        self._metadata = base._metadata  # noqa: SLF001
        self._is_pytorch_model = False
        self._pytorch_device = None
        self._rng = base._rng  # noqa: SLF001

        self._config = config
        if action_stats.q01 is None or action_stats.q99 is None:
            raise ValueError("Action norm stats must include quantiles (pi0.5 uses quantile normalization).")
        self._q01 = np.asarray(action_stats.q01[:3], dtype=np.float32)
        self._q99 = np.asarray(action_stats.q99[:3], dtype=np.float32)
        logging.info("Guidance action quantiles: q01=%s q99=%s", self._q01, self._q99)

        w, m = make_schedule(config.repair_schedule, config.num_denoise_steps)
        self._repair_weight, self._margin_scale = w, m
        logging.info("Repair schedule %s: w=%s", config.repair_schedule, np.round(w, 2))

        bound = types.MethodType(guided_sample_actions, self._model)
        # Shape-determining kwargs must be static under jit (noise shape,
        # loop bounds, selection slice) — one compile per distinct K.
        self._sample_actions = nnx_utils.module_jit(
            bound, static_argnames=("num_candidates", "prefix_len", "num_steps")
        )

    def _make_params(self, payload: dict | None) -> GuidanceParams:
        cfg = self._config
        if payload is None:
            payload = {}
        enabled = float(payload.get("enabled", 0.0))
        eef = np.asarray(payload.get("eef_pos", np.zeros(3)), dtype=np.float32)
        # Place the dummy obstacle far away when disabled so diagnostics stay finite.
        obs_pos = np.asarray(payload.get("obstacle_pos", np.array([10.0, 10.0, 10.0])), dtype=np.float32)
        r_obs = float(payload.get("obstacle_radius", cfg.default_obstacle_radius))
        return GuidanceParams(
            enabled=jnp.float32(enabled),
            eef_pos=jnp.asarray(eef),
            obstacle_pos=jnp.asarray(obs_pos),
            r_eff=jnp.float32(r_obs + cfg.eef_radius + cfg.d_safe),
            inflation_slope=jnp.float32(cfg.inflation_slope),
            gamma=jnp.float32(cfg.gamma),
            q01=jnp.asarray(self._q01),
            q99=jnp.asarray(self._q99),
            translation_scale=jnp.float32(cfg.translation_scale),
            repair_weight=jnp.asarray(self._repair_weight),
            margin_scale=jnp.asarray(self._margin_scale),
            target_pos=jnp.asarray(np.asarray(payload.get("target_pos", FAR), dtype=np.float32)),
            dest_pos=jnp.asarray(np.asarray(payload.get("dest_pos", FAR), dtype=np.float32)),
            corridor_radius=jnp.float32(cfg.corridor_radius),
            corridor_relax=jnp.float32(cfg.corridor_relax),
            companion_scale=jnp.float32(1.0 if cfg.use_companions else 0.0),
            dbnr_beta0=jnp.float32(cfg.dbnr_beta0),
        )

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        obs = dict(obs)
        guidance = self._make_params(obs.pop("guidance", None))

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs: dict[str, Any] = dict(self._sample_kwargs)
        if noise is not None:
            n = jnp.asarray(noise)
            sample_kwargs["noise"] = n[None, ...] if n.ndim == 2 else n

        observation = _model.Observation.from_dict(inputs)
        start = time.monotonic()
        actions, diag = self._sample_actions(
            sample_rng, observation, guidance=guidance,
            num_candidates=self._config.num_candidates, **sample_kwargs,
        )
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["guidance"] = {k: float(np.asarray(v)[0]) for k, v in diag.items()}
        return outputs


def make_guided_policy(base: _policy.Policy, norm_stats: dict[str, _transforms.NormStats], config: GuidanceConfig):
    if "actions" not in norm_stats:
        raise KeyError(f"'actions' missing from norm stats (keys: {list(norm_stats)})")
    return GuidedPolicy(base, norm_stats["actions"], config)
