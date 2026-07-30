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
    cmd_clip: float = 1.0  # per-step command clamp (1.0 = [-1,1] command space;
    # 0.05 with translation_scale=1.0 for metric-delta checkpoints like the
    # LIBERO-Safety finetune, keeping the actuation model at 5 cm/step)
    default_obstacle_radius: float = 0.065
    num_denoise_steps: int = 10
    num_candidates: int = 1  # best-of-K noise seeds with executed-prefix selection
    # Enforcement candidate A (adjoint noise-space optimization). adjoint_steps
    # > 0 REPLACES the in-denoising repair path with K-seed selection + gated
    # gradient ascent of the certified prefix margin w.r.t. the initial noise.
    adjoint_steps: int = 0
    adjoint_lr: float = 0.05
    adjoint_tau: float = 0.02  # ascend only while margin < tau [m]
    # Enforcement candidates B and C (0.0 = inert, bit-exact legacy path).
    fk_beta: float = 0.0  # FK/SMC particle tilt strength
    repulsor_eta: float = 0.0  # soft repulsor strength [m/denoise-step]
    # Enforcement candidate D: certificate-triggered RENOISE repair (0 = off).
    # mode 'escalate' = SDEdit depth schedule (0.3/0.6/1.0) from the rejected
    # chunk; 'reject' = fresh full resampling each attempt — the equal-compute
    # rejection-sampling kill-baseline the novelty audit demands.
    renoise_attempts: int = 0
    renoise_mode: str = "escalate"
    corridor_radius: float = 0.07  # sanctioned-approach cylinder radius [m]
    corridor_relax: float = 0.6  # margin relaxation inside corridors (0 = off)
    use_companions: bool = True  # False = EEF-point-only (SOTA-reimpl fidelity arm)
    repair_schedule: str = "ramp"  # "ramp" (trust early, exact late) | "uniform" (r3 behavior)
    dbnr_beta0: float = 0.0  # DBNR restitution budget (0 = off; KILLED 2026-07-16, kept for ablation reproduction)
    hdc_scale: float = 0.0  # HDC lateral detour bias, meters per action step (0 = off)
    sq_eps: float = 1.0  # superquadric exponent for anisotropic obstacles (1 = ellipsoid)
    # Revision knobs (spec 2026-07-27, both 0 = bit-exact legacy selection):
    progress_weight: float = 0.0  # A1 directed-progress term in best-of-K selection
    lookahead_weight: float = 0.0  # C tail-margin (dead-end) term in best-of-K selection
    tilt_lambda: float = 0.0  # derived barrier-gradient guidance strength [m/denoise-step]


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
    if kind == "none":
        # Repair fully off: enforcement-pilot arms (FK tilt / repulsor /
        # select-only) run the unmodified flow; selection still applies.
        return np.zeros(n, dtype=np.float32), margin
    if kind == "posthoc":
        # Post-hoc shield baseline (AEGIS architecture class): the flow is
        # never guided — one exact projection sweep repairs the FINAL decoded
        # chunk only. Same barrier/criterion/policy as every other row; the
        # delta vs in-denoising schedules isolates the architecture.
        w = np.zeros(n, dtype=np.float32)
        w[-1] = 1.0
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

        if config.renoise_attempts > 0:
            from openpi.models.pi0_guided import flow_decode_actions
            self._flow_decode = nnx_utils.module_jit(
                types.MethodType(flow_decode_actions, self._model),
                static_argnames=("num_candidates", "prefix_len",
                                 "num_steps", "start_time"))
        if config.adjoint_steps > 0:
            from openpi.models.pi0_guided import adjoint_sample_actions
            bound = types.MethodType(adjoint_sample_actions, self._model)
            self._sample_actions = nnx_utils.module_jit(
                bound, static_argnames=("num_candidates", "prefix_len",
                                        "num_steps", "ascent_steps"))
            self._extra_sample_kwargs = {
                "ascent_steps": config.adjoint_steps,
                "ascent_lr": config.adjoint_lr,
                "ascent_tau": config.adjoint_tau,
            }
        else:
            bound = types.MethodType(guided_sample_actions, self._model)
            # Shape-determining kwargs must be static under jit (noise shape,
            # loop bounds, selection slice) — one compile per distinct K.
            self._sample_actions = nnx_utils.module_jit(
                bound, static_argnames=("num_candidates", "prefix_len", "num_steps")
            )
            self._extra_sample_kwargs = {}

    def _make_params(self, payload: dict | None) -> GuidanceParams:
        cfg = self._config
        if payload is None:
            payload = {}
        enabled = float(payload.get("enabled", 0.0))
        eef = np.asarray(payload.get("eef_pos", np.zeros(3)), dtype=np.float32)
        # Place the dummy obstacle far away when disabled so diagnostics stay finite.
        obs_pos = np.asarray(payload.get("obstacle_pos", np.array([10.0, 10.0, 10.0])), dtype=np.float32)
        r_obs = float(payload.get("obstacle_radius", cfg.default_obstacle_radius))
        # Anisotropic shape: semi-axes from the client; zeros = sphere mode.
        scales = np.asarray(payload.get("obstacle_scales", np.zeros(3)), dtype=np.float32)
        # Guard-set second obstacle (optional payload dict {pos, radius, scales});
        # absent -> FAR + zero scales = its barrier never wins the min.
        extra = payload.get("obstacle2") or None
        extra_pos = (np.asarray(extra["pos"], dtype=np.float32) if extra
                     else np.array([100.0, 100.0, 100.0], dtype=np.float32))
        extra_r = float(extra.get("radius", r_obs)) if extra else r_obs
        extra_scales = (np.asarray(extra.get("scales", np.zeros(3)), dtype=np.float32) if extra
                        else np.zeros(3, dtype=np.float32))
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
            cmd_clip=jnp.float32(cfg.cmd_clip),
            repair_weight=jnp.asarray(self._repair_weight),
            margin_scale=jnp.asarray(self._margin_scale),
            target_pos=jnp.asarray(np.asarray(payload.get("target_pos", FAR), dtype=np.float32)),
            dest_pos=jnp.asarray(np.asarray(payload.get("dest_pos", FAR), dtype=np.float32)),
            corridor_radius=jnp.float32(cfg.corridor_radius),
            corridor_relax=jnp.float32(cfg.corridor_relax),
            companion_scale=jnp.float32(1.0 if cfg.use_companions else 0.0),
            dbnr_beta0=jnp.float32(cfg.dbnr_beta0),
            obstacle_scales=jnp.asarray(scales),
            sq_eps=jnp.float32(cfg.sq_eps),
            r_obs_base=jnp.float32(r_obs),
            # hdc_scale / repulsor_eta accept per-request payload overrides so
            # client-side controllers (stall-recovery escalation, dual-adaptive
            # eta — spec 2026-07-27) can modulate them per replan while the
            # server stays stateless. Absent -> server config, legacy behavior.
            hdc_scale=jnp.float32(float(payload.get("hdc_scale", cfg.hdc_scale))),
            extra_obstacle_pos=jnp.asarray(extra_pos),
            extra_r_obs=jnp.float32(extra_r),
            extra_obstacle_scales=jnp.asarray(extra_scales),
            fk_beta=jnp.float32(cfg.fk_beta),
            repulsor_eta=jnp.float32(float(payload.get("repulsor_eta", cfg.repulsor_eta))),
            progress_weight=jnp.float32(cfg.progress_weight),
            lookahead_weight=jnp.float32(cfg.lookahead_weight),
            tilt_lambda=jnp.float32(float(payload.get("tilt_lambda", cfg.tilt_lambda))),
        )

    def _renoise_infer(self, rng, observation, guidance):
        """Candidate D host loop: K-seed decode -> if the certificate rejects,
        renoise the best chunk to escalating depth (or fresh-resample in
        'reject' mode) and re-decode, keeping best-so-far (anytime)."""
        cfg = self._config
        x0, d = self._flow_decode(rng, observation, guidance=guidance,
                                  num_candidates=cfg.num_candidates,
                                  start_time=1.0)
        m = np.asarray(d["min_margin"])
        best = int(np.argmax(m))
        chunk, margin = x0[best], float(m[best])
        depths = (0.3, 0.6, 1.0)
        attempts = 0
        for a in range(cfg.renoise_attempts):
            if margin >= 0.0:
                break
            attempts += 1
            rng, sub = jax.random.split(rng)
            if cfg.renoise_mode == "reject":
                x1, d1 = self._flow_decode(sub, observation, guidance=guidance,
                                           num_candidates=1, start_time=1.0)
            else:
                t_star = depths[min(a, len(depths) - 1)]
                x1, d1 = self._flow_decode(sub, observation, guidance=guidance,
                                           num_candidates=1, start_time=t_star,
                                           start_chunk=chunk)
            m1 = float(np.asarray(d1["min_margin"])[0])
            if m1 > margin:
                chunk, margin = x1[0], m1
        diag = {"selected_margin": np.asarray([margin], dtype=np.float32),
                "renoise_attempts": np.asarray([float(attempts)], dtype=np.float32),
                "feasible_count": np.asarray([float(margin >= 0.0)], dtype=np.float32)}
        return chunk[None], diag

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        obs = dict(obs)
        payload = obs.get("guidance", None)
        guidance = self._make_params(obs.pop("guidance", None))
        # ROCS release barrier (semantic-outcome steering, LIBERO-Safety):
        # active only when the client supplies a forbidden release region.
        rocs_center = None
        rocs_radius = 0.0
        rocs_eef = None
        if isinstance(payload, dict) and "forbidden_center" in payload:
            rocs_center = np.asarray(payload["forbidden_center"], dtype=np.float32)
            rocs_radius = float(payload.get("forbidden_radius", 0.10))
            rocs_eef = np.asarray(payload.get("eef_pos", np.zeros(3)), dtype=np.float32)

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
        if self._config.renoise_attempts > 0:
            actions, diag = self._renoise_infer(sample_rng, observation, guidance)
        else:
            actions, diag = self._sample_actions(
                sample_rng, observation, guidance=guidance,
                num_candidates=self._config.num_candidates,
                **self._extra_sample_kwargs, **sample_kwargs,
            )
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["guidance"] = {
            k: (np.asarray(v).tolist() if k.startswith("cand_") else float(np.asarray(v)[0]))
            for k, v in diag.items()
        }
        if rocs_center is not None and rocs_radius > 0.0:
            from openpi.models import rocs as _rocs

            chunk = jnp.asarray(outputs["actions"], dtype=jnp.float32)[None, ...]
            chunk_new, rdiag = _rocs.delay_release(
                chunk, jnp.asarray(rocs_eef), self._config.translation_scale,
                jnp.asarray(rocs_center), jnp.float32(rocs_radius), jnp.float32(1.0),
            )
            outputs["actions"] = np.asarray(chunk_new[0])
            outputs["guidance"].update(
                {k: float(np.asarray(v)[0]) for k, v in rdiag.items()}
            )
        return outputs


def make_guided_policy(base: _policy.Policy, norm_stats: dict[str, _transforms.NormStats], config: GuidanceConfig):
    if "actions" not in norm_stats:
        raise KeyError(f"'actions' missing from norm stats (keys: {list(norm_stats)})")
    return GuidedPolicy(base, norm_stats["actions"], config)
