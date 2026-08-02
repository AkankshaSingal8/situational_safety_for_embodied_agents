"""Serve a pi0.5 checkpoint with in-denoising DCBF safety guidance.

Identical to serve_policy.py except the created policy is wrapped in
GuidedPolicy, which consumes an optional per-request `guidance` payload
(popped before input transforms). Requests without the payload behave as the
plain policy (correction scaled by enabled=0), so one server handles both
guided and baseline evaluation.
"""

import dataclasses
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.policies.guided_policy import GuidanceConfig, make_guided_policy
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    policy: Checkpoint
    default_prompt: str | None = None
    port: int = 8123
    # Guidance knobs (Tier-GT geometric defaults).
    gamma: float = 0.9
    d_safe: float = 0.01
    eef_radius: float = 0.09
    inflation_slope: float = 0.004
    translation_scale: float = 0.05
    # Per-step command clamp in COMMAND units. 1.0 = the [-1,1] command
    # convention (base pi05_libero). For checkpoints whose actions unnormalize
    # to METRIC deltas (LIBERO-Safety finetune: q99 ~ 0.02 m), serve with
    # translation_scale=1.0 cmd_clip=0.05 so the repair's actuation model stays
    # "max 5 cm per control step".
    cmd_clip: float = 1.0
    # Enforcement candidate A: >0 replaces in-denoising repair with K-seed
    # selection + gated noise-space gradient ascent of the certified margin.
    adjoint_steps: int = 0
    adjoint_lr: float = 0.05
    adjoint_tau: float = 0.02
    # Enforcement candidates B / C (0.0 = inert legacy path).
    fk_beta: float = 0.0
    repulsor_eta: float = 0.0
    # Candidate D: certificate-triggered renoise repair; mode 'reject' = the
    # equal-compute rejection-sampling kill-baseline.
    renoise_attempts: int = 0
    renoise_mode: str = "escalate"
    repair_schedule: str = "ramp"
    num_candidates: int = 1
    corridor_radius: float = 0.07
    corridor_relax: float = 0.6
    use_companions: bool = True
    dbnr_beta0: float = 0.0
    hdc_scale: float = 0.0
    sq_eps: float = 1.0
    # Revision knobs (spec 2026-07-27): A1 directed-progress and C tail-margin
    # terms in best-of-K selection. 0 = legacy score.
    progress_weight: float = 0.0
    lookahead_weight: float = 0.0
    tilt_lambda: float = 0.0
    # Consequence-steering selection layer (spec 2026-08-01, Task 3).
    # None (default) = off, byte-identical to the pre-existing selection.
    consequence_select: str | None = None
    consequence_n_members: int = 3
    consequence_pessimism: float = 1.0
    consequence_threshold: float = 0.10
    consequence_domain: str = "SL"
    consequence_action_scale: float = 1.0
    consequence_action_clip: float = 0.0


def main(args: Args) -> None:
    train_config = _config.get_config(args.policy.config)
    base_policy = _policy_config.create_trained_policy(
        train_config, args.policy.dir, default_prompt=args.default_prompt
    )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.policy.dir) / "assets", data_config.asset_id)
    if norm_stats is None:
        raise ValueError(f"No norm stats found under {args.policy.dir}/assets/{data_config.asset_id}")

    guidance_config = GuidanceConfig(
        gamma=args.gamma,
        d_safe=args.d_safe,
        eef_radius=args.eef_radius,
        inflation_slope=args.inflation_slope,
        translation_scale=args.translation_scale,
        cmd_clip=args.cmd_clip,
        adjoint_steps=args.adjoint_steps,
        adjoint_lr=args.adjoint_lr,
        adjoint_tau=args.adjoint_tau,
        fk_beta=args.fk_beta,
        repulsor_eta=args.repulsor_eta,
        renoise_attempts=args.renoise_attempts,
        renoise_mode=args.renoise_mode,
        repair_schedule=args.repair_schedule,
        num_candidates=args.num_candidates,
        corridor_radius=args.corridor_radius,
        corridor_relax=args.corridor_relax,
        use_companions=args.use_companions,
        dbnr_beta0=args.dbnr_beta0,
        hdc_scale=args.hdc_scale,
        sq_eps=args.sq_eps,
        progress_weight=args.progress_weight,
        lookahead_weight=args.lookahead_weight,
        tilt_lambda=args.tilt_lambda,
        consequence_select=args.consequence_select,
        consequence_n_members=args.consequence_n_members,
        consequence_pessimism=args.consequence_pessimism,
        consequence_threshold=args.consequence_threshold,
        consequence_domain=args.consequence_domain,
        consequence_action_scale=args.consequence_action_scale,
        consequence_action_clip=args.consequence_action_clip,
    )
    policy = make_guided_policy(base_policy, norm_stats, guidance_config)
    logging.info("Guided policy ready (gamma=%s, d_safe=%s)", args.gamma, args.d_safe)

    # Warm up JIT before opening the port so the first real request doesn't sit
    # behind a ~60-90s compile (which trips the client's websocket keepalive).
    from openpi.policies.libero_policy import make_libero_example

    example = make_libero_example()
    example["guidance"] = {"enabled": 0.0}
    t0 = __import__("time").monotonic()
    policy.infer(example)
    logging.info("Warmup infer done in %.1fs — opening port.", __import__("time").monotonic() - t0)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating guided server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=base_policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
