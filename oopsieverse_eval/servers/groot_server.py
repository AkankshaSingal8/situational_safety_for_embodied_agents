"""ZMQ policy server wrapping NVIDIA's off-the-shelf `nvidia/GR00T-N1.6-3B`
checkpoint, using the `ROBOCASA_PANDA_OMRON` embodiment tag that ships
pretrained (not finetune-only) in that checkpoint's own `statistics.json` /
`processor_config.json` -- see ../results/groot_risk_gate_report.md for the
full research trail on how this was confirmed.

Deliberately reuses `Isaac-GR00T`'s OWN `gr00t.policy.server_client.PolicyServer`
+ ZMQ REQ/REP wire protocol (confirmed by reading that module at git tag
`n1.6.1-release`) rather than hand-rolling a socket/pickle protocol like
cosmos_server.py -- this is NVIDIA's own recommended serving pattern for
GR00T specifically (`getting_started/policy.md` / examples/robocasa/README.md),
and `PolicyServer` already implements it correctly.

Run inside the `oopsieverse_groot_srv` conda env (see
../setup/setup_groot_server_env.sh -- installs `Isaac-GR00T` pinned to git
tag `n1.6.1-release`, NOT `main`, which has already moved on to N1.7 and no
longer ships the `gr00t_n1d6` model implementation this checkpoint needs):

    conda activate oopsieverse_groot_srv
    export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf
    python oopsieverse_eval/servers/groot_server.py --port 5555

Wire protocol (ZMQ REQ/REP via `gr00t.policy.server_client.PolicyServer` /
`PolicyClient` -- confirmed by reading both classes at git tag
`n1.6.1-release`): request is
    {"endpoint": "get_action", "data": {"observation": <obs dict>, "options": {}}}
and `Gr00tPolicy.get_action()` expects `<obs dict>` NESTED by modality group
(NOT flat "video.<key>" strings -- that was this file's first-draft
assumption, corrected after reading gr00t/policy/gr00t_policy.py directly):
    {
      "video": {"res256_image_side_0": <B,T,H,W,3> uint8,   # agentview_right
                 "res256_image_side_1": <B,T,H,W,3> uint8,   # agentview_left
                 "res256_image_wrist_0": <B,T,H,W,3> uint8}, # eye_in_hand
      "state": {"end_effector_position_relative": <B,T,3> float32,
                "end_effector_rotation_relative": <B,T,4> float32 (quat),
                "gripper_qpos": <B,T,2> float32,
                "base_position": <B,T,3> float32,
                "base_rotation": <B,T,4> float32 (quat)},
      "language": {"task_description": [[str]] shape (B,1)},
    }
Response: `(action_dict, info_dict)` where action_dict has
    {"end_effector_position": <B,T,3>, "end_effector_rotation": <B,T,3>,
     "gripper_close": <B,T,1>, "base_motion": <B,T,4>, "control_mode": <B,T,1>}
(T = action horizon; modality_configs["robocasa_panda_omron"]["action"]
["delta_indices"] = 16 steps. NVIDIA's own robocasa365 gymnasium_groot.py
wrapper executes 8 of these open-loop by default (`--n-action-steps 8`)
before re-querying -- see run_robocasa_groot_eval.py for the same choice.)

NOTE: this file's per-request obs/action shapes are PolicyServer's transport
concern, not this server's -- `groot_server.py` only loads the policy and
hands it to `PolicyServer`, which already implements the (de)serialization.
See ../eval/run_robocasa_groot_eval.py for the client-side dict construction.
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groot_server")

EMBODIMENT_TAG = "ROBOCASA_PANDA_OMRON"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="nvidia/GR00T-N1.6-3B",
        help="HF repo id or local snapshot dir (resolved via HF_HOME cache if a repo id).",
    )
    parser.add_argument("--embodiment_tag", default=EMBODIMENT_TAG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    embodiment_tag = EmbodimentTag[args.embodiment_tag]

    logger.info("Loading Gr00tPolicy from %s (embodiment_tag=%s, device=%s)...",
                args.model_path, args.embodiment_tag, args.device)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=args.strict,
    )
    logger.info("Policy loaded. Starting PolicyServer on %s:%d ...", args.host, args.port)

    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
