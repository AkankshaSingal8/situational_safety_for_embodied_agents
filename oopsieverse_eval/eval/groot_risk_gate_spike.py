"""Risk-gate spike for GR00T-N1.6 on OopsieVerse (high-risk spike, see
../results/groot_risk_gate_report.md for full context).

Goal: confirm `gr00t.policy.gr00t_policy.Gr00tPolicy` loads
`nvidia/GR00T-N1.6-3B` with `embodiment_tag=ROBOCASA_PANDA_OMRON` and
produces NON-DEGENERATE actions -- WITHOUT ever constructing a
robosuite/RoboCasa env in this process -- using synthetic observations
shaped exactly like the checkpoint's own `processor_config.json`
modality config for this embodiment (3 cameras @ 256x256, 5 state keys,
1 language key).

Mirrors cosmos_risk_gate_spike.py's philosophy: this does NOT touch the
sim at all. If this fails to load or produces degenerate/constant output,
STOP -- do not build groot_server.py's ZMQ path further or attempt real
RoboCasa episodes until this is resolved.

Degeneracy check: query the policy with 5 different synthetic observations
(varying images and state) plus, separately, 5 truly random-noise
observations. A healthy, embodiment-aware policy should produce actions
that (a) are not constant across different inputs, (b) are not saturated
at the same extreme value every step, and (c) differ in aggregate
statistics between "structured" and "pure noise" inputs more than they
differ between two draws of pure noise (i.e. the policy is doing more
than emitting a fixed prior).

Run inside the `oopsieverse_groot_srv` conda env (see
../setup/setup_groot_server_env.sh), via SLURM (see
../slurm/spike_groot_risk_gate.slurm):

    conda activate oopsieverse_groot_srv
    export HF_HOME=/ocean/projects/cis250185p/asingal/oopsieverse_eval_cache/hf
    python oopsieverse_eval/eval/groot_risk_gate_spike.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

MODEL_PATH = "nvidia/GR00T-N1.6-3B"
CAMERA_KEYS = ["res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0"]
IMG_H = IMG_W = 256
STATE_DIMS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}


def make_obs(rng: np.random.Generator, structured: bool, instruction: str) -> dict:
    """Build one `Gr00tPolicy.get_action()` observation, batch=1, time=1.

    `structured=True` -> a smooth gradient image + a plausible EEF pose
    (matches statistics.json's mean/std order of magnitude for this
    embodiment: eef pos ~[0.27,-0.04,0.54], quat components in [-1,1],
    gripper_qpos ~[0.03,-0.03]).
    `structured=False` -> uniform random noise image + random-scale state,
    the "is this policy just emitting noise regardless of input" control.
    """
    video = {}
    for i, key in enumerate(CAMERA_KEYS):
        if structured:
            # Smooth per-camera gradient + a bright "object" patch that
            # moves between draws -- visually structured, not noise.
            base = np.linspace(0, 255, IMG_W, dtype=np.uint8)
            img = np.tile(base, (IMG_H, 1))
            img = np.stack([img, np.roll(img, i * 20, axis=1), np.zeros_like(img)], axis=-1)
            cx, cy = rng.integers(40, IMG_W - 40), rng.integers(40, IMG_H - 40)
            img[cy - 20 : cy + 20, cx - 20 : cx + 20] = [255, 0, 0]
        else:
            img = rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        video[key] = img[None, None]  # (B=1, T=1, H, W, 3)

    state = {}
    if structured:
        state["end_effector_position_relative"] = np.array([[[0.27, -0.04, 0.54]]], dtype=np.float32) \
            + rng.normal(0, 0.05, (1, 1, 3)).astype(np.float32)
        quat = rng.normal(0, 0.3, (1, 1, 4)).astype(np.float32)
        quat[..., -1] += 0.9
        state["end_effector_rotation_relative"] = quat
        state["gripper_qpos"] = np.array([[[0.03, -0.03]]], dtype=np.float32)
        state["base_position"] = rng.normal([2.5, -1.5, 0.7], 0.1, (1, 1, 3)).astype(np.float32)
        base_quat = np.zeros((1, 1, 4), dtype=np.float32)
        base_quat[..., -1] = 1.0
        state["base_rotation"] = base_quat
    else:
        for key, dim in STATE_DIMS.items():
            state[key] = rng.uniform(-5, 5, (1, 1, dim)).astype(np.float32)

    return {
        "video": video,
        "state": state,
        # Key must exactly match nvidia/GR00T-N1.6-3B's processor_config.json
        # modality_configs["robocasa_panda_omron"]["language"]["modality_keys"]
        # (confirmed by reading that file directly, and by gr00t_policy.py's
        # check_observation() iterating self.modality_configs["language"]
        # .modality_keys) -- NOT the simplified "task_description" this
        # script first used, which fails validation with a KeyError.
        "language": {"annotation.human.action.task_description": [[instruction]]},
    }


def summarize_action(action: dict) -> dict:
    out = {}
    for k, v in action.items():
        arr = np.asarray(v, dtype=np.float32)
        out[k] = {
            "shape": list(arr.shape),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return out


def main():
    print("=== GR00T-N1.6 risk-gate spike (no sim construction) ===")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    print(f"Loading Gr00tPolicy from {MODEL_PATH} (embodiment_tag=ROBOCASA_PANDA_OMRON)...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.ROBOCASA_PANDA_OMRON,
        model_path=MODEL_PATH,
        device="cuda",
        strict=True,
    )
    print("Policy loaded OK.")

    rng = np.random.default_rng(0)
    instructions = [
        "pick up the egg from the counter",
        "open the microwave door",
        "turn on the stove",
        "close the drawer",
        "place the plate on the counter",
    ]

    structured_summaries = []
    noise_summaries = []
    for i, instr in enumerate(instructions):
        obs_s = make_obs(rng, structured=True, instruction=instr)
        action_s, _info_s = policy.get_action(obs_s)
        summary_s = summarize_action(action_s)
        structured_summaries.append(summary_s)
        print(f"[structured #{i}] instr={instr!r}")
        for k, v in summary_s.items():
            print(f"    {k}: {v}")

        obs_n = make_obs(rng, structured=False, instruction=instr)
        action_n, _info_n = policy.get_action(obs_n)
        summary_n = summarize_action(action_n)
        noise_summaries.append(summary_n)
        print(f"[noise      #{i}]")
        for k, v in summary_n.items():
            print(f"    {k}: {v}")

    # Degeneracy checks.
    eef_pos_means_structured = [s["end_effector_position"]["mean"] for s in structured_summaries]
    eef_pos_stds_structured = [s["end_effector_position"]["std"] for s in structured_summaries]
    variance_across_inputs = float(np.std(eef_pos_means_structured))
    within_query_std = float(np.mean(eef_pos_stds_structured))

    print("\n=== Degeneracy summary ===")
    print(f"end_effector_position mean, across 5 structured queries: {eef_pos_means_structured}")
    print(f"std of those means (cross-query variation): {variance_across_inputs:.6f}")
    print(f"mean within-query std (chunk-internal variation): {within_query_std:.6f}")

    all_zero = all(abs(m) < 1e-6 for m in eef_pos_means_structured) and within_query_std < 1e-6
    all_identical = variance_across_inputs < 1e-6

    print(f"All-zero output: {all_zero}")
    print(f"Identical output regardless of input (all_identical): {all_identical}")

    if all_zero or all_identical:
        print("\n=== RISK-GATE SPIKE: DEGENERATE OUTPUT -- LIKELY NO-GO ===")
        sys.exit(2)

    print("\n=== RISK-GATE SPIKE: PASS (non-degenerate action output) ===")
    print("Safe to proceed to real RoboCasa episodes via groot_server.py + run_robocasa_groot_eval.py.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== RISK-GATE SPIKE: FAIL (exception) ===")
        traceback.print_exc()
        print("\nSTOP: do not build further on groot_server.py until this is resolved.")
        sys.exit(1)
