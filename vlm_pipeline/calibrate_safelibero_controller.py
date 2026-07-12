"""Measure SafeLIBERO's actual action-to-EEF controller response.

This is an additive diagnostic. It does not load a VLA, write benchmark
results, or change any environment configuration. The measurements are used to
decide whether action-space prediction is accurate enough for a safety filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from libero.libero import benchmark

from safelibero_utils import get_safelibero_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="safelibero_spatial")
    parser.add_argument("--level", default="I", choices=("I", "II"))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--state-count", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reset_to_state(env, state: np.ndarray) -> dict:
    env.reset()
    obs = env.set_init_state(state)
    # Match the evaluator's stabilization, without a policy or filter.
    for _ in range(20):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    return obs


def main() -> None:
    args = parse_args()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite](args.level)
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, instruction = get_safelibero_env(
        task,
        "openvla",
        resolution=args.resolution,
        include_wrist_camera=False,
    )

    commands = (0.10, 0.25, 0.50, 0.75, 1.00)
    records: list[dict] = []
    try:
        for state_index, state in enumerate(initial_states[: args.state_count]):
            for axis, axis_name in enumerate(("x", "y", "z")):
                for sign in (-1.0, 1.0):
                    for magnitude in commands:
                        obs = reset_to_state(env, state)
                        before = np.asarray(obs["robot0_eef_pos"], dtype=float)
                        action = np.zeros(7, dtype=np.float32)
                        action[axis] = sign * magnitude
                        obs, _, _, _ = env.step(action)
                        after = np.asarray(obs["robot0_eef_pos"], dtype=float)
                        delta = after - before
                        records.append(
                            {
                                "state_index": state_index,
                                "axis": axis_name,
                                "command": float(action[axis]),
                                "eef_delta": delta.tolist(),
                                "axis_response": float(delta[axis]),
                                "cross_axis_norm": float(
                                    np.linalg.norm(np.delete(delta, axis))
                                ),
                            }
                        )
    finally:
        env.close()

    summary = {}
    for axis, axis_name in enumerate(("x", "y", "z")):
        axis_records = [record for record in records if record["axis"] == axis_name]
        command = np.asarray([record["command"] for record in axis_records])
        response = np.asarray([record["axis_response"] for record in axis_records])
        least_squares_scale = float(np.dot(command, response) / np.dot(command, command))
        residual = response - least_squares_scale * command
        summary[axis_name] = {
            "meters_per_normalized_action": least_squares_scale,
            "rmse_meters": float(np.sqrt(np.mean(residual**2))),
            "max_abs_error_meters": float(np.max(np.abs(residual))),
            "mean_cross_axis_motion_meters": float(
                np.mean([record["cross_axis_norm"] for record in axis_records])
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": args.suite,
        "level": args.level,
        "task_id": args.task_id,
        "instruction": instruction,
        "states": min(args.state_count, len(initial_states)),
        "assumed_static_filter_scale_meters": 0.050,
        "summary": summary,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
