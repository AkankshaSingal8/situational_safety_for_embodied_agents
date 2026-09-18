#!/usr/bin/env python
"""Bit-exact parity harness for ``FOLSafetyFilter._apply_cbf``.

Every number in ``fol_safety_filter/README.md`` was produced by the hardcoded
obstacle loop in ``_apply_cbf``.  Routing those decisions through loaded rule
records is only a restructuring if the emitted actions do not move at all, so
this harness pins them: it drives ``_apply_cbf`` over a fixed synthetic
trajectory (no simulator, no policy, no GPU), sweeps the env-var matrix that
selects the geometry branches, and compares against a committed golden trace.

    # capture, from an unmodified checkout
    PYTHONPATH=<checkout> python tools/fol_parity_trace.py --emit golden.npz

    # verify, from the refactored tree
    python tools/fol_parity_trace.py --check golden.npz

Exact equality is the bar, not a tolerance: the refactor performs the same
float operations in the same order, so any nonzero difference is a behaviour
change that has to be explained rather than absorbed.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, Iterator, List, Tuple

import numpy as np

# Appended, not inserted: an explicit PYTHONPATH must still win, so the golden
# trace can be captured from a different checkout than the one holding this
# script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
# Generated from a fixed seed, then frozen as float64 arrays.  The seed is used
# only to lay out plausible geometry; the trace itself is what is committed, so
# a different numpy RNG implementation cannot silently move the golden values.

SEED = 0
N_STEPS = 24

# (FOL_ELLIPSOID, FOL_RAY_RESCALE, FOL_TASK_BIAS)
ENV_COMBOS: List[Tuple[str, str, str]] = list(
    itertools.product(["0", "1"], ["0", "1", "target"], ["0", "1"])
)

_OBSTACLE_R = 0.22          # exceeds the 0.20 floor, so the record value bites
_ELLIPSOID_WARN = np.array([0.24, 0.18, 0.30])
_ELLIPSOID_HARD = np.array([0.12, 0.09, 0.15])


def _frame_R(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_scenarios() -> List[Dict]:
    """Deterministic scenario list.  Order is part of the trace identity."""
    rng = np.random.default_rng(SEED)
    table_z = 0.83
    scenarios: List[Dict] = []

    # Scenario families, each exercising a different branch of _apply_cbf:
    #   oracle_single   — one oracle-named obstacle, radial sphere path
    #   oracle_multi    — two obstacles, so obstacle-major ordering matters
    #   visual_corridor — 'visual_' prefix + vision targets -> cancel_scale 0.6
    #   ellipsoid       — rotated ellipsoid zones, incl. the ray-rescale branch
    #   arm_only        — EEF clear of every zone, an arm checkpoint inside one
    specs = [
        ("oracle_single", ["moka_pot_obstacle"], False, False),
        ("oracle_multi", ["moka_pot_obstacle", "wine_bottle_obstacle"], False, False),
        ("visual_corridor", ["visual_obstacle_0"], True, False),
        ("visual_multi_corridor", ["visual_obstacle_0", "visual_obstacle_1"], True, False),
        ("ellipsoid", ["moka_pot_obstacle"], False, True),
        ("ellipsoid_visual", ["visual_obstacle_0"], True, True),
        ("arm_only", ["moka_pot_obstacle"], False, False),
    ]

    for name, obstacles, is_visual, with_ellipsoid in specs:
        obs_pos = {}
        for i, obs in enumerate(obstacles):
            obs_pos[obs] = np.array(
                [-0.05 + 0.18 * i, 0.10 - 0.12 * i, table_z + 0.05], dtype=np.float64
            )

        ellipsoids = {}
        if with_ellipsoid:
            for i, obs in enumerate(obstacles):
                ellipsoids[obs] = {
                    "frame_R": _frame_R(0.35 + 0.2 * i),
                    "warn_radii": _ELLIPSOID_WARN.copy(),
                    "hard_radii": _ELLIPSOID_HARD.copy(),
                }

        # A straight sweep through the first obstacle, so the trajectory
        # crosses warning_r and hard_r rather than grazing them.
        first = obs_pos[obstacles[0]]
        start = first + np.array([-0.38, -0.05, 0.10])
        end = first + np.array([0.30, 0.06, -0.02])

        if name == "arm_only":
            # Keep the EEF far away for every step; only the arm checkpoint
            # enters a zone.  This is the fixture that proves the arm channel
            # is a separable rule rather than a side effect of the EEF one.
            start = first + np.array([-0.70, -0.40, 0.30])
            end = first + np.array([-0.55, -0.34, 0.26])

        steps = []
        for k in range(N_STEPS):
            frac = k / (N_STEPS - 1)
            ee_pos = (1.0 - frac) * start + frac * end
            # Commanded deltas at OpenVLA scale (~cm per step), with a small
            # deterministic jitter and a vertical descent component.
            u_cmd = np.zeros(6, dtype=np.float64)
            u_cmd[:3] = (end - start) / (N_STEPS - 1)
            u_cmd[:3] += rng.normal(0.0, 0.002, 3)
            u_cmd[2] -= 0.004
            u_cmd[3:6] = rng.normal(0.0, 0.05, 3)
            steps.append(
                {
                    "ee_pos": np.round(ee_pos, 12),
                    "u_cmd": np.round(u_cmd, 12),
                }
            )

        arm_checkpoints = []
        if name == "arm_only":
            # Elbow/wrist parked just inside the 0.15 m warning radius.
            for i, (body, wr, hr, push) in enumerate(
                [("robot0_link4", 0.15, 0.08, 0.06), ("robot0_link6", 0.15, 0.08, 0.06)]
            ):
                arm_checkpoints.append(
                    {
                        "pos": first + np.array([-0.11 + 0.01 * i, 0.04, 0.02]),
                        "warning_r": wr,
                        "hard_r": hr,
                        "push": push,
                        "name": body,
                    }
                )
        else:
            for i, (body, wr, hr, push) in enumerate(
                [("robot0_link4", 0.15, 0.08, 0.06), ("robot0_link6", 0.15, 0.08, 0.06)]
            ):
                arm_checkpoints.append(
                    {
                        "pos": first + np.array([-0.26 + 0.05 * i, -0.13, 0.13]),
                        "warning_r": wr,
                        "hard_r": hr,
                        "push": push,
                        "name": body,
                    }
                )

        vision_targets = []
        if is_visual:
            # Two mentioned objects near the obstacle: the corridor relaxation
            # only applies when the command aims at one of them.
            vision_targets = [
                (first + np.array([0.09, 0.02, 0.0]))[:2].copy(),
                (first + np.array([0.02, -0.14, 0.0]))[:2].copy(),
            ]

        scenarios.append(
            {
                "name": name,
                "obstacles": obstacles,
                "obstacle_pos": obs_pos,
                "obstacle_radii": {o: _OBSTACLE_R for o in obstacles},
                "ellipsoids": ellipsoids,
                "vision_targets": vision_targets,
                "arm_checkpoints": arm_checkpoints,
                "steps": steps,
                # velocity / rotation channels, supplied through MappedCBFSet
                "velocity_limit": 0.12 if name.startswith("oracle") else None,
                "angular_limit": 0.20 if name == "ellipsoid" else None,
                "target_name": obstacles[0] if name.endswith("corridor") else None,
            }
        )

    return scenarios


# --------------------------------------------------------------------------- #
# Driving _apply_cbf
# --------------------------------------------------------------------------- #

def _make_filter(scenario: Dict):
    from fol_safety_filter.filter import FOLSafetyFilter

    f = FOLSafetyFilter(use_vlm_grounding=False, use_vlm_obstacle=False)
    f._obstacle_names = list(scenario["obstacles"])
    f._obstacle_radii = dict(scenario["obstacle_radii"])
    f._obstacle_ellipsoids = {
        k: {kk: vv.copy() for kk, vv in v.items()}
        for k, v in scenario["ellipsoids"].items()
    }
    f._vision_targets_xy = [t.copy() for t in scenario["vision_targets"]]
    f._vision_target_xy = (
        scenario["vision_targets"][0].copy() if scenario["vision_targets"] else None
    )
    f._target_name = scenario["target_name"]
    f._vision_fallback_active = False
    f._initialized = True
    return f


def _make_state(scenario: Dict, ee_pos: np.ndarray):
    from fol_safety_filter.primitives import ObjectState, RobotState

    objects = {}
    for obs, pos in scenario["obstacle_pos"].items():
        objects[obs] = ObjectState(
            name=obs,
            pos=pos.copy(),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            bbox_half=np.array([0.04, 0.04, 0.04]),
        )
    return RobotState(
        ee_pos=ee_pos.copy(),
        ee_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        ee_vel=np.zeros(3),
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        joint_limits_lo=np.full(7, -2.8),
        joint_limits_hi=np.full(7, 2.8),
        gripper_width=0.02,
        objects=objects,
    )


def _make_cbf_set(scenario: Dict):
    from fol_safety_filter.cbf_mapper import MappedCBFSet

    return MappedCBFSet(
        semantic_envelopes=[],
        collision_envelopes=[],
        pose_constrained=scenario["angular_limit"] is not None,
        desired_orientation=None,
        velocity_limit=scenario["velocity_limit"],
        angular_limit=scenario["angular_limit"],
        active_rule_names=[],
    )


def _iter_rows(scenarios: List[Dict]) -> Iterator[Tuple[str, int, str, np.ndarray]]:
    """Yield (scenario, step, env_key, u_safe) for the whole matrix."""
    saved = {k: os.environ.get(k) for k in
             ("FOL_ELLIPSOID", "FOL_RAY_RESCALE", "FOL_TASK_BIAS")}
    try:
        for ellip, ray, bias in ENV_COMBOS:
            os.environ["FOL_ELLIPSOID"] = ellip
            os.environ["FOL_RAY_RESCALE"] = ray
            os.environ["FOL_TASK_BIAS"] = bias
            env_key = f"e{ellip}_r{ray}_b{bias}"
            for scenario in scenarios:
                f = _make_filter(scenario)
                cbf_set = _make_cbf_set(scenario)
                for idx, step in enumerate(scenario["steps"]):
                    state = _make_state(scenario, step["ee_pos"])
                    arm = [
                        {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                         for kk, vv in cp.items()}
                        for cp in scenario["arm_checkpoints"]
                    ]
                    u_safe, _ = f._apply_cbf(
                        step["u_cmd"].copy(), state, cbf_set, arm
                    )
                    yield scenario["name"], idx, env_key, np.asarray(u_safe, dtype=np.float64)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def compute_trace() -> Dict[str, np.ndarray]:
    scenarios = build_scenarios()
    keys: List[str] = []
    u_cmd_rows: List[np.ndarray] = []
    u_safe_rows: List[np.ndarray] = []
    by_name = {s["name"]: s for s in scenarios}

    for name, idx, env_key, u_safe in _iter_rows(scenarios):
        keys.append(f"{env_key}|{name}|{idx}")
        u_cmd_rows.append(by_name[name]["steps"][idx]["u_cmd"])
        u_safe_rows.append(u_safe)

    return {
        "keys": np.array(keys, dtype=object),
        "u_cmd": np.stack(u_cmd_rows),
        "u_safe": np.stack(u_safe_rows),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _active_fraction(trace: Dict[str, np.ndarray]) -> float:
    diff = np.abs(trace["u_safe"] - trace["u_cmd"]).max(axis=1)
    return float((diff > 1e-12).mean())


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--emit", metavar="NPZ", help="write a golden trace")
    g.add_argument("--check", metavar="NPZ", help="compare against a golden trace")
    args = ap.parse_args(argv)

    trace = compute_trace()
    frac = _active_fraction(trace)
    print(f"rows={len(trace['keys'])}  filter-active rows={frac:.1%}")

    if args.emit:
        os.makedirs(os.path.dirname(os.path.abspath(args.emit)), exist_ok=True)
        np.savez_compressed(
            args.emit,
            keys=trace["keys"],
            u_cmd=trace["u_cmd"],
            u_safe=trace["u_safe"],
        )
        print(f"wrote {args.emit}")
        if frac <= 0.0:
            print("ERROR: golden trace has no filter-active rows — it would "
                  "prove nothing", file=sys.stderr)
            return 2
        return 0

    golden = np.load(args.check, allow_pickle=True)
    g_keys = list(golden["keys"])
    t_keys = list(trace["keys"])
    if g_keys != t_keys:
        only_g = [k for k in g_keys if k not in set(t_keys)][:5]
        only_t = [k for k in t_keys if k not in set(g_keys)][:5]
        print(f"FAIL: scenario keys differ.\n  golden-only: {only_g}\n"
              f"  current-only: {only_t}", file=sys.stderr)
        return 1

    delta = np.abs(trace["u_safe"] - golden["u_safe"])
    max_delta = float(delta.max())
    if max_delta != 0.0:
        row = int(delta.max(axis=1).argmax())
        print(
            f"FAIL: max|delta| = {max_delta:.3e} (must be exactly 0)\n"
            f"  first/worst mismatch at {t_keys[row]}\n"
            f"  golden  u_safe = {golden['u_safe'][row]}\n"
            f"  current u_safe = {trace['u_safe'][row]}",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: max|delta| = 0 over {len(t_keys)} rows "
          f"({len(ENV_COMBOS)} env combos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
