"""Shared motion-degeneracy statistics: distinguishes "the policy is doing
something" from "the policy is emitting noise" by comparing real-episode
motion against a matched random-action baseline (same task/seed).

Used by run_robocasa_groot_eval.py (and its random-baseline counterpart,
groot_random_baseline_eval.py) to populate each episode record with
`eef_total_displacement` (sum of per-step L2 eef-position deltas, in
meters) and `mean_frame_diff` (mean absolute pixel difference between
consecutive `agentview_image` frames, 0-255 scale) -- reused directly by
../results/groot_risk_gate_report.md's real-episode degeneracy verdict.
"""

from __future__ import annotations

from typing import List

import numpy as np


def eef_total_displacement(eef_positions: List[np.ndarray]) -> float:
    """Sum of frame-to-frame L2 displacement of the end-effector, in meters."""
    if len(eef_positions) < 2:
        return 0.0
    arr = np.asarray(eef_positions, dtype=np.float64)
    deltas = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=-1)))


def mean_frame_diff(frames: List[np.ndarray]) -> float:
    """Mean absolute pixel difference between consecutive frames (0-255 scale)."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        diffs.append(
            float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
        )
    return float(np.mean(diffs))
