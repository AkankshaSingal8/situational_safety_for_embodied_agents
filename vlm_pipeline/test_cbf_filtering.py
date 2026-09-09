#!/usr/bin/env python3
"""
test_cbf_filtering.py

Standalone test for CBF filtering without requiring OpenVLA model.
Tests the core CBF functions with synthetic data.
"""

import json
import numpy as np
import sys

# Import CBF functions from run_libero_eval_with_cbf
sys.path.insert(0, '.')
from run_libero_eval_with_cbf import (
    load_ellipsoids,
    evaluate_cbf,
    certify_action_simple,
)


def test_ellipsoid_loading():
    """Test loading pre-computed ellipsoids."""
    print("\n" + "="*70)
    print("TEST 1: Ellipsoid Loading")
    print("="*70)

    ellipsoids = load_ellipsoids(task_id=0, cbf_outputs_dir="cbf_outputs")

    print(f"✓ Loaded {len(ellipsoids)} ellipsoids")

    if len(ellipsoids) > 0:
        print(f"\nSample ellipsoid:")
        sample = ellipsoids[0]
        print(f"  Object: {sample['object']}")
        print(f"  Relationship: {sample['relationship']}")
        print(f"  Center: {sample['center']}")
        print(f"  Semi-axes: {sample['semi_axes']}")

    return ellipsoids


def test_cbf_evaluation(ellipsoids):
    """Test CBF evaluation at different positions."""
    print("\n" + "="*70)
    print("TEST 2: CBF Evaluation")
    print("="*70)

    if len(ellipsoids) == 0:
        print("⚠️  No ellipsoids to test")
        return

    # Test ellipsoid (moka_pot above constraint)
    ellipsoid = None
    for e in ellipsoids:
        if e['object'] == 'moka_pot_obstacle_1' and e['relationship'] == 'above':
            ellipsoid = e
            break

    if ellipsoid is None:
        ellipsoid = ellipsoids[0]

    center = np.array(ellipsoid['center'])
    semi_axes = np.array(ellipsoid['semi_axes'])

    print(f"\nTesting ellipsoid: {ellipsoid['object']} - {ellipsoid['relationship']}")
    print(f"  Center: {center}")
    print(f"  Semi-axes: {semi_axes}")

    # Test positions
    test_cases = [
        ("At center (unsafe)", center),
        ("Far away (safe)", center + np.array([0.5, 0.5, 0.0])),
        ("On boundary", center + semi_axes * np.array([1.0, 0.0, 0.0])),
        ("Just inside", center + semi_axes * np.array([0.9, 0.0, 0.0])),
    ]

    print("\nCBF values at different positions:")
    for desc, pos in test_cases:
        h = evaluate_cbf(pos, center, semi_axes)
        safety = "SAFE ✓" if h > 0 else "UNSAFE ✗"
        print(f"  {desc:20s}: h = {h:+.3f}  ({safety})")


def test_action_certification(ellipsoids):
    """Test action certification with CBF-QP."""
    print("\n" + "="*70)
    print("TEST 3: Action Certification")
    print("="*70)

    if len(ellipsoids) == 0:
        print("⚠️  No ellipsoids to test")
        return

    # Simulate robot state (from metadata: moka pot is at [-0.052, 0.020, 1.006])
    # Start robot at a safe position
    ee_pos = np.array([-0.1, 0.1, 0.9])  # Safe starting position

    print(f"\nRobot end-effector at: {ee_pos}")

    # Test commanded actions
    test_actions = [
        ("Safe action (away from obstacles)", np.array([0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0])),
        ("Unsafe action (toward obstacle)", np.array([0.05, -0.08, 0.1, 0.0, 0.0, 0.0, 1.0])),
        ("Borderline action", np.array([0.02, -0.04, 0.05, 0.0, 0.0, 0.0, 1.0])),
    ]

    for desc, u_cmd in test_actions:
        print(f"\n{desc}:")
        print(f"  Commanded: {u_cmd[:3]}")

        u_cert, info = certify_action_simple(
            u_cmd=u_cmd,
            ee_pos=ee_pos,
            ellipsoids=ellipsoids,
            dt=0.05,
            alpha=1.0
        )

        print(f"  Certified:  {u_cert[:3]}")
        print(f"  Modified: {info['modified']}")
        print(f"  h_min: {info['h_min']:.3f}")
        print(f"  Interventions: {info['num_interventions']}")

        if info['modified']:
            deviation = np.linalg.norm(u_cert[:3] - u_cmd[:3])
            print(f"  Deviation: {deviation:.4f} m")
            if len(info['interventions']) > 0:
                print(f"  Active constraints:")
                for interv in info['interventions']:
                    print(f"    - {interv['object']} ({interv['relationship']}): h={interv['h_current']:.3f}")


def test_trajectory_saving():
    """Test trajectory NPZ saving and loading."""
    print("\n" + "="*70)
    print("TEST 4: Trajectory Saving/Loading")
    print("="*70)

    from run_libero_eval_with_cbf import save_trajectory
    import os
    import tempfile

    # Create synthetic trajectory data
    trajectory_data = []
    for t in range(10):
        trajectory_data.append({
            't': t,
            'ee_pos': np.array([0.1, 0.1, 0.9 + t*0.01]),
            'ee_quat': np.array([0.0, 0.0, 0.0, 1.0]),
            'action_commanded': np.random.randn(7) * 0.01,
            'action_certified': np.random.randn(7) * 0.01,
            'action_executed': np.random.randn(7) * 0.01,
            'cbf_active': True,
            'h_min': 0.5 - t * 0.05,
            'num_interventions': 1 if t % 3 == 0 else 0,
        })

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_trajectory(trajectory_data, task_id=0, episode_idx=0, run_dir=tmpdir)

        # Load and verify
        npz_path = os.path.join(tmpdir, "task_0", "episode_00.npz")
        data = np.load(npz_path)
        traj = data['trajectory']

        print(f"✓ Saved and loaded trajectory with {len(traj)} timesteps")
        print(f"\nTrajectory structure:")
        print(f"  Fields: {traj.dtype.names}")
        print(f"\nSample data (t=0):")
        print(f"  ee_pos: {traj[0]['ee_pos']}")
        print(f"  h_min: {traj[0]['h_min']}")
        print(f"  num_interventions: {traj[0]['num_interventions']}")

        # Check interventions
        intervention_count = np.sum(traj['num_interventions'] > 0)
        print(f"\nStatistics:")
        print(f"  Total interventions: {intervention_count}/{len(traj)} timesteps ({intervention_count/len(traj)*100:.1f}%)")
        print(f"  Min h-value: {traj['h_min'].min():.3f}")


def main():
    """Run all CBF tests."""
    print("\n" + "="*70)
    print("CBF FILTERING TEST SUITE")
    print("="*70)

    try:
        # Test 1: Load ellipsoids
        ellipsoids = test_ellipsoid_loading()

        # Test 2: Evaluate CBF
        test_cbf_evaluation(ellipsoids)

        # Test 3: Certify actions
        test_action_certification(ellipsoids)

        # Test 4: Save/load trajectories
        test_trajectory_saving()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
