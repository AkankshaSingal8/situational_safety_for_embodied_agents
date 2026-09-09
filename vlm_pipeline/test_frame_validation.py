#!/usr/bin/env python3
"""Test script to validate the frame corruption detection logic."""

import numpy as np

def test_frame_detection():
    """Test different types of frames to see if variance threshold works."""

    # Test 1: Normal rendered scene (should have high variance)
    normal_frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    # Add some structure (not pure noise)
    normal_frame[:512, :] = 100  # Dark region
    normal_frame[512:, :] = 200  # Light region

    # Test 2: Grey noise (uniform random grey - what corrupted rendering produces)
    # More uniform distribution - all values 0-255 appear with equal probability
    grey_noise = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

    # Test 3: Solid grey (failed render - zero variance)
    solid_grey = np.ones((1024, 1024, 3), dtype=np.uint8) * 128

    # Test 4: Black frame (rendering failure - zero variance)
    black_frame = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Test 5: Uninitialized memory (could be anything)
    uninit_memory = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    frames = {
        "Normal scene": normal_frame,
        "Grey noise": grey_noise,
        "Solid grey": solid_grey,
        "Black frame": black_frame,
        "Uninitialized": uninit_memory,
    }

    print("Frame Analysis:")
    print(f"{'Frame Type':<20} {'Std Dev':<10} {'Hist Std':<12} {'Decision'}")
    print("=" * 60)

    for name, frame in frames.items():
        img_std = frame.std()
        hist, _ = np.histogram(frame.flatten(), bins=32, range=(0, 256))
        hist_std = hist.std()
        hist_threshold = frame.size / 300

        # Apply validation logic
        decision = "ACCEPTED"
        if img_std < 5.0:
            decision = "REJECTED (solid color)"
        elif hist_std < hist_threshold:
            decision = "REJECTED (uniform noise)"

        print(f"{name:<20} {img_std:>8.2f}   {hist_std:>10.0f}   {decision}")

if __name__ == "__main__":
    test_frame_detection()
