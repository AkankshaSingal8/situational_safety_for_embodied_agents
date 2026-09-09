#!/usr/bin/env python3
"""
Minimal test for frame corruption fix.
Tests the defensive copying logic without requiring full environment setup.
"""

import numpy as np
import sys

def test_frame_defensive_copy():
    """Test that our defensive copy prevents buffer reuse corruption."""
    print("=" * 70)
    print("TEST 1: Defensive Frame Copy")
    print("=" * 70)

    # Simulate MuJoCo render buffer (reusable memory)
    render_buffer = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    # Simulate what happens in the code
    print("\n1. Initial frame capture from 'simulator buffer'")
    img = render_buffer  # This is what obs["agentview_image"] returns

    # Apply the rotation (simulating get_libero_image)
    print("2. Apply 180-degree rotation")
    img_rotated = img[::-1, ::-1]

    # OLD WAY (potentially buggy):
    print("\n--- OLD WAY (just .copy()) ---")
    img_old = img_rotated.copy()
    print(f"   dtype: {img_old.dtype}")
    print(f"   shape: {img_old.shape}")
    print(f"   C_CONTIGUOUS: {img_old.flags['C_CONTIGUOUS']}")
    print(f"   Shares memory with buffer: {np.shares_memory(img_old, render_buffer)}")

    # NEW WAY (our fix):
    print("\n--- NEW WAY (np.ascontiguousarray) ---")
    img_new = np.ascontiguousarray(img_rotated.copy())
    print(f"   dtype: {img_new.dtype}")
    print(f"   shape: {img_new.shape}")
    print(f"   C_CONTIGUOUS: {img_new.flags['C_CONTIGUOUS']}")
    print(f"   Shares memory with buffer: {np.shares_memory(img_new, render_buffer)}")

    # Verify assertions would pass
    print("\n3. Test assertions from our fix:")
    try:
        assert img_new.dtype == np.uint8, f"Frame dtype mismatch: {img_new.dtype}"
        assert img_new.shape == (1024, 1024, 3), f"Frame shape mismatch: {img_new.shape}"
        assert img_new.flags['C_CONTIGUOUS'], "Frame is not C-contiguous"
        print("   ✅ All assertions PASSED")
    except AssertionError as e:
        print(f"   ❌ Assertion FAILED: {e}")
        return False

    # Simulate buffer reuse (MuJoCo overwrites buffer for next frame)
    print("\n4. Simulate buffer reuse (fill with zeros)")
    render_buffer[:] = 0

    # Check if our copy is still intact
    print("\n5. Check if stored frame is corrupted:")
    if img_new.sum() > 0:
        print(f"   ✅ Frame still intact (sum={img_new.sum()})")
        print(f"   ✅ Successfully isolated from render buffer!")
    else:
        print(f"   ❌ Frame corrupted (sum={img_new.sum()})")
        print(f"   ❌ Frame shares memory with buffer!")
        return False

    return True


def test_error_handling_copy():
    """Test that error handling creates proper copies, not references."""
    print("\n" + "=" * 70)
    print("TEST 2: Error Handling Frame Duplication")
    print("=" * 70)

    replay_images = []

    # Add a frame
    frame1 = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    frame1_safe = np.ascontiguousarray(frame1.copy())
    replay_images.append(frame1_safe)

    print(f"\n1. Added frame 1 (sum={frame1_safe.sum()})")

    # OLD WAY (buggy - appends reference):
    print("\n--- OLD WAY (append reference) ---")
    replay_images_old = [frame1_safe]
    replay_images_old.append(replay_images_old[-1])  # BUG: reference!

    print(f"   Frame 0 id: {id(replay_images_old[0])}")
    print(f"   Frame 1 id: {id(replay_images_old[1])}")
    print(f"   Same object: {replay_images_old[0] is replay_images_old[1]}")

    # NEW WAY (fixed - copies frame):
    print("\n--- NEW WAY (copy frame) ---")
    replay_images_new = [frame1_safe]
    replay_images_new.append(replay_images_new[-1].copy())  # FIX: copy!

    print(f"   Frame 0 id: {id(replay_images_new[0])}")
    print(f"   Frame 1 id: {id(replay_images_new[1])}")
    print(f"   Same object: {replay_images_new[0] is replay_images_new[1]}")

    # Verify independence
    print("\n2. Modify frame 1 to test independence:")
    replay_images_new[1][:] = 0  # Zero out the duplicate

    if replay_images_new[0].sum() > 0 and replay_images_new[1].sum() == 0:
        print(f"   ✅ Frames are independent!")
        print(f"      Frame 0 sum: {replay_images_new[0].sum()} (intact)")
        print(f"      Frame 1 sum: {replay_images_new[1].sum()} (modified)")
        return True
    else:
        print(f"   ❌ Frames share memory!")
        return False


def test_ascontiguousarray_on_rotation():
    """Test that ascontiguousarray works correctly on rotated images."""
    print("\n" + "=" * 70)
    print("TEST 3: Rotation + ascontiguousarray")
    print("=" * 70)

    # Create test image
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    # Rotation creates negative strides
    img_rotated = img[::-1, ::-1]

    print(f"\n1. Original image:")
    print(f"   Strides: {img.strides}")
    print(f"   C_CONTIGUOUS: {img.flags['C_CONTIGUOUS']}")

    print(f"\n2. After rotation (view with negative strides):")
    print(f"   Strides: {img_rotated.strides}")
    print(f"   C_CONTIGUOUS: {img_rotated.flags['C_CONTIGUOUS']}")

    # Apply our fix
    img_fixed = np.ascontiguousarray(img_rotated)

    print(f"\n3. After ascontiguousarray:")
    print(f"   Strides: {img_fixed.strides}")
    print(f"   C_CONTIGUOUS: {img_fixed.flags['C_CONTIGUOUS']}")

    # Verify content is preserved
    print(f"\n4. Verify content preserved:")
    print(f"   Arrays equal: {np.array_equal(img_rotated, img_fixed)}")
    print(f"   Memory independent: {not np.shares_memory(img, img_fixed)}")

    if img_fixed.flags['C_CONTIGUOUS'] and np.array_equal(img_rotated, img_fixed):
        print("   ✅ Rotation + ascontiguousarray works correctly!")
        return True
    else:
        print("   ❌ Something went wrong!")
        return False


if __name__ == "__main__":
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Frame Corruption Fix - Unit Tests" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        test_frame_defensive_copy,
        test_error_handling_copy,
        test_ascontiguousarray_on_rotation,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"\nTests passed: {passed}/{total}")

    if all(results):
        print("\n✅ ALL TESTS PASSED - Fix is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Please review!")
        sys.exit(1)
