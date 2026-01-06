"""
Diagnostic test for LivePortrait with test character.
This will help identify what's not working as expected.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
import time

print("="*70)
print("LIVEPORTRAIT DIAGNOSTIC TEST")
print("="*70)
print()

# Test 1: Check character loading
print("[1/6] Testing character loading...")
try:
    from character_manager_v2 import MultiBatchCharacterManager

    manager = MultiBatchCharacterManager(
        characters_path="assets/characters",
        target_size=(512, 512),
        auto_crop=True,
        preload_all=False,
        enable_video_processing=True
    )

    print(f"  ✓ Character manager loaded")
    print(f"  ✓ Total characters: {manager.get_character_count()}")

    if manager.get_character_count() == 0:
        print("  ✗ ERROR: No characters found!")
        print("  Please add characters to assets/characters/")
        sys.exit(1)

    # Get current character
    char = manager.get_current_character()
    if char:
        print(f"  ✓ Current character: {char.name}")
        print(f"    - Images: {len(char.image_files)}")
        print(f"    - Videos: {len(char.video_files)}")

    # Get character image
    char_image = manager.get_current_character_image()
    if char_image is not None:
        print(f"  ✓ Character image loaded: {char_image.shape}")
    else:
        print(f"  ✗ ERROR: Character image is None!")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check motion tracker
print()
print("[2/6] Testing motion tracker...")
try:
    from motion_tracker import MotionTracker

    tracker = MotionTracker()
    print(f"  ✓ Motion tracker initialized")
    print(f"    - Using legacy API: {tracker.use_legacy_api}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check AI animator
print()
print("[3/6] Testing AI animator...")
try:
    from ai_animator import AIAnimator

    animator = AIAnimator(
        model_type="liveportrait",
        model_path="models/liveportrait",
        device="cuda",
        fp16=True
    )

    if animator.initialize():
        print(f"  ✓ AI animator initialized")
    else:
        print(f"  ⚠ AI animator initialized with mock model")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check camera
print()
print("[4/6] Testing camera...")
try:
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Camera 1 working: {frame.shape}")

            # Test face tracking
            landmarks = tracker.process_frame(frame)
            if landmarks:
                print(f"  ✓ Face detected in camera feed")
            else:
                print(f"  ⚠ No face detected (make sure face is visible)")
        else:
            print(f"  ✗ Camera opened but cannot read frames")
        cap.release()
    else:
        print(f"  ✗ Failed to open camera 1")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# Test 5: Test animation
print()
print("[5/6] Testing animation...")
try:
    # Create test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame[:] = (128, 128, 128)  # Gray

    # Animate
    start_time = time.time()
    animated = animator.animate_frame(char_image, test_frame, None)
    elapsed = (time.time() - start_time) * 1000

    print(f"  ✓ Animation test complete")
    print(f"    - Input shape: {char_image.shape}")
    print(f"    - Output shape: {animated.shape}")
    print(f"    - Time: {elapsed:.1f}ms")

    # Check if animation did anything
    if np.array_equal(animated, char_image):
        print(f"  ⚠ Output is identical to input (mock model?)")
    else:
        print(f"  ✓ Animation produced different output")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check model status
print()
print("[6/6] Checking LivePortrait model status...")
try:
    from pathlib import Path

    model_path = Path("models/liveportrait")

    print(f"  Model path: {model_path}")
    print(f"  Exists: {model_path.exists()}")

    if model_path.exists():
        files = list(model_path.glob("*"))
        print(f"  Files in model directory: {len(files)}")
        for f in files[:5]:
            print(f"    - {f.name}")
        if len(files) > 5:
            print(f"    ... and {len(files)-5} more")

        if len(files) <= 1:  # Only README
            print()
            print("  ⚠ WARNING: Real LivePortrait model not installed!")
            print("  ⚠ Using MOCK model (no actual animation)")
            print()
            print("  To use real LivePortrait:")
            print("    1. Download LivePortrait model")
            print("    2. Place files in models/liveportrait/")
            print("    3. Restart application")
    else:
        print(f"  ⚠ Model directory does not exist")

except Exception as e:
    print(f"  ✗ ERROR: {e}")

# Summary
print()
print("="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print()
print("Issues found:")
print()

issues_found = False

# Check if using mock model
model_path = Path("models/liveportrait")
if model_path.exists():
    files = list(model_path.glob("*"))
    if len(files) <= 1:
        issues_found = True
        print("❌ Real LivePortrait model NOT installed")
        print("   Current: Using MOCK model (character displayed but not animated)")
        print("   Solution: Install real LivePortrait model")
        print()

if not issues_found:
    print("✓ No critical issues found")
    print()
    print("If animation is not working as expected:")
    print("  1. Make sure face is visible in camera")
    print("  2. Check lighting (face should be well-lit)")
    print("  3. Try pressing 'T' to toggle stats and see FPS")
    print("  4. Try switching characters with arrow keys")

print()
print("="*70)

