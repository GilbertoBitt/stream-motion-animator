"""Simple test to diagnose the issue."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*60)
print("DIAGNOSTIC TEST - Stream Motion Animator")
print("="*60)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    import cv2
    print("  ✓ OpenCV:", cv2.__version__)
except ImportError as e:
    print("  ✗ OpenCV failed:", e)
    sys.exit(1)

try:
    import mediapipe as mp
    print("  ✓ MediaPipe:", mp.__version__)
except ImportError as e:
    print("  ✗ MediaPipe failed:", e)
    sys.exit(1)

try:
    import torch
    print("  ✓ PyTorch:", torch.__version__)
    print("  ✓ CUDA available:", torch.cuda.is_available())
except ImportError as e:
    print("  ✗ PyTorch failed:", e)

try:
    import numpy as np
    print("  ✓ NumPy:", np.__version__)
except ImportError as e:
    print("  ✗ NumPy failed:", e)
    sys.exit(1)

# Test 2: Config
print("\n[2/5] Testing configuration...")
try:
    from config import load_config
    config = load_config()
    print(f"  ✓ Config loaded")
    print(f"    - Characters path: {config.characters_path}")
    print(f"    - Model type: {config.model_type}")
    print(f"    - Device: {config.device}")
except Exception as e:
    print(f"  ✗ Config failed: {e}")
    sys.exit(1)

# Test 3: Character Manager
print("\n[3/5] Testing character manager...")
try:
    from character_manager import CharacterManager
    char_mgr = CharacterManager(
        characters_path=config.characters_path,
        target_size=(512, 512),
        auto_crop=True,
        preload_all=True
    )
    print(f"  ✓ Character manager initialized")
    print(f"    - Characters found: {char_mgr.get_character_count()}")

    if char_mgr.get_character_count() > 0:
        character = char_mgr.get_current_character()
        if character is not None:
            print(f"    - Current character: {char_mgr.get_current_character_name()}")
            print(f"    - Shape: {character.shape}, dtype: {character.dtype}")
            print(f"    - Min/Max values: {character.min()}/{character.max()}")
        else:
            print(f"  ⚠ Character loaded but is None!")
    else:
        print(f"  ⚠ No characters found in {config.characters_path}")
except Exception as e:
    print(f"  ✗ Character manager failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Motion Tracker
print("\n[4/5] Testing motion tracker...")
try:
    from motion_tracker import MotionTracker
    tracker = MotionTracker()
    print(f"  ✓ Motion tracker initialized")
    print(f"    - Using legacy API: {tracker.use_legacy_api}")
except Exception as e:
    print(f"  ✗ Motion tracker failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Camera
print("\n[5/5] Testing camera...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"  ✓ Camera 0 opened successfully")
        ret, frame = cap.read()
        if ret:
            print(f"    - Frame shape: {frame.shape}")
            print(f"    - Frame dtype: {frame.dtype}")
        else:
            print(f"  ⚠ Camera opened but cannot read frames!")
        cap.release()
    else:
        print(f"  ✗ Failed to open camera 0")
except Exception as e:
    print(f"  ✗ Camera test failed: {e}")

# Test 6: Display test
print("\n[6/6] Testing OpenCV display...")
try:
    # Create a simple test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (0, 255, 0)  # Green
    cv2.putText(test_img, "TEST IMAGE", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Test Window", test_img)
    print("  ✓ Display test window shown")
    print("    Press any key in the window to continue...")
    cv2.waitKey(2000)  # Wait 2 seconds
    cv2.destroyAllWindows()
    print("  ✓ Display test passed")
except Exception as e:
    print(f"  ✗ Display test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC TEST COMPLETE")
print("="*60)
print("\nIf all tests passed, the application should work.")
print("If display test failed, there may be an issue with your display environment.")

