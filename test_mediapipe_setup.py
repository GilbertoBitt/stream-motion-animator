"""
Test script to verify MediaPipe installation and model download.
"""
import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print()

print("Testing MediaPipe import...")
try:
    import mediapipe as mp
    print(f"✓ MediaPipe {mp.__version__} imported successfully")
    print(f"  Has 'solutions' attribute: {hasattr(mp, 'solutions')}")

    if hasattr(mp, 'solutions'):
        print("  → Using legacy solutions API (0.10.9-0.10.14)")
        print(f"  Has 'face_mesh': {hasattr(mp.solutions, 'face_mesh')}")
    else:
        print("  → Using Tasks API (0.10.30+)")

        # Test Tasks API import
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            print("  ✓ Tasks API imports successful")
        except ImportError as e:
            print(f"  ✗ Tasks API import failed: {e}")

except ImportError as e:
    print(f"✗ MediaPipe import failed: {e}")
    print("\nPlease install MediaPipe:")
    print("  pip install mediapipe>=0.10.30 --user")
    sys.exit(1)

print()
print("Testing motion_tracker import...")
try:
    sys.path.insert(0, 'src')
    from motion_tracker import MotionTracker, MEDIAPIPE_INITIALIZED, USE_LEGACY_API
    print("✓ motion_tracker imported successfully")
    print(f"  MediaPipe initialized: {MEDIAPIPE_INITIALIZED}")
    print(f"  Using legacy API: {USE_LEGACY_API}")

    print()
    print("Testing MotionTracker initialization...")
    tracker = MotionTracker()
    print("✓ MotionTracker initialized successfully!")
    print(f"  Face mesh: {tracker.face_mesh is not None}")
    print(f"  Face landmarker: {tracker.face_landmarker is not None}")

    del tracker
    print()
    print("=" * 60)
    print("SUCCESS! Everything is working correctly.")
    print("=" * 60)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

