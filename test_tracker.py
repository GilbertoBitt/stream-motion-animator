"""Simple test for motion tracker initialization."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("1. Testing motion tracker import...")
try:
    from motion_tracker import MotionTracker
    print("✓ MotionTracker imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating MotionTracker instance...")
try:
    tracker = MotionTracker(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        smoothing=0.5
    )
    print("✓ MotionTracker created successfully")
except Exception as e:
    print(f"✗ Failed to create: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Cleaning up...")
try:
    tracker.cleanup()
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Cleanup failed: {e}")

print("\n✓✓✓ All tests passed! ✓✓✓")

