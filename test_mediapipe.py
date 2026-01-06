"""Quick test to verify MediaPipe imports work correctly."""

import sys
print("Python version:", sys.version)

print("\n1. Testing basic mediapipe import...")
try:
    import mediapipe as mp
    print(f"✓ MediaPipe imported (version {mp.__version__})")
except Exception as e:
    print(f"✗ Failed to import mediapipe: {e}")
    sys.exit(1)

print("\n2. Testing solutions attribute...")
if hasattr(mp, 'solutions'):
    print("✓ mp.solutions exists")
    mp_face_mesh = mp.solutions.face_mesh
else:
    print("✗ mp.solutions does not exist, trying alternative import...")
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        print("✓ Imported via mediapipe.python.solutions")
    except ImportError:
        try:
            import mediapipe.python.solutions.face_mesh as mp_face_mesh
            print("✓ Imported via direct module path")
        except ImportError as e:
            print(f"✗ All import methods failed: {e}")
            sys.exit(1)

print("\n3. Testing FaceMesh initialization...")
try:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("✓ FaceMesh initialized successfully")
    face_mesh.close()
    print("✓ FaceMesh closed successfully")
except Exception as e:
    print(f"✗ Failed to initialize FaceMesh: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓✓✓ All MediaPipe tests passed! ✓✓✓")

