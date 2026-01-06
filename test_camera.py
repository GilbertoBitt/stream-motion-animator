"""Test camera feed with character display."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
from config import load_config
from character_manager import CharacterManager
from motion_tracker import MotionTracker

def test_camera(camera_index):
    """Test a specific camera."""
    print(f"\n{'='*60}")
    print(f"Testing Camera {camera_index}")
    print('='*60)
    
    # Load config
    config = load_config()
    
    # Initialize character manager
    print("Loading characters...")
    char_mgr = CharacterManager(
        characters_path=config.characters_path,
        target_size=(512, 512),
        auto_crop=True,
        preload_all=True
    )
    
    if char_mgr.get_character_count() == 0:
        print("ERROR: No characters found!")
        return False
    
    print(f"Loaded {char_mgr.get_character_count()} characters")
    character = char_mgr.get_current_character()
    print(f"Current: {char_mgr.get_current_character_name()}")
    
    # Initialize motion tracker
    print("Initializing motion tracker...")
    tracker = MotionTracker()
    print("Motion tracker ready")
    
    # Open camera
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Failed to open camera {camera_index}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {width}x{height} @ {fps:.0f}fps")
    
    # Test reading frames
    print("Testing frame capture...")
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frames from camera!")
        cap.release()
        return False
    
    print(f"✓ Frame captured: {frame.shape}")
    
    # Main loop
    print("\nStarting preview (press 'q' to quit)...")
    print("Press arrow keys to switch characters")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame")
            break
        
        frame_count += 1
        
        # Process with motion tracker
        landmarks = tracker.process_frame(frame)
        
        # Prepare character for display (with white background)
        if character.shape[2] == 4:
            h, w = character.shape[:2]
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255
            alpha = character[:, :, 3:4] / 255.0
            rgb = character[:, :, :3]
            display_char = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        else:
            display_char = character[:, :, :3]
        
        # Add text overlay to webcam
        display_webcam = frame.copy()
        cv2.putText(display_webcam, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_webcam, f"Landmarks: {'Yes' if landmarks else 'No'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if landmarks else (0, 0, 255), 2)
        
        # Show both windows
        cv2.imshow("Webcam Feed", display_webcam)
        cv2.imshow("Character", display_char)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 81:  # Left arrow
            char_mgr.prev_character()
            character = char_mgr.get_current_character()
            print(f"Switched to: {char_mgr.get_current_character_name()}")
        elif key == 83:  # Right arrow
            char_mgr.next_character()
            character = char_mgr.get_current_character()
            print(f"Switched to: {char_mgr.get_current_character_name()}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    
    print(f"\nTest complete! Processed {frame_count} frames")
    return True

if __name__ == "__main__":
    # Test all available cameras
    cameras_to_test = [0, 1, 2]
    
    if len(sys.argv) > 1:
        # Test specific camera from command line
        camera_idx = int(sys.argv[1])
        test_camera(camera_idx)
    else:
        # Try each camera
        for cam_idx in cameras_to_test:
            print(f"\nTrying camera {cam_idx}...")
            if test_camera(cam_idx):
                print(f"✓ Camera {cam_idx} works!")
                break
            else:
                print(f"✗ Camera {cam_idx} failed")
        else:
            print("\nERROR: No working camera found!")

