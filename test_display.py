"""Test script to check if display is working."""
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import load_config
from character_manager import CharacterManager
from motion_tracker import MotionTracker

def main():
    print("Loading configuration...")
    config = load_config()

    print("\nInitializing character manager...")
    character_manager = CharacterManager(
        characters_path=config.characters_path,
        target_size=(512, 512),
        auto_crop=True,
        preload_all=True
    )

    print(f"Characters found: {character_manager.get_character_count()}")

    if character_manager.get_character_count() == 0:
        print("ERROR: No characters found!")
        return

    print(f"Current character: {character_manager.get_current_character_name()}")

    # Get current character
    character = character_manager.get_current_character()
    if character is None:
        print("ERROR: Failed to load character!")
        return

    print(f"Character shape: {character.shape}, dtype: {character.dtype}")

    # Initialize motion tracker
    print("\nInitializing motion tracker...")
    try:
        motion_tracker = MotionTracker()
        print("Motion tracker initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize motion tracker: {e}")
        return

    # Open webcam
    print("\nOpening camera 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Failed to open camera!")
        return

    print("Camera opened successfully")

    # Main loop
    print("\nStarting display loop (press 'q' to quit)...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame!")
            break

        frame_count += 1

        # Process frame
        landmarks = motion_tracker.process_frame(frame)

        # Convert character to BGR for display
        if character.shape[2] == 4:
            # Create white background
            h, w = character.shape[:2]
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255

            # Alpha blend
            alpha = character[:, :, 3:4] / 255.0
            rgb = character[:, :, :3]
            display_char = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        else:
            display_char = character[:, :, :3]

        # Show character
        cv2.imshow("Character", display_char)

        # Show webcam
        cv2.imshow("Webcam", frame)

        # Show stats every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Webcam {frame.shape}, Character {character.shape}, Landmarks: {landmarks is not None}")

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    motion_tracker.cleanup()
    print("Test complete!")

if __name__ == "__main__":
    main()

