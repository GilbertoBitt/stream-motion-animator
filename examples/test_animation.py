#!/usr/bin/env python3
"""
Example: Test Stream Motion Animator with a static image or generated frames.
This script demonstrates the application pipeline without requiring a webcam.
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from motion_tracker import MotionTracker
from sprite_animator import SpriteAnimator
from output_manager import OutputManager


def generate_test_frame(width=1280, height=720, frame_num=0):
    """Generate a test frame with a gradient and frame number."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient
    for y in range(height):
        frame[y, :, 0] = int(100 + (y / height) * 100)  # Blue
        frame[y, :, 1] = int(150 - (y / height) * 50)   # Green
        frame[y, :, 2] = int(200)                        # Red
    
    # Add frame number
    cv2.putText(
        frame, 
        f"Frame: {frame_num}", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    return frame


def main():
    """Run test animation."""
    print("Stream Motion Animator - Test Mode")
    print("="*60)
    
    # Load configuration
    config = Config()
    
    # Initialize components
    print("Initializing components...")
    tracker = MotionTracker(config)
    animator = SpriteAnimator(config)
    output = OutputManager(config)
    
    print("\nRunning animation test...")
    print("Press 'Q' to quit\n")
    
    frame_num = 0
    
    try:
        while True:
            # Generate test frame
            test_frame = generate_test_frame(frame_num=frame_num)
            
            # Process through pipeline
            tracking_data = tracker.process_frame(test_frame)
            animator.update(tracking_data)
            rendered_frame = animator.render()
            output.send_frame(rendered_frame)
            
            # Check for quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            frame_num += 1
            
            if frame_num % 30 == 0:
                print(f"Processed {frame_num} frames...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("\nCleaning up...")
        tracker.close()
        animator.close()
        output.close()
        print("Done!")


if __name__ == "__main__":
    main()
