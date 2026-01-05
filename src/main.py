"""
Stream Motion Animator - Main Application
Real-time motion tracking and 2D sprite animation for streaming.
"""

import cv2
import time
import sys
import os
from config import Config
from motion_tracker import MotionTracker
from sprite_animator import SpriteAnimator
from output_manager import OutputManager


class StreamMotionAnimator:
    """Main application class."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        print("="*60)
        print("Stream Motion Animator")
        print("="*60)
        
        # Load configuration
        self.config = Config(config_path)
        
        # Initialize video capture
        video_source = self.config.get('video.source', 0)
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")
        
        # Set video properties
        width = self.config.get('video.width', 1280)
        height = self.config.get('video.height', 720)
        fps = self.config.get('video.fps', 30)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        print(f"Video capture initialized: {width}x{height} @ {fps}fps")
        
        # Initialize tracking system
        print("Initializing motion tracking...")
        self.tracker = MotionTracker(self.config)
        
        # Initialize sprite animator
        print("Initializing sprite animator...")
        self.animator = SpriteAnimator(self.config)
        
        # Initialize output manager
        print("Initializing output manager...")
        self.output = OutputManager(self.config)
        
        # Performance tracking
        self.show_fps = self.config.get('performance.show_fps', True)
        self.show_tracking_overlay = self.config.get('performance.show_tracking_overlay', False)
        self.frame_times = []
        self.max_frame_times = 30
        
        # Application state
        self.running = False
        
        print("\nInitialization complete!")
        print("\nControls:")
        print("  Q or ESC - Quit application")
        print("  R - Reload configuration")
        print("  T - Toggle tracking overlay")
        print("  F - Toggle FPS display")
        print("  P - Toggle preview window")
        print("="*60)
    
    def run(self):
        """Run the main application loop."""
        self.running = True
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process tracking
                tracking_data = self.tracker.process_frame(frame)
                
                # Update animation
                self.animator.update(tracking_data)
                
                # Render sprites
                output_frame = self.animator.render()
                
                # Show tracking overlay if enabled
                if self.show_tracking_overlay:
                    overlay_frame = self.tracker.draw_tracking_overlay(frame, tracking_data)
                    cv2.imshow('Tracking Overlay', overlay_frame)
                
                # Add FPS overlay if enabled
                if self.show_fps:
                    output_frame = self._add_fps_overlay(output_frame)
                
                # Send to outputs
                self.output.send_frame(output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
                
                # Track frame time
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                if len(self.frame_times) > self.max_frame_times:
                    self.frame_times.pop(0)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _add_fps_overlay(self, frame):
        """Add FPS counter to frame."""
        if not self.frame_times:
            return frame
        
        # Calculate FPS
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Convert frame to BGR for OpenCV text rendering if needed
        if frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr_frame = frame.copy()
        
        # Add FPS text
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            bgr_frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Convert back to BGRA if needed
        if frame.shape[2] == 4:
            return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
        return bgr_frame
    
    def _handle_keyboard(self, key) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            True to continue, False to exit
        """
        if key == ord('q') or key == 27:  # Q or ESC
            print("\nExiting...")
            return False
        elif key == ord('r'):
            print("\nReloading configuration...")
            self.config.reload()
        elif key == ord('t'):
            self.show_tracking_overlay = not self.show_tracking_overlay
            print(f"\nTracking overlay: {'ON' if self.show_tracking_overlay else 'OFF'}")
            if not self.show_tracking_overlay:
                cv2.destroyWindow('Tracking Overlay')
        elif key == ord('f'):
            self.show_fps = not self.show_fps
            print(f"\nFPS display: {'ON' if self.show_fps else 'OFF'}")
        elif key == ord('p'):
            self.output.preview_enabled = not self.output.preview_enabled
            print(f"\nPreview window: {'ON' if self.output.preview_enabled else 'OFF'}")
            if not self.output.preview_enabled:
                cv2.destroyWindow('Stream Motion Animator')
        
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if hasattr(self, 'tracker'):
            self.tracker.close()
        
        if hasattr(self, 'animator'):
            self.animator.close()
        
        if hasattr(self, 'output'):
            self.output.close()
        
        print("Cleanup complete. Goodbye!")


def main():
    """Main entry point."""
    # Check for config file argument
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return 1
    
    try:
        app = StreamMotionAnimator(config_path)
        app.run()
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
