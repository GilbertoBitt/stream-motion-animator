"""
Main application for Stream Motion Animator.

Coordinates webcam capture, tracking, AI inference, and output.
"""

import cv2
import numpy as np
import logging
import time
import threading
import queue
from typing import Optional
from pynput import keyboard

from config import load_config
from motion_tracker import MotionTracker, FacialLandmarks
from character_manager import CharacterManager
from ai_animator import AIAnimator
from output_manager import OutputManager
from performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamMotionAnimator:
    """Main application class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize application.
        
        Args:
            config_path: Path to config file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.motion_tracker: Optional[MotionTracker] = None
        self.character_manager: Optional[CharacterManager] = None
        self.ai_animator: Optional[AIAnimator] = None
        self.output_manager: Optional[OutputManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Webcam
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Pipeline queues for async processing
        self.capture_queue = queue.Queue(maxsize=2)
        self.tracking_queue = queue.Queue(maxsize=2)
        self.inference_queue = queue.Queue(maxsize=2)
        
        # Control flags
        self.running_event = threading.Event()
        self.show_stats = True
        self.async_pipeline = self.config.async_pipeline
        
        # Threads
        self.threads = []
        
        logger.info("Stream Motion Animator initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing components...")
            
            # Initialize webcam
            self.cap = cv2.VideoCapture(self.config.video_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.video_fps)
            
            if not self.cap.isOpened():
                logger.error("Failed to open webcam")
                return False
            
            logger.info(f"Webcam opened: {self.config.video_width}x{self.config.video_height} @ {self.config.video_fps}fps")
            
            # Initialize motion tracker
            self.motion_tracker = MotionTracker(
                min_detection_confidence=self.config.get('tracking.min_detection_confidence', 0.7),
                min_tracking_confidence=self.config.get('tracking.min_tracking_confidence', 0.5),
                smoothing=self.config.get('tracking.smoothing', 0.5)
            )
            logger.info("Motion tracker initialized")
            
            # Initialize character manager
            self.character_manager = CharacterManager(
                characters_path=self.config.characters_path,
                target_size=tuple(self.config.get('character.target_size', [512, 512])),
                auto_crop=self.config.get('character.auto_crop', True),
                preload_all=self.config.get('character.preload_all', True)
            )
            logger.info(f"Character manager initialized: {self.character_manager.get_character_count()} characters")
            
            # Initialize AI animator
            self.ai_animator = AIAnimator(
                model_type=self.config.model_type,
                model_path=self.config.model_path,
                device=self.config.device,
                fp16=self.config.use_fp16,
                use_tensorrt=self.config.get('ai_model.use_tensorrt', False),
                warmup_frames=self.config.get('ai_model.warmup_frames', 10)
            )
            
            if not self.ai_animator.initialize():
                logger.error("Failed to initialize AI animator")
                return False
            
            logger.info("AI animator initialized")
            
            # Initialize output manager
            self.output_manager = OutputManager(
                width=self.config.output_width,
                height=self.config.output_height,
                spout_enabled=self.config.spout_enabled,
                spout_name=self.config.get('output.spout_name', 'StreamMotionAnimator'),
                ndi_enabled=self.config.ndi_enabled,
                ndi_name=self.config.get('output.ndi_name', 'Stream Motion Animator')
            )
            logger.info("Output manager initialized")
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                window_size=30,
                enable_gpu=(self.config.device == 'cuda')
            )
            logger.info("Performance monitor initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def capture_thread(self) -> None:
        """Thread for capturing frames from webcam."""
        while self.running_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.capture_queue.put(frame, timeout=0.1)
                    self.performance_monitor.tick_capture()
                except queue.Full:
                    pass  # Skip frame if queue is full
    
    def tracking_thread(self) -> None:
        """Thread for face tracking."""
        while self.running_event.is_set():
            try:
                frame = self.capture_queue.get(timeout=0.1)
                landmarks = self.motion_tracker.process_frame(frame)
                self.tracking_queue.put((frame, landmarks), timeout=0.1)
                self.performance_monitor.tick_tracking()
            except (queue.Empty, queue.Full):
                pass  # Skip on queue timeout
    
    def inference_thread(self) -> None:
        """Thread for AI inference."""
        while self.running_event.is_set():
            try:
                frame, landmarks = self.tracking_queue.get(timeout=0.1)
                character = self.character_manager.get_current_character()
                
                if character is not None:
                    animated = self.ai_animator.animate_frame(character, frame, landmarks)
                    self.inference_queue.put(animated, timeout=0.1)
                    self.performance_monitor.tick_inference()
            except (queue.Empty, queue.Full):
                pass  # Skip on queue timeout
    
    def output_thread(self) -> None:
        """Thread for output streaming."""
        while self.running_event.is_set():
            try:
                animated_frame = self.inference_queue.get(timeout=0.1)
                self.output_manager.send_frame(animated_frame)
                self.performance_monitor.tick_output()
            except queue.Empty:
                pass  # No frame available yet
    
    def run_sync(self) -> None:
        """Run in synchronous mode (simpler, for debugging)."""
        logger.info("Running in synchronous mode")
        
        while self.running_event.is_set():
            # Capture
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            self.performance_monitor.tick_capture()
            
            # Track
            landmarks = self.motion_tracker.process_frame(frame)
            self.performance_monitor.tick_tracking()
            
            # Get character
            character = self.character_manager.get_current_character()
            if character is None:
                logger.warning("No character available")
                time.sleep(0.1)
                continue
            
            # Animate
            animated = self.ai_animator.animate_frame(character, frame, landmarks)
            self.performance_monitor.tick_inference()
            
            # Output
            self.output_manager.send_frame(animated)
            self.performance_monitor.tick_output()
            
            # Total
            self.performance_monitor.tick_total()
            
            # Show stats periodically
            if self.show_stats:
                self._print_stats_periodic()
            
            # Show preview window
            self._show_preview(animated, landmarks)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    def run_async(self) -> None:
        """Run in asynchronous mode (better performance)."""
        logger.info("Running in asynchronous mode")
        
        # Start threads
        self.threads = [
            threading.Thread(target=self.capture_thread, name="Capture"),
            threading.Thread(target=self.tracking_thread, name="Tracking"),
            threading.Thread(target=self.inference_thread, name="Inference"),
            threading.Thread(target=self.output_thread, name="Output")
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        logger.info("All threads started")
        
        # Main loop for monitoring and display
        last_stats_time = time.time()
        
        while self.running_event.is_set():
            # Get latest frame for preview
            try:
                animated = self.inference_queue.get(timeout=0.1)
                self.performance_monitor.tick_total()
                
                # Show preview
                self._show_preview(animated, None)
                
            except queue.Empty:
                pass  # No frame available yet
            
            # Print stats periodically
            if self.show_stats and time.time() - last_stats_time > 2.0:
                self._print_stats()
                last_stats_time = time.time()
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Wait for threads to finish
        logger.info("Waiting for threads to finish...")
        for thread in self.threads:
            thread.join(timeout=1.0)
    
    def _show_preview(self, frame: np.ndarray, landmarks: Optional[FacialLandmarks]) -> None:
        """Show preview window with animated frame."""
        if frame is None:
            return
        
        # Convert RGBA to BGR for display
        if frame.shape[2] == 4:
            # Create checkered background for transparency
            h, w = frame.shape[:2]
            checker = np.zeros((h, w, 3), dtype=np.uint8)
            checker[::20, ::20] = 128
            checker[10::20, 10::20] = 128
            
            # Alpha blend
            alpha = frame[:, :, 3:4] / 255.0
            rgb = frame[:, :, :3]
            display = (rgb * alpha + checker * (1 - alpha)).astype(np.uint8)
        else:
            display = frame
        
        # Add stats overlay
        if self.show_stats:
            stats = self.performance_monitor.get_stats()
            cv2.putText(display, f"FPS: {stats.total_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Character: {self.character_manager.get_current_character_name()}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Stream Motion Animator", display)
    
    def _print_stats(self) -> None:
        """Print performance statistics."""
        stats = self.performance_monitor.get_stats(
            max_inference_time_ms=self.config.get('performance.max_inference_time_ms', 15)
        )
        
        print("\n" + "="*60)
        print(f"FPS: Capture={stats.capture_fps:.1f} | Tracking={stats.tracking_fps:.1f} | "
              f"Inference={stats.inference_fps:.1f} | Output={stats.output_fps:.1f} | "
              f"Total={stats.total_fps:.1f}")
        print(f"Inference Time: {stats.inference_time_ms:.1f}ms")
        print(f"Character: {self.character_manager.get_current_character_name()} "
              f"({self.character_manager.current_character_index + 1}/"
              f"{self.character_manager.get_character_count()})")
        print("="*60)
    
    def _print_stats_periodic(self) -> None:
        """Print stats less frequently in sync mode."""
        if not hasattr(self, '_last_stats_print'):
            self._last_stats_print = 0
        
        if time.time() - self._last_stats_print > 2.0:
            self._print_stats()
            self._last_stats_print = time.time()
    
    def on_key_press(self, key) -> None:
        """Handle keyboard events."""
        try:
            # Number keys for character selection
            if hasattr(key, 'char') and key.char is not None:
                if key.char.isdigit():
                    index = int(key.char) - 1
                    if index >= 0:
                        self.character_manager.switch_character(index)
                elif key.char == 'q':
                    logger.info("Quit requested")
                    self.running_event.clear()
                elif key.char == 'r':
                    logger.info("Reloading characters")
                    self.character_manager.reload_characters()
                elif key.char == 't':
                    self.show_stats = not self.show_stats
                    logger.info(f"Stats display: {self.show_stats}")
                elif key.char == 's':
                    self.output_manager.toggle_spout()
                elif key.char == 'n':
                    self.output_manager.toggle_ndi()
            
            # Arrow keys
            if key == keyboard.Key.right:
                self.character_manager.next_character()
            elif key == keyboard.Key.left:
                self.character_manager.prev_character()
                
        except AttributeError:
            pass  # Ignore keys without char attribute (e.g., special keys)
    
    def run(self) -> None:
        """Main run method."""
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return
        
        # Setup keyboard listener
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()
        
        # Print initial info
        print("\n" + "="*60)
        print("STREAM MOTION ANIMATOR")
        print("="*60)
        print(f"Characters: {self.character_manager.get_character_count()}")
        print(f"Current: {self.character_manager.get_current_character_name()}")
        print(f"Mode: {'Async' if self.async_pipeline else 'Sync'}")
        print(f"Device: {self.config.device}")
        print("\nHotkeys:")
        print("  1-9: Switch character")
        print("  Left/Right: Previous/Next character")
        print("  R: Reload characters")
        print("  T: Toggle stats")
        print("  S: Toggle Spout")
        print("  N: Toggle NDI")
        print("  Q: Quit")
        print("="*60 + "\n")
        
        # Start main loop
        self.running_event.set()
        
        try:
            if self.async_pipeline:
                self.run_async()
            else:
                self.run_sync()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
            listener.stop()
    
    def cleanup(self) -> None:
        """Cleanup all resources."""
        logger.info("Cleaning up...")
        
        self.running_event.clear()
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.motion_tracker is not None:
            self.motion_tracker.cleanup()
        
        if self.ai_animator is not None:
            self.ai_animator.cleanup()
        
        if self.output_manager is not None:
            self.output_manager.cleanup()
        
        if self.performance_monitor is not None:
            self.performance_monitor.cleanup()
        
        logger.info("Cleanup complete")


def main():
    """Entry point."""
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    app = StreamMotionAnimator(config_path)
    app.run()


if __name__ == "__main__":
    main()
