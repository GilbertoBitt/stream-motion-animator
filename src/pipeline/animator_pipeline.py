"""
Main animation pipeline that coordinates all components
"""
import time
import cv2
import numpy as np
from typing import List, Optional, Dict
import logging
from pathlib import Path

from src.capture import WebcamCapture
from src.tracking import MediaPipeTracker
from src.animation import LivePortraitModel
from src.output import DisplayOutput, SpoutOutput, NDIOutput
from src.utils import Config, PerformanceMetrics, load_image, add_fps_overlay


class AnimatorPipeline:
    """Main pipeline coordinating capture, tracking, animation, and output"""
    
    def __init__(self, config: Config):
        """
        Initialize animator pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.capture: Optional[WebcamCapture] = None
        self.tracker: Optional[MediaPipeTracker] = None
        self.animator: Optional[LivePortraitModel] = None
        self.outputs: List = []
        
        # Source images
        self.source_images: List[np.ndarray] = []
        self.current_source_idx = 0
        
        # Performance metrics
        self.metrics = PerformanceMetrics(
            window_size=100,
            enabled=config.get('metrics.enabled', True)
        )
        
        # Control flags
        self.running = False
        self.paused = False
        self.show_fps = config.get('output.display.show_fps', True)
        self.show_landmarks = config.get('output.display.show_landmarks', False)
        
        # Frame timing
        self.target_fps = config.get('performance.target_fps', 60)
        self.frame_time = 1.0 / self.target_fps
        self.last_metrics_report = time.time()
        self.metrics_report_interval = config.get('metrics.report_interval', 30)
    
    def initialize(self) -> bool:
        """
        Initialize all pipeline components
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing animator pipeline...")
            
            # Initialize capture
            self.logger.info("Initializing video capture...")
            self.capture = WebcamCapture(
                device_id=self.config.get('capture.device_id', 0),
                width=self.config.get('capture.width', 1920),
                height=self.config.get('capture.height', 1080),
                fps=self.config.get('capture.fps', 60),
                backend=self.config.get('capture.backend', 'auto')
            )
            
            if not self.capture.start():
                self.logger.error("Failed to start video capture")
                return False
            
            # Initialize tracker
            self.logger.info("Initializing face tracker...")
            self.tracker = MediaPipeTracker(
                max_num_faces=self.config.get('tracking.max_num_faces', 1),
                min_detection_confidence=self.config.get('tracking.min_detection_confidence', 0.5),
                min_tracking_confidence=self.config.get('tracking.min_tracking_confidence', 0.5),
                refine_landmarks=self.config.get('tracking.refine_landmarks', True)
            )
            
            # Initialize animator
            self.logger.info("Initializing AI animator...")
            self.animator = LivePortraitModel(
                model_path=self.config.get('animation.model_path'),
                gpu_id=self.config.get('animation.gpu_id', 0),
                batch_size=self.config.get('animation.batch_size', 4),
                use_fp16=self.config.get('animation.use_half_precision', True),
                input_size=tuple(self.config.get('animation.input_resolution', [512, 512]))
            )
            
            if not self.animator.load_model():
                self.logger.error("Failed to load animation model")
                return False
            
            # Initialize outputs
            self.logger.info("Initializing outputs...")
            enabled_outputs = self.config.get('output.enabled', ['display'])
            
            for output_type in enabled_outputs:
                if output_type == 'display':
                    output = DisplayOutput(
                        window_name=self.config.get('output.display.window_name', 'AI Motion Animator')
                    )
                elif output_type == 'spout':
                    output = SpoutOutput(
                        sender_name=self.config.get('output.spout.sender_name', 'AI_Avatar'),
                        width=self.config.get('output.spout.resolution', [1920, 1080])[0],
                        height=self.config.get('output.spout.resolution', [1920, 1080])[1]
                    )
                elif output_type == 'ndi':
                    output = NDIOutput(
                        stream_name=self.config.get('output.ndi.stream_name', 'AI_Avatar'),
                        width=self.config.get('output.ndi.resolution', [1920, 1080])[0],
                        height=self.config.get('output.ndi.resolution', [1920, 1080])[1],
                        fps=self.target_fps
                    )
                else:
                    self.logger.warning(f"Unknown output type: {output_type}")
                    continue
                
                if output.start():
                    self.outputs.append(output)
                else:
                    self.logger.warning(f"Failed to start {output_type} output")
            
            if not self.outputs:
                self.logger.error("No outputs initialized")
                return False
            
            self.logger.info("Pipeline initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}", exc_info=True)
            return False
    
    def load_source_images(self, image_paths: List[str]) -> bool:
        """
        Load source images for animation
        
        Args:
            image_paths: List of paths to source images
            
        Returns:
            True if at least one image loaded successfully
        """
        self.source_images.clear()
        
        for path in image_paths:
            try:
                self.logger.info(f"Loading source image: {path}")
                image = load_image(path)
                self.source_images.append(image)
            except Exception as e:
                self.logger.error(f"Failed to load image {path}: {e}")
        
        if not self.source_images:
            self.logger.error("No source images loaded")
            return False
        
        self.logger.info(f"Loaded {len(self.source_images)} source image(s)")
        
        # Set first image as current source
        self.set_source_image(0)
        
        return True
    
    def set_source_image(self, index: int) -> bool:
        """
        Set current source image by index
        
        Args:
            index: Image index
            
        Returns:
            True if image set successfully
        """
        if not (0 <= index < len(self.source_images)):
            self.logger.error(f"Invalid image index: {index}")
            return False
        
        self.current_source_idx = index
        current_image = self.source_images[index]
        
        if self.animator:
            self.animator.set_source_image(current_image)
            self.logger.info(f"Switched to source image {index + 1}/{len(self.source_images)}")
        
        return True
    
    def process_frame(self) -> Optional[np.ndarray]:
        """
        Process single frame through the pipeline
        
        Returns:
            Processed frame or None on error
        """
        try:
            # Capture frame
            t_capture_start = time.time()
            ret, frame = self.capture.read()
            if not ret or frame is None:
                return None
            t_capture_end = time.time()
            self.metrics.record_stage('capture', t_capture_end - t_capture_start)
            
            # Track face
            t_tracking_start = time.time()
            detection = self.tracker.detect(frame)
            t_tracking_end = time.time()
            self.metrics.record_stage('tracking', t_tracking_end - t_tracking_start)
            
            # Animate
            t_animation_start = time.time()
            if detection and self.animator:
                animated_frame = self.animator.animate(
                    None,  # Use cached source image
                    detection
                )
            else:
                # No detection, use source image
                if self.source_images:
                    animated_frame = self.source_images[self.current_source_idx].copy()
                    # Resize to match capture resolution
                    animated_frame = cv2.resize(animated_frame, (frame.shape[1], frame.shape[0]))
                else:
                    animated_frame = frame
            t_animation_end = time.time()
            self.metrics.record_stage('animation', t_animation_end - t_animation_start)
            
            # Add overlays
            if self.show_landmarks and detection:
                animated_frame = self.tracker.draw_landmarks(animated_frame, detection, False)
            
            if self.show_fps:
                animated_frame = add_fps_overlay(animated_frame, self.metrics.get_fps())
            
            return animated_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            return None
    
    def run(self) -> None:
        """Main processing loop"""
        if not self.running:
            self.logger.error("Pipeline not initialized")
            return
        
        self.logger.info("Starting animation loop...")
        self.logger.info("Controls:")
        self.logger.info("  1-9: Switch source image")
        self.logger.info("  L: Load new image")
        self.logger.info("  F: Toggle FPS display")
        self.logger.info("  M: Toggle landmarks display")
        self.logger.info("  P: Pause/Resume")
        self.logger.info("  S: Save current frame")
        self.logger.info("  R: Print performance report")
        self.logger.info("  Q/ESC: Quit")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Start frame timing
                self.metrics.start_frame()
                
                # Process frame
                if not self.paused:
                    output_frame = self.process_frame()
                    
                    if output_frame is not None:
                        # Send to outputs
                        t_output_start = time.time()
                        for output in self.outputs:
                            output.send_frame(output_frame)
                        t_output_end = time.time()
                        self.metrics.record_stage('output', t_output_end - t_output_start)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    if not self._handle_key(key):
                        break  # Exit requested
                
                # Report metrics periodically
                if time.time() - self.last_metrics_report > self.metrics_report_interval:
                    self._report_metrics()
                    self.last_metrics_report = time.time()
                
                # Frame rate limiting
                loop_time = time.time() - loop_start
                if loop_time < self.frame_time:
                    time.sleep(self.frame_time - loop_time)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.logger.info("Animation loop stopped")
    
    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input
        
        Args:
            key: Key code
            
        Returns:
            True to continue, False to exit
        """
        # Quit
        if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
            return False
        
        # Switch source image (1-9)
        if ord('1') <= key <= ord('9'):
            index = key - ord('1')
            if index < len(self.source_images):
                self.set_source_image(index)
        
        # Load new image
        elif key in [ord('l'), ord('L')]:
            self.logger.info("Load image feature not implemented in headless mode")
        
        # Toggle FPS
        elif key in [ord('f'), ord('F')]:
            self.show_fps = not self.show_fps
            self.logger.info(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
        
        # Toggle landmarks
        elif key in [ord('m'), ord('M')]:
            self.show_landmarks = not self.show_landmarks
            self.logger.info(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
        
        # Pause/Resume
        elif key in [ord('p'), ord('P')]:
            self.paused = not self.paused
            self.logger.info(f"Pipeline: {'PAUSED' if self.paused else 'RUNNING'}")
        
        # Save frame
        elif key in [ord('s'), ord('S')]:
            self.logger.info("Save frame feature not fully implemented")
        
        # Print report
        elif key in [ord('r'), ord('R')]:
            self.metrics.print_summary()
            if self.tracker:
                self.logger.info(f"Tracker stats: {self.tracker.get_stats()}")
        
        return True
    
    def _report_metrics(self) -> None:
        """Report performance metrics"""
        summary = self.metrics.get_summary()
        self.logger.info(
            f"Performance: {summary['fps']:.1f} FPS, "
            f"{summary['latency_ms']:.1f}ms latency, "
            f"{summary['dropped_frames']} dropped"
        )
        
        # Save to file if enabled
        if self.config.get('metrics.save_to_file', False):
            metrics_file = self.config.get('metrics.metrics_file', 'logs/metrics.csv')
            self.metrics.save_to_csv(metrics_file)
    
    def start(self) -> bool:
        """
        Start the pipeline
        
        Returns:
            True if started successfully
        """
        if not self.initialize():
            return False
        
        self.running = True
        return True
    
    def stop(self) -> None:
        """Stop the pipeline and cleanup resources"""
        self.logger.info("Stopping pipeline...")
        self.running = False
        
        # Print final metrics
        self.metrics.print_summary()
        
        # Stop outputs
        for output in self.outputs:
            output.stop()
        self.outputs.clear()
        
        # Stop capture
        if self.capture:
            self.capture.stop()
        
        # Cleanup tracker
        if self.tracker:
            self.logger.info(f"Tracker final stats: {self.tracker.get_stats()}")
        
        # Unload animator
        if self.animator:
            self.animator.unload_model()
        
        self.logger.info("Pipeline stopped")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
