"""
Video capture module for webcam input
"""
import cv2
import numpy as np
import threading
import queue
from typing import Optional, Tuple
import logging


class WebcamCapture:
    """Webcam video capture with thread-safe frame buffering"""
    
    def __init__(
        self,
        device_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        backend: str = 'auto',
        buffer_size: int = 2
    ):
        """
        Initialize webcam capture
        
        Args:
            device_id: Camera device ID (0 for default)
            width: Capture width
            height: Capture height
            fps: Target frames per second
            backend: Video backend ('auto', 'dshow', 'v4l2')
            buffer_size: Number of frames to buffer
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.backend = backend
        self.buffer_size = buffer_size
        self.max_capture_errors = 10  # Configurable error threshold before stopping
        
        self.logger = logging.getLogger(__name__)
        
        # Video capture object
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.capture_errors = 0
    
    def start(self) -> bool:
        """
        Start video capture
        
        Returns:
            True if capture started successfully
        """
        try:
            # Select backend
            backend_api = cv2.CAP_ANY
            if self.backend == 'dshow':
                backend_api = cv2.CAP_DSHOW
            elif self.backend == 'v4l2':
                backend_api = cv2.CAP_V4L2
            
            # Open video capture
            self.cap = cv2.VideoCapture(self.device_id, backend_api)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera device {self.device_id}")
                return False
            
            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Verify properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps "
                f"(requested: {self.width}x{self.height} @ {self.target_fps}fps)"
            )
            
            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting capture: {e}")
            return False
    
    def _capture_loop(self) -> None:
        """Capture loop running in separate thread"""
        while self.running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.capture_errors += 1
                    if self.capture_errors > self.max_capture_errors:
                        self.logger.error(f"Too many capture errors ({self.max_capture_errors}), stopping")
                        self.running = False
                        break
                    continue
                
                # Reset error counter on successful read
                self.capture_errors = 0
                
                # Try to add frame to queue (non-blocking)
                try:
                    # If queue is full, remove oldest frame
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(frame, block=False)
                    self.frame_count += 1
                    
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                self.capture_errors += 1
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame from buffer
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.running:
            return False, None
        
        try:
            # Get latest frame (with timeout)
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self) -> None:
        """Stop video capture and release resources"""
        self.running = False
        
        # Wait for capture thread to finish
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info(f"Capture stopped. Total frames captured: {self.frame_count}")
    
    def is_running(self) -> bool:
        """Check if capture is running"""
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def get_properties(self) -> dict:
        """Get current capture properties"""
        if self.cap is None or not self.cap.isOpened():
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': self.frame_count,
            'running': self.running
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
