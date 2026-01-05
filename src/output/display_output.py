"""
Display output handler
Shows frames in an OpenCV window
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

from .base_output import BaseOutput


class DisplayOutput(BaseOutput):
    """Display output using OpenCV window"""
    
    def __init__(
        self,
        window_name: str = "AI Motion Animator",
        width: Optional[int] = None,
        height: Optional[int] = None
    ):
        """
        Initialize display output
        
        Args:
            window_name: Name of the display window
            width: Optional display width (None = auto)
            height: Optional display height (None = auto)
        """
        super().__init__(window_name)
        self.window_name = window_name
        self.display_width = width
        self.display_height = height
    
    def start(self) -> bool:
        """Start display output"""
        try:
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            # Set window size if specified
            if self.display_width and self.display_height:
                cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            
            self.initialized = True
            self.logger.info(f"Display output started: {self.window_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start display output: {e}")
            return False
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Display frame in window
        
        Args:
            frame: Frame to display (BGR format)
            
        Returns:
            True if frame displayed successfully
        """
        if not self.initialized:
            return False
        
        try:
            # Resize if dimensions specified
            if self.display_width and self.display_height:
                frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            cv2.imshow(self.window_name, frame)
            self.frame_count += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error displaying frame: {e}")
            return False
    
    def stop(self) -> None:
        """Stop display output"""
        if self.initialized:
            cv2.destroyWindow(self.window_name)
            self.initialized = False
            self.logger.info(f"Display output stopped. Frames displayed: {self.frame_count}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
