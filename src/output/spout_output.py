"""
Spout output handler (Windows only)

Spout is a video sharing framework for Windows
https://spout.zeal.co/
"""
import numpy as np
from typing import Tuple
import logging
import sys

from .base_output import BaseOutput

# Try to import Spout
SPOUT_AVAILABLE = False
if sys.platform == 'win32':
    try:
        import SpoutGL
        SPOUT_AVAILABLE = True
    except ImportError:
        pass


class SpoutOutput(BaseOutput):
    """
    Spout output for sharing video with OBS and other applications
    
    Note: Requires SpoutGL library on Windows
    Install: pip install SpoutGL
    """
    
    def __init__(
        self,
        sender_name: str = "AI_Avatar",
        width: int = 1920,
        height: int = 1080
    ):
        """
        Initialize Spout output
        
        Args:
            sender_name: Spout sender name (visible in receivers)
            width: Output width
            height: Output height
        """
        super().__init__(sender_name)
        
        if not SPOUT_AVAILABLE:
            self.logger.warning(
                "Spout is not available. This requires Windows and SpoutGL library. "
                "Install with: pip install SpoutGL"
            )
        
        self.sender_name = sender_name
        self.width = width
        self.height = height
        self.sender = None
    
    def start(self) -> bool:
        """Start Spout sender"""
        if not SPOUT_AVAILABLE:
            self.logger.error("Spout is not available on this platform")
            return False
        
        try:
            # Create Spout sender
            self.sender = SpoutGL.SpoutSender()
            self.sender.setSenderName(self.sender_name)
            
            self.initialized = True
            self.logger.info(
                f"Spout sender started: {self.sender_name} ({self.width}x{self.height})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Spout sender: {e}")
            return False
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send frame via Spout
        
        Args:
            frame: Frame to send (BGR format)
            
        Returns:
            True if frame sent successfully
        """
        if not self.initialized or self.sender is None:
            return False
        
        try:
            # Resize if necessary
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert BGR to RGB (Spout expects RGB)
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Send frame
            self.sender.sendImage(rgb_frame, self.width, self.height, 0)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Spout frame: {e}")
            return False
    
    def stop(self) -> None:
        """Stop Spout sender"""
        if self.initialized and self.sender is not None:
            try:
                self.sender.releaseSender()
            except Exception as e:
                self.logger.error(f"Error releasing Spout sender: {e}")
            
            self.sender = None
            self.initialized = False
            self.logger.info(f"Spout sender stopped. Frames sent: {self.frame_count}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
