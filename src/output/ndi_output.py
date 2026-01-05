"""
NDI output handler

NDI (Network Device Interface) is a protocol for video over IP
https://www.ndi.tv/
"""
import numpy as np
from typing import Tuple
import logging

from .base_output import BaseOutput

# Try to import NDI
NDI_AVAILABLE = False
try:
    import NDIlib as ndi
    NDI_AVAILABLE = True
except ImportError:
    pass


class NDIOutput(BaseOutput):
    """
    NDI output for streaming video over network
    
    Note: Requires NDI SDK and ndi-python library
    Install NDI SDK from: https://www.ndi.tv/sdk/
    Install ndi-python: pip install ndi-python
    """
    
    def __init__(
        self,
        stream_name: str = "AI_Avatar",
        width: int = 1920,
        height: int = 1080,
        fps: int = 60
    ):
        """
        Initialize NDI output
        
        Args:
            stream_name: NDI stream name (visible on network)
            width: Output width
            height: Output height
            fps: Target frames per second
        """
        super().__init__(stream_name)
        
        if not NDI_AVAILABLE:
            self.logger.warning(
                "NDI is not available. This requires NDI SDK and ndi-python library. "
                "Install NDI SDK from https://www.ndi.tv/sdk/ then: pip install ndi-python"
            )
        
        self.stream_name = stream_name
        self.width = width
        self.height = height
        self.fps = fps
        self.ndi_send = None
    
    def start(self) -> bool:
        """Start NDI sender"""
        if not NDI_AVAILABLE:
            self.logger.error("NDI is not available on this system")
            return False
        
        try:
            # Initialize NDI
            if not ndi.initialize():
                self.logger.error("Failed to initialize NDI")
                return False
            
            # Create NDI send instance
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = self.stream_name
            send_settings.clock_video = True
            send_settings.clock_audio = False
            
            self.ndi_send = ndi.send_create(send_settings)
            
            if self.ndi_send is None:
                self.logger.error("Failed to create NDI sender")
                return False
            
            self.initialized = True
            self.logger.info(
                f"NDI sender started: {self.stream_name} ({self.width}x{self.height} @ {self.fps}fps)"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start NDI sender: {e}")
            return False
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send frame via NDI
        
        Args:
            frame: Frame to send (BGR format)
            
        Returns:
            True if frame sent successfully
        """
        if not self.initialized or self.ndi_send is None:
            return False
        
        try:
            # Resize if necessary
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert BGR to RGBA (NDI expects BGRA or RGBA)
            import cv2
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            
            # Create NDI video frame
            video_frame = ndi.VideoFrameV2()
            video_frame.xres = self.width
            video_frame.yres = self.height
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
            video_frame.frame_rate_N = self.fps
            video_frame.frame_rate_D = 1
            video_frame.data = rgba_frame
            video_frame.line_stride_in_bytes = self.width * 4
            
            # Send frame
            ndi.send_send_video_v2(self.ndi_send, video_frame)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending NDI frame: {e}")
            return False
    
    def stop(self) -> None:
        """Stop NDI sender"""
        if self.initialized and self.ndi_send is not None:
            try:
                ndi.send_destroy(self.ndi_send)
                ndi.destroy()
            except Exception as e:
                self.logger.error(f"Error stopping NDI sender: {e}")
            
            self.ndi_send = None
            self.initialized = False
            self.logger.info(f"NDI sender stopped. Frames sent: {self.frame_count}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
