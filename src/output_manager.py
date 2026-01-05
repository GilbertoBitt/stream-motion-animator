"""
Output manager for Spout and NDI streaming.

Handles sending animated frames to OBS and other streaming software.
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages Spout and NDI output streams."""
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        spout_enabled: bool = True,
        spout_name: str = "StreamMotionAnimator",
        ndi_enabled: bool = True,
        ndi_name: str = "Stream Motion Animator"
    ):
        """
        Initialize output manager.
        
        Args:
            width: Output width
            height: Output height
            spout_enabled: Enable Spout output
            spout_name: Spout sender name
            ndi_enabled: Enable NDI output
            ndi_name: NDI sender name
        """
        self.width = width
        self.height = height
        self.spout_enabled = spout_enabled
        self.spout_name = spout_name
        self.ndi_enabled = ndi_enabled
        self.ndi_name = ndi_name
        
        # Spout sender
        self.spout_sender = None
        if spout_enabled:
            self._init_spout()
        
        # NDI sender
        self.ndi_sender = None
        if ndi_enabled:
            self._init_ndi()
    
    def _init_spout(self) -> None:
        """Initialize Spout sender."""
        try:
            import SpoutGL
            self.spout_sender = SpoutGL.SpoutSender()
            self.spout_sender.setSenderName(self.spout_name)
            logger.info(f"Spout sender initialized: {self.spout_name}")
        except ImportError:
            logger.warning("SpoutGL not available. Install with: pip install SpoutGL")
            self.spout_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Spout: {e}")
            self.spout_enabled = False
    
    def _init_ndi(self) -> None:
        """Initialize NDI sender."""
        try:
            # Try to import NDI library
            # Note: NDI setup is platform-specific and may require additional configuration
            import NDIlib as ndi
            
            if not ndi.initialize():
                logger.warning("Failed to initialize NDI library")
                self.ndi_enabled = False
                return
            
            # Create NDI sender
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = self.ndi_name
            
            self.ndi_sender = ndi.send_create(send_settings)
            
            if self.ndi_sender is None:
                logger.warning("Failed to create NDI sender")
                self.ndi_enabled = False
            else:
                logger.info(f"NDI sender initialized: {self.ndi_name}")
                
        except ImportError:
            logger.warning(
                "NDI library not available. "
                "See docs/INSTALLATION.md for NDI setup instructions."
            )
            self.ndi_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize NDI: {e}")
            self.ndi_enabled = False
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send frame to all enabled outputs.
        
        Args:
            frame: RGBA frame to send
            
        Returns:
            True if at least one output succeeded
        """
        success = False
        
        # Ensure frame is correct size
        if frame.shape[:2] != (self.height, self.width):
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Send to Spout
        if self.spout_enabled and self.spout_sender is not None:
            try:
                # Spout expects RGBA with uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Ensure RGBA format
                if frame.shape[2] == 3:
                    import cv2
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                
                self.spout_sender.sendImage(
                    frame,
                    self.width,
                    self.height,
                    0  # GL_RGBA
                )
                success = True
            except Exception as e:
                logger.error(f"Spout send failed: {e}")
        
        # Send to NDI
        if self.ndi_enabled and self.ndi_sender is not None:
            try:
                import NDIlib as ndi
                
                # Convert to RGBA uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Create NDI video frame
                video_frame = ndi.VideoFrameV2()
                video_frame.data = frame
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
                
                ndi.send_send_video_v2(self.ndi_sender, video_frame)
                success = True
            except Exception as e:
                logger.error(f"NDI send failed: {e}")
        
        return success
    
    def toggle_spout(self) -> bool:
        """
        Toggle Spout output on/off.
        
        Returns:
            New Spout state
        """
        self.spout_enabled = not self.spout_enabled
        
        if self.spout_enabled and self.spout_sender is None:
            self._init_spout()
        
        logger.info(f"Spout output: {'enabled' if self.spout_enabled else 'disabled'}")
        return self.spout_enabled
    
    def toggle_ndi(self) -> bool:
        """
        Toggle NDI output on/off.
        
        Returns:
            New NDI state
        """
        self.ndi_enabled = not self.ndi_enabled
        
        if self.ndi_enabled and self.ndi_sender is None:
            self._init_ndi()
        
        logger.info(f"NDI output: {'enabled' if self.ndi_enabled else 'disabled'}")
        return self.ndi_enabled
    
    def cleanup(self) -> None:
        """Cleanup output resources."""
        if self.spout_sender is not None:
            try:
                self.spout_sender.releaseSender()
            except Exception as e:
                logger.error(f"Error cleaning up Spout: {e}")
        
        if self.ndi_sender is not None:
            try:
                import NDIlib as ndi
                ndi.send_destroy(self.ndi_sender)
                ndi.destroy()
            except Exception as e:
                logger.error(f"Error cleaning up NDI: {e}")
    
    def __del__(self):
        """Destructor to cleanup resources."""
        self.cleanup()
    
    def get_info(self) -> dict:
        """
        Get information about output state.
        
        Returns:
            Dictionary with output info
        """
        return {
            "width": self.width,
            "height": self.height,
            "spout_enabled": self.spout_enabled,
            "spout_name": self.spout_name,
            "ndi_enabled": self.ndi_enabled,
            "ndi_name": self.ndi_name
        }
