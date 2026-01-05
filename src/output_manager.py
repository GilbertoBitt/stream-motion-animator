"""
Output manager for Spout and NDI streaming.
Handles sending rendered frames to external applications.
"""

import numpy as np
import cv2
from typing import Optional


class OutputManager:
    """Manages output streams (Spout and NDI)."""
    
    def __init__(self, config):
        """
        Initialize output manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        self.width = config.get('output.width', 1920)
        self.height = config.get('output.height', 1080)
        
        # Initialize Spout
        self.spout_enabled = config.get('output.enable_spout', False)
        self.spout_sender = None
        
        if self.spout_enabled:
            self.spout_sender = self._init_spout()
        
        # Initialize NDI
        self.ndi_enabled = config.get('output.enable_ndi', False)
        self.ndi_sender = None
        
        if self.ndi_enabled:
            self.ndi_sender = self._init_ndi()
        
        # Preview window
        self.preview_enabled = True
    
    def _init_spout(self) -> Optional[object]:
        """Initialize Spout sender."""
        try:
            import SpoutGL
            spout_name = self.config.get('output.spout_name', 'StreamMotionAnimator')
            sender = SpoutGL.SpoutSender()
            sender.setSenderName(spout_name)
            print(f"Spout sender initialized: {spout_name}")
            return sender
        except ImportError:
            print("Warning: SpoutGL not available. Install with: pip install SpoutGL")
            return None
        except Exception as e:
            print(f"Warning: Could not initialize Spout: {e}")
            return None
    
    def _init_ndi(self) -> Optional[object]:
        """Initialize NDI sender."""
        try:
            import NDIlib as ndi
            ndi_name = self.config.get('output.ndi_name', 'Stream Motion Animator')
            
            if not ndi.initialize():
                print("Warning: Could not initialize NDI")
                return None
            
            # Create NDI sender
            send_settings = ndi.SendCreate()
            send_settings.ndi_name = ndi_name
            sender = ndi.send_create(send_settings)
            
            if sender is None:
                print("Warning: Could not create NDI sender")
                return None
            
            print(f"NDI sender initialized: {ndi_name}")
            return sender
        except ImportError:
            print("Warning: NDIlib not available. Install with: pip install ndi-python")
            return None
        except Exception as e:
            print(f"Warning: Could not initialize NDI: {e}")
            return None
    
    def send_frame(self, frame: np.ndarray):
        """
        Send frame to all enabled outputs.
        
        Args:
            frame: Frame to send (BGRA format expected)
        """
        # Ensure frame is correct size
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Send to Spout
        if self.spout_sender:
            self._send_spout(frame)
        
        # Send to NDI
        if self.ndi_sender:
            self._send_ndi(frame)
        
        # Show preview window
        if self.preview_enabled:
            self._show_preview(frame)
    
    def _send_spout(self, frame: np.ndarray):
        """Send frame via Spout."""
        try:
            # Spout expects RGBA format
            if frame.shape[2] == 4:  # BGRA
                rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            else:  # BGR
                rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            
            # Send frame
            self.spout_sender.sendImage(
                rgba_frame,
                self.width,
                self.height,
                gl_format=0x1908  # GL_RGBA
            )
        except Exception as e:
            print(f"Error sending Spout frame: {e}")
    
    def _send_ndi(self, frame: np.ndarray):
        """Send frame via NDI."""
        try:
            import NDIlib as ndi
            
            # NDI expects BGRA or UYVY format
            # We'll use BGRA
            if frame.shape[2] == 3:  # BGR
                bgra_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            else:  # Already BGRA
                bgra_frame = frame
            
            # Create video frame
            video_frame = ndi.VideoFrameV2()
            video_frame.data = bgra_frame
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
            video_frame.xres = self.width
            video_frame.yres = self.height
            video_frame.frame_rate_N = 30000
            video_frame.frame_rate_D = 1001
            
            # Send frame
            ndi.send_send_video_v2(self.ndi_sender, video_frame)
        except Exception as e:
            print(f"Error sending NDI frame: {e}")
    
    def _show_preview(self, frame: np.ndarray):
        """Show preview window."""
        try:
            # Convert BGRA to BGR for display if needed
            if frame.shape[2] == 4:
                # Composite alpha channel onto white background for preview
                bgr = frame[:, :, :3]
                alpha = frame[:, :, 3:4] / 255.0
                white_bg = np.ones_like(bgr) * 255
                preview_frame = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            else:
                preview_frame = frame
            
            # Resize for display if too large
            display_height = 720
            if preview_frame.shape[0] > display_height:
                scale = display_height / preview_frame.shape[0]
                new_width = int(preview_frame.shape[1] * scale)
                preview_frame = cv2.resize(preview_frame, (new_width, display_height))
            
            cv2.imshow('Stream Motion Animator', preview_frame)
        except Exception as e:
            print(f"Error showing preview: {e}")
    
    def close(self):
        """Cleanup resources."""
        if self.spout_sender:
            try:
                self.spout_sender.releaseSender()
            except:
                pass
        
        if self.ndi_sender:
            try:
                import NDIlib as ndi
                ndi.send_destroy(self.ndi_sender)
                ndi.destroy()
            except:
                pass
        
        cv2.destroyAllWindows()
