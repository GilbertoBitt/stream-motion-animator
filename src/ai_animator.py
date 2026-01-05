"""
AI Animator - Main inference engine.

Coordinates the AI model to animate character images with webcam input.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any

from .models.model_loader import ModelLoader
from .models.base_model import BaseAnimationModel
from .motion_tracker import FacialLandmarks

logger = logging.getLogger(__name__)


class AIAnimator:
    """AI-powered character animator."""
    
    def __init__(
        self,
        model_type: str = "liveportrait",
        model_path: str = "models/liveportrait",
        device: str = "cuda",
        fp16: bool = True,
        use_tensorrt: bool = False,
        warmup_frames: int = 10
    ):
        """
        Initialize AI animator.
        
        Args:
            model_type: Type of AI model to use
            model_path: Path to model weights
            device: Device to run on ('cuda' or 'cpu')
            fp16: Use FP16 precision
            use_tensorrt: Use TensorRT acceleration
            warmup_frames: Number of warmup frames
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.fp16 = fp16
        self.use_tensorrt = use_tensorrt
        self.warmup_frames = warmup_frames
        
        self.model: Optional[BaseAnimationModel] = None
        self.is_initialized = False
        
        # Frame cache for optimization
        self.frame_cache = {}
        self.cache_enabled = False
    
    def initialize(self) -> bool:
        """
        Initialize the AI model.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Initializing AI animator...")
            
            # Load model
            self.model = ModelLoader.load_model(
                self.model_type,
                self.model_path,
                self.device,
                fp16=self.fp16,
                use_tensorrt=self.use_tensorrt
            )
            
            if self.model is None:
                logger.error("Failed to load model")
                return False
            
            # Warm up GPU
            if self.warmup_frames > 0:
                logger.info(f"Warming up GPU with {self.warmup_frames} frames...")
                self.model.warmup(self.warmup_frames)
            
            self.is_initialized = True
            logger.info("AI animator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI animator: {e}")
            return False
    
    def animate_frame(
        self,
        character_image: np.ndarray,
        webcam_frame: np.ndarray,
        landmarks: Optional[FacialLandmarks] = None
    ) -> np.ndarray:
        """
        Animate character with webcam input.
        
        Args:
            character_image: Character image (RGBA)
            webcam_frame: Webcam frame (BGR)
            landmarks: Optional facial landmarks for optimization
            
        Returns:
            Animated frame (RGBA)
        """
        if not self.is_initialized or self.model is None:
            logger.warning("Animator not initialized")
            return character_image
        
        try:
            # Prepare landmarks dict for model
            landmarks_dict = None
            if landmarks is not None:
                landmarks_dict = {
                    'landmarks': landmarks.landmarks,
                    'head_rotation': landmarks.head_rotation,
                    'left_eye_state': landmarks.left_eye_state,
                    'right_eye_state': landmarks.right_eye_state,
                    'mouth_open': landmarks.mouth_open,
                    'bbox': landmarks.bbox
                }
            
            # Run inference
            animated_frame = self.model.animate(
                character_image,
                webcam_frame,
                landmarks_dict
            )
            
            return animated_frame
            
        except Exception as e:
            logger.error(f"Animation failed: {e}")
            return character_image
    
    def is_ready(self) -> bool:
        """
        Check if animator is ready.
        
        Returns:
            True if ready to animate
        """
        return self.is_initialized and self.model is not None and self.model.is_loaded()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get animator information.
        
        Returns:
            Dictionary with animator info
        """
        info = {
            "initialized": self.is_initialized,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.device,
            "fp16": self.fp16,
            "tensorrt": self.use_tensorrt
        }
        
        if self.model is not None:
            info["model_info"] = self.model.get_info()
        
        return info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model is not None:
            self.model.cleanup()
        
        self.frame_cache.clear()
        self.is_initialized = False
        logger.info("AI animator cleaned up")
    
    def __del__(self):
        """Destructor."""
        self.cleanup()
