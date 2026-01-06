"""
AI Animator - Main inference engine.

Coordinates the AI model to animate character images with webcam input.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List

from models.model_loader import ModelLoader
from models.base_model import BaseAnimationModel
from motion_tracker import FacialLandmarks

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
        self.custom_animator = None  # For custom ONNX character model
        self.onnx_model = None  # For generic ONNX model
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
            
            # Handle custom_onnx model type
            if self.model_type == 'custom_onnx':
                logger.info("Loading custom ONNX character model...")
                from pathlib import Path
                from custom_character_animator import CustomCharacterAnimator

                # Create custom animator
                self.custom_animator = CustomCharacterAnimator("Test")

                # Load character model
                if not self.custom_animator.load_character_model():
                    logger.error("Failed to load custom character model")
                    logger.info("Falling back to ONNX model")
                    self.model_type = 'onnx'
                else:
                    # Load landmark detector
                    landmark_path = Path("models/liveportrait/landmark.onnx")
                    if not self.custom_animator.load_landmark_detector(landmark_path):
                        logger.error("Failed to load landmark detector")
                        logger.info("Falling back to mock model")
                        self.model_type = 'mock'
                    else:
                        logger.info("Custom ONNX character model loaded successfully")
                        self.is_initialized = True
                        return True

            # Handle onnx model type
            if self.model_type == 'onnx':
                logger.info("Loading ONNX model...")
                from pathlib import Path
                from onnx_liveportrait import ONNXLivePortrait

                landmark_path = Path("models/liveportrait/landmark.onnx")
                if not landmark_path.exists():
                    logger.error("landmark.onnx not found")
                    logger.info("Falling back to mock model")
                    self.model_type = 'mock'
                else:
                    self.onnx_model = ONNXLivePortrait(Path("models/liveportrait"), device=self.device)
                    if self.onnx_model.load_models():
                        logger.info("ONNX model loaded successfully")
                        self.is_initialized = True
                        return True
                    else:
                        logger.error("Failed to load ONNX model")
                        logger.info("Falling back to mock model")
                        self.model_type = 'mock'

            # Load regular model (or mock)
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
            import traceback
            traceback.print_exc()
            return False
    
    def animate_frame(
        self,
        character_image: np.ndarray,
        webcam_frame: np.ndarray,
        landmarks: Optional[FacialLandmarks] = None,
        preprocessed_data: Optional[Dict] = None,
        reference_images: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Animate character with webcam input.
        
        Args:
            character_image: Primary character image (RGBA)
            webcam_frame: Webcam frame (BGR)
            landmarks: Optional facial landmarks for optimization
            preprocessed_data: Optional preprocessed character data for faster inference
            reference_images: Optional list of additional reference images for better quality

        Returns:
            Animated frame (RGBA)
        """
        if not self.is_initialized:
            logger.warning("Animator not initialized, returning original character")
            return character_image

        try:
            # Use custom ONNX animator if available
            if hasattr(self, 'custom_animator') and self.custom_animator:
                return self.custom_animator.animate_character(
                    character_image,
                    webcam_frame,
                    frame_index=0
                )

            # Use ONNX model if available
            if hasattr(self, 'onnx_model') and self.onnx_model:
                return self.onnx_model.animate_character(
                    character_image,
                    webcam_frame
                )

            # Use standard model
            if self.model:
                # Convert landmarks dict to appropriate format if needed
                landmarks_dict = None
                if landmarks:
                    landmarks_dict = {
                        'head_rotation': (landmarks.pitch, landmarks.yaw, landmarks.roll),
                        'mouth_open': landmarks.mouth_open_ratio,
                        'eye_openness': (landmarks.left_eye_openness, landmarks.right_eye_openness)
                    }

                return self.model.animate(
                    character_image,
                    webcam_frame,
                    landmarks=landmarks_dict,
                    character_tensor=preprocessed_data['tensor'] if preprocessed_data else None,
                    reference_images=reference_images
                )

            return character_image

        except Exception as e:
            logger.error(f"Animation failed: {e}")
            return character_image
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
            
            # Use preprocessed data if available (faster inference)
            if preprocessed_data is not None and 'tensor' in preprocessed_data:
                # Run inference with preprocessed tensor
                animated_frame = self.model.animate(
                    character_image,
                    webcam_frame,
                    landmarks_dict,
                    character_tensor=preprocessed_data['tensor'],
                    reference_images=reference_images
                )
            else:
                # Run inference with regular image
                animated_frame = self.model.animate(
                    character_image,
                    webcam_frame,
                    landmarks_dict,
                    reference_images=reference_images
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
        
        if self.optimizer is not None:
            self.optimizer.cleanup()

        self.frame_cache.clear()
        self.is_initialized = False
        logger.info("AI animator cleaned up")
    
    def __del__(self):
        """Destructor."""
        self.cleanup()
