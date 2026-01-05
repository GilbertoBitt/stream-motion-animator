"""
Base model interface for AI animation models.

Provides abstract interface that all models must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class BaseAnimationModel(ABC):
    """Abstract base class for animation models."""
    
    @abstractmethod
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize the model.
        
        Args:
            model_path: Path to model weights
            device: Device to run on ('cuda' or 'cpu')
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load model weights and initialize.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def warmup(self, num_frames: int = 10) -> None:
        """
        Warm up GPU with dummy inference.
        
        Args:
            num_frames: Number of warmup frames
        """
        pass
    
    @abstractmethod
    def animate(
        self,
        source_image: np.ndarray,
        driving_frame: np.ndarray,
        landmarks: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Animate source image with driving frame.
        
        Args:
            source_image: Character image (RGBA)
            driving_frame: Webcam frame (BGR or RGB)
            landmarks: Optional facial landmarks for optimization
            
        Returns:
            Animated frame (RGBA)
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if model is loaded and ready.
        
        Returns:
            True if model is loaded
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup model resources."""
        pass
