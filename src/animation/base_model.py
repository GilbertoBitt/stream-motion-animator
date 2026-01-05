"""
Base class for AI animation models
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class BaseAnimationModel(ABC):
    """Abstract base class for AI animation models"""
    
    def __init__(self, model_path: Optional[str] = None, gpu_id: int = 0):
        """
        Initialize animation model
        
        Args:
            model_path: Path to model weights
            gpu_id: GPU device ID
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialized = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the animation model
        
        Returns:
            True if model loaded successfully
        """
        pass
    
    @abstractmethod
    def preprocess_source_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess source image for the model
        
        Args:
            image: Source image (BGR format)
            
        Returns:
            Preprocessed image or tensor
        """
        pass
    
    @abstractmethod
    def animate(
        self, 
        source_image: np.ndarray,
        driving_landmarks: Dict,
        return_intermediate: bool = False
    ) -> np.ndarray:
        """
        Generate animated frame from source image and driving landmarks
        
        Args:
            source_image: Source portrait image
            driving_landmarks: Facial landmarks from tracking
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Animated frame (BGR format)
        """
        pass
    
    def unload_model(self) -> None:
        """Unload model and free resources"""
        self.initialized = False
        self.logger.info("Model unloaded")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.__class__.__name__,
            'model_path': self.model_path,
            'gpu_id': self.gpu_id,
            'initialized': self.initialized
        }
    
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        return self.initialized
