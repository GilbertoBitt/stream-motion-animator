"""
Model loader and downloader for AI animation models.

Handles automatic model downloads and initialization.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from .base_model import BaseAnimationModel
from .liveportrait_model import LivePortraitModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and downloading animation models."""
    
    SUPPORTED_MODELS = {
        'liveportrait': LivePortraitModel,
        # Add more models here
        # 'animateanyone': AnimateAnyoneModel,
        # 'sadtalker': SadTalkerModel,
    }
    
    @staticmethod
    def load_model(
        model_type: str,
        model_path: str,
        device: str = "cuda",
        **kwargs
    ) -> Optional[BaseAnimationModel]:
        """
        Load an animation model.
        
        Args:
            model_type: Type of model ('liveportrait', etc.)
            model_path: Path to model weights
            device: Device to run on
            **kwargs: Additional model-specific parameters
            
        Returns:
            Loaded model or None if failed
        """
        model_type = model_type.lower()
        
        if model_type not in ModelLoader.SUPPORTED_MODELS:
            logger.error(
                f"Unsupported model type: {model_type}\n"
                f"Supported models: {list(ModelLoader.SUPPORTED_MODELS.keys())}"
            )
            return None
        
        try:
            # Get model class
            model_class = ModelLoader.SUPPORTED_MODELS[model_type]
            
            # Create model instance
            model = model_class(model_path, device, **kwargs)
            
            # Load model weights
            if not model.load_model():
                logger.error("Failed to load model")
                return None
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def check_model_exists(model_path: str) -> bool:
        """
        Check if model files exist.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if model exists
        """
        path = Path(model_path)
        return path.exists() and path.is_dir()
    
    @staticmethod
    def get_model_size(model_path: str) -> int:
        """
        Get total size of model files in bytes.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Total size in bytes
        """
        path = Path(model_path)
        if not path.exists():
            return 0
        
        total_size = 0
        for file in path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size
    
    @staticmethod
    def list_available_models() -> list:
        """
        List all supported model types.
        
        Returns:
            List of model type names
        """
        return list(ModelLoader.SUPPORTED_MODELS.keys())
