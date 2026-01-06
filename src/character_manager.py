"""
Character management system for loading and switching between character images.

Handles image preprocessing, validation, and caching with optimization support.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CharacterManager:
    """Manages character images for animation."""
    
    def __init__(
        self,
        characters_path: str,
        target_size: Tuple[int, int] = (512, 512),
        auto_crop: bool = True,
        preload_all: bool = True,
        use_preprocessing_cache: bool = True
    ):
        """
        Initialize character manager.
        
        Args:
            characters_path: Path to characters directory
            target_size: Target size for character images (width, height)
            auto_crop: Whether to auto-detect and crop faces
            preload_all: Load all characters into memory
            use_preprocessing_cache: Use optimized preprocessing cache
        """
        self.characters_path = Path(characters_path)
        self.target_size = target_size
        self.auto_crop = auto_crop
        self.preload_all = preload_all
        self.use_preprocessing_cache = use_preprocessing_cache

        # Character storage
        self.character_files: List[Path] = []
        self.character_images: Dict[str, np.ndarray] = {}
        self.current_character_index: int = 0
        
        # Preprocessing cache
        self.preprocessor = None
        if use_preprocessing_cache:
            try:
                from image_preprocessor import ImagePreprocessor
                self.preprocessor = ImagePreprocessor(
                    cache_dir="cache/preprocessed",
                    device="cuda",
                    fp16=True
                )
                logger.info("Image preprocessing cache enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize preprocessor: {e}")
                self.use_preprocessing_cache = False

        # Face detector for auto-crop
        self.face_cascade = None
        if auto_crop:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logger.warning(f"Failed to load face detector: {e}")
                self.auto_crop = False
        
        # Load characters
        self.load_characters()
    
    def load_characters(self) -> None:
        """Scan directory and load character images."""
        if not self.characters_path.exists():
            logger.warning(f"Characters directory not found: {self.characters_path}")
            return
        
        # Find all image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.character_files = sorted([
            f for f in self.characters_path.iterdir()
            if f.suffix.lower() in valid_extensions and f.is_file()
        ])
        
        if not self.character_files:
            logger.warning(f"No character images found in {self.characters_path}")
            return
        
        logger.info(f"Found {len(self.character_files)} character images")
        
        # Preprocess all images if cache is enabled
        if self.use_preprocessing_cache and self.preprocessor:
            logger.info("Preprocessing character images for optimized inference...")
            self.preprocessor.preprocess_batch(
                [str(f) for f in self.character_files],
                self.target_size
            )

        # Preload if enabled
        if self.preload_all:
            for char_file in self.character_files:
                self._load_character_image(char_file)
            logger.info(f"Preloaded {len(self.character_images)} characters")
    
    def _load_character_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a character image.
        
        Args:
            image_path: Path to character image
            
        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Load image with PIL to handle transparency
            pil_image = Image.open(image_path)
            
            # Convert to RGBA if not already
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Auto-crop face if enabled
            if self.auto_crop and self.face_cascade is not None:
                image = self._auto_crop_face(image)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Cache the image
            self.character_images[str(image_path)] = image
            
            logger.info(f"Loaded character: {image_path.name}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load character {image_path}: {e}")
            return None
    
    def _auto_crop_face(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-detect and crop face from image.
        
        Args:
            image: Input RGBA image
            
        Returns:
            Cropped image or original if no face detected
        """
        try:
            # Convert to grayscale for face detection
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            if len(faces) == 0:
                logger.warning("No face detected for auto-crop")
                return image
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding around face (20%)
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Crop
            cropped = image[y1:y2, x1:x2]
            
            logger.info(f"Auto-cropped face: {w}x{h} at ({x}, {y})")
            return cropped
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return image
    
    def get_current_character(self) -> Optional[np.ndarray]:
        """
        Get the current character image.
        
        Returns:
            Current character image or None
        """
        if not self.character_files:
            return None
        
        current_file = self.character_files[self.current_character_index]
        
        # Load from cache or disk
        if str(current_file) in self.character_images:
            return self.character_images[str(current_file)]
        else:
            return self._load_character_image(current_file)
    
    def get_current_character_name(self) -> str:
        """Get the name of the current character."""
        if not self.character_files:
            return "None"
        return self.character_files[self.current_character_index].name
    
    def switch_character(self, index: int) -> bool:
        """
        Switch to character at given index.
        
        Args:
            index: Character index (0-based)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.character_files:
            logger.warning("No characters available")
            return False
        
        if 0 <= index < len(self.character_files):
            self.current_character_index = index
            logger.info(f"Switched to character {index + 1}: {self.get_current_character_name()}")
            return True
        else:
            logger.warning(f"Invalid character index: {index}")
            return False
    
    def next_character(self) -> bool:
        """
        Switch to next character.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.character_files:
            return False
        
        next_index = (self.current_character_index + 1) % len(self.character_files)
        return self.switch_character(next_index)
    
    def prev_character(self) -> bool:
        """
        Switch to previous character.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.character_files:
            return False
        
        prev_index = (self.current_character_index - 1) % len(self.character_files)
        return self.switch_character(prev_index)
    
    def reload_characters(self) -> None:
        """Reload all characters from disk."""
        logger.info("Reloading characters...")
        self.character_images.clear()
        self.load_characters()
    
    def get_character_count(self) -> int:
        """Get number of available characters."""
        return len(self.character_files)
    
    def get_character_list(self) -> List[str]:
        """Get list of character names."""
        return [f.name for f in self.character_files]
    
    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """
        Validate a character image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not image_path.exists():
                return False, "File not found"
            
            # Try to load image
            image = Image.open(image_path)
            
            # Check format
            if image.format not in ['PNG', 'JPEG', 'BMP']:
                return False, f"Unsupported format: {image.format}"
            
            # Check size
            width, height = image.size
            if width < 256 or height < 256:
                return False, f"Image too small: {width}x{height} (minimum 256x256)"
            
            # Check if face is detected (if auto_crop enabled)
            if self.auto_crop and self.face_cascade is not None:
                image_array = np.array(image.convert('RGB'))
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                if len(faces) == 0:
                    return False, "No face detected in image"
            
            return True, "Image is valid"
            
        except Exception as e:
            return False, f"Error loading image: {str(e)}"
    
    def get_info(self) -> Dict:
        """
        Get information about character manager state.
        
        Returns:
            Dictionary with manager info
        """
        info = {
            "characters_path": str(self.characters_path),
            "character_count": self.get_character_count(),
            "current_character": self.get_current_character_name(),
            "current_index": self.current_character_index,
            "preloaded": len(self.character_images),
            "target_size": self.target_size,
            "auto_crop": self.auto_crop,
            "preprocessing_cache": self.use_preprocessing_cache
        }

        # Add cache stats if available
        if self.preprocessor:
            info["cache_stats"] = self.preprocessor.get_cache_stats()

        return info

    def get_preprocessed_data(self, character_index: Optional[int] = None):
        """
        Get preprocessed data for fast inference.

        Args:
            character_index: Character index (uses current if None)

        Returns:
            Preprocessed data dictionary or None
        """
        if not self.use_preprocessing_cache or not self.preprocessor:
            return None

        if character_index is None:
            character_index = self.current_character_index

        if 0 <= character_index < len(self.character_files):
            image_path = self.character_files[character_index]
            return self.preprocessor.preprocess_character_image(
                str(image_path),
                self.target_size
            )

        return None
