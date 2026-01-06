"""
Image preprocessing and caching system for optimized inference.

Pre-computes image features and encodings to speed up real-time animation
and reduce CPU/GPU usage during runtime.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging
import pickle
import hashlib

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Pre-computes and caches image features for fast inference.

    This reduces runtime computation by:
    - Pre-computing image tensors in correct format
    - Caching face detection/alignment results
    - Pre-encoding images with model's encoder (if available)
    - Creating indexed lookup for fast frame generation
    """

    def __init__(
        self,
        cache_dir: str = "cache/preprocessed",
        device: str = "cuda",
        fp16: bool = True
    ):
        """
        Initialize image preprocessor.

        Args:
            cache_dir: Directory to store cached preprocessed data
            device: Device for tensor operations
            fp16: Use FP16 precision
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 and device == "cuda" else torch.float32

        # Cache storage
        self.preprocessed_cache: Dict[str, Dict[str, Any]] = {}
        self.tensor_cache: Dict[str, torch.Tensor] = {}

        logger.info(f"Image preprocessor initialized (device={device}, fp16={fp16})")

    def preprocess_character_image(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (512, 512),
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess and cache character image data.

        Args:
            image_path: Path to character image
            target_size: Target image size (width, height)
            force_recompute: Force recomputation even if cached

        Returns:
            Dictionary containing preprocessed data:
            - 'tensor': PyTorch tensor (normalized, ready for model)
            - 'numpy': NumPy array (RGBA format)
            - 'bbox': Face bounding box if detected
            - 'landmarks': Face landmarks if detected
            - 'hash': Image hash for cache validation
        """
        image_path = Path(image_path)
        cache_key = self._get_cache_key(image_path, target_size)

        # Check cache
        if not force_recompute and cache_key in self.preprocessed_cache:
            logger.debug(f"Using cached preprocessed data for {image_path.name}")
            return self.preprocessed_cache[cache_key]

        # Try to load from disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not force_recompute and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.preprocessed_cache[cache_key] = cached_data
                logger.debug(f"Loaded cached data from disk: {image_path.name}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")

        # Compute preprocessing
        logger.info(f"Preprocessing image: {image_path.name}")

        try:
            # Load image
            from PIL import Image
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')

            # Resize
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy
            numpy_array = np.array(pil_image, dtype=np.uint8)

            # Detect face (optional, for alignment)
            bbox, landmarks = self._detect_face(numpy_array)

            # Convert to tensor and normalize
            tensor = self._image_to_tensor(numpy_array)

            # Compute hash for cache validation
            image_hash = self._compute_image_hash(numpy_array)

            # Package preprocessed data
            preprocessed_data = {
                'tensor': tensor,
                'numpy': numpy_array,
                'bbox': bbox,
                'landmarks': landmarks,
                'hash': image_hash,
                'size': target_size,
                'path': str(image_path)
            }

            # Cache in memory
            self.preprocessed_cache[cache_key] = preprocessed_data

            # Save to disk
            self._save_cache(cache_key, preprocessed_data)

            logger.info(f"Preprocessed and cached: {image_path.name}")
            return preprocessed_data

        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise

    def preprocess_batch(
        self,
        image_paths: list,
        target_size: Tuple[int, int] = (512, 512),
        force_recompute: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess multiple images in batch.

        Args:
            image_paths: List of image paths
            target_size: Target size for all images
            force_recompute: Force recomputation

        Returns:
            Dictionary mapping image paths to preprocessed data
        """
        logger.info(f"Preprocessing batch of {len(image_paths)} images...")

        results = {}
        for image_path in image_paths:
            try:
                preprocessed = self.preprocess_character_image(
                    image_path,
                    target_size,
                    force_recompute
                )
                results[str(image_path)] = preprocessed
            except Exception as e:
                logger.error(f"Failed to preprocess {image_path}: {e}")

        logger.info(f"Successfully preprocessed {len(results)}/{len(image_paths)} images")
        return results

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor.

        Args:
            image: NumPy array (H, W, C) in range [0, 255]

        Returns:
            PyTorch tensor (C, H, W) normalized to [0, 1] or [-1, 1]
        """
        # Convert to float and normalize to [0, 1]
        tensor = torch.from_numpy(image).float() / 255.0

        # Rearrange to (C, H, W)
        tensor = tensor.permute(2, 0, 1)

        # Normalize to [-1, 1] (common for many models)
        tensor = (tensor - 0.5) / 0.5

        # Move to device and convert precision
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        return tensor

    def _detect_face(self, image: np.ndarray) -> Tuple[Optional[Tuple], Optional[np.ndarray]]:
        """
        Detect face in image and extract landmarks.

        Args:
            image: Image array (RGBA)

        Returns:
            Tuple of (bbox, landmarks) or (None, None)
        """
        try:
            # Convert to RGB for detection
            rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Simple face detection with OpenCV
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(rgb, 1.1, 4)

            if len(faces) > 0:
                # Take largest face
                bbox = max(faces, key=lambda f: f[2] * f[3])
                return tuple(bbox), None

        except Exception as e:
            logger.debug(f"Face detection failed: {e}")

        return None, None

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute hash of image for cache validation."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def _get_cache_key(self, image_path: Path, target_size: Tuple[int, int]) -> str:
        """Generate cache key for image."""
        path_hash = hashlib.md5(str(image_path).encode()).hexdigest()[:16]
        size_str = f"{target_size[0]}x{target_size[1]}"
        return f"{image_path.stem}_{path_hash}_{size_str}"

    def _save_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save preprocessed data to disk cache."""
        try:
            # Don't save tensors to disk (device-specific)
            disk_data = {k: v for k, v in data.items() if k != 'tensor'}

            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(disk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug(f"Saved cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_tensor(self, cache_key: str) -> Optional[torch.Tensor]:
        """Get cached tensor for fast inference."""
        if cache_key in self.preprocessed_cache:
            return self.preprocessed_cache[cache_key].get('tensor')
        return None

    def clear_cache(self, memory_only: bool = False) -> None:
        """
        Clear preprocessed cache.

        Args:
            memory_only: If True, only clear memory cache (keep disk cache)
        """
        self.preprocessed_cache.clear()
        self.tensor_cache.clear()

        if not memory_only:
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_cache_count = len(list(self.cache_dir.glob("*.pkl")))
        memory_cache_count = len(self.preprocessed_cache)

        # Estimate memory usage
        memory_mb = sum(
            data['numpy'].nbytes / (1024 * 1024)
            for data in self.preprocessed_cache.values()
        )

        return {
            'disk_cache_count': disk_cache_count,
            'memory_cache_count': memory_cache_count,
            'memory_usage_mb': memory_mb,
            'cache_dir': str(self.cache_dir)
        }

