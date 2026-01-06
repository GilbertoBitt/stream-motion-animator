"""
Inference Optimizer for Stream Motion Animator.

Pre-processes character images to extract features and cache them for faster inference.
Uses tensor operations and indexed lookups instead of full image processing each frame.
"""

import numpy as np
import torch
import cv2
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from dataclasses import dataclass
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CharacterFeatures:
    """Pre-computed features for a character."""

    character_id: str
    source_tensor: torch.Tensor  # Pre-processed source image as tensor
    appearance_features: Optional[torch.Tensor] = None  # Appearance embedding
    canonical_keypoints: Optional[torch.Tensor] = None  # Canonical facial keypoints
    motion_basis: Optional[torch.Tensor] = None  # Motion deformation basis
    alpha_mask: Optional[torch.Tensor] = None  # Alpha channel mask
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class MotionCache:
    """Cache for motion vectors to reuse similar motions."""

    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.95):
        """
        Initialize motion cache.

        Args:
            max_size: Maximum number of cached motions
            similarity_threshold: Threshold for motion similarity (0-1)
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def _compute_motion_hash(self, landmarks: np.ndarray) -> Tuple[int, ...]:
        """Compute a hash for landmark positions."""
        # Quantize landmarks to reduce precision and improve cache hits
        quantized = (landmarks * 10).astype(np.int32)
        return tuple(quantized.flatten())

    def get(self, landmarks: np.ndarray) -> Optional[torch.Tensor]:
        """
        Get cached motion vector if available.

        Args:
            landmarks: Facial landmarks

        Returns:
            Cached motion tensor or None
        """
        motion_hash = self._compute_motion_hash(landmarks)

        if motion_hash in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(motion_hash)
            return self.cache[motion_hash]

        return None

    def put(self, landmarks: np.ndarray, motion_tensor: torch.Tensor) -> None:
        """
        Store motion vector in cache.

        Args:
            landmarks: Facial landmarks
            motion_tensor: Computed motion tensor
        """
        motion_hash = self._compute_motion_hash(landmarks)

        # Add to cache
        self.cache[motion_hash] = motion_tensor

        # Remove oldest if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class InferenceOptimizer:
    """Optimizes AI inference through pre-processing and caching."""

    def __init__(
        self,
        device: str = "cuda",
        fp16: bool = True,
        cache_dir: Optional[str] = None,
        enable_motion_cache: bool = True,
        batch_size: int = 1
    ):
        """
        Initialize inference optimizer.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            fp16: Use FP16 precision
            cache_dir: Directory to save/load cached features
            enable_motion_cache: Enable motion vector caching
            batch_size: Batch size for processing
        """
        self.device = device
        self.fp16 = fp16 and device == "cuda"
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Feature cache (in-memory)
        self.character_features: Dict[str, CharacterFeatures] = {}

        # Motion cache
        self.motion_cache = MotionCache() if enable_motion_cache else None

        # Batch processing
        self.batch_size = batch_size

        # Performance stats
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Inference optimizer initialized (device={device}, fp16={fp16})")

    def preprocess_character(
        self,
        character_id: str,
        character_image: np.ndarray,
        model: Optional[Any] = None,
        force_recompute: bool = False
    ) -> CharacterFeatures:
        """
        Pre-process a character image and extract features.

        Args:
            character_id: Unique identifier for character
            character_image: Character image (RGBA)
            model: Optional AI model to extract features
            force_recompute: Force recomputation even if cached

        Returns:
            CharacterFeatures object
        """
        # Check if already in memory
        if not force_recompute and character_id in self.character_features:
            logger.debug(f"Character {character_id} already in cache")
            return self.character_features[character_id]

        # Try to load from disk cache
        if not force_recompute:
            cached_features = self._load_cached_features(character_id)
            if cached_features is not None:
                self.character_features[character_id] = cached_features
                logger.info(f"Loaded cached features for {character_id}")
                return cached_features

        logger.info(f"Pre-processing character: {character_id}")

        # Convert to tensor
        source_tensor = self._image_to_tensor(character_image)

        # Extract features if model is provided
        appearance_features = None
        canonical_keypoints = None
        motion_basis = None
        alpha_mask = None

        if model is not None and hasattr(model, 'extract_features'):
            try:
                features = model.extract_features(source_tensor)
                appearance_features = features.get('appearance')
                canonical_keypoints = features.get('keypoints')
                motion_basis = features.get('motion_basis')
            except Exception as e:
                logger.warning(f"Failed to extract features: {e}")

        # Extract alpha mask
        if character_image.shape[2] == 4:
            alpha_channel = character_image[:, :, 3]
            alpha_mask = torch.from_numpy(alpha_channel).to(self.device)
            if self.fp16:
                alpha_mask = alpha_mask.half()
            alpha_mask = alpha_mask.unsqueeze(0).unsqueeze(0) / 255.0

        # Create feature object
        features = CharacterFeatures(
            character_id=character_id,
            source_tensor=source_tensor,
            appearance_features=appearance_features,
            canonical_keypoints=canonical_keypoints,
            motion_basis=motion_basis,
            alpha_mask=alpha_mask,
            metadata={
                'original_shape': character_image.shape,
                'device': self.device,
                'fp16': self.fp16
            }
        )

        # Cache in memory
        self.character_features[character_id] = features

        # Save to disk
        self._save_cached_features(character_id, features)

        logger.info(f"Character {character_id} pre-processed and cached")
        return features

    def get_character_features(self, character_id: str) -> Optional[CharacterFeatures]:
        """
        Get cached character features.

        Args:
            character_id: Character identifier

        Returns:
            CharacterFeatures or None if not cached
        """
        return self.character_features.get(character_id)

    def optimize_inference(
        self,
        character_features: CharacterFeatures,
        driving_frame: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        model: Optional[Any] = None
    ) -> np.ndarray:
        """
        Perform optimized inference using cached features.

        Args:
            character_features: Pre-computed character features
            driving_frame: Webcam frame
            landmarks: Optional facial landmarks
            model: AI model for inference

        Returns:
            Animated frame
        """
        if model is None:
            logger.warning("No model provided for inference")
            return self._tensor_to_image(character_features.source_tensor)

        try:
            # Check motion cache first
            motion_tensor = None
            if self.motion_cache is not None and landmarks is not None:
                motion_tensor = self.motion_cache.get(landmarks)
                if motion_tensor is not None:
                    self.cache_hits += 1
                    logger.debug("Motion cache hit")
                else:
                    self.cache_misses += 1

            # Convert driving frame to tensor
            driving_tensor = self._image_to_tensor(driving_frame)

            # Use model's optimized inference if available
            if hasattr(model, 'inference_with_features'):
                # Fast path: use pre-computed features
                result_tensor = model.inference_with_features(
                    source_features=character_features,
                    driving_tensor=driving_tensor,
                    cached_motion=motion_tensor,
                    landmarks=landmarks
                )
            elif hasattr(model, 'animate_optimized'):
                # Alternative fast path
                result_tensor = model.animate_optimized(
                    character_features.source_tensor,
                    driving_tensor,
                    appearance_features=character_features.appearance_features,
                    motion_basis=character_features.motion_basis,
                    landmarks=landmarks
                )
            else:
                # Fallback to standard inference
                result = model.animate(
                    self._tensor_to_image(character_features.source_tensor),
                    driving_frame,
                    landmarks={'landmarks': landmarks} if landmarks is not None else None
                )
                return result

            # Cache motion if computed
            if motion_tensor is None and self.motion_cache is not None and landmarks is not None:
                # Extract motion from result if possible
                if hasattr(model, 'last_motion_tensor'):
                    self.motion_cache.put(landmarks, model.last_motion_tensor)

            # Convert result back to image
            result_image = self._tensor_to_image(result_tensor)

            return result_image

        except Exception as e:
            logger.error(f"Optimized inference failed: {e}")
            # Fallback to source image
            return self._tensor_to_image(character_features.source_tensor)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to PyTorch tensor."""
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to CHW format
        if len(image.shape) == 3:
            tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            tensor = torch.from_numpy(image).unsqueeze(0)

        # Move to device
        tensor = tensor.to(self.device)

        # Convert to FP16 if enabled
        if self.fp16:
            tensor = tensor.half()

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy image."""
        # Remove batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Move to CPU and convert to float32
        tensor = tensor.detach().cpu().float()

        # Convert from CHW to HWC
        if tensor.dim() == 3:
            image = tensor.permute(1, 2, 0).numpy()
        else:
            image = tensor.numpy()

        # Convert to uint8
        image = (image * 255).clip(0, 255).astype(np.uint8)

        return image

    def _save_cached_features(self, character_id: str, features: CharacterFeatures) -> None:
        """Save features to disk cache."""
        try:
            cache_file = self.cache_dir / f"{character_id}.pkl"

            # Convert tensors to CPU for serialization
            save_data = {
                'character_id': features.character_id,
                'source_tensor': features.source_tensor.cpu(),
                'appearance_features': features.appearance_features.cpu() if features.appearance_features is not None else None,
                'canonical_keypoints': features.canonical_keypoints.cpu() if features.canonical_keypoints is not None else None,
                'motion_basis': features.motion_basis.cpu() if features.motion_basis is not None else None,
                'alpha_mask': features.alpha_mask.cpu() if features.alpha_mask is not None else None,
                'metadata': features.metadata
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(save_data, f)

            logger.debug(f"Saved features to {cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cached features: {e}")

    def _load_cached_features(self, character_id: str) -> Optional[CharacterFeatures]:
        """Load features from disk cache."""
        try:
            cache_file = self.cache_dir / f"{character_id}.pkl"

            if not cache_file.exists():
                return None

            with open(cache_file, 'rb') as f:
                save_data = pickle.load(f)

            # Move tensors to device
            features = CharacterFeatures(
                character_id=save_data['character_id'],
                source_tensor=save_data['source_tensor'].to(self.device),
                appearance_features=save_data['appearance_features'].to(self.device) if save_data['appearance_features'] is not None else None,
                canonical_keypoints=save_data['canonical_keypoints'].to(self.device) if save_data['canonical_keypoints'] is not None else None,
                motion_basis=save_data['motion_basis'].to(self.device) if save_data['motion_basis'] is not None else None,
                alpha_mask=save_data['alpha_mask'].to(self.device) if save_data['alpha_mask'] is not None else None,
                metadata=save_data['metadata']
            )

            # Apply FP16 if needed
            if self.fp16:
                features.source_tensor = features.source_tensor.half()
                if features.appearance_features is not None:
                    features.appearance_features = features.appearance_features.half()
                if features.canonical_keypoints is not None:
                    features.canonical_keypoints = features.canonical_keypoints.half()
                if features.motion_basis is not None:
                    features.motion_basis = features.motion_basis.half()
                if features.alpha_mask is not None:
                    features.alpha_mask = features.alpha_mask.half()

            return features

        except Exception as e:
            logger.warning(f"Failed to load cached features: {e}")
            return None

    def clear_cache(self, disk_only: bool = False) -> None:
        """
        Clear the feature cache.

        Args:
            disk_only: If True, only clear disk cache, keep memory cache
        """
        if not disk_only:
            self.character_features.clear()
            logger.info("Memory cache cleared")

        if self.motion_cache is not None:
            self.motion_cache.clear()
            logger.info("Motion cache cleared")

        # Clear disk cache
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Disk cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        cache_hit_rate = 0.0
        if self.cache_hits + self.cache_misses > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)

        return {
            'cached_characters': len(self.character_features),
            'motion_cache_size': len(self.motion_cache.cache) if self.motion_cache else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'device': self.device,
            'fp16': self.fp16
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.character_features.clear()
        if self.motion_cache is not None:
            self.motion_cache.clear()

        logger.info("Inference optimizer cleaned up")

