"""
Multi-batch Character Manager with Video Support.

Supports character folders containing:
- Multiple reference images (PNG, JPG, etc.)
- Multiple reference videos (MP4, AVI, etc.)
- Frame extraction from videos
- Batch feature extraction for better LivePortrait training

Structure:
assets/characters/
  ├── character1/
  │   ├── reference_image_1.png
  │   ├── reference_image_2.jpg
  │   ├── reference_video_1.mp4
  │   └── reference_video_2.mp4
  ├── character2/
  │   ├── image1.png
  │   └── video1.mp4
  └── ...
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image
import logging
import hashlib
import pickle

logger = logging.getLogger(__name__)


class Character:
    """Represents a character with multiple reference images and videos."""

    def __init__(self, name: str, folder_path: Path):
        """
        Initialize character.

        Args:
            name: Character name
            folder_path: Path to character folder
        """
        self.name = name
        self.folder_path = folder_path

        # Reference materials
        self.image_files: List[Path] = []
        self.video_files: List[Path] = []
        self.extracted_frames: List[np.ndarray] = []

        # Processed data
        self.reference_images: List[np.ndarray] = []  # All reference images
        self.primary_image: Optional[np.ndarray] = None  # Main display image
        self.feature_embeddings: Optional[Dict] = None  # Cached features

        # Statistics
        self.total_references = 0
        self.frames_from_video = 0

    def __repr__(self):
        return f"Character({self.name}, {self.total_references} refs, {len(self.video_files)} videos)"


class MultiBatchCharacterManager:
    """
    Enhanced character manager supporting multiple images/videos per character.

    Features:
    - Character folders with multiple reference materials
    - Video frame extraction
    - Batch feature extraction for better AI training
    - Smart frame sampling from videos
    - Cached preprocessing
    """

    # Supported file extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(
        self,
        characters_path: str,
        target_size: Tuple[int, int] = (512, 512),
        auto_crop: bool = True,
        preload_all: bool = True,
        use_preprocessing_cache: bool = True,
        max_frames_per_video: int = 30,
        video_sample_rate: int = 10,
        enable_video_processing: bool = True
    ):
        """
        Initialize multi-batch character manager.

        Args:
            characters_path: Path to characters directory
            target_size: Target size for images (width, height)
            auto_crop: Auto-detect and crop faces
            preload_all: Load all characters into memory
            use_preprocessing_cache: Use cached preprocessing
            max_frames_per_video: Maximum frames to extract per video
            video_sample_rate: Extract every Nth frame from video
            enable_video_processing: Enable video frame extraction
        """
        self.characters_path = Path(characters_path)
        self.target_size = target_size
        self.auto_crop = auto_crop
        self.preload_all = preload_all
        self.use_preprocessing_cache = use_preprocessing_cache
        self.max_frames_per_video = max_frames_per_video
        self.video_sample_rate = video_sample_rate
        self.enable_video_processing = enable_video_processing

        # Character storage
        self.characters: List[Character] = []
        self.current_character_index: int = 0
        self.character_cache: Dict[str, Character] = {}

        # Cache directories
        self.cache_dir = Path("cache/characters")
        self.frame_cache_dir = self.cache_dir / "frames"
        self.feature_cache_dir = self.cache_dir / "features"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_cache_dir.mkdir(exist_ok=True)
        self.feature_cache_dir.mkdir(exist_ok=True)

        # Preprocessing
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
                logger.info("Face detection enabled for auto-crop")
            except Exception as e:
                logger.warning(f"Failed to load face detector: {e}")
                self.auto_crop = False

        # Load characters
        self.load_characters()

    def load_characters(self) -> None:
        """Scan directory and load all character folders."""
        if not self.characters_path.exists():
            logger.warning(f"Characters directory not found: {self.characters_path}")
            return

        # Find character folders
        character_folders = [
            d for d in self.characters_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

        if not character_folders:
            logger.warning("No character folders found. Falling back to legacy mode...")
            self._load_legacy_characters()
            return

        logger.info(f"Found {len(character_folders)} character folders")

        # Load each character
        for folder in sorted(character_folders):
            try:
                character = self._load_character_folder(folder)
                if character and character.total_references > 0:
                    self.characters.append(character)
                    logger.info(f"Loaded: {character}")
            except Exception as e:
                logger.error(f"Failed to load character from {folder.name}: {e}")

        if not self.characters:
            logger.warning("No valid characters loaded")
            return

        logger.info(f"Successfully loaded {len(self.characters)} characters")

        # Preload if enabled
        if self.preload_all:
            self._preload_all_characters()

    def _load_character_folder(self, folder: Path) -> Optional[Character]:
        """
        Load a character from a folder.

        Args:
            folder: Path to character folder

        Returns:
            Character object or None
        """
        character = Character(folder.name, folder)

        # Find image files
        for ext in self.IMAGE_EXTENSIONS:
            character.image_files.extend(folder.glob(f"*{ext}"))
            character.image_files.extend(folder.glob(f"*{ext.upper()}"))

        # Find video files
        if self.enable_video_processing:
            for ext in self.VIDEO_EXTENSIONS:
                character.video_files.extend(folder.glob(f"*{ext}"))
                character.video_files.extend(folder.glob(f"*{ext.upper()}"))

        character.image_files = sorted(character.image_files)
        character.video_files = sorted(character.video_files)

        character.total_references = len(character.image_files) + len(character.video_files)

        if character.total_references == 0:
            logger.warning(f"No reference materials found in {folder.name}")
            return None

        logger.info(
            f"  {folder.name}: {len(character.image_files)} images, "
            f"{len(character.video_files)} videos"
        )

        return character

    def _load_legacy_characters(self) -> None:
        """Load characters from legacy flat structure (single images)."""
        logger.info("Loading characters in legacy mode...")

        # Find all image files directly in characters_path
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(self.characters_path.glob(f"*{ext}"))
            image_files.extend(self.characters_path.glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)

        if not image_files:
            logger.warning("No image files found in characters directory")
            return

        # Create a character for each image
        for img_file in image_files:
            character = Character(img_file.stem, self.characters_path)
            character.image_files = [img_file]
            character.total_references = 1
            self.characters.append(character)

        logger.info(f"Loaded {len(self.characters)} characters (legacy mode)")

    def _preload_all_characters(self) -> None:
        """Preload all character reference materials."""
        logger.info("Preloading all characters...")

        for i, character in enumerate(self.characters):
            logger.info(f"  [{i+1}/{len(self.characters)}] Loading {character.name}...")
            self._load_character_data(character)

        logger.info("All characters preloaded")

    def _load_character_data(self, character: Character) -> None:
        """
        Load all reference data for a character.

        Args:
            character: Character to load
        """
        # Check cache first
        cache_file = self.frame_cache_dir / f"{character.name}_frames.pkl"
        if cache_file.exists() and self.use_preprocessing_cache:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    character.reference_images = cached_data['images']
                    character.primary_image = cached_data['primary']
                    character.frames_from_video = cached_data.get('video_frames', 0)
                    logger.info(f"    Loaded from cache: {len(character.reference_images)} references")
                    return
            except Exception as e:
                logger.warning(f"    Failed to load cache: {e}")

        # Load images
        for img_file in character.image_files:
            img = self._load_and_process_image(img_file)
            if img is not None:
                character.reference_images.append(img)

        # Extract frames from videos
        if self.enable_video_processing and character.video_files:
            for video_file in character.video_files:
                frames = self._extract_video_frames(video_file, character.name)
                character.reference_images.extend(frames)
                character.frames_from_video += len(frames)

        # Set primary image (first image or first video frame)
        if character.reference_images:
            character.primary_image = character.reference_images[0]

        logger.info(
            f"    Loaded {len(character.reference_images)} references "
            f"({len(character.image_files)} images + {character.frames_from_video} video frames)"
        )

        # Save to cache
        if self.use_preprocessing_cache and character.reference_images:
            try:
                cache_data = {
                    'images': character.reference_images,
                    'primary': character.primary_image,
                    'video_frames': character.frames_from_video
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"    Cached to: {cache_file.name}")
            except Exception as e:
                logger.warning(f"    Failed to save cache: {e}")

    def _load_and_process_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and process a single image.

        Args:
            image_path: Path to image

        Returns:
            Processed image array or None
        """
        try:
            # Load with PIL
            pil_image = Image.open(image_path)

            # Convert to RGBA
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')

            # Convert to numpy
            image = np.array(pil_image)

            # Auto-crop face
            if self.auto_crop and self.face_cascade is not None:
                image = self._auto_crop_face(image)

            # Resize
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

            return image

        except Exception as e:
            logger.error(f"Failed to load {image_path.name}: {e}")
            return None

    def _extract_video_frames(self, video_path: Path, character_name: str) -> List[np.ndarray]:
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file
            character_name: Name of character (for cache)

        Returns:
            List of extracted frame arrays
        """
        frames = []

        try:
            # Check cache
            video_hash = self._get_file_hash(video_path)
            cache_file = self.frame_cache_dir / f"{character_name}_{video_hash}_frames.pkl"

            if cache_file.exists() and self.use_preprocessing_cache:
                try:
                    with open(cache_file, 'rb') as f:
                        frames = pickle.load(f)
                        logger.info(f"      Video cache: {video_path.name} ({len(frames)} frames)")
                        return frames
                except Exception as e:
                    logger.warning(f"      Failed to load video cache: {e}")

            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"      Failed to open video: {video_path.name}")
                return frames

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"      Extracting from {video_path.name} ({total_frames} frames @ {fps:.1f}fps)")

            frame_count = 0
            extracted_count = 0

            while extracted_count < self.max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample every Nth frame
                if frame_count % self.video_sample_rate == 0:
                    # Convert BGR to RGB then to RGBA
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)

                    # Auto-crop face
                    if self.auto_crop and self.face_cascade is not None:
                        frame_rgba = self._auto_crop_face(frame_rgba)

                    # Resize
                    frame_rgba = cv2.resize(frame_rgba, self.target_size, interpolation=cv2.INTER_LANCZOS4)

                    frames.append(frame_rgba)
                    extracted_count += 1

                frame_count += 1

            cap.release()

            logger.info(f"      Extracted {len(frames)} frames from {video_path.name}")

            # Save to cache
            if self.use_preprocessing_cache and frames:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(frames, f)
                except Exception as e:
                    logger.warning(f"      Failed to save video cache: {e}")

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path.name}: {e}")

        return frames

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for caching."""
        stat = file_path.stat()
        hash_input = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def _auto_crop_face(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-detect and crop face from image.

        Args:
            image: Input RGBA image

        Returns:
            Cropped image or original if no face detected
        """
        try:
            # Convert to grayscale
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            if len(faces) == 0:
                return image

            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # Add padding (30%)
            padding = int(max(w, h) * 0.3)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            # Crop
            cropped = image[y1:y2, x1:x2]

            return cropped

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return image

    # Public API methods

    def get_character_count(self) -> int:
        """Get total number of characters."""
        return len(self.characters)

    def get_current_character(self) -> Optional[Character]:
        """Get current character object."""
        if not self.characters:
            return None
        return self.characters[self.current_character_index]

    def get_current_character_image(self) -> Optional[np.ndarray]:
        """Get primary image of current character."""
        character = self.get_current_character()
        if not character:
            return None

        # Load if not loaded
        if character.primary_image is None:
            self._load_character_data(character)

        return character.primary_image

    def get_current_character_references(self) -> List[np.ndarray]:
        """
        Get all reference images for current character.

        Returns:
            List of all reference images (from images + video frames)
        """
        character = self.get_current_character()
        if not character:
            return []

        # Load if not loaded
        if not character.reference_images:
            self._load_character_data(character)

        return character.reference_images

    def get_current_character_name(self) -> str:
        """Get name of current character."""
        character = self.get_current_character()
        return character.name if character else "None"

    def switch_character(self, index: int) -> bool:
        """
        Switch to character at index.

        Args:
            index: Character index

        Returns:
            True if successful
        """
        if not self.characters or not (0 <= index < len(self.characters)):
            return False

        self.current_character_index = index
        logger.info(f"Switched to: {self.get_current_character_name()}")
        return True

    def next_character(self) -> bool:
        """Switch to next character."""
        if not self.characters:
            return False
        self.current_character_index = (self.current_character_index + 1) % len(self.characters)
        logger.info(f"Next: {self.get_current_character_name()}")
        return True

    def prev_character(self) -> bool:
        """Switch to previous character."""
        if not self.characters:
            return False
        self.current_character_index = (self.current_character_index - 1) % len(self.characters)
        logger.info(f"Previous: {self.get_current_character_name()}")
        return True

    def reload_characters(self) -> None:
        """Reload all characters from disk."""
        logger.info("Reloading characters...")
        self.characters.clear()
        self.character_cache.clear()
        self.current_character_index = 0
        self.load_characters()

    def get_character_stats(self) -> Dict:
        """Get statistics about loaded characters."""
        stats = {
            'total_characters': len(self.characters),
            'current_character': self.get_current_character_name(),
            'total_images': sum(len(c.image_files) for c in self.characters),
            'total_videos': sum(len(c.video_files) for c in self.characters),
            'total_references': sum(len(c.reference_images) for c in self.characters),
            'video_frames_extracted': sum(c.frames_from_video for c in self.characters),
        }
        return stats

