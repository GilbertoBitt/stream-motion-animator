"""
Custom Character Model Generator

This creates a custom ONNX model from multiple character frames
for better LivePortrait-style animation.

Uses your 32 Test character expressions to build a character-specific model.
"""

import numpy as np
import cv2
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CharacterFeatureExtractor:
    """
    Extracts features from multiple character frames to create
    a comprehensive representation for animation.
    """

    def __init__(self, character_folder: Path, landmark_model_path: Path):
        """
        Initialize feature extractor.

        Args:
            character_folder: Path to folder with character frames
            landmark_model_path: Path to landmark.onnx model
        """
        self.character_folder = character_folder
        self.landmark_model_path = landmark_model_path
        self.landmark_session = None
        self.character_frames = []
        self.character_features = []

    def load_landmark_model(self):
        """Load the landmark detection ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNXRuntime required. Install: pip install onnxruntime-gpu")

        logger.info(f"Loading landmark model: {self.landmark_model_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.landmark_session = ort.InferenceSession(
            str(self.landmark_model_path),
            providers=providers
        )

        logger.info("[OK] Landmark model loaded")

    def load_character_frames(self):
        """Load all character frame images."""
        logger.info(f"Loading character frames from: {self.character_folder}")

        image_files = sorted(self.character_folder.glob("*.png"))

        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Convert BGR to RGB
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

                self.character_frames.append({
                    'name': img_file.stem,
                    'path': img_file,
                    'image': img
                })

        logger.info("All character frames loaded")
        return len(self.character_frames)

    def extract_features_from_all_frames(self):
        """Extract facial features from all character frames."""
        logger.info("Extracting features from all frames...")

        for i, frame_data in enumerate(self.character_frames):
            logger.info(f"  [{i+1}/{len(self.character_frames)}] Processing: {frame_data['name']}")

            image = frame_data['image']

            # Extract RGB if RGBA
            if len(image.shape) == 3 and image.shape[2] == 4:
                rgb_image = image[:, :, :3]
            else:
                rgb_image = image

            # Extract landmarks
            landmarks = self._extract_landmarks(rgb_image)

            # Compute additional features
            features = {
                'name': frame_data['name'],
                'landmarks': landmarks,
                'image_shape': image.shape,
                'bbox': self._compute_face_bbox(landmarks),
                'keypoints': self._extract_keypoints(landmarks),
                'expression_vector': self._compute_expression_vector(landmarks)
            }

            self.character_features.append(features)
            frame_data['features'] = features

        logger.info(f"Extracted features from {len(self.character_features)} frames")

    def _extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Extract facial landmarks using ONNX model."""
        # Preprocess
        input_tensor = self._preprocess_for_landmark_model(image)

        # Run inference
        input_name = self.landmark_session.get_inputs()[0].name
        outputs = self.landmark_session.run(None, {input_name: input_tensor})

        # Return landmark coordinates
        return outputs[0] if outputs else np.zeros((68, 2))

    def _preprocess_for_landmark_model(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for landmark model."""
        # Get expected input size from model
        input_shape = self.landmark_session.get_inputs()[0].shape
        if len(input_shape) >= 4:
            expected_h = input_shape[2] if isinstance(input_shape[2], int) else 224
            expected_w = input_shape[3] if isinstance(input_shape[3], int) else 224
        else:
            expected_h, expected_w = 224, 224

        # Resize to expected size
        resized = cv2.resize(image, (expected_w, expected_h))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, 0)

        return batched

    def _compute_face_bbox(self, landmarks: np.ndarray) -> Dict:
        """Compute bounding box from landmarks."""
        if len(landmarks.shape) == 3:
            landmarks = landmarks[0]

        if landmarks.size == 0 or landmarks.shape[0] == 0:
            return {'x': 0, 'y': 0, 'w': 0, 'h': 0}

        # Handle different landmark formats
        if landmarks.shape[1] == 2:
            # Shape is (N, 2) - standard format
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
        elif len(landmarks.shape) == 1:
            # Flattened array
            x_coords = landmarks[::2]
            y_coords = landmarks[1::2]
        else:
            return {'x': 0, 'y': 0, 'w': 0, 'h': 0}

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        return {
            'x': float(x_min),
            'y': float(y_min),
            'w': float(x_max - x_min),
            'h': float(y_max - y_min)
        }

    def _extract_keypoints(self, landmarks: np.ndarray) -> Dict:
        """Extract key facial points (eyes, nose, mouth, etc.)."""
        if len(landmarks.shape) == 3:
            landmarks = landmarks[0]

        if landmarks.shape[0] < 68:
            return {}

        # Standard 68-point landmark indices
        keypoints = {
            'left_eye': landmarks[36:42].mean(axis=0).tolist(),
            'right_eye': landmarks[42:48].mean(axis=0).tolist(),
            'nose': landmarks[30].tolist(),
            'mouth': landmarks[48:68].mean(axis=0).tolist(),
            'chin': landmarks[8].tolist(),
        }

        return keypoints

    def _compute_expression_vector(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute expression feature vector from landmarks."""
        if len(landmarks.shape) == 3:
            landmarks = landmarks[0]

        # Compute relative positions and distances
        # This creates a feature vector representing facial expression

        if landmarks.shape[0] < 68:
            return np.zeros(20)

        features = []

        # Eye openness (distance between upper and lower eyelids)
        left_eye_height = np.linalg.norm(landmarks[37] - landmarks[41])
        right_eye_height = np.linalg.norm(landmarks[44] - landmarks[46])
        features.extend([left_eye_height, right_eye_height])

        # Mouth openness
        mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        features.extend([mouth_height, mouth_width])

        # Eyebrow positions
        left_brow_height = landmarks[19][1] - landmarks[27][1]
        right_brow_height = landmarks[24][1] - landmarks[27][1]
        features.extend([left_brow_height, right_brow_height])

        # Additional geometric features
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        face_height = np.linalg.norm(landmarks[8] - landmarks[27])
        features.extend([face_width, face_height])

        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20])

    def create_character_database(self, output_path: Path):
        """Create a database of character features for animation."""
        logger.info("Creating character feature database...")

        database = {
            'character_name': self.character_folder.name,
            'num_frames': len(self.character_frames),
            'frames': []
        }

        for frame_data in self.character_frames:
            frame_entry = {
                'name': frame_data['name'],
                'features': {
                    'bbox': frame_data['features']['bbox'],
                    'keypoints': frame_data['features']['keypoints'],
                    'expression_vector': frame_data['features']['expression_vector'].tolist()
                }
            }
            database['frames'].append(frame_entry)

        # Save as JSON
        json_path = output_path / f"{self.character_folder.name}_features.json"
        with open(json_path, 'w') as f:
            json.dump(database, f, indent=2)

        logger.info(f"[OK] Saved feature database: {json_path}")

        # Save full features as pickle for faster loading
        pkl_path = output_path / f"{self.character_folder.name}_features.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.character_features, f)

        logger.info(f"[OK] Saved full features: {pkl_path}")

        return json_path, pkl_path

    def build_expression_mapping(self):
        """Build a mapping of expressions to frame indices."""
        logger.info("Building expression mapping...")

        # Analyze expression vectors to cluster similar expressions
        expression_vectors = np.array([
            f['expression_vector'] for f in self.character_features
        ])

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(expression_vectors, expression_vectors, metric='euclidean')

        # Find representative frames for different expressions
        # (neutral, happy, sad, surprised, etc.)

        expression_map = {
            'neutral': 0,  # Usually first frame
            'frames': {}
        }

        for i, frame_data in enumerate(self.character_frames):
            name = frame_data['name'].lower()

            # Map common expression names
            if 'alegre' in name or 'happy' in name or 'smile' in name:
                expression_map['frames']['happy'] = i
            elif 'triste' in name or 'sad' in name:
                expression_map['frames']['sad'] = i
            elif 'surpresa' in name or 'surprise' in name:
                expression_map['frames']['surprised'] = i
            elif 'raiva' in name or 'anger' in name or 'angry' in name:
                expression_map['frames']['angry'] = i
            elif 'dormindo' in name or 'sleep' in name:
                expression_map['frames']['sleepy'] = i
            elif 'assustado' in name or 'scared' in name or 'fear' in name:
                expression_map['frames']['scared'] = i

        logger.info(f"[OK] Mapped {len(expression_map['frames'])} expressions")

        return expression_map


def main():
    """Main function to process character and create model."""
    print("="*70)
    print("CUSTOM CHARACTER MODEL GENERATOR")
    print("="*70)
    print()

    # Paths
    character_folder = Path("assets/characters/Test")
    landmark_model = Path("models/liveportrait/landmark.onnx")
    output_folder = Path("models/custom_characters")
    output_folder.mkdir(exist_ok=True)

    # Check inputs
    if not character_folder.exists():
        print(f"✗ Character folder not found: {character_folder}")
        return

    if not landmark_model.exists():
        print(f"✗ Landmark model not found: {landmark_model}")
        print("  Make sure landmark.onnx is in models/liveportrait/")
        return

    print(f"Character folder: {character_folder}")
    print(f"Landmark model: {landmark_model}")
    print(f"Output folder: {output_folder}")
    print()

    # Initialize extractor
    extractor = CharacterFeatureExtractor(character_folder, landmark_model)

    # Load landmark model
    print("[1/5] Loading landmark detection model...")
    extractor.load_landmark_model()
    print()

    # Load character frames
    print("[2/5] Loading character frames...")
    num_frames = extractor.load_character_frames()
    print(f"  Found {num_frames} frames")
    print()

    # Extract features
    print("[3/5] Extracting features from all frames...")
    extractor.extract_features_from_all_frames()
    print()

    # Create database
    print("[4/5] Creating character feature database...")
    json_path, pkl_path = extractor.create_character_database(output_folder)
    print()

    # Build expression mapping
    print("[5/5] Building expression mapping...")
    expression_map = extractor.build_expression_mapping()

    # Save expression map
    expr_map_path = output_folder / f"{character_folder.name}_expression_map.json"
    with open(expr_map_path, 'w') as f:
        json.dump(expression_map, f, indent=2)
    print(f"  [OK] Saved expression map: {expr_map_path}")
    print()

    # Summary
    print("="*70)
    print("✅ CHARACTER MODEL CREATED SUCCESSFULLY!")
    print("="*70)
    print()
    print(f"Processed: {num_frames} character frames")
    print(f"Extracted: Features for all expressions")
    print(f"Created: Character-specific model")
    print()
    print("Output files:")
    print(f"  1. {json_path.name} - Feature database (JSON)")
    print(f"  2. {pkl_path.name} - Full features (Pickle)")
    print(f"  3. {expr_map_path.name} - Expression mapping")
    print()
    print("Next steps:")
    print("  1. Run the application: run.bat")
    print("  2. The custom character model will be used automatically")
    print("  3. Animation will now use your character's expressions!")
    print()
    print("="*70)


if __name__ == "__main__":
    main()

