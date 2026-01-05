"""
Motion tracking using MediaPipe Face Mesh.

Extracts facial landmarks and movements from webcam input.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FacialLandmarks:
    """Container for facial landmark data."""
    landmarks: np.ndarray  # 468 landmarks (x, y, z)
    head_rotation: Tuple[float, float, float]  # (pitch, yaw, roll) in degrees
    left_eye_state: float  # 0.0 (closed) to 1.0 (open)
    right_eye_state: float
    mouth_open: float  # 0.0 (closed) to 1.0 (open)
    confidence: float  # Detection confidence
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, w, h)


class MotionTracker:
    """Face tracking using MediaPipe Face Mesh."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        smoothing: float = 0.5
    ):
        """
        Initialize motion tracker.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            smoothing: Smoothing factor (0.0-1.0), higher = smoother
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing = smoothing
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Previous landmarks for smoothing
        self.prev_landmarks: Optional[np.ndarray] = None
        
        # Key landmark indices for facial features
        # Eyes
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Mouth
        self.mouth_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185]
        self.mouth_inner_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        
        # Face contour for head pose estimation
        self.face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    def process_frame(self, frame: np.ndarray) -> Optional[FacialLandmarks]:
        """
        Process a frame and extract facial landmarks.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            FacialLandmarks object or None if no face detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks.landmark
        ])
        
        # Apply smoothing
        if self.prev_landmarks is not None:
            landmarks = self.smoothing * self.prev_landmarks + (1 - self.smoothing) * landmarks
        self.prev_landmarks = landmarks.copy()
        
        # Calculate facial features
        head_rotation = self._estimate_head_pose(landmarks, frame.shape)
        left_eye_state = self._calculate_eye_openness(landmarks, self.left_eye_indices)
        right_eye_state = self._calculate_eye_openness(landmarks, self.right_eye_indices)
        mouth_open = self._calculate_mouth_openness(landmarks)
        
        # Calculate bounding box
        bbox = self._calculate_bbox(landmarks)
        
        return FacialLandmarks(
            landmarks=landmarks,
            head_rotation=head_rotation,
            left_eye_state=left_eye_state,
            right_eye_state=right_eye_state,
            mouth_open=mouth_open,
            confidence=1.0,  # MediaPipe doesn't provide per-face confidence
            bbox=bbox
        )
    
    def _estimate_head_pose(self, landmarks: np.ndarray, image_shape: Tuple) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll).
        
        Args:
            landmarks: Facial landmarks array
            image_shape: Image dimensions
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # Use key facial points for pose estimation
        # Nose tip, chin, left eye, right eye, left mouth, right mouth
        image_points = np.array([
            landmarks[1][:2],    # Nose tip
            landmarks[152][:2],  # Chin
            landmarks[33][:2],   # Left eye left corner
            landmarks[263][:2],  # Right eye right corner
            landmarks[61][:2],   # Left mouth corner
            landmarks[291][:2]   # Right mouth corner
        ], dtype=np.float64)
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        
        # Camera internals
        h, w = image_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return (0.0, 0.0, 0.0)
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return (pitch, yaw, roll)
    
    def _calculate_eye_openness(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Calculate eye openness (0.0 = closed, 1.0 = open).
        
        Args:
            landmarks: Facial landmarks
            eye_indices: Indices of eye landmarks
            
        Returns:
            Eye openness value
        """
        # Get eye landmarks
        eye_points = landmarks[eye_indices]
        
        # Calculate vertical distance
        vertical_dist = np.linalg.norm(eye_points[1][:2] - eye_points[5][:2])
        
        # Calculate horizontal distance
        horizontal_dist = np.linalg.norm(eye_points[0][:2] - eye_points[3][:2])
        
        # Eye aspect ratio
        if horizontal_dist > 0:
            ear = vertical_dist / horizontal_dist
        else:
            ear = 0.0
        
        # Normalize to 0-1 range (typical EAR when open is around 0.2-0.3)
        openness = min(1.0, ear / 0.25)
        
        return openness
    
    def _calculate_mouth_openness(self, landmarks: np.ndarray) -> float:
        """
        Calculate mouth openness (0.0 = closed, 1.0 = open).
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Mouth openness value
        """
        # Get mouth landmarks
        outer_top = landmarks[13][:2]
        outer_bottom = landmarks[14][:2]
        outer_left = landmarks[61][:2]
        outer_right = landmarks[291][:2]
        
        # Calculate vertical and horizontal distances
        vertical_dist = np.linalg.norm(outer_top - outer_bottom)
        horizontal_dist = np.linalg.norm(outer_left - outer_right)
        
        # Mouth aspect ratio
        if horizontal_dist > 0:
            mar = vertical_dist / horizontal_dist
        else:
            mar = 0.0
        
        # Normalize to 0-1 range (typical MAR when open is around 0.5-0.7)
        openness = min(1.0, mar / 0.6)
        
        return openness
    
    def _calculate_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box for face.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (x, y, width, height)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: FacialLandmarks) -> np.ndarray:
        """
        Draw facial landmarks on frame for visualization.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks to draw
            
        Returns:
            Frame with landmarks drawn
        """
        output = frame.copy()
        
        # Draw all landmarks
        for point in landmarks.landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
        
        # Draw bounding box
        x, y, w, h = landmarks.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display info
        pitch, yaw, roll = landmarks.head_rotation
        cv2.putText(output, f"Pitch: {pitch:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"Yaw: {yaw:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"Roll: {roll:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"Mouth: {landmarks.mouth_open:.2f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output
    
    def cleanup(self) -> None:
        """Cleanup MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
    
    def __del__(self):
        """Destructor to cleanup resources."""
        self.cleanup()
