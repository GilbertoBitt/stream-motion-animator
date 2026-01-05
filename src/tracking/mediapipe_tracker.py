"""
MediaPipe face tracking implementation
"""
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class MediaPipeTracker:
    """Face landmark detection using MediaPipe"""
    
    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True
    ):
        """
        Initialize MediaPipe face tracker
        
        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            refine_landmarks: Whether to refine landmark positions
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )
        
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key landmark indices for face regions
        self.key_landmarks = {
            'left_eye': [33, 133, 157, 158, 159, 160, 161, 163, 144, 145, 153, 154, 155],
            'right_eye': [362, 263, 387, 386, 385, 384, 398, 382, 381, 373, 374, 380],
            'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312],
            'nose': [1, 2, 98, 327],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        # Landmark smoothing
        self.smoothing_enabled = True
        self.smoothing_window = 5
        self.landmark_history: List[np.ndarray] = []
        
        # Statistics
        self.detection_count = 0
        self.no_detection_count = 0
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face landmarks in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing landmarks and metadata, or None if no face detected
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.no_detection_count += 1
                return None
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            h, w = frame.shape[:2]
            landmarks = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Apply smoothing
            if self.smoothing_enabled:
                landmarks = self._smooth_landmarks(landmarks)
            
            # Extract key landmark groups
            key_points = self._extract_key_landmarks(landmarks)
            
            # Calculate bounding box
            bbox = self._calculate_bbox(landmarks)
            
            # Calculate face angle
            angles = self._estimate_pose(landmarks)
            
            self.detection_count += 1
            
            return {
                'landmarks': landmarks,
                'key_points': key_points,
                'bbox': bbox,
                'angles': angles,
                'confidence': 1.0  # MediaPipe doesn't provide per-face confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting landmarks: {e}")
            return None
    
    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to landmarks"""
        self.landmark_history.append(landmarks)
        
        # Keep only recent history
        if len(self.landmark_history) > self.smoothing_window:
            self.landmark_history.pop(0)
        
        # Average over history
        if len(self.landmark_history) > 1:
            smoothed = np.mean(self.landmark_history, axis=0)
            return smoothed.astype(np.float32)
        
        return landmarks
    
    def _extract_key_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract key landmark groups"""
        key_points = {}
        
        for name, indices in self.key_landmarks.items():
            try:
                key_points[name] = landmarks[indices]
            except IndexError:
                # If indices are out of range, skip this group
                pass
        
        return key_points
    
    def _calculate_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from landmarks
        
        Returns:
            (x, y, width, height)
        """
        x_min = int(landmarks[:, 0].min())
        y_min = int(landmarks[:, 1].min())
        x_max = int(landmarks[:, 0].max())
        y_max = int(landmarks[:, 1].max())
        
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)
    
    def _estimate_pose(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Estimate head pose angles (yaw, pitch, roll)
        Simple estimation based on landmark positions
        
        Returns:
            Dictionary with 'yaw', 'pitch', 'roll' angles in degrees
        """
        # Use key points for pose estimation
        # Simplified approach - for production, use proper 3D pose estimation
        
        # Nose tip
        nose = landmarks[1] if len(landmarks) > 1 else landmarks[0]
        
        # Eye centers
        left_eye_indices = self.key_landmarks['left_eye']
        right_eye_indices = self.key_landmarks['right_eye']
        
        if len(landmarks) > max(max(left_eye_indices), max(right_eye_indices)):
            left_eye = landmarks[left_eye_indices].mean(axis=0)
            right_eye = landmarks[right_eye_indices].mean(axis=0)
            
            # Calculate yaw (horizontal rotation)
            eye_center = (left_eye + right_eye) / 2
            yaw = np.arctan2(nose[0] - eye_center[0], 100) * 180 / np.pi
            
            # Calculate roll (tilt)
            roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
            
            # Calculate pitch (vertical rotation) - simplified
            pitch = (nose[1] - eye_center[1]) / 10.0
            
        else:
            yaw = 0.0
            pitch = 0.0
            roll = 0.0
        
        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll)
        }
    
    def draw_landmarks(self, frame: np.ndarray, detection: Dict, 
                       draw_connections: bool = True) -> np.ndarray:
        """
        Draw landmarks on frame for visualization
        
        Args:
            frame: Input frame
            detection: Detection result from detect()
            draw_connections: Whether to draw connections between landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if detection is None:
            return frame
        
        result = frame.copy()
        landmarks = detection['landmarks']
        
        # Draw landmarks
        for x, y in landmarks:
            cv2.circle(result, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        # Draw bounding box
        if 'bbox' in detection:
            x, y, w, h = detection['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw angles
        if 'angles' in detection:
            angles = detection['angles']
            y_offset = 30
            for name, value in angles.items():
                text = f"{name}: {value:.1f}"
                cv2.putText(result, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        return result
    
    def reset(self) -> None:
        """Reset tracker state"""
        self.landmark_history.clear()
        self.detection_count = 0
        self.no_detection_count = 0
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        total = self.detection_count + self.no_detection_count
        detection_rate = (self.detection_count / total * 100) if total > 0 else 0
        
        return {
            'detections': self.detection_count,
            'no_detections': self.no_detection_count,
            'detection_rate': detection_rate
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
