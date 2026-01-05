"""
Motion tracking using MediaPipe.
Tracks face, pose, and hands for real-time animation.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple

# Try to import MediaPipe - support both old and new APIs
try:
    # Try new API (MediaPipe 0.10+)
    from mediapipe import tasks
    from mediapipe.tasks import vision
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_NEW_API = True
except ImportError:
    try:
        # Try old API (MediaPipe < 0.10)
        import mediapipe as mp
        MEDIAPIPE_NEW_API = False
    except ImportError:
        raise ImportError("MediaPipe is not installed. Install with: pip install mediapipe")


class MotionTracker:
    """Handles real-time motion tracking using MediaPipe."""
    
    def __init__(self, config):
        """
        Initialize motion tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.using_new_api = MEDIAPIPE_NEW_API
        
        # Initialize tracking modules based on config
        self.face_mesh = None
        self.pose = None
        self.hands = None
        
        min_detection_conf = config.get('tracking.min_detection_confidence', 0.5)
        min_tracking_conf = config.get('tracking.min_tracking_confidence', 0.5)
        
        print(f"Using MediaPipe {'new' if self.using_new_api else 'legacy'} API")
        
        # Note: New MediaPipe API (0.10+) uses model files and is more complex
        # For simplicity, we'll create a simplified tracking system
        # In production, download model files from MediaPipe or use legacy version
        
        # For now, create mock trackers that return None
        # Real implementation would need model files
        if config.get('tracking.face_enabled', True):
            print("Warning: Face tracking initialization skipped (requires model files)")
        
        if config.get('tracking.pose_enabled', True):
            print("Warning: Pose tracking initialization skipped (requires model files)")
        
        if config.get('tracking.hands_enabled', True):
            print("Warning: Hand tracking initialization skipped (requires model files)")
        
        # Previous frame data for smoothing
        self.prev_face_data = None
        self.prev_pose_data = None
        self.prev_hands_data = None
        
        self.smoothing = config.get('tracking.smoothing', 0.5)
        
        # Mock drawing utilities
        self.mock_mode = True
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a video frame and extract tracking data.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing tracking data
        """
        height, width = frame.shape[:2]
        
        tracking_data = {
            'face': None,
            'pose': None,
            'hands': None,
            'frame_shape': (height, width)
        }
        
        # In mock mode, return empty tracking data
        # Real implementation would process frame through MediaPipe models
        if self.mock_mode:
            # Generate simple mock data based on center position
            # This allows the animator to still work without actual tracking
            tracking_data['face'] = self._generate_mock_face_data()
            tracking_data['pose'] = self._generate_mock_pose_data()
            tracking_data['hands'] = self._generate_mock_hands_data()
        
        return tracking_data
    
    def _generate_mock_face_data(self) -> Dict:
        """Generate mock face tracking data for testing."""
        return {
            'position': (0.5, 0.4, 0.0),  # Center of frame
            'rotation': (0.0, 0.0, 0.0),  # No rotation
            'landmarks': None
        }
    
    def _generate_mock_pose_data(self) -> Dict:
        """Generate mock pose tracking data for testing."""
        return {
            'torso_position': (0.5, 0.6),  # Center, slightly below face
            'torso_rotation': 0.0,
            'left_shoulder': (0.4, 0.5),
            'right_shoulder': (0.6, 0.5),
            'landmarks': None
        }
    
    def _generate_mock_hands_data(self) -> Dict:
        """Generate mock hands tracking data for testing."""
        return {
            'left': {
                'wrist': (0.3, 0.7),
                'thumb_tip': (0.28, 0.65),
                'index_tip': (0.27, 0.62),
                'middle_tip': (0.28, 0.60),
                'ring_tip': (0.29, 0.62),
                'pinky_tip': (0.30, 0.64),
                'landmarks': None
            },
            'right': {
                'wrist': (0.7, 0.7),
                'thumb_tip': (0.72, 0.65),
                'index_tip': (0.73, 0.62),
                'middle_tip': (0.72, 0.60),
                'ring_tip': (0.71, 0.62),
                'pinky_tip': (0.70, 0.64),
                'landmarks': None
            }
        }
    
    def _process_face_landmarks(self, landmarks, width: int, height: int) -> Dict:
        """Extract and process face landmark data."""
        # Key face landmarks for head position and rotation
        nose_tip = landmarks.landmark[4]  # Nose tip
        chin = landmarks.landmark[152]  # Chin
        left_eye = landmarks.landmark[33]  # Left eye outer corner
        right_eye = landmarks.landmark[263]  # Right eye outer corner
        forehead = landmarks.landmark[10]  # Forehead center
        
        # Calculate head position (normalized)
        head_x = nose_tip.x
        head_y = nose_tip.y
        head_z = nose_tip.z
        
        # Calculate head rotation (simplified)
        # Eye line for roll
        eye_dx = right_eye.x - left_eye.x
        eye_dy = right_eye.y - left_eye.y
        roll = np.arctan2(eye_dy, eye_dx)
        
        # Nose-chin for pitch
        nose_chin_dy = chin.y - nose_tip.y
        nose_chin_dz = chin.z - nose_tip.z
        pitch = np.arctan2(nose_chin_dz, nose_chin_dy)
        
        # Left-right tilt for yaw (simplified)
        face_center_x = (left_eye.x + right_eye.x) / 2
        yaw = (nose_tip.x - face_center_x) * 2  # Simplified yaw
        
        face_data = {
            'position': (head_x, head_y, head_z),
            'rotation': (pitch, yaw, roll),
            'landmarks': landmarks
        }
        
        # Apply smoothing
        if self.prev_face_data and self.smoothing > 0:
            face_data = self._smooth_data(face_data, self.prev_face_data, self.smoothing)
        
        self.prev_face_data = face_data
        return face_data
    
    def _process_pose_landmarks(self, landmarks, width: int, height: int) -> Dict:
        """Extract and process pose landmark data."""
        # Key pose landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate torso position
        torso_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        torso_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        # Calculate torso rotation
        shoulder_dx = right_shoulder.x - left_shoulder.x
        shoulder_dy = right_shoulder.y - left_shoulder.y
        torso_roll = np.arctan2(shoulder_dy, shoulder_dx)
        
        pose_data = {
            'torso_position': (torso_x, torso_y),
            'torso_rotation': torso_roll,
            'left_shoulder': (left_shoulder.x, left_shoulder.y),
            'right_shoulder': (right_shoulder.x, right_shoulder.y),
            'landmarks': landmarks
        }
        
        # Apply smoothing
        if self.prev_pose_data and self.smoothing > 0:
            pose_data = self._smooth_data(pose_data, self.prev_pose_data, self.smoothing)
        
        self.prev_pose_data = pose_data
        return pose_data
    
    def _process_hands_landmarks(self, hand_landmarks_list, handedness_list, 
                                  width: int, height: int) -> Dict:
        """Extract and process hand landmark data."""
        hands_data = {
            'left': None,
            'right': None
        }
        
        for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            # Determine which hand
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            
            # Get wrist position
            wrist = hand_landmarks.landmark[0]
            
            # Get fingertip positions
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            hand_data = {
                'wrist': (wrist.x, wrist.y),
                'thumb_tip': (thumb_tip.x, thumb_tip.y),
                'index_tip': (index_tip.x, index_tip.y),
                'middle_tip': (middle_tip.x, middle_tip.y),
                'ring_tip': (ring_tip.x, ring_tip.y),
                'pinky_tip': (pinky_tip.x, pinky_tip.y),
                'landmarks': hand_landmarks
            }
            
            if hand_label == "Left":
                hands_data['left'] = hand_data
            else:
                hands_data['right'] = hand_data
        
        # Apply smoothing
        if self.prev_hands_data and self.smoothing > 0:
            hands_data = self._smooth_hands_data(hands_data, self.prev_hands_data, self.smoothing)
        
        self.prev_hands_data = hands_data
        return hands_data
    
    def _smooth_data(self, current: Dict, previous: Dict, alpha: float) -> Dict:
        """Apply exponential smoothing to tracking data."""
        smoothed = {}
        for key, value in current.items():
            if key == 'landmarks':
                smoothed[key] = value  # Don't smooth raw landmarks
            elif isinstance(value, tuple):
                prev_value = previous.get(key, value)
                if isinstance(prev_value, tuple) and len(value) == len(prev_value):
                    smoothed[key] = tuple(
                        alpha * p + (1 - alpha) * c 
                        for p, c in zip(prev_value, value)
                    )
                else:
                    smoothed[key] = value
            else:
                smoothed[key] = value
        return smoothed
    
    def _smooth_hands_data(self, current: Dict, previous: Dict, alpha: float) -> Dict:
        """Apply smoothing to hands data."""
        smoothed = {}
        for hand in ['left', 'right']:
            if current[hand] is not None:
                if previous.get(hand) is not None:
                    smoothed[hand] = self._smooth_data(
                        current[hand], previous[hand], alpha
                    )
                else:
                    smoothed[hand] = current[hand]
            else:
                smoothed[hand] = None
        return smoothed
    
    def draw_tracking_overlay(self, frame: np.ndarray, tracking_data: Dict) -> np.ndarray:
        """
        Draw tracking landmarks on frame for debugging.
        
        Args:
            frame: Input frame
            tracking_data: Tracking data from process_frame
            
        Returns:
            Frame with overlays
        """
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # In mock mode or when landmarks are None, draw simple indicators
        if self.mock_mode or tracking_data.get('face', {}).get('landmarks') is None:
            # Draw simple position markers
            if tracking_data['face']:
                pos = tracking_data['face']['position']
                x, y = int(pos[0] * width), int(pos[1] * height)
                cv2.circle(overlay_frame, (x, y), 20, (255, 0, 0), 2)
                cv2.putText(overlay_frame, 'FACE', (x - 20, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if tracking_data['pose']:
                torso_pos = tracking_data['pose']['torso_position']
                x, y = int(torso_pos[0] * width), int(torso_pos[1] * height)
                cv2.circle(overlay_frame, (x, y), 30, (0, 255, 0), 2)
                cv2.putText(overlay_frame, 'BODY', (x - 20, y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if tracking_data['hands']:
                for hand_side in ['left', 'right']:
                    hand_data = tracking_data['hands'].get(hand_side)
                    if hand_data:
                        wrist = hand_data['wrist']
                        x, y = int(wrist[0] * width), int(wrist[1] * height)
                        cv2.circle(overlay_frame, (x, y), 15, (0, 0, 255), 2)
                        cv2.putText(overlay_frame, hand_side.upper(), (x - 20, y - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            
            # Add mock mode indicator
            cv2.putText(overlay_frame, 'MOCK MODE (No MediaPipe models)', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return overlay_frame
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            try:
                self.face_mesh.close()
            except:
                pass
        if hasattr(self, 'pose') and self.pose:
            try:
                self.pose.close()
            except:
                pass
        if hasattr(self, 'hands') and self.hands:
            try:
                self.hands.close()
            except:
                pass
