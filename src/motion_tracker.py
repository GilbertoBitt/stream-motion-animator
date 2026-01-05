"""
Motion tracking using MediaPipe.
Tracks face, pose, and hands for real-time animation.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Tuple


class MotionTracker:
    """Handles real-time motion tracking using MediaPipe."""
    
    def __init__(self, config):
        """
        Initialize motion tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize tracking modules based on config
        self.face_mesh = None
        self.pose = None
        self.hands = None
        
        min_detection_conf = config.get('tracking.min_detection_confidence', 0.5)
        min_tracking_conf = config.get('tracking.min_tracking_confidence', 0.5)
        
        if config.get('tracking.face_enabled', True):
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf
            )
        
        if config.get('tracking.pose_enabled', True):
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf
            )
        
        if config.get('tracking.hands_enabled', True):
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf
            )
        
        # Previous frame data for smoothing
        self.prev_face_data = None
        self.prev_pose_data = None
        self.prev_hands_data = None
        
        self.smoothing = config.get('tracking.smoothing', 0.5)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a video frame and extract tracking data.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing tracking data
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        tracking_data = {
            'face': None,
            'pose': None,
            'hands': None,
            'frame_shape': (height, width)
        }
        
        # Process face tracking
        if self.face_mesh:
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                tracking_data['face'] = self._process_face_landmarks(
                    face_landmarks, width, height
                )
        
        # Process pose tracking
        if self.pose:
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                tracking_data['pose'] = self._process_pose_landmarks(
                    pose_results.pose_landmarks, width, height
                )
        
        # Process hand tracking
        if self.hands:
            hands_results = self.hands.process(rgb_frame)
            if hands_results.multi_hand_landmarks:
                tracking_data['hands'] = self._process_hands_landmarks(
                    hands_results.multi_hand_landmarks,
                    hands_results.multi_handedness,
                    width, height
                )
        
        return tracking_data
    
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
        
        # Draw face landmarks
        if tracking_data['face'] and 'landmarks' in tracking_data['face']:
            self.mp_drawing.draw_landmarks(
                overlay_frame,
                tracking_data['face']['landmarks'],
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Draw pose landmarks
        if tracking_data['pose'] and 'landmarks' in tracking_data['pose']:
            self.mp_drawing.draw_landmarks(
                overlay_frame,
                tracking_data['pose']['landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if tracking_data['hands']:
            for hand in ['left', 'right']:
                if tracking_data['hands'][hand] and 'landmarks' in tracking_data['hands'][hand]:
                    self.mp_drawing.draw_landmarks(
                        overlay_frame,
                        tracking_data['hands'][hand]['landmarks'],
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                    )
        
        return overlay_frame
    
    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
