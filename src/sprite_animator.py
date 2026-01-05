"""
2D Sprite-based animation system.
Maps motion tracking data to sprite transformations.
"""

import numpy as np
import pygame
import os
from typing import Dict, Optional, Tuple
from PIL import Image


class SpriteAnimator:
    """Handles sprite loading, transformation, and rendering."""
    
    def __init__(self, config):
        """
        Initialize sprite animator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize pygame for rendering
        pygame.init()
        
        # Set up a display (can be hidden) to enable image loading
        # Use HIDDEN flag if available, otherwise use NOFRAME
        try:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
        except:
            try:
                pygame.display.set_mode((1, 1), pygame.NOFRAME)
            except:
                # Fallback for headless environments
                pass
        
        # Get output dimensions
        self.width = config.get('output.width', 1920)
        self.height = config.get('output.height', 1080)
        
        # Create surface with per-pixel alpha
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Animation settings
        self.sprite_scale = config.get('animation.sprite_scale', 1.0)
        self.movement_sensitivity = config.get('animation.movement_sensitivity', 1.0)
        self.rotation_sensitivity = config.get('animation.rotation_sensitivity', 1.0)
        self.interpolation_speed = config.get('animation.interpolation_speed', 0.3)
        
        # Background alpha
        self.bg_alpha = config.get('output.background_alpha', 0)
        
        # Load sprites
        self.sprites = self._load_sprites()
        
        # Current and target transformations for smooth interpolation
        self.current_transforms = {
            'head': {'pos': (self.width // 2, self.height // 2), 'rotation': 0, 'scale': 1.0},
            'body': {'pos': (self.width // 2, self.height // 2), 'rotation': 0, 'scale': 1.0},
            'left_arm': {'pos': (self.width // 2, self.height // 2), 'rotation': 0, 'scale': 1.0},
            'right_arm': {'pos': (self.width // 2, self.height // 2), 'rotation': 0, 'scale': 1.0},
            'left_hand': {'pos': (self.width // 2, self.height // 2), 'rotation': 0, 'scale': 1.0},
            'right_hand': {'pos': (self.width // 2, self.height // 2), 'rotation': 0, 'scale': 1.0}
        }
        
        self.target_transforms = self.current_transforms.copy()
    
    def _load_sprites(self) -> Dict[str, Optional[pygame.Surface]]:
        """Load sprite images from configuration."""
        sprites = {}
        sprite_paths = self.config.get('sprites', {})
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for sprite_name, sprite_path in sprite_paths.items():
            full_path = os.path.join(base_dir, sprite_path)
            
            if os.path.exists(full_path):
                try:
                    # Load with pygame to preserve alpha
                    sprite = pygame.image.load(full_path).convert_alpha()
                    sprites[sprite_name] = sprite
                    print(f"Loaded sprite: {sprite_name} from {full_path}")
                except Exception as e:
                    print(f"Warning: Could not load sprite {sprite_name}: {e}")
                    sprites[sprite_name] = None
            else:
                print(f"Warning: Sprite not found: {full_path}")
                sprites[sprite_name] = self._create_placeholder_sprite(sprite_name)
        
        return sprites
    
    def _create_placeholder_sprite(self, name: str) -> pygame.Surface:
        """Create a placeholder sprite for testing."""
        # Create a simple colored rectangle as placeholder
        size = 100
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Different colors for different sprite types
        colors = {
            'head': (255, 200, 200, 255),
            'body': (200, 255, 200, 255),
            'left_arm': (200, 200, 255, 255),
            'right_arm': (200, 200, 255, 255),
            'left_hand': (255, 255, 200, 255),
            'right_hand': (255, 255, 200, 255)
        }
        
        color = colors.get(name, (200, 200, 200, 255))
        pygame.draw.circle(surface, color, (size // 2, size // 2), size // 2)
        
        # Add label
        font = pygame.font.Font(None, 20)
        text = font.render(name[:4], True, (0, 0, 0))
        text_rect = text.get_rect(center=(size // 2, size // 2))
        surface.blit(text, text_rect)
        
        return surface
    
    def update(self, tracking_data: Dict):
        """
        Update sprite transformations based on tracking data.
        
        Args:
            tracking_data: Tracking data from MotionTracker
        """
        # Update head transform based on face tracking
        if tracking_data['face']:
            self._update_head_transform(tracking_data['face'])
        
        # Update body transform based on pose tracking
        if tracking_data['pose']:
            self._update_body_transform(tracking_data['pose'])
        
        # Update hand transforms based on hand tracking
        if tracking_data['hands']:
            self._update_hands_transform(tracking_data['hands'])
        
        # Interpolate current transforms towards target
        self._interpolate_transforms()
    
    def _update_head_transform(self, face_data: Dict):
        """Update head sprite transformation."""
        pos = face_data['position']
        rotation = face_data['rotation']
        
        # Map normalized position to screen coordinates
        screen_x = pos[0] * self.width * self.movement_sensitivity
        screen_y = pos[1] * self.height * self.movement_sensitivity
        
        # Apply rotation (convert from radians to degrees)
        roll_deg = np.degrees(rotation[2]) * self.rotation_sensitivity
        
        self.target_transforms['head'] = {
            'pos': (screen_x, screen_y),
            'rotation': -roll_deg,  # Negative for correct direction
            'scale': self.sprite_scale
        }
    
    def _update_body_transform(self, pose_data: Dict):
        """Update body sprite transformation."""
        torso_pos = pose_data['torso_position']
        torso_rotation = pose_data['torso_rotation']
        
        # Map to screen coordinates
        screen_x = torso_pos[0] * self.width * self.movement_sensitivity
        screen_y = torso_pos[1] * self.height * self.movement_sensitivity
        
        # Apply rotation
        rotation_deg = np.degrees(torso_rotation) * self.rotation_sensitivity
        
        self.target_transforms['body'] = {
            'pos': (screen_x, screen_y),
            'rotation': -rotation_deg,
            'scale': self.sprite_scale
        }
        
        # Update arm positions based on shoulders
        left_shoulder = pose_data['left_shoulder']
        right_shoulder = pose_data['right_shoulder']
        
        self.target_transforms['left_arm'] = {
            'pos': (left_shoulder[0] * self.width * self.movement_sensitivity,
                   left_shoulder[1] * self.height * self.movement_sensitivity),
            'rotation': -rotation_deg * 0.5,
            'scale': self.sprite_scale * 0.8
        }
        
        self.target_transforms['right_arm'] = {
            'pos': (right_shoulder[0] * self.width * self.movement_sensitivity,
                   right_shoulder[1] * self.height * self.movement_sensitivity),
            'rotation': -rotation_deg * 0.5,
            'scale': self.sprite_scale * 0.8
        }
    
    def _update_hands_transform(self, hands_data: Dict):
        """Update hand sprite transformations."""
        for hand_side in ['left', 'right']:
            hand_data = hands_data[hand_side]
            if hand_data:
                wrist_pos = hand_data['wrist']
                
                screen_x = wrist_pos[0] * self.width * self.movement_sensitivity
                screen_y = wrist_pos[1] * self.height * self.movement_sensitivity
                
                sprite_key = f'{hand_side}_hand'
                self.target_transforms[sprite_key] = {
                    'pos': (screen_x, screen_y),
                    'rotation': 0,
                    'scale': self.sprite_scale * 0.6
                }
    
    def _interpolate_transforms(self):
        """Smoothly interpolate current transforms towards target."""
        alpha = self.interpolation_speed
        
        for key in self.current_transforms:
            current = self.current_transforms[key]
            target = self.target_transforms.get(key, current)
            
            # Interpolate position
            current['pos'] = (
                current['pos'][0] * (1 - alpha) + target['pos'][0] * alpha,
                current['pos'][1] * (1 - alpha) + target['pos'][1] * alpha
            )
            
            # Interpolate rotation
            current['rotation'] = current['rotation'] * (1 - alpha) + target['rotation'] * alpha
            
            # Interpolate scale
            current['scale'] = current['scale'] * (1 - alpha) + target['scale'] * alpha
    
    def render(self) -> np.ndarray:
        """
        Render sprites to output frame.
        
        Returns:
            Rendered frame as numpy array (RGBA)
        """
        # Clear surface with transparent or specified background
        bg_color = (0, 0, 0, self.bg_alpha)
        self.surface.fill(bg_color)
        
        # Render sprites in order (back to front)
        render_order = ['body', 'left_arm', 'right_arm', 'head', 'left_hand', 'right_hand']
        
        for sprite_name in render_order:
            sprite = self.sprites.get(sprite_name)
            if sprite:
                transform = self.current_transforms[sprite_name]
                self._render_sprite(sprite, transform)
        
        # Convert pygame surface to numpy array
        # Get string buffer from surface
        frame_str = pygame.image.tostring(self.surface, 'RGBA')
        frame_array = np.frombuffer(frame_str, dtype=np.uint8)
        frame_array = frame_array.reshape((self.height, self.width, 4))
        
        return frame_array
    
    def _render_sprite(self, sprite: pygame.Surface, transform: Dict):
        """Render a single sprite with transformation."""
        # Apply scale
        scale = transform['scale']
        new_width = int(sprite.get_width() * scale)
        new_height = int(sprite.get_height() * scale)
        
        if new_width > 0 and new_height > 0:
            scaled_sprite = pygame.transform.scale(sprite, (new_width, new_height))
            
            # Apply rotation
            rotation = transform['rotation']
            rotated_sprite = pygame.transform.rotate(scaled_sprite, rotation)
            
            # Get position (center)
            pos = transform['pos']
            rect = rotated_sprite.get_rect(center=pos)
            
            # Blit to surface
            self.surface.blit(rotated_sprite, rect)
    
    def get_frame_bgr(self) -> np.ndarray:
        """
        Get rendered frame in BGR format for OpenCV compatibility.
        
        Returns:
            Frame in BGR format with alpha channel
        """
        rgba_frame = self.render()
        # Convert RGBA to BGRA for OpenCV
        bgra_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
        return bgra_frame
    
    def close(self):
        """Cleanup resources."""
        pygame.quit()


# OpenCV import for color conversion
import cv2
