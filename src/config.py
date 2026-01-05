"""
Configuration loader for Stream Motion Animator.
Loads and validates configuration from YAML file.
"""

import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to assets/config.yaml
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "assets", "config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config is not None else {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}, using defaults")
            return self._default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'video': {
                'source': 0,
                'width': 1280,
                'height': 720,
                'fps': 30
            },
            'output': {
                'width': 1920,
                'height': 1080,
                'spout_name': 'StreamMotionAnimator',
                'ndi_name': 'Stream Motion Animator',
                'background_alpha': 0,
                'enable_spout': False,
                'enable_ndi': False
            },
            'tracking': {
                'face_enabled': True,
                'pose_enabled': True,
                'hands_enabled': True,
                'smoothing': 0.5,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            'animation': {
                'sprite_scale': 1.0,
                'movement_sensitivity': 1.0,
                'rotation_sensitivity': 1.0,
                'interpolation_speed': 0.3
            },
            'sprites': {
                'head': 'assets/sprites/head.png',
                'body': 'assets/sprites/body.png',
                'left_arm': 'assets/sprites/left_arm.png',
                'right_arm': 'assets/sprites/right_arm.png',
                'left_hand': 'assets/sprites/left_hand.png',
                'right_hand': 'assets/sprites/right_hand.png'
            },
            'performance': {
                'target_fps': 30,
                'show_fps': True,
                'show_tracking_overlay': False
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key path (e.g., 'video.width')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        print(f"Configuration reloaded from {self.config_path}")
