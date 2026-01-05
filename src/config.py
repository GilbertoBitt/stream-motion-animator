"""
Configuration loader for the Stream Motion Animator.

This module handles loading and validating configuration from config.yaml.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to assets/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "assets" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file or use the default template."
            )
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['video', 'ai_model', 'character', 'output', 
                           'performance', 'tracking', 'hotkeys']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section '{section}' in config")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'video.width')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'video.width')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save to. If None, uses original config_path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    @property
    def video_source(self) -> int:
        """Get video source (webcam index)."""
        return self.get('video.source', 0)
    
    @property
    def video_width(self) -> int:
        """Get video capture width."""
        return self.get('video.width', 1280)
    
    @property
    def video_height(self) -> int:
        """Get video capture height."""
        return self.get('video.height', 720)
    
    @property
    def video_fps(self) -> int:
        """Get target video FPS."""
        return self.get('video.fps', 60)
    
    @property
    def model_type(self) -> str:
        """Get AI model type."""
        return self.get('ai_model.type', 'liveportrait')
    
    @property
    def model_path(self) -> str:
        """Get AI model path."""
        return self.get('ai_model.model_path', 'models/liveportrait')
    
    @property
    def device(self) -> str:
        """Get compute device (cuda/cpu)."""
        return self.get('ai_model.device', 'cuda')
    
    @property
    def use_fp16(self) -> bool:
        """Check if FP16 precision is enabled."""
        return self.get('ai_model.fp16', True)
    
    @property
    def characters_path(self) -> str:
        """Get characters directory path."""
        return self.get('character.images_path', 'assets/characters/')
    
    @property
    def output_width(self) -> int:
        """Get output width."""
        return self.get('output.width', 1920)
    
    @property
    def output_height(self) -> int:
        """Get output height."""
        return self.get('output.height', 1080)
    
    @property
    def target_fps(self) -> int:
        """Get target FPS."""
        return self.get('performance.target_fps', 60)
    
    @property
    def async_pipeline(self) -> bool:
        """Check if async pipeline is enabled."""
        return self.get('performance.async_pipeline', True)
    
    @property
    def spout_enabled(self) -> bool:
        """Check if Spout output is enabled."""
        return self.get('output.spout_enabled', True)
    
    @property
    def ndi_enabled(self) -> bool:
        """Check if NDI output is enabled."""
        return self.get('output.ndi_enabled', True)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Config object
    """
    return Config(config_path)
