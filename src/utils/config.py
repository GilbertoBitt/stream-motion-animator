"""
Configuration management utilities
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager with support for defaults and overrides"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml
        """
        self.config_dir = Path(__file__).parent.parent.parent
        
        # Load default configuration
        default_config_path = self.config_dir / "config.yaml"
        self.config = self._load_yaml(default_config_path)
        
        # Load custom configuration if provided
        if config_path:
            custom_config = self._load_yaml(config_path)
            self.config = self._merge_configs(self.config, custom_config)
        
        # Load local overrides if they exist
        local_config_path = self.config_dir / "config_local.yaml"
        if local_config_path.exists():
            local_config = self._load_yaml(local_config_path)
            self.config = self._merge_configs(self.config, local_config)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {path}: {e}")
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'capture.fps')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'capture.fps')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file
        
        Args:
            path: Path to save config. If None, saves to config_local.yaml
        """
        if path is None:
            path = self.config_dir / "config_local.yaml"
        else:
            path = Path(path)
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment"""
        self.set(key, value)
