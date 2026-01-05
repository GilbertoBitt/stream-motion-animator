"""
Basic tests for the animator pipeline
"""
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config
from src.utils.config import Config as ConfigClass


def test_config_loading():
    """Test configuration loading"""
    config = Config()
    assert config is not None
    assert isinstance(config, ConfigClass)


def test_config_get():
    """Test configuration get method"""
    config = Config()
    
    # Test existing key
    device_id = config.get('capture.device_id', 0)
    assert device_id is not None
    
    # Test non-existing key with default
    value = config.get('non.existing.key', 'default')
    assert value == 'default'


def test_config_set():
    """Test configuration set method"""
    config = Config()
    
    # Set value
    config.set('test.key', 123)
    assert config.get('test.key') == 123


def test_config_nested():
    """Test nested configuration access"""
    config = Config()
    
    # Test nested structure
    config.set('level1.level2.level3', 'value')
    assert config.get('level1.level2.level3') == 'value'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
