"""
Stream Motion Animator - AI-Driven Live Portrait Animation System

A high-performance AI-based character animation system for real-time streaming.
"""

__version__ = "1.0.0"
__author__ = "GilbertoBitt"
__license__ = "MIT"

from .config import Config, load_config
from .motion_tracker import MotionTracker, FacialLandmarks
from .character_manager import CharacterManager
from .ai_animator import AIAnimator
from .output_manager import OutputManager
from .performance_monitor import PerformanceMonitor, PerformanceStats

__all__ = [
    'Config',
    'load_config',
    'MotionTracker',
    'FacialLandmarks',
    'CharacterManager',
    'AIAnimator',
    'OutputManager',
    'PerformanceMonitor',
    'PerformanceStats',
]
