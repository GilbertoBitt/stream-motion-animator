"""
Base class for output handlers
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import logging


class BaseOutput(ABC):
    """Abstract base class for output handlers"""
    
    def __init__(self, name: str = "Output"):
        """
        Initialize output handler
        
        Args:
            name: Output handler name
        """
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{name}")
        self.initialized = False
        self.frame_count = 0
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start the output handler
        
        Returns:
            True if started successfully
        """
        pass
    
    @abstractmethod
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send frame to output
        
        Args:
            frame: Frame to send (BGR format)
            
        Returns:
            True if frame sent successfully
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the output handler and release resources"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if output is initialized"""
        return self.initialized
    
    def get_stats(self) -> dict:
        """Get output statistics"""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'frames_sent': self.frame_count
        }
