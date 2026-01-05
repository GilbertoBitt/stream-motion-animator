"""
Performance monitoring and profiling for the animation system.

Tracks FPS, GPU usage, memory consumption, and provides real-time statistics.
"""

import time
import psutil
from collections import deque
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class PerformanceStats:
    """Container for performance statistics."""
    capture_fps: float = 0.0
    tracking_fps: float = 0.0
    inference_fps: float = 0.0
    output_fps: float = 0.0
    total_fps: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    inference_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, window_size: int = 30, enable_gpu: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of samples to average over
            enable_gpu: Whether to monitor GPU stats (requires pynvml)
        """
        self.window_size = window_size
        self.enable_gpu = enable_gpu
        
        # FPS tracking for different stages
        self.capture_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.output_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
        
        # Last timestamp for each stage
        self.last_capture_time = None
        self.last_tracking_time = None
        self.last_inference_time = None
        self.last_output_time = None
        self.last_total_time = None
        
        # GPU monitoring
        self.gpu_handle = None
        if enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml = pynvml
            except (ImportError, Exception) as e:
                print(f"Warning: GPU monitoring disabled: {e}")
                self.enable_gpu = False
    
    def tick_capture(self) -> None:
        """Mark a capture frame tick."""
        current_time = time.time()
        if self.last_capture_time is not None:
            delta = current_time - self.last_capture_time
            self.capture_times.append(delta)
        self.last_capture_time = current_time
    
    def tick_tracking(self) -> None:
        """Mark a tracking frame tick."""
        current_time = time.time()
        if self.last_tracking_time is not None:
            delta = current_time - self.last_tracking_time
            self.tracking_times.append(delta)
        self.last_tracking_time = current_time
    
    def tick_inference(self) -> None:
        """Mark an inference frame tick."""
        current_time = time.time()
        if self.last_inference_time is not None:
            delta = current_time - self.last_inference_time
            self.inference_times.append(delta)
        self.last_inference_time = current_time
    
    def tick_output(self) -> None:
        """Mark an output frame tick."""
        current_time = time.time()
        if self.last_output_time is not None:
            delta = current_time - self.last_output_time
            self.output_times.append(delta)
        self.last_output_time = current_time
    
    def tick_total(self) -> None:
        """Mark a total pipeline frame tick."""
        current_time = time.time()
        if self.last_total_time is not None:
            delta = current_time - self.last_total_time
            self.total_times.append(delta)
        self.last_total_time = current_time
    
    def _calculate_fps(self, times: deque) -> float:
        """Calculate FPS from time deltas."""
        if len(times) == 0:
            return 0.0
        avg_time = sum(times) / len(times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_gpu_stats(self) -> tuple:
        """
        Get GPU usage and memory statistics.
        
        Returns:
            Tuple of (usage_percent, memory_used_mb, memory_total_mb)
        """
        if not self.enable_gpu or self.gpu_handle is None:
            return 0.0, 0.0, 0.0
        
        try:
            # GPU utilization
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_usage = utilization.gpu
            
            # Memory info
            memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
            memory_total = memory_info.total / (1024 ** 2)
            
            return gpu_usage, memory_used, memory_total
        except Exception as e:
            print(f"Warning: Failed to get GPU stats: {e}")
            return 0.0, 0.0, 0.0
    
    def get_stats(self, max_inference_time_ms: float = 15.0) -> PerformanceStats:
        """
        Get current performance statistics.
        
        Args:
            max_inference_time_ms: Maximum acceptable inference time in ms
            
        Returns:
            PerformanceStats object
        """
        stats = PerformanceStats()
        
        # Calculate FPS for each stage
        stats.capture_fps = self._calculate_fps(self.capture_times)
        stats.tracking_fps = self._calculate_fps(self.tracking_times)
        stats.inference_fps = self._calculate_fps(self.inference_times)
        stats.output_fps = self._calculate_fps(self.output_times)
        stats.total_fps = self._calculate_fps(self.total_times)
        
        # GPU stats
        stats.gpu_usage, stats.gpu_memory_used, stats.gpu_memory_total = self.get_gpu_stats()
        
        # CPU and RAM (non-blocking CPU percent)
        stats.cpu_usage = psutil.cpu_percent(interval=None)
        stats.ram_usage = psutil.virtual_memory().percent
        
        # Inference time
        if len(self.inference_times) > 0:
            stats.inference_time_ms = (sum(self.inference_times) / len(self.inference_times)) * 1000
        
        # Generate warnings
        stats.warnings = []
        if stats.inference_time_ms > max_inference_time_ms:
            stats.warnings.append(f"Inference time ({stats.inference_time_ms:.1f}ms) exceeds target ({max_inference_time_ms}ms)")
        
        if stats.gpu_memory_used > stats.gpu_memory_total * 0.9:
            stats.warnings.append(f"GPU memory usage high: {stats.gpu_memory_used:.0f}MB / {stats.gpu_memory_total:.0f}MB")
        
        if stats.total_fps < 30:
            stats.warnings.append(f"Low FPS: {stats.total_fps:.1f}")
        
        return stats
    
    def print_stats(self, stats: Optional[PerformanceStats] = None) -> None:
        """
        Print performance statistics to console.
        
        Args:
            stats: Stats to print. If None, gets current stats.
        """
        if stats is None:
            stats = self.get_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        print(f"FPS:")
        print(f"  Capture:   {stats.capture_fps:6.1f}")
        print(f"  Tracking:  {stats.tracking_fps:6.1f}")
        print(f"  Inference: {stats.inference_fps:6.1f}")
        print(f"  Output:    {stats.output_fps:6.1f}")
        print(f"  Total:     {stats.total_fps:6.1f}")
        print(f"\nInference Time: {stats.inference_time_ms:.1f}ms")
        
        if self.enable_gpu:
            print(f"\nGPU:")
            print(f"  Usage:     {stats.gpu_usage:.1f}%")
            print(f"  Memory:    {stats.gpu_memory_used:.0f}MB / {stats.gpu_memory_total:.0f}MB")
        
        print(f"\nSystem:")
        print(f"  CPU:       {stats.cpu_usage:.1f}%")
        print(f"  RAM:       {stats.ram_usage:.1f}%")
        
        if stats.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in stats.warnings:
                print(f"  - {warning}")
        
        print("="*60)
    
    def cleanup(self) -> None:
        """Cleanup GPU monitoring resources.

        This method is safe to call multiple times. For deterministic cleanup,
        prefer calling this method explicitly or using the context manager
        protocol (``with PerformanceMonitor(...) as monitor: ...``).
        """
        if self.enable_gpu and self.gpu_handle is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                # Best-effort cleanup; ignore errors during shutdown
                pass
            finally:
                # Mark GPU monitoring as cleaned up to make this idempotent
                self.gpu_handle = None
                self.enable_gpu = False

    def __enter__(self) -> "PerformanceMonitor":
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context and clean up resources."""
        self.cleanup()
    
    def __del__(self):
        """Destructor to cleanup resources (best-effort, non-deterministic)."""
        try:
            self.cleanup()
        except Exception:
            # Avoid raising exceptions during garbage collection
            pass
