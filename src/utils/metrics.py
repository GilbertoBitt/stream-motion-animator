"""
Performance metrics tracking
"""
import time
import psutil
from typing import Dict, List, Optional
from collections import deque
from pathlib import Path

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class PerformanceMetrics:
    """Track and report performance metrics"""
    
    def __init__(self, window_size: int = 100, enabled: bool = True):
        """
        Initialize metrics tracker
        
        Args:
            window_size: Number of samples for rolling average
            enabled: Whether metrics tracking is enabled
        """
        self.enabled = enabled
        self.window_size = window_size
        
        # Frame timing
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        
        # Processing stage timings
        self.stage_times = {
            'capture': deque(maxlen=window_size),
            'tracking': deque(maxlen=window_size),
            'animation': deque(maxlen=window_size),
            'output': deque(maxlen=window_size),
        }
        
        # Counters
        self.total_frames = 0
        self.dropped_frames = 0
        
        # Start time
        self.start_time = time.time()
        
        # System resources
        self.process = psutil.Process()
    
    def start_frame(self) -> None:
        """Mark the start of a new frame"""
        if not self.enabled:
            return
        
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        self.total_frames += 1
    
    def record_stage(self, stage: str, duration: float) -> None:
        """
        Record timing for a processing stage
        
        Args:
            stage: Stage name (capture, tracking, animation, output)
            duration: Duration in seconds
        """
        if not self.enabled or stage not in self.stage_times:
            return
        self.stage_times[stage].append(duration)
    
    def record_dropped_frame(self) -> None:
        """Record a dropped frame"""
        if not self.enabled:
            return
        self.dropped_frames += 1
    
    def get_fps(self) -> float:
        """Get current FPS (rolling average)"""
        if not self.enabled or len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_latency(self) -> float:
        """Get current frame latency in milliseconds"""
        if not self.enabled or len(self.frame_times) == 0:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times) * 1000
    
    def get_stage_times(self) -> Dict[str, float]:
        """Get average processing time for each stage in milliseconds"""
        result = {}
        for stage, times in self.stage_times.items():
            if len(times) > 0:
                result[stage] = (sum(times) / len(times)) * 1000
            else:
                result[stage] = 0.0
        return result
    
    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU utilization and memory statistics"""
        if not GPUTIL_AVAILABLE:
            return {'utilization': 0.0, 'memory_used': 0.0, 'memory_total': 0.0}
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0.0,
                    'temperature': gpu.temperature
                }
        except Exception:
            pass
        
        return {'utilization': 0.0, 'memory_used': 0.0, 'memory_total': 0.0}
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU and RAM statistics"""
        try:
            return {
                'cpu_percent': self.process.cpu_percent(),
                'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'system_memory_percent': psutil.virtual_memory().percent
            }
        except Exception:
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'system_memory_percent': 0.0}
    
    def get_summary(self) -> Dict:
        """Get complete metrics summary"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'fps': self.get_fps(),
            'latency_ms': self.get_latency(),
            'stage_times_ms': self.get_stage_times(),
            'gpu': self.get_gpu_stats(),
            'cpu': self.get_cpu_stats()
        }
    
    def print_summary(self) -> None:
        """Print metrics summary to console"""
        if not self.enabled:
            return
        
        summary = self.get_summary()
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"Uptime: {summary['uptime_seconds']:.1f}s")
        print(f"FPS: {summary['fps']:.1f}")
        print(f"Latency: {summary['latency_ms']:.1f}ms")
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Dropped Frames: {summary['dropped_frames']}")
        print("\nStage Timings:")
        for stage, time_ms in summary['stage_times_ms'].items():
            print(f"  {stage}: {time_ms:.2f}ms")
        
        if GPUTIL_AVAILABLE:
            gpu = summary['gpu']
            print(f"\nGPU:")
            print(f"  Utilization: {gpu['utilization']:.1f}%")
            print(f"  Memory: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB ({gpu.get('memory_percent', 0):.1f}%)")
        
        cpu = summary['cpu']
        print(f"\nCPU:")
        print(f"  Process: {cpu['cpu_percent']:.1f}%")
        print(f"  Memory: {cpu['memory_mb']:.0f}MB")
        print("="*60 + "\n")
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save current metrics to CSV file
        
        Args:
            filepath: Path to CSV file
        """
        if not self.enabled:
            return
        
        import csv
        from datetime import datetime
        
        summary = self.get_summary()
        
        # Create file with headers if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = path.exists()
        
        with open(path, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'uptime', 'fps', 'latency_ms', 'total_frames', 
                'dropped_frames', 'gpu_util', 'gpu_memory_mb', 'cpu_percent'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'uptime': f"{summary['uptime_seconds']:.1f}",
                'fps': f"{summary['fps']:.1f}",
                'latency_ms': f"{summary['latency_ms']:.1f}",
                'total_frames': summary['total_frames'],
                'dropped_frames': summary['dropped_frames'],
                'gpu_util': f"{summary['gpu'].get('utilization', 0):.1f}",
                'gpu_memory_mb': f"{summary['gpu'].get('memory_used', 0):.0f}",
                'cpu_percent': f"{summary['cpu']['cpu_percent']:.1f}"
            })
