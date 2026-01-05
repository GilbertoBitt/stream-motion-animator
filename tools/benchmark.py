#!/usr/bin/env python3
"""
Benchmark tool for Stream Motion Animator.

Tests performance with different configurations.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from performance_monitor import PerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Benchmark:
    """Performance benchmarking tool."""
    
    def __init__(self, config_path: Optional[str] = None, duration: int = 30):
        """
        Initialize benchmark.
        
        Args:
            config_path: Path to config file
            duration: Benchmark duration in seconds
        """
        self.config = load_config(config_path)
        self.duration = duration
        self.monitor = PerformanceMonitor(
            window_size=30,
            enable_gpu=(self.config.device == 'cuda')
        )
    
    def run_system_info(self) -> dict:
        """Get system information."""
        import platform
        import psutil
        
        info = {
            'platform': platform.system(),
            'python': platform.python_version(),
            'cpu': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'ram': f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        }
        
        # GPU info
        if self.config.device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    info['gpu'] = torch.cuda.get_device_name(0)
                    info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
                else:
                    info['gpu'] = 'CUDA not available'
            except ImportError:
                info['gpu'] = 'PyTorch not installed'
        else:
            info['gpu'] = 'CPU mode'
        
        return info
    
    def print_system_info(self, info: dict) -> None:
        """Print system information."""
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        for key, value in info.items():
            print(f"{key.upper():15s}: {value}")
        print("="*60)
    
    def run_mock_benchmark(self) -> dict:
        """Run a mock benchmark (without full model)."""
        print("\n" + "="*60)
        print("RUNNING MOCK BENCHMARK")
        print("="*60)
        print(f"Duration: {self.duration} seconds")
        print(f"Configuration: {self.config.model_type}, FP16={self.config.use_fp16}")
        print("="*60)
        
        start_time = time.time()
        frame_count = 0
        
        # Simulate pipeline
        while time.time() - start_time < self.duration:
            # Simulate different stages
            self.monitor.tick_capture()
            time.sleep(0.001)  # ~1ms capture
            
            self.monitor.tick_tracking()
            time.sleep(0.005)  # ~5ms tracking
            
            self.monitor.tick_inference()
            time.sleep(0.015)  # ~15ms inference
            
            self.monitor.tick_output()
            time.sleep(0.001)  # ~1ms output
            
            self.monitor.tick_total()
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {elapsed:.1f}s / {self.duration}s - "
                      f"Frames: {frame_count} - "
                      f"FPS: {frame_count / elapsed:.1f}")
        
        # Get final stats
        stats = self.monitor.get_stats()
        
        # Calculate results
        results = {
            'duration': time.time() - start_time,
            'frame_count': frame_count,
            'avg_fps': frame_count / (time.time() - start_time),
            'capture_fps': stats.capture_fps,
            'tracking_fps': stats.tracking_fps,
            'inference_fps': stats.inference_fps,
            'output_fps': stats.output_fps,
            'total_fps': stats.total_fps,
            'inference_time_ms': stats.inference_time_ms,
            'gpu_usage': stats.gpu_usage,
            'gpu_memory_used': stats.gpu_memory_used,
            'gpu_memory_total': stats.gpu_memory_total,
            'cpu_usage': stats.cpu_usage,
            'ram_usage': stats.ram_usage
        }
        
        return results
    
    def print_results(self, results: dict) -> None:
        """Print benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Duration:       {results['duration']:.1f}s")
        print(f"Total Frames:   {results['frame_count']}")
        print(f"Average FPS:    {results['avg_fps']:.1f}")
        print()
        print("Pipeline Breakdown:")
        print(f"  Capture:      {results['capture_fps']:6.1f} FPS")
        print(f"  Tracking:     {results['tracking_fps']:6.1f} FPS")
        print(f"  Inference:    {results['inference_fps']:6.1f} FPS  ← Bottleneck")
        print(f"  Output:       {results['output_fps']:6.1f} FPS")
        print(f"  Total:        {results['total_fps']:6.1f} FPS")
        print()
        print(f"Inference Time: {results['inference_time_ms']:.1f}ms")
        
        if results['gpu_usage'] > 0:
            print()
            print("GPU Statistics:")
            print(f"  Usage:        {results['gpu_usage']:.1f}%")
            print(f"  Memory:       {results['gpu_memory_used']:.0f} MB / {results['gpu_memory_total']:.0f} MB")
        
        print()
        print("System:")
        print(f"  CPU:          {results['cpu_usage']:.1f}%")
        print(f"  RAM:          {results['ram_usage']:.1f}%")
        print("="*60)
    
    def generate_recommendations(self, results: dict) -> list:
        """Generate performance recommendations."""
        recommendations = []
        
        target_fps = self.config.target_fps
        
        if results['total_fps'] >= target_fps:
            recommendations.append(f"✅ Target achieved! ({results['total_fps']:.1f} FPS)")
        elif results['total_fps'] >= target_fps * 0.9:
            recommendations.append(f"⚠️  Close to target ({results['total_fps']:.1f} / {target_fps} FPS)")
        else:
            recommendations.append(f"❌ Below target ({results['total_fps']:.1f} / {target_fps} FPS)")
        
        # Inference bottleneck
        if results['inference_fps'] < results['tracking_fps'] * 0.8:
            recommendations.append("- Inference is the bottleneck")
            if not self.config.use_fp16:
                recommendations.append("  → Enable FP16 for 2x speed boost")
            if not self.config.get('ai_model.use_tensorrt', False):
                recommendations.append("  → Consider enabling TensorRT for 10-15% speedup")
        
        # GPU usage
        if results['gpu_usage'] > 0:
            if results['gpu_usage'] < 50:
                recommendations.append("- Low GPU usage - CPU may be bottleneck")
                recommendations.append("  → Check if async_pipeline is enabled")
            elif results['gpu_usage'] > 95:
                recommendations.append("- GPU fully saturated")
                recommendations.append("  → Consider reducing resolution or using frame skipping")
            
            # Memory
            memory_percent = (results['gpu_memory_used'] / results['gpu_memory_total']) * 100
            if memory_percent > 90:
                recommendations.append("- GPU memory usage high")
                recommendations.append("  → Reduce batch size or preloaded characters")
        
        # CPU usage
        if results['cpu_usage'] > 80:
            recommendations.append("- High CPU usage")
            recommendations.append("  → Close background applications")
        
        return recommendations
    
    def print_recommendations(self, recommendations: list) -> None:
        """Print recommendations."""
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        for rec in recommendations:
            print(rec)
        print("="*60)
    
    def run(self) -> None:
        """Run complete benchmark."""
        # System info
        sys_info = self.run_system_info()
        self.print_system_info(sys_info)
        
        # Run benchmark
        results = self.run_mock_benchmark()
        self.print_results(results)
        
        # Recommendations
        recommendations = self.generate_recommendations(results)
        self.print_recommendations(recommendations)
        
        print("\n✅ Benchmark complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Stream Motion Animator performance"
    )
    parser.add_argument(
        '--config',
        '-c',
        help='Path to config file'
    )
    parser.add_argument(
        '--duration',
        '-d',
        type=int,
        default=30,
        help='Benchmark duration in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    benchmark = Benchmark(args.config, args.duration)
    benchmark.run()


if __name__ == "__main__":
    main()
