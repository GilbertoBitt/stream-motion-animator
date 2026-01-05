#!/usr/bin/env python3
"""
Performance benchmarking script for the AI animator
"""
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config, setup_logger
from src.pipeline import AnimatorPipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Benchmark AI Motion Animator Performance'
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to test image'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=30,
        help='Benchmark duration in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--auto-tune',
        action='store_true',
        help='Test different batch sizes to find optimal settings'
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='Batch sizes to test in auto-tune mode'
    )
    
    return parser.parse_args()


def run_benchmark(config: Config, image_path: str, duration: int) -> dict:
    """
    Run benchmark test
    
    Args:
        config: Configuration
        image_path: Path to test image
        duration: Test duration in seconds
        
    Returns:
        Performance statistics
    """
    logger = setup_logger('benchmark', level='INFO')
    
    logger.info(f"Running benchmark for {duration} seconds...")
    logger.info(f"Configuration: Batch size={config.get('animation.batch_size')}, GPU={config.get('animation.gpu_id')}")
    
    # Initialize pipeline
    pipeline = AnimatorPipeline(config)
    
    if not pipeline.start():
        logger.error("Failed to start pipeline")
        return None
    
    if not pipeline.load_source_images([image_path]):
        logger.error("Failed to load image")
        pipeline.stop()
        return None
    
    # Run for specified duration
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            pipeline.metrics.start_frame()
            frame = pipeline.process_frame()
            
            if frame is not None:
                frame_count += 1
            
            # Small delay to prevent spinning
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted")
    
    finally:
        # Get results
        summary = pipeline.metrics.get_summary()
        pipeline.stop()
    
    logger.info(f"Benchmark complete: {frame_count} frames processed")
    
    return summary


def auto_tune(config: Config, image_path: str, batch_sizes: list) -> None:
    """
    Test different batch sizes to find optimal settings
    
    Args:
        config: Base configuration
        image_path: Path to test image
        batch_sizes: List of batch sizes to test
    """
    logger = setup_logger('benchmark', level='INFO')
    
    logger.info("="*60)
    logger.info("AUTO-TUNE MODE")
    logger.info("="*60)
    logger.info(f"Testing batch sizes: {batch_sizes}")
    
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"\nTesting batch size: {batch_size}")
        
        # Update config
        test_config = Config()
        test_config.config = config.config.copy()
        test_config.set('animation.batch_size', batch_size)
        test_config.set('output.enabled', ['display'])  # Only display output for testing
        
        # Run benchmark
        summary = run_benchmark(test_config, image_path, duration=15)
        
        if summary:
            results.append({
                'batch_size': batch_size,
                'fps': summary['fps'],
                'latency_ms': summary['latency_ms'],
                'gpu_util': summary['gpu'].get('utilization', 0),
                'gpu_memory_mb': summary['gpu'].get('memory_used', 0)
            })
            
            logger.info(f"Results: FPS={summary['fps']:.1f}, Latency={summary['latency_ms']:.1f}ms")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("AUTO-TUNE RESULTS")
    logger.info("="*60)
    
    if results:
        logger.info(f"{'Batch Size':<12} {'FPS':<8} {'Latency':<12} {'GPU Util':<12} {'GPU Mem':<12}")
        logger.info("-"*60)
        
        best_fps = 0
        best_batch = 1
        
        for result in results:
            logger.info(
                f"{result['batch_size']:<12} "
                f"{result['fps']:<8.1f} "
                f"{result['latency_ms']:<12.1f} "
                f"{result['gpu_util']:<12.1f} "
                f"{result['gpu_memory_mb']:<12.0f}"
            )
            
            if result['fps'] > best_fps:
                best_fps = result['fps']
                best_batch = result['batch_size']
        
        logger.info("-"*60)
        logger.info(f"\nRECOMMENDED: Batch size = {best_batch} (achieved {best_fps:.1f} FPS)")
    else:
        logger.error("No valid results obtained")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    logger = setup_logger('benchmark', level='INFO')
    
    logger.info("AI Motion Animator - Performance Benchmark")
    
    # Validate image
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1
    
    # Load configuration
    config = Config(args.config)
    
    # Run benchmarks
    if args.auto_tune:
        auto_tune(config, str(image_path), args.batch_sizes)
    else:
        summary = run_benchmark(config, str(image_path), args.duration)
        
        if summary:
            logger.info("\n" + "="*60)
            logger.info("BENCHMARK RESULTS")
            logger.info("="*60)
            logger.info(f"FPS: {summary['fps']:.1f}")
            logger.info(f"Latency: {summary['latency_ms']:.1f}ms")
            logger.info(f"Total Frames: {summary['total_frames']}")
            logger.info(f"Dropped Frames: {summary['dropped_frames']}")
            
            stage_times = summary['stage_times_ms']
            logger.info(f"\nStage Timings:")
            for stage, time_ms in stage_times.items():
                logger.info(f"  {stage}: {time_ms:.2f}ms")
            
            gpu = summary['gpu']
            logger.info(f"\nGPU:")
            logger.info(f"  Utilization: {gpu.get('utilization', 0):.1f}%")
            logger.info(f"  Memory: {gpu.get('memory_used', 0):.0f}MB / {gpu.get('memory_total', 0):.0f}MB")
            
            cpu = summary['cpu']
            logger.info(f"\nCPU:")
            logger.info(f"  Process: {cpu['cpu_percent']:.1f}%")
            logger.info(f"  Memory: {cpu['memory_mb']:.0f}MB")
            logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
