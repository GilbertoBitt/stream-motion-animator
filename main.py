#!/usr/bin/env python3
"""
AI-Driven Live Portrait Motion Animator
Main entry point for the application
"""
import sys
import argparse
from pathlib import Path

from src.utils import Config, setup_logger
from src.pipeline import AnimatorPipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AI-Driven Live Portrait Motion Animator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single image
  python main.py --image portrait.jpg
  
  # With multiple images for switching
  python main.py --image img1.jpg img2.jpg img3.jpg
  
  # Custom configuration
  python main.py --image portrait.jpg --config config_rtx3080.yaml
  
  # Enable multiple outputs
  python main.py --image portrait.jpg --output display spout ndi
  
  # Optimize for performance
  python main.py --image portrait.jpg --gpu-id 0 --batch-size 4 --target-fps 60

Controls:
  1-9        : Switch between loaded source images
  L          : Load new image (not implemented in CLI mode)
  F          : Toggle FPS display
  M          : Toggle landmarks display
  P          : Pause/Resume
  S          : Save current frame
  R          : Print performance report
  Q or ESC   : Quit
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--image', '-i',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to source portrait image(s)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Video capture options
    parser.add_argument(
        '--camera',
        type=int,
        help='Camera device ID (overrides config)'
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='Capture resolution (overrides config)'
    )
    
    # GPU options
    parser.add_argument(
        '--gpu-id',
        type=int,
        help='GPU device ID (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for inference (overrides config)'
    )
    
    # Performance options
    parser.add_argument(
        '--target-fps',
        type=int,
        help='Target FPS (overrides config)'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision (overrides config)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        nargs='+',
        choices=['display', 'spout', 'ndi'],
        help='Output types (overrides config)'
    )
    
    parser.add_argument(
        '--spout-name',
        type=str,
        help='Spout sender name (overrides config)'
    )
    
    parser.add_argument(
        '--ndi-name',
        type=str,
        help='NDI stream name (overrides config)'
    )
    
    # Logging and metrics
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--enable-metrics',
        action='store_true',
        help='Enable performance metrics (overrides config)'
    )
    
    parser.add_argument(
        '--disable-metrics',
        action='store_true',
        help='Disable performance metrics (overrides config)'
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    """
    Apply command-line argument overrides to configuration
    
    Args:
        config: Configuration object
        args: Parsed command-line arguments
    """
    # Camera options
    if args.camera is not None:
        config.set('capture.device_id', args.camera)
    
    if args.resolution is not None:
        config.set('capture.width', args.resolution[0])
        config.set('capture.height', args.resolution[1])
    
    # GPU options
    if args.gpu_id is not None:
        config.set('animation.gpu_id', args.gpu_id)
    
    if args.batch_size is not None:
        config.set('animation.batch_size', args.batch_size)
    
    # Performance options
    if args.target_fps is not None:
        config.set('performance.target_fps', args.target_fps)
    
    if args.fp16:
        config.set('animation.use_half_precision', True)
    
    # Output options
    if args.output is not None:
        config.set('output.enabled', args.output)
    
    if args.spout_name is not None:
        config.set('output.spout.sender_name', args.spout_name)
    
    if args.ndi_name is not None:
        config.set('output.ndi.stream_name', args.ndi_name)
    
    # Metrics
    if args.enable_metrics:
        config.set('metrics.enabled', True)
    elif args.disable_metrics:
        config.set('metrics.enabled', False)


def validate_images(image_paths: list) -> list:
    """
    Validate that image paths exist
    
    Args:
        image_paths: List of image paths
        
    Returns:
        List of valid image paths
        
    Raises:
        FileNotFoundError: If no valid images found
    """
    valid_paths = []
    
    for path in image_paths:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            valid_paths.append(str(path_obj.absolute()))
        else:
            print(f"Warning: Image not found: {path}")
    
    if not valid_paths:
        raise FileNotFoundError("No valid source images provided")
    
    return valid_paths


def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(
        name='animator',
        level=args.log_level,
        log_file='logs/animator.log',
        use_colors=True
    )
    
    logger.info("="*60)
    logger.info("AI-Driven Live Portrait Motion Animator")
    logger.info("="*60)
    
    try:
        # Validate images
        logger.info("Validating source images...")
        image_paths = validate_images(args.image)
        logger.info(f"Found {len(image_paths)} valid image(s)")
        
        # Load configuration
        logger.info("Loading configuration...")
        config = Config(args.config)
        
        # Apply CLI overrides
        apply_cli_overrides(config, args)
        
        # Log configuration summary
        logger.info("Configuration:")
        logger.info(f"  Capture: {config.get('capture.width')}x{config.get('capture.height')} @ {config.get('capture.fps')}fps")
        logger.info(f"  GPU ID: {config.get('animation.gpu_id')}")
        logger.info(f"  Batch Size: {config.get('animation.batch_size')}")
        logger.info(f"  Target FPS: {config.get('performance.target_fps')}")
        logger.info(f"  Outputs: {', '.join(config.get('output.enabled'))}")
        
        # Create and initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = AnimatorPipeline(config)
        
        if not pipeline.start():
            logger.error("Failed to start pipeline")
            return 1
        
        # Load source images
        logger.info("Loading source images...")
        if not pipeline.load_source_images(image_paths):
            logger.error("Failed to load source images")
            pipeline.stop()
            return 1
        
        # Run main loop
        logger.info("Starting animation...")
        pipeline.run()
        
        # Cleanup
        logger.info("Shutting down...")
        pipeline.stop()
        
        logger.info("Goodbye!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
